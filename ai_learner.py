import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple
import signal
import hashlib
import getopt
import sys
from rubiks import Cube, face_relations

# Number of squares along each edge of the Cube.
edge_length = 2

# Number of layers to use in the NN.
num_layers = 2

# Starting attempt's random seed. Will be used if != None.
root_seed = None

# If True, attempts to learn through reinforcement learning.
# Otherwise, random actions are taken.
flag_deliberate_attempt = True

# Tracks used seeds and solution statistics.
attempt_seeds = []
solved_stats = []

# If True, discard current Cube and create a new one at iter (term_iter).
flag_term_point = True
term_iter = 600000

# Used to determine the number of identical past states.
cube_hash_dict = {}
cube_hash_memory_size = 500
# Tracks last (cube_hash_memory_size) hashes, so as to decrement/delete old entries in cube_hash_dict.
cube_hash_memory = None

# Extra randomness is introduced when duplicate states are encountered.
extra_rand = 0.
# Max dynamic randomness inducable by periodic evaluation.
extra_rand_max = 0.35
# Used as an exponent in extra rand's calculation.
# Larger values requires more duplicates to approach extra_rand_max,
# steepening the curve and pushing it rightward.
extra_rand_scaling_power = 0.8

# Used in ReplayMemory class.
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# Parameters specific to the AI.
BATCH_SIZE = 1024
GAMMA = 0.999
EPS_START = 0.975
EPS_END = 0.15
EPS_DECAY = 100000
TARGET_UPDATE = BATCH_SIZE * 100
TRANSITION_MEMORY_SIZE = 10000
total_iterations = 0


def exit_gracefully(signum, frame):
	"""
	More elegant handling of keyboard interrupts.
	Enter 'Ctrl+C', then 'y' to terminate early.
	"""
	global flag_continue_attempt
	global flag_continue_main
	global time_start
	global best_cube
	global max_correct

	signal.signal(signal.SIGINT, original_sigint)
	time_paused = time.time()
	print('\n\n---------- PAUSING ----------')
	try:
		if 'best_cube' in globals():
			show_best_cube_statistics(best_cube, max_correct)

		if len(solved_stats) > 0:
			show_solved_stats()

		input_text = input('\nReally Quit? (y/n)> ').lower().lstrip()
		if input_text != '' and input_text[0] == 'y':
			print('Exiting.')
			flag_continue_main = False
			flag_continue_attempt = False
	
	except KeyboardInterrupt:
		print('\nExiting.')
		flag_continue_main = False
		flag_continue_attempt = False
	
	print('\n---------- RESUMING ----------')
	if 'time_start' in globals():
		time_start += time.time() - time_paused
	signal.signal(signal.SIGINT, exit_gracefully)


original_sigint = signal.getsignal(signal.SIGINT)
signal.signal(signal.SIGINT, exit_gracefully)


class NN(nn.Module):
	""" Simple NN model. """
	def __init__(self):
		super(NN, self).__init__()
		num_squares = 6 * edge_length * edge_length
		num_states = 6 * num_squares
		num_outputs = 6 * 2

		self.drop1 = nn.AlphaDropout(0.01, True)

		if num_layers == 1:
			self.lin1 = nn.Linear(num_states, num_outputs)
		elif num_layers == 2:
			self.lin1 = nn.Linear(num_states, 2 * num_outputs)
			self.lin2 = nn.Linear(2 * num_outputs, num_outputs)
		elif num_layers == 3:
			self.lin1 = nn.Linear(num_states, num_squares)
			self.lin2 = nn.Linear(num_squares, 2 * num_outputs)
			self.lin3 = nn.Linear(2 * num_outputs, num_outputs)
		else:
			print('Error: Invalid number of layers! (Choose 1 -> 3)')
			quit()

	def forward(self, x):
		if num_layers >= 1:
			x = self.lin1(x)
		if num_layers >= 2:
			x = self.drop1(x)
			x = self.lin2(x)
		if num_layers >= 3:
			x = self.lin3(x)
		return x


class ReplayMemory(object):
	""" Memory of past state transitions and their associated rewards. """
	def __init__(self, capacity):
		self.capacity = capacity
		self.memory = []
		self.position = 0

	def push(self, *args):
		"""Saves a transition."""
		if len(self.memory) < self.capacity:
			self.memory.append(None)
		self.memory[self.position] = Transition(*args)
		self.position = (self.position + 1) % self.capacity

	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)

	def __len__(self):
		return len(self.memory)


def select_action(state):
	""" Generate an action, depending on the current state. """
	global total_iterations
	global eps_threshold
	global extra_rand

	sample = random.random()
	eps_threshold = EPS_END + (EPS_START - EPS_END) * \
		np.exp(-1. * total_iterations / EPS_DECAY)

	if att_iter >= cube_hash_memory_size:
		# Update extra_rand once in a while.
		if att_iter % (cube_hash_memory_size/20) == 0:
			extra_rand = update_extra_rand()
		eps_threshold = max(eps_threshold, extra_rand)

	total_iterations += 1
	if sample > eps_threshold:
		with torch.no_grad():
			return np.argmax(policy_net(state)).view(1)
	else:
		return torch.randint(low=0, high=6*2, size=(1,), dtype=torch.long)


def update_extra_rand():
	"""
	Determine how much extra randomness should be introduced,
	depending on the number of identical recent states.
	Introduces considerable cost when >> 1000 dict elements.
	"""
	# Using PyTorch takes slightly longer for small dictionaries (1000 elems).
	# cube_hash_dict_values = torch.tensor(list(cube_hash_dict.values()))
	# num_duplicates = torch.sum(cube_hash_dict_values > 1)

	num_duplicates = sum([x - 1 for x in cube_hash_dict.values()])
	extra_rand = (num_duplicates ** extra_rand_scaling_power) / (cube_hash_memory_size ** extra_rand_scaling_power)
	extra_rand *= extra_rand_max

	return extra_rand


def clear_cube_hashes():
	""" Clear the past cube state memory. """
	global cube_hash_memory
	global cube_hash_dict

	cube_hash_memory = [0] * cube_hash_memory_size
	cube_hash_dict = {}


def optimize_model():
	""" Update the model. """
	if len(memory) < BATCH_SIZE:
		return
	transitions = memory.sample(BATCH_SIZE)
	batch = Transition(*zip(*transitions))

	state_batch = torch.cat(batch.state).reshape(-1, 6 * 6 * edge_length * edge_length)
	action_batch = torch.cat(batch.action)
	reward_batch = torch.cat(batch.reward)

	# Compute Q(s_t, a):
	# Computes Q(s_t), then selects the columns of actions taken.
	state_action_values = policy_net(state_batch).gather(1, action_batch.unsqueeze(1))

	# Compute V(s_{t+1}) for all next states.
	next_state_values = target_net(state_batch).max(1)[0].detach()
	# Compute the expected Q values.
	expected_state_action_values = (next_state_values * GAMMA) + reward_batch

	# Compute Huber loss.
	loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

	# Optimize the model.
	optimizer.zero_grad()
	loss.backward()
	for param in policy_net.parameters():
		param.grad.data.clamp_(-1, 1)
	optimizer.step()


def reward_function(cube):
	""" Calculates a cube's reward, which is a scalar measurement of how good it is. """
	total_correct = 0.
	reward = 0.

	# Check corner correctness.
	total_correct_corners = 0
	total_correct_corners += check_corners(cube.faces)
	total_correct_corners *= 3
	total_correct += total_correct_corners

	if cube.edge_length >= 3:
		# Check edge correctness.
		total_correct_edges = 0
		total_correct_edges += check_edges(cube.faces, cube.edge_length)
		total_correct_edges *= 2
		total_correct += total_correct_edges

		# Check inner correctness.
		total_correct_inner = check_inner(cube.faces, cube.edge_length)
		total_correct += total_correct_inner

	reward += total_correct

	for face_idx,face in enumerate(cube.faces):
		# Provides up to 1 extra point for each face, depending on colour correctness.
		num_correct_colours = np.sum(face.flatten() == face_idx)
		reward += num_correct_colours / (cube.edge_length * cube.edge_length)

	return total_correct, reward


def check_inner(faces, edge_length):
	"""
	Returns the number of inner blocks of a cube that are correct.
	"""
	num_correct_inner = 0

	for face_idx in range(6):
		num_correct_inner += np.sum(faces[face_idx, 1:-1, 1:-1] == face_idx)
	return num_correct_inner


def check_edges(faces, edge_length):
	"""
	Returns the number of edge blocks of a cube that are entirely correct.
	"""
	num_correct_edges = 0

	for face_idx in [0, 5]:
		# Face above.
		u_face_colour = face_relations['_'.join([str(face_idx), 'u'])]
		u_face = faces[u_face_colour]

		# Face to the left.
		l_face_colour = face_relations['_'.join([str(face_idx), 'l'])]
		l_face = faces[l_face_colour]

		# Face below.
		d_face_colour = face_relations['_'.join([str(face_idx), 'd'])]
		d_face = faces[d_face_colour]

		# Face to the right.
		r_face_colour = face_relations['_'.join([str(face_idx), 'r'])]
		r_face = faces[r_face_colour]

		if face_idx == 0:
			for edge_idx in range(1, edge_length - 1):
				# Upper edge
				if faces[face_idx, 0, edge_idx] == face_idx:
					u_face_edge = u_face[-1, edge_idx]
					if u_face_edge == u_face_colour:
						num_correct_edges += 1

				# Left edge
				if faces[face_idx, edge_idx, 0] == face_idx:
					l_face_edge = l_face[edge_idx, -1]
					if l_face_edge == l_face_colour:
						num_correct_edges += 1

				# Bottom edge
				if faces[face_idx, -1, edge_idx] == face_idx:
					d_face_edge = d_face[0, edge_idx]
					if d_face_edge == d_face_colour:
						num_correct_edges += 1

				# Right edge
				if faces[face_idx, edge_idx, -1] == face_idx:
					r_face_edge = r_face[edge_idx, 0]
					if r_face_edge == r_face_colour:
						num_correct_edges += 1

		if face_idx == 5:
			for edge_idx in range(1, edge_length - 1):
				# Upper edge
				if faces[face_idx, 0, edge_idx] == face_idx:
					u_face_edge = u_face[0, -1-edge_idx]
					if u_face_edge == u_face_colour:
						num_correct_edges += 1

				# Left edge
				if faces[face_idx, edge_idx, 0] == face_idx:
					l_face_edge = l_face[edge_idx, -1]
					if l_face_edge == l_face_colour:
						num_correct_edges += 1

				# Bottom edge
				if faces[face_idx, -1, edge_idx] == face_idx:
					d_face_edge = d_face[-1, -1-edge_idx]
					if d_face_edge == d_face_colour:
						num_correct_edges += 1

				# Right edge
				if faces[face_idx, edge_idx, -1] == face_idx:
					r_face_edge = r_face[edge_idx, -1]
					if r_face_edge == r_face_colour:
						num_correct_edges += 1

	return num_correct_edges


def check_corners(faces):
	"""
	Returns the number of corner blocks of a cube that are entirely correct.
	"""
	num_correct_corners = 0

	for face_idx in [0, 5]:
		# Face above.
		u_face_colour = face_relations['_'.join([str(face_idx), 'u'])]
		u_face = faces[u_face_colour]

		# Face to the left.
		l_face_colour = face_relations['_'.join([str(face_idx), 'l'])]
		l_face = faces[l_face_colour]

		# Face below.
		d_face_colour = face_relations['_'.join([str(face_idx), 'd'])]
		d_face = faces[d_face_colour]

		# Face to the right.
		r_face_colour = face_relations['_'.join([str(face_idx), 'r'])]
		r_face = faces[r_face_colour]

		if face_idx == 0:
			# Top-left
			if faces[face_idx, 0, 0] == face_idx:
				l_face_corner = l_face[0, -1]
				if l_face_corner == l_face_colour:
					u_face_corner = u_face[-1, 0]
					if u_face_corner == u_face_colour:
						num_correct_corners += 1

			# Top-right
			if faces[face_idx, 0, -1] == face_idx:
				r_face_corner = r_face[0, 0]
				if r_face_corner == r_face_colour:
					u_face_corner = u_face[-1, -1]
					if u_face_corner == u_face_colour:
						num_correct_corners += 1

			# Bottom-left
			if faces[face_idx, -1, 0] == face_idx:
				l_face_corner = l_face[-1, -1]
				if l_face_corner == l_face_colour:
					d_face_corner = d_face[0, 0]
					if d_face_corner == d_face_colour:
						num_correct_corners += 1

			# Bottom-right
			if faces[face_idx, -1, -1] == face_idx:
				r_face_corner = r_face[-1, 0]
				if r_face_corner == r_face_colour:
					d_face_corner = d_face[0, -1]
					if d_face_corner == d_face_colour:
						num_correct_corners += 1

		elif face_idx == 5:
			# Top-left
			if faces[face_idx, 0, 0] == face_idx:
				l_face_corner = l_face[0, -1]
				if l_face_corner == l_face_colour:
					u_face_corner = u_face[0, -1]
					if u_face_corner == u_face_colour:
						num_correct_corners += 1

			# Top-right
			if faces[face_idx, 0, -1] == face_idx:
				r_face_corner = r_face[0, 0]
				if r_face_corner == r_face_colour:
					u_face_corner = u_face[0, 0]
					if u_face_corner == u_face_colour:
						num_correct_corners += 1

			# Bottom-left
			if faces[face_idx, -1, 0] == face_idx:
				l_face_corner = l_face[-1, -1]
				if l_face_corner == l_face_colour:
					d_face_corner = d_face[-1, -1]
					if d_face_corner == d_face_colour:
						num_correct_corners += 1

			# Bottom-right
			if faces[face_idx, -1, -1] == face_idx:
				r_face_corner = r_face[-1, 0]
				if r_face_corner == r_face_colour:
					d_face_corner = d_face[-1, 0]
					if d_face_corner == d_face_colour:
						num_correct_corners += 1

	return num_correct_corners


def show_stats(
		cube, running_stats_length, num_correct, running_num_correct, max_correct,
		reward, running_reward, max_reward, att_iter, time_start):
	""" Format and display the provided cube and statistics. """
	print(cube)
	print('Correct:     (Current: {0:3}, Running: {1:7.3f}, Max: {2:4})'.format(
		int(num_correct), sum(running_num_correct)/running_stats_length, int(max_correct)))
	print('Reward:      (Current: {0:3}, Running: {1:7.3f}, Max: {2:4})'.format(
		int(reward), sum(running_reward)/running_stats_length, int(max_reward)))
	if flag_deliberate_attempt:
		print('Randomness: {0:.2f}%'.format(100 * eps_threshold))
	print('Current Iter: {0}'.format(att_iter))
	print('Current Time: {0} seconds'.format(int(time.time() - time_start)))


def show_solved_stats():
	""" Show the statistics of past solutions. """
	print()
	print('Solved statistics:')
	for idx,stat in enumerate(solved_stats):
		print('Cube {0:>3} (Attempt {1:>3}): Iterations: {2:>7}, Time: {3:>5}, Seed: {4:>10}'.format(
			idx, stat[0], stat[1], int(stat[2]), stat[3]))
	print()


def show_best_cube_statistics(cube, max_correct):
	""" Show the given best cube and its number of correct squares. """
	print('\n\nAttempt seed: {0}'.format(attempt_seeds[-1]))
	print('Best cube ({0} correct blocks):'.format(int(max_correct)))
	print(cube)


def solution_attempt():
	"""
	Attempt to solve a Rubik's Cube.
	Will use an AI if flag_deliberate_attempt == True,
	otherwise will take random actions.
	"""
	global attempt_num
	global time_start
	global best_cube
	global max_correct
	global att_iter
	global cube_hash_dict
	global flag_continue_attempt

	# Dictates continuation of each attempt.
	flag_continue_attempt = True

	cube = Cube(edge_length=edge_length)
	print('\nOriginal cube:')
	print(cube)

	print('\n\nScrambling...')
	cube.scramble()
	
	num_squares = 6 * cube.edge_length * cube.edge_length
	num_states = num_squares * 6
	att_iter = 0
	max_correct = 0
	best_cube = cube.copy()
	max_reward = 0
	time_start = time.time()
	running_stats_length = 1000
	running_num_correct = [0] * running_stats_length
	running_reward = [0] * running_stats_length

	clear_cube_hashes()

	if flag_deliberate_attempt:
		# Used to translate the cube's state into boolean values.
		state_grid = torch.arange(start=0, end=num_states, step=6, dtype=torch.int64)

		# Gather the cube's current state.
		squares = torch.as_tensor(cube.faces.flatten(), dtype=torch.int64)
		state = torch.zeros([num_states], dtype=torch.float32)
		state[state_grid + squares] = 1

	while flag_continue_attempt:
		if flag_deliberate_attempt:
			# Select an action, depending on the current state.
			action = select_action(state)
			face = int(action / 2)
			rotation = action % 2
			# Take the action.
			cube.rotate(face, rotation)

			# Hash the cube's current state.
			cube_hash = hashlib.sha1(cube.faces).digest()

			if att_iter > cube_hash_memory_size:
				# Decrement or delete oldest hash, depending on its number of occurences.
				if cube_hash_dict[cube_hash_memory[att_iter % cube_hash_memory_size]] == 1:
					del cube_hash_dict[cube_hash_memory[att_iter % cube_hash_memory_size]]
				else:
					cube_hash_dict[cube_hash_memory[att_iter % cube_hash_memory_size]] -= 1

			# Overwrite oldest space with newest hash.
			cube_hash_memory[att_iter % cube_hash_memory_size] = cube_hash

			# Increment or create the newest hash in the hash dict.
			if cube_hash in cube_hash_dict:
				cube_hash_dict[cube_hash] += 1
			else:
				cube_hash_dict[cube_hash] = 1

		else:
			# Take a random action.
			cube.random_rotation()

		# Compute the number of correctly-coloured squares and corresponding reward.
		num_correct, reward = reward_function(cube)
		running_num_correct[att_iter % running_stats_length] = num_correct
		running_reward[att_iter % running_stats_length] = reward

		if num_correct > max_correct:
			max_correct = num_correct
			best_cube = cube.copy()

		if num_correct == num_squares:
			print('\n\nSolved!')
			solved_stats.append([attempt_num, att_iter, time.time() - time_start, attempt_seeds[-1]])
			break

		if reward > max_reward:
			max_reward = reward

		if flag_deliberate_attempt:
			# Gather the cube's current state.
			squares = torch.as_tensor(cube.faces.flatten(), dtype=torch.int64)
			next_state = torch.zeros([num_states], dtype=torch.float32)
			next_state[state_grid + squares] = 1
			# Store the transition in memory
			memory.push(state, action, next_state, torch.as_tensor([reward]))
			state = next_state

			if att_iter % BATCH_SIZE == 0:
				# Perform one step of the optimization.
				optimize_model()

			if att_iter % TARGET_UPDATE == 0:
				# Update the target network.
				target_net.load_state_dict(policy_net.state_dict())

		if att_iter % 1000 == 0:
			show_stats(
				cube, running_stats_length, num_correct, running_num_correct, max_correct,
				reward, running_reward, max_reward, att_iter, time_start)

		att_iter += 1

		# If set, discards the current cube and creates a new one at term_iter iteration.
		if flag_term_point and att_iter % term_iter == 0:
			print('\n\nTermination point reached.')
			show_best_cube_statistics(best_cube, max_correct)
			clear_cube_hashes()
			attempt_num += 1
			print('Creating a new cube.')
			cube = Cube(edge_length=edge_length)
			print('Scrambling...')
			cube.scramble()
			att_iter = 0
			max_correct = 0
			best_cube = cube.copy()
			max_reward = 0
			time_start = time.time()
			running_num_correct = [0] * running_stats_length

	if flag_continue_main:
		if len(solved_stats) > 0:
			show_solved_stats()
		show_best_cube_statistics(best_cube, max_correct)


def main():
	global edge_length
	global num_layers
	global root_seed
	global flag_deliberate_attempt

	global memory
	global policy_net
	global target_net
	global optimizer

	# Read all command-line parameters.
	try:
		opts, args = getopt.getopt(sys.argv[1:], 's:l:h', ['size=', 'layers=', 'seed=', 'random', 'help'])
		for opt, arg in opts:
			if opt in ('-h', '--help'):
				print('Options:')
				print(' -s N, --size=N      number of squares per Cube edge (default: 2)')
				print(' -l N, --layers=N    number of layers in the NN (default: 2)')
				print(' --seed=N            set the RNG seed')
				print(' --random            only use random choices, no AI')
				print(' -h, --help          display this help page and exit')
				print('\n')
				quit()

			elif opt in ('-s', '--size'):
				try:
					arg = int(arg)
					if arg >= 2:
						edge_length = int(arg)
						print('Cube size set to {0}.'.format(edge_length))
					else:
						print('Provided cube size is not valid. '
							+ 'Setting to default of {0}.'.format(edge_length))
				except ValueError:
					print('Provided cube size is not valid. '
						+ 'Setting to default of {0}.'.format(edge_length))

			elif opt in ('-l', '--layers'):
				try:
					arg = int(arg)
					if arg >= 1:
						num_layers = int(arg)
						print('NN layers set to {0}.'.format(num_layers))
					else:
						print('Provided number of layers is not valid. '
							+ 'Setting to default of {0}.'.format(num_layers))
				except ValueError:
					print('Provided number of layers is not valid. '
						+ 'Setting to default of {0}.'.format(num_layers))

			elif opt in ('--seed'):
				try:
					arg = int(arg)
					root_seed = arg
					print('RNG seed set to {0}'.format(root_seed))
				except ValueError:
					print('Provided RNG seed is not valid.')

			elif opt in ('--random'):
				flag_deliberate_attempt = False
				print('Only using random choices during solution attempts.')

	except getopt.GetoptError:
		print('\nError: Unrecognized option provided,'
			+ ' or an option that requires an argument was given none.')
		print('Call with option \'--help\' for the help page.\n\n')
		quit()

	# Creation of NNs and related objects.
	policy_net = NN()
	target_net = NN()
	target_net.load_state_dict(policy_net.state_dict())
	target_net.eval()
	optimizer = optim.RMSprop(policy_net.parameters())
	memory = ReplayMemory(TRANSITION_MEMORY_SIZE)

	# Dictates continuation of main loop.
	global flag_continue_main
	flag_continue_main = True

	global attempt_num
	attempt_num = 0

	last_random_state = random.getstate()

	while flag_continue_main:
		# Set the RNG seed.
		if attempt_num == 0 and root_seed != None:
			attempt_seed = root_seed
			random.seed(root_seed)
		else:
			random.setstate(last_random_state)
			random.seed(random.randint(0, 2 ** 30))
			attempt_seed = random.randint(0, 2 ** 30)

		last_random_state = random.getstate()
		attempt_seeds.append(attempt_seed)

		# Setting PyTorch's RNG seed doesn't seem to affect the results.
		torch.manual_seed(attempt_seed)

		# Start a solution attempt on a new cube.
		solution_attempt()
		attempt_num += 1


if __name__ == '__main__':
	main()