import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple
import signal
import getopt
import sys
from rubiks import Cube, face_relations

# Number of blocks along each edge of the Cube.
edge_length = 2

# Number of layers to use in the NN.
num_layers = 3

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

# Used in ReplayMemory class.
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# Parameters specific to the AI.
BATCH_SIZE = 4096
GAMMA = 0.95
EPS_START = 0.975
EPS_END = 0.025
EPS_DECAY = 1000000
TRANSITION_MEMORY_SIZE = 10000
total_iterations = 0

# Memory of the past 3 actions, used for simple rule enforcement.
recent_actions = [-1, -1, -1]


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
			flag_continue_main = False
			flag_continue_attempt = False
	
	except KeyboardInterrupt:
		flag_continue_main = False
		flag_continue_attempt = False

	if flag_continue_main:
		print('\n---------- RESUMING ----------')
	else:
		print('\n---------- QUITTING ----------')
	if 'time_start' in globals():
		time_start += time.time() - time_paused
	signal.signal(signal.SIGINT, exit_gracefully)


original_sigint = signal.getsignal(signal.SIGINT)
signal.signal(signal.SIGINT, exit_gracefully)


class DQN(nn.Module):
	""" Deep Q-Network. """
	def __init__(self):
		super(DQN, self).__init__()
		num_squares = 6 * edge_length * edge_length
		num_states = 6 * num_squares
		num_outputs = 6 * 2

		if num_layers == 2:
			self.layers = nn.Sequential(
				nn.Linear(num_states, 2 * num_outputs),
				nn.SELU(),
				nn.Linear(2 * num_outputs, num_outputs)
			)
		elif num_layers == 3:
			self.layers = nn.Sequential(
				nn.Linear(num_states, num_squares),
				nn.SELU(),
				nn.Linear(num_squares, 2 * num_outputs),
				nn.SELU(),
				nn.Linear(2 * num_outputs, num_outputs)
			)

	def forward(self, x):
		return self.layers(x)


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

	sample = random.random()
	eps_threshold = EPS_END + (EPS_START - EPS_END) * \
		np.exp(-1. * total_iterations / EPS_DECAY)

	total_iterations += 1
	if sample > eps_threshold:
		with torch.no_grad():
			# Take max valid action
			actions = torch.argsort(policy_net.forward(state), descending=True)
			actions_idx = 0
			action = actions[actions_idx]
			while not check_valid_action(action):
				actions_idx += 1
				action = actions[actions_idx]
	else:
		# Take random valid action
		action = torch.randint(low=0, high=6*2, size=(1,), dtype=torch.long)[0]
		while not check_valid_action(action):
			action = torch.randint(low=0, high=6*2, size=(1,), dtype=torch.long)[0]

	recent_actions[total_iterations % 3] = int(action)
	return action.view(1)


def check_valid_action(action):
	""" Returns whether the given action is valid, given the rule-set. """
	flag_valid_action = True

	# Disallow moves that counter the previous move.
	prev_action = recent_actions[(total_iterations - 1) % 3]
	if action % 2 == 0:
		if prev_action == action + 1:
			flag_valid_action = False
	else:
		if prev_action == action - 1:
			flag_valid_action = False

	# Disallow 4 similar consecutive moves.
	if recent_actions[0] == action and len(set(recent_actions)) == 1:
		flag_valid_action = False

	return flag_valid_action


def optimize_model():
	""" Update the model. """
	if len(memory) < BATCH_SIZE:
		return
	transitions = memory.sample(BATCH_SIZE)
	batch = Transition(*zip(*transitions))

	state_batch = torch.cat(batch.state).reshape(-1, 6 * 6 * edge_length * edge_length)
	next_state_batch = torch.cat(batch.next_state).reshape(-1, 6 * 6 * edge_length * edge_length)
	action_batch = torch.cat(batch.action)
	reward_batch = torch.cat(batch.reward)

	# Compute Q(s_t, a):
	# Computes Q(s_t), then selects the columns of actions taken.
	state_action_values = policy_net(state_batch).gather(1, action_batch.unsqueeze(1))

	# Compute V(s_{t+1}) for all next states.
	next_state_values = policy_net(next_state_batch).max(1)[0].detach()
	# Compute the expected Q values.
	expected_state_action_values = reward_batch + GAMMA * next_state_values

	# Compute Huber loss.
	loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

	# Optimize the model.
	optimizer.zero_grad()
	loss.backward()
	for param in policy_net.parameters():
		param.grad.data.clamp_(-6, 6)
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
		# Provides up to 2 extra points for each face, depending on colour correctness.
		num_correct_colours = np.sum(face.flatten() == face_idx)
		reward += 2 * num_correct_colours / (cube.edge_length * cube.edge_length)

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
		print('Cube {0:>3} (Attempt {1:>3}): Rotations: {2:>7}, Time: {3:>5}, Seed: {4:>10}'.format(
			idx, stat[0], stat[1], int(stat[2]), stat[3]))
	print()


def show_best_cube_statistics(cube, max_correct):
	""" Show the given best cube and its number of correct squares. """
	print('\n\nAttempt {0} (seed: {1})'.format(attempt_num, attempt_seeds[-1]))
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

			if total_iterations % BATCH_SIZE == 0:
				# Perform one step of the optimization.
				optimize_model()

		if att_iter % 1000 == 0:
			show_stats(
				cube, running_stats_length, num_correct, running_num_correct, max_correct,
				reward, running_reward, max_reward, att_iter, time_start)

		att_iter += 1

		# If set, discards the current cube and creates a new one at term_iter iteration.
		if flag_term_point and att_iter % term_iter == 0:
			print('\n\nTermination point reached.')
			show_best_cube_statistics(best_cube, max_correct)
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
	global optimizer

	# Read all command-line parameters.
	try:
		opts, args = getopt.getopt(sys.argv[1:], 's:l:h', ['size=', 'layers=', 'seed=', 'random', 'help'])
		for opt, arg in opts:
			if opt in ('-h', '--help'):
				print('Options:')
				print(' -s N, --size=N      number of squares per Cube edge (default: 2)')
				print(' -l N, --layers=N    number of layers in the NN (default: 2, allowable: 2, 3)')
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
						time.sleep(3)
				except ValueError:
					print('Provided cube size is not valid. '
						+ 'Setting to default of {0}.'.format(edge_length))
					time.sleep(3)

			elif opt in ('-l', '--layers'):
				try:
					arg = int(arg)
					if arg in [2, 3]:
						num_layers = int(arg)
						print('NN layers set to {0}.'.format(num_layers))
					else:
						print('Provided number of layers is not valid. '
							+ 'Setting to default of {0}.'.format(num_layers))
						time.sleep(3)
				except ValueError:
					print('Provided number of layers is not valid. '
						+ 'Setting to default of {0}.'.format(num_layers))
					time.sleep(3)

			elif opt in ('--seed'):
				try:
					arg = int(arg)
					root_seed = arg
					print('RNG seed set to {0}'.format(root_seed))
				except ValueError:
					print('Provided RNG seed is not valid.')
					time.sleep(3)

			elif opt in ('--random'):
				flag_deliberate_attempt = False
				print('Only using random choices during solution attempts.')

	except getopt.GetoptError:
		print('\nError: Unrecognized option provided,'
			+ ' or an option that requires an argument was given none.')
		print('Call with option \'--help\' for the help page.\n\n')
		quit()

	# Creation of DQN and related objects.
	policy_net = DQN()
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