import numpy as np
import math
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple
import signal
from rubiks import Cube, face_relations

# Number of squares along each edge of the Cube.
edge_length = 3

# If True, attempts to learn through reinforcement learning.
# Otherwise, random actions are taken.
flag_deliberate_attempt = True

# Number of layers to use in the NN.
num_layers = 2

# Starting attempt's random seed. Will be used if != None.
root_seed = None

# Tracks used seeds and solution statistics.
attempt_seeds = []
solved_statistics = []

# If True, discards the current Cube and creates a new one to solve.
flag_termination_point = False
# Iteration at which to create a new Cube, if flag_termination_point == True.
termination_iteration = 100000

# Used in ReplayMemory class.
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# Parameters specific to the AI.
BATCH_SIZE = 16
GAMMA = 0.999
EPS_START = 0.975
EPS_END = 0.025
EPS_DECAY = 50000
TARGET_UPDATE = 1000
MEMORY_SIZE = 5000
total_iterations = 0


def exit_gracefully(signum, frame):
	"""
	More elegant handling of keyboard interrupts.
	Enter 'Ctrl+C', then 'y' to terminate early.
	"""
	global flag_continue_attempt
	global flag_continue_main
	signal.signal(signal.SIGINT, original_sigint)
	try:
		input_text = input('\n\nReally Quit? (y/n)> ').lower().lstrip()
		if input_text != '' and input_text[0] == 'y':
			print('Exiting.')
			flag_continue_main = False
			flag_continue_attempt = False
	except KeyboardInterrupt:
		print('\nExiting.')
		flag_continue_main = False
		flag_continue_attempt = False
	signal.signal(signal.SIGINT, exit_gracefully)


original_sigint = signal.getsignal(signal.SIGINT)
signal.signal(signal.SIGINT, exit_gracefully)


class NN(nn.Module):
	""" Simple NN model. """
	def __init__(self):
		super(NN, self).__init__()
		num_squares = 6 * edge_length * edge_length
		num_states = num_squares * 6
		num_outputs = 6 * 2

		if num_layers == 1:
			self.lin1 = nn.Linear(num_states, num_outputs)
		elif num_layers == 2:
			self.lin1 = nn.Linear(num_states, 24)
			self.lin2 = nn.Linear(24, num_outputs)
		elif num_layers == 3:
			self.lin1 = nn.Linear(num_states, num_squares)
			self.lin2 = nn.Linear(num_squares, 24)
			self.lin3 = nn.Linear(24, num_outputs)
		else:
			print('Invalid number of layers! (Choose 1 -> 3)')
			quit()

	def forward(self, x):
		if num_layers >= 1:
			x = self.lin1(x)
		if num_layers >= 2:
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


# Creation of NNs and related objects.
policy_net = NN()
target_net = NN()
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(MEMORY_SIZE)


def select_action(state):
	""" Generate an action, depending on the current state. """
	global total_iterations
	sample = random.random()
	eps_threshold = EPS_END + (EPS_START - EPS_END) * \
		math.exp(-1. * total_iterations / EPS_DECAY)
	total_iterations += 1
	if sample > eps_threshold:
		with torch.no_grad():
			return np.argmax(policy_net(state)).view(1)
	else:
		return torch.tensor(random.randrange(6 * 2), dtype=torch.long).view(1)


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
	edge_length = cube.edge_length
	total_correct_corners = 0

	# Look at each face of the given cube.
	for face_idx,face in enumerate(cube.faces):
		# Count the number of correctly coloured squares.
		num_correct_colours = np.sum(face.flatten() == face_idx)
		total_correct += num_correct_colours
		reward += num_correct_colours

		if num_correct_colours == edge_length * edge_length:
			# All squares are the correct colour.
			reward += 2 * (edge_length - 1)
		elif num_correct_colours >= edge_length * edge_length / 2:
			# Half of the squares are the correct colour.
			reward += 1 * (edge_length - 1)

		num_correct_corners = check_corners(cube.faces, face_idx)
		if num_correct_corners == 4:
			# Reward more if all corners of a face are correct.
			reward += 1
		total_correct_corners += num_correct_corners

	reward += (total_correct_corners / 3) * edge_length
	# Increase disparity between top solutions
	reward **= 1.25
	return total_correct, reward


def check_corners(faces, face_idx):
	""" Returns the number of corners of a face that are correct for all 3 of their sides. """
	num_correct_corners = 0

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

	elif face_idx == 1:
		# Top-left
		if faces[face_idx, 0, 0] == face_idx:
			l_face_corner = l_face[0, 0]
			if l_face_corner == l_face_colour:
				u_face_corner = u_face[0, -1]
				if u_face_corner == u_face_colour:
					num_correct_corners += 1

		# Top-right
		if faces[face_idx, 0, -1] == face_idx:
			r_face_corner = r_face[0, -1]
			if r_face_corner == r_face_colour:
				u_face_corner = u_face[0, 0]
				if u_face_corner == u_face_colour:
					num_correct_corners += 1

		# Bottom-left
		if faces[face_idx, -1, 0] == face_idx:
			l_face_corner = l_face[0, -1]
			if l_face_corner == l_face_colour:
				d_face_corner = d_face[0, 0]
				if d_face_corner == d_face_colour:
					num_correct_corners += 1

		# Bottom-right
		if faces[face_idx, -1, -1] == face_idx:
			r_face_corner = r_face[0, 0]
			if r_face_corner == r_face_colour:
				d_face_corner = d_face[0, -1]
				if d_face_corner == d_face_colour:
					num_correct_corners += 1

	elif face_idx == 2:
		# Top-left
		if faces[face_idx, 0, 0] == face_idx:
			l_face_corner = l_face[0, -1]
			if l_face_corner == l_face_colour:
				u_face_corner = u_face[0, 0]
				if u_face_corner == u_face_colour:
					num_correct_corners += 1

		# Top-right
		if faces[face_idx, 0, -1] == face_idx:
			r_face_corner = r_face[0, 0]
			if r_face_corner == r_face_colour:
				u_face_corner = u_face[-1, 0]
				if u_face_corner == u_face_colour:
					num_correct_corners += 1

		# Bottom-left
		if faces[face_idx, -1, 0] == face_idx:
			l_face_corner = l_face[-1, -1]
			if l_face_corner == l_face_colour:
				d_face_corner = d_face[-1, 0]
				if d_face_corner == d_face_colour:
					num_correct_corners += 1

		# Bottom-right
		if faces[face_idx, -1, -1] == face_idx:
			r_face_corner = r_face[-1, 0]
			if r_face_corner == r_face_colour:
				d_face_corner = d_face[0, 0]
				if d_face_corner == d_face_colour:
					num_correct_corners += 1

	elif face_idx == 3:
		# Top-left
		if faces[face_idx, 0, 0] == face_idx:
			l_face_corner = l_face[-1, -1]
			if l_face_corner == l_face_colour:
				u_face_corner = u_face[-1, 0]
				if u_face_corner == u_face_colour:
					num_correct_corners += 1

		# Top-right
		if faces[face_idx, 0, -1] == face_idx:
			r_face_corner = r_face[-1, 0]
			if r_face_corner == r_face_colour:
				u_face_corner = u_face[-1, -1]
				if u_face_corner == u_face_colour:
					num_correct_corners += 1

		# Bottom-left
		if faces[face_idx, -1, 0] == face_idx:
			l_face_corner = l_face[-1, 0]
			if l_face_corner == l_face_colour:
				d_face_corner = d_face[-1, -1]
				if d_face_corner == d_face_colour:
					num_correct_corners += 1

		# Bottom-right
		if faces[face_idx, -1, -1] == face_idx:
			r_face_corner = r_face[-1, -1]
			if r_face_corner == r_face_colour:
				d_face_corner = d_face[-1, 0]
				if d_face_corner == d_face_colour:
					num_correct_corners += 1

	elif face_idx == 4:
		# Top-left
		if faces[face_idx, 0, 0] == face_idx:
			l_face_corner = l_face[0, -1]
			if l_face_corner == l_face_colour:
				u_face_corner = u_face[-1, -1]
				if u_face_corner == u_face_colour:
					num_correct_corners += 1

		# Top-right
		if faces[face_idx, 0, -1] == face_idx:
			r_face_corner = r_face[0, 0]
			if r_face_corner == r_face_colour:
				u_face_corner = u_face[0, -1]
				if u_face_corner == u_face_colour:
					num_correct_corners += 1

		# Bottom-left
		if faces[face_idx, -1, 0] == face_idx:
			l_face_corner = l_face[-1, -1]
			if l_face_corner == l_face_colour:
				d_face_corner = d_face[0, -1]
				if d_face_corner == d_face_colour:
					num_correct_corners += 1

		# Bottom-right
		if faces[face_idx, -1, -1] == face_idx:
			r_face_corner = r_face[-1, 0]
			if r_face_corner == r_face_colour:
				d_face_corner = d_face[-1, -1]
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


def show_statistics(
		cube, running_stats_length, num_correct, running_num_correct, max_correct,
		reward, running_reward, max_reward, iteration, time_start):
	""" Format and display the provided cube and statistics. """
	print(cube)
	print('Correct:     (Current: {0:3}, Running: {1:7.3f}, Max: {2:4})'.format(
		int(num_correct), sum(running_num_correct)/running_stats_length, int(max_correct)))
	print('Reward:      (Current: {0:3}, Running: {1:7.3f}, Max: {2:4})'.format(
		int(reward), sum(running_reward)/running_stats_length, int(max_reward)))
	randomness = EPS_END + (EPS_START - EPS_END) * \
		math.exp(-1. * total_iterations / EPS_DECAY)
	print('Randomness: {0:.2f}%'.format(100 * randomness))
	print('Current Iter: {0}'.format(iteration))
	print('Current Time: {0} seconds'.format(int(time.time() - time_start)))


def show_solved_statistics():
	""" Show the statistics of past solutions. """
	print()
	print('Solved statistics:')
	for idx,stat in enumerate(solved_statistics):
		print('Cube {0}: Iterations: {1}, Time: {2}, Seed: {3}'.format(
			idx, stat[0], int(stat[1]), stat[2]))
	print()


def show_best_cube_statistics(cube, max_correct):
	""" Show the given best cube and its number of correct squares. """
	print('\n\n\nAttempt seed: {0}'.format(attempt_seeds[-1]))
	print('Best cube ({0} correct colours):'.format(int(max_correct)))
	print(cube)


def solution_attempt():
	"""
	Attempt to solve a Rubik's Cube.
	Depending on flag_deliberate_attempt, may use an AI to learn,
	otherwise will take random actions.
	"""
	global flag_continue_attempt
	# Dictates continuation of each attempt.
	flag_continue_attempt = True

	cube = Cube(edge_length=edge_length)
	print('Original cube:')
	print(cube)

	print('\n\nScrambling...')
	cube.scramble()
	
	num_squares = 6 * cube.edge_length * cube.edge_length
	num_states = num_squares * 6
	iteration = 0
	max_correct = 0
	best_cube = cube.copy()
	max_reward = 0
	time_start = time.time()
	running_stats_length = 1000
	running_num_correct = [0] * running_stats_length
	running_reward = [0] * running_stats_length

	# Used to translate the cube's state into boolean values.

	if flag_deliberate_attempt:
		# Gather the cube's current state.
		state_grid = torch.arange(start=0, end=num_states, step=6, dtype=torch.int64)
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
			cube.take_action(face, rotation)
		else:
			# Take a random action.
			cube.take_random_action()

		# Compute the number of correctly-coloured squares and corresponding reward.
		num_correct, reward = reward_function(cube)
		running_num_correct[iteration % running_stats_length] = num_correct
		running_reward[iteration % running_stats_length] = reward
		reward = torch.as_tensor([reward])

		if num_correct > max_correct:
			max_correct = num_correct
			best_cube = cube.copy()

		if num_correct == num_squares:
			print('\n\nSolved!')
			solved_statistics.append([iteration, time.time() - time_start, attempt_seeds[-1]])
			break

		if reward > max_reward:
			max_reward = reward

		if flag_deliberate_attempt:
			# Gather the cube's current state.
			squares = torch.as_tensor(cube.faces.flatten(), dtype=torch.int64)
			next_state = torch.zeros([num_states], dtype=torch.float32)
			next_state[state_grid + squares] = 1
			# Store the transition in memory
			memory.push(state, action, next_state, reward)
			state = next_state

			# Perform one step of the optimization (on the target network)
			optimize_model()

			# Update the target network.
			if iteration % TARGET_UPDATE == 0:
				target_net.load_state_dict(policy_net.state_dict())

		if iteration % 1000 == 0:
			show_statistics(
				cube, running_stats_length, num_correct, running_num_correct, max_correct,
				reward, running_reward, max_reward, iteration, time_start)

		iteration += 1
		
		# If set, discards the current cube and create a new one.
		if flag_termination_point and iteration % termination_iteration == 0:
			print('\n\nCreating a new cube.')
			cube = Cube(edge_length=edge_length)
			print('Scrambling...')
			cube.scramble()
			iteration = 0
			max_correct = 0
			best_cube = cube.copy()
			max_reward = 0
			time_start = time.time()
			running_num_correct = [0] * running_stats_length

	if len(solved_statistics) > 0:
		show_solved_statistics()
	show_best_cube_statistics(best_cube, max_correct)


def main():
	global flag_continue_main
	# Dictates continuation of main loop.
	flag_continue_main = True

	attempt_num = 0
	while flag_continue_main:
		# Set the seed.
		if attempt_num == 0 and root_seed != None:
			attempt_seed = root_seed
		else:
			attempt_seed = random.randint(0, 2 ** 30)
		attempt_seeds.append(attempt_seed)
		random.seed(attempt_seed)
		torch.manual_seed(attempt_seed)

		# Start a solution attempt on a new cube.
		solution_attempt()
		attempt_num += 1


if __name__ == '__main__':
	main()