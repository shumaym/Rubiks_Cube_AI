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
from rubiks import Cube

# Number of squares along each edge of the Cube.
edge_length = 3

# If True, attempts to learn through reinforcement learning.
# Otherwise, random actions are taken.
flag_deliberate_attempt = True

# Whether to use multiple layers in the NN.
flag_multiple_layers = True

# Starting seed for python's and PyTorch's random.
# If != None, will be used.
root_seed = None

# Tracks used seeds and solution statistics.
attempt_seeds = []
solved_statistics = []

# Used in ReplayMemory class.
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

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
		num_outputs = 6 * 2
		if flag_multiple_layers:
			self.lin1 = nn.Linear(num_squares, 24)
			self.lin2 = nn.Linear(24, num_outputs)
		else:
			self.head = nn.Linear(num_squares, num_outputs)

	def forward(self, x):
		# x = F.selu(x)
		if flag_multiple_layers:
			x = self.lin1(x.view(-1, x.size(0)))
			x = self.lin2(x)
			return x
		else:
			return self.head(x.view(-1, x.size(0)))


class ReplayMemory(object):
	""" Memory of past state transitions and their associated reward deltas. """
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


# Parameters specific to the AI.
BATCH_SIZE = 16
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.025
EPS_DECAY = 200
TARGET_UPDATE = 2500

policy_net = NN()
target_net = NN()
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)
steps_done = 0


def select_action(state):
	""" Generate an action, depending on the current state. """
	global steps_done
	sample = random.random()
	eps_threshold = EPS_END + (EPS_START - EPS_END) * \
		math.exp(-1. * steps_done / EPS_DECAY)
	steps_done += 1
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

	state_batch = torch.cat(batch.state).reshape(6 * edge_length * edge_length, -1)
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
	""" Calculates a cube's number of correctly-coloured squares, along with a reward. """
	total_correct = 0.
	reward = 0.
	edge_length = cube.edge_length

	# Look at each face of the given cube.
	for face_idx,face in enumerate(cube.faces):
		# Count the number of correctly coloured squares.
		num_correct = np.sum(face.flatten() == face_idx)
		total_correct += num_correct
		reward += num_correct

		if num_correct == edge_length * edge_length:
			# All squares are the correct colour.
			reward += 2 * (edge_length - 1)
		elif num_correct >= edge_length * edge_length / 2:
			# Half of the squares are the correct colour.
			reward += 1 * (edge_length - 1)

		# Rewards for corners and edges of the correct colour, if the cube is 3x3x3 or larger.
		if cube.edge_length >= 3:
			num_correct_corners = 0
			if face[0, 0] == face_idx:
				num_correct_corners += 1
			if face[0, -1] == face_idx:
				num_correct_corners += 1
			if face[-1, 0] == face_idx:
				num_correct_corners += 1
			if face[-1, -1] == face_idx:
				num_correct_corners += 1

			reward += 2 * num_correct_corners
			if num_correct_corners == 4:
				# Reward more if all corners of a face are the correct colour.
				reward += num_correct_corners + 2

			# Rewards for each row of the correct colour.
			rows_correct = np.sum(np.sum(face == face_idx, axis=1) == edge_length)
			reward += rows_correct * (edge_length - 1)

			# Rewards for each column of the correct colour.
			cols_correct = np.sum(np.sum(face.transpose() == face_idx, axis=1) == edge_length)
			reward += cols_correct * (edge_length - 1)

	return total_correct, reward


def show_statistics(
		cube, num_correct, running_num_correct, running_stats_length, max_correct,
		reward, max_reward, max_reward_iter, iteration, time_start):
	""" Format and display the provided cube and statistics. """
	print(cube)
	print('Currently Correct: {0}'.format(int(num_correct)))
	print('Running Correctness: {0:.3f}'.format(sum(running_num_correct)/running_stats_length))
	print('Max Correct: {0}'.format(int(max_correct)))
	print('Current Reward: {0}'.format(int(reward)))
	print('Max Reward: {0} at iter {1}'.format(int(max_reward), int(max_reward_iter)))
	print('Current Iter: {0}'.format(iteration))
	print('Current Time: {0} seconds'.format(int(time.time() - time_start)))


def show_solved_statistics():
	""" Show the statistics of past solutions. """
	print()
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
	# Dictates continuation each attempt.
	flag_continue_attempt = True

	cube = Cube(edge_length=edge_length)
	_, max_reward = reward_function(cube)
	print('Original cube:')
	print(cube)

	print('\n\nScrambling...')
	cube.scramble()
	
	num_squares = 6 * cube.edge_length * cube.edge_length
	iteration = 0
	max_correct = 0
	best_cube = cube.copy()
	max_reward = 0
	time_start = time.time()
	max_reward_iter = -1
	running_stats_length = 1000
	running_num_correct = [0] * running_stats_length

	while flag_continue_attempt:
		if flag_deliberate_attempt:
			# Gather the cube's current state.
			state = torch.as_tensor(cube.faces.flatten(), dtype=torch.float32)

			# Select an action, depending on the current state.
			action = select_action(state)
			face = int(action / 2)
			rotation = action % 2
			# Take the action.
			cube.take_action(face, rotation)
		else:
			cube.take_random_action()

		# Compute the number of correctly-coloured squares and corresponding reward.
		num_correct, reward = reward_function(cube)
		running_num_correct[iteration % running_stats_length] = num_correct

		if num_correct > max_correct:
			max_correct = num_correct
			best_cube = cube.copy()

		if num_correct == num_squares:
			print('\n\nSolved!')
			solved_statistics.append([iteration, time.time() - time_start, attempt_seeds[-1]])
			break

		reward = torch.as_tensor([reward])
		if reward > max_reward:
			max_reward = reward
			max_reward_iter = iteration

		if flag_deliberate_attempt:
			next_state = torch.as_tensor(cube.faces.flatten(), dtype=torch.float32)
			# Store the transition in memory
			memory.push(state, action, next_state, reward)
			state = next_state

			# Perform one step of the optimization (on the target network)
			# if iteration % BATCH_SIZE == 0:
			optimize_model()

			# Update the target network
			if iteration % TARGET_UPDATE == 0:
				target_net.load_state_dict(policy_net.state_dict())

		if iteration % 1000 == 0:
			show_statistics(
				cube, num_correct, running_num_correct, running_stats_length, max_correct,
				reward, max_reward, max_reward_iter, iteration, time_start)

		iteration += 1

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