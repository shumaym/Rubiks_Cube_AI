"""
This script emulates a Rubik's Cube of arbitrary size,
supporting all forms of its manipulation.
Cubes are visually represented as coloured blocks in standard output.

Author: Mykel Shumay
"""

import numpy as np
import random
import time
import getopt
import sys

# Number of squares along each edge of the Cube.
edge_length = 3
# Unicode character used in a Cube's visual representation.
console_block = u'\u25a0'
# Default console text colour.
text_colour_white  = '\033[0m'
# Number of random rotations to make during a Cube's scrambling.
scramble_iterations = 25000

# Colour values of the Rubik's Cube.
colours = {
	'red': 0,
	'white': 1,
	'green': 2,
	'yellow': 3,
	'blue': 4,
	'orange': 5
}

# Colours for console output.
console_colours = {
	colours['red']: '\033[31m',
	colours['white']: '\033[37m',
	colours['green']: '\033[92m',
	colours['yellow']: '\033[93m',
	colours['blue']: '\033[94m',
	colours['orange']: '\033[91m',
}

# For each face, stores the index of each adjacent face.
face_relations = {
	'_'.join([str(colours['red']), 'u']): colours['white'],
	'_'.join([str(colours['red']), 'l']): colours['green'],
	'_'.join([str(colours['red']), 'd']): colours['yellow'],
	'_'.join([str(colours['red']), 'r']): colours['blue'],

	'_'.join([str(colours['green']), 'u']): colours['white'],
	'_'.join([str(colours['green']), 'l']): colours['orange'],
	'_'.join([str(colours['green']), 'd']): colours['yellow'],
	'_'.join([str(colours['green']), 'r']): colours['red'],

	'_'.join([str(colours['orange']), 'u']): colours['white'],
	'_'.join([str(colours['orange']), 'l']): colours['blue'],
	'_'.join([str(colours['orange']), 'd']): colours['yellow'],
	'_'.join([str(colours['orange']), 'r']): colours['green'],

	'_'.join([str(colours['blue']), 'u']): colours['white'],
	'_'.join([str(colours['blue']), 'l']): colours['red'],
	'_'.join([str(colours['blue']), 'd']): colours['yellow'],
	'_'.join([str(colours['blue']), 'r']): colours['orange'],

	'_'.join([str(colours['white']), 'u']): colours['orange'],
	'_'.join([str(colours['white']), 'l']): colours['green'],
	'_'.join([str(colours['white']), 'd']): colours['red'],
	'_'.join([str(colours['white']), 'r']): colours['blue'],

	'_'.join([str(colours['yellow']), 'u']): colours['red'],
	'_'.join([str(colours['yellow']), 'l']): colours['green'],
	'_'.join([str(colours['yellow']), 'd']): colours['orange'],
	'_'.join([str(colours['yellow']), 'r']): colours['blue']
}


class Cube():
	"""Emulates a Rubik's Cube."""

	def __init__(self, edge_length):
		self.edge_length = edge_length
		self.faces = np.zeros([6,self.edge_length,self.edge_length], dtype=np.uint8)

		# cw_rotate_take_idxs specifies how to rearrange a face upon clockwise rotation.
		cw_rotate_take_idxs = np.arange(self.edge_length * self.edge_length)
		cw_rotate_take_idxs = cw_rotate_take_idxs.reshape(self.edge_length, self.edge_length)
		self.cw_rotate_take_idxs = np.flip(cw_rotate_take_idxs.transpose(), axis=[1])

		# Initialize each face with a single colour.
		for face_idx in range(self.faces.shape[0]):
			self.faces[face_idx].fill(face_idx)

	def rotate(self, action):
		"""Given an action number, rotates a section of the Cube appropriately.

		Action description:
			0 -> 11:
				Each face may be rotated either clockwise/counter-clockwise.
				-Examples-
				0: Rotate face 0 clockwise
				1: Rotate face 0 counter-clockwise
				2: Rotate face 1 clockwise
				...

			12+:
				If the Cube is larger than 2x2x2, each inner column may
				be rotated upwards/downwards, and each inner row may
				be rotated leftwards/rightwards.
		"""

		if action < 12:
			# Rotates the given face in the given direction, followed by its sides.
			self.rotate_face(action)
			self.rotate_edges(action)
		else:
			# Rotate an inner row/column.
			self.rotate_middle(action)
			pass

	def rotate_face(self, action):
		"""Rotate a single face 90 degrees in the given direction (side edges not accounted for)."""
		# Face to rotate.
		face_idx = action // 2
		# Direction to rotate (0: clockwise; 1: counter-clockwise)
		rotation = action % 2

		if rotation == 0:
			# Rotate the face's contents clockwise.
			indices = self.cw_rotate_take_idxs
		else:
			# Rotate the face's contents counter-clockwise.
			indices = np.flip(self.cw_rotate_take_idxs.flatten(), axis=[0])

		# Reassign the face's values based on (indices).
		self.faces[face_idx] = np.take(
			a=self.faces[face_idx],
			indices=indices).reshape(self.edge_length, self.edge_length)

	def rotate_edges(self, action):
		"""Rotate a face's adjacent edges 90 degrees in the given direction (face not accounted for)."""
		# Face to rotate about.
		face_idx = action // 2
		# Direction to rotate (0: clockwise; 1: counter-clockwise)
		rotation = action % 2

		### Gather information on the affected faces.
		# Face above.
		u_face = self.faces[face_relations['_'.join([str(face_idx), 'u'])]]
		u_face_copy = np.copy(u_face)

		# Left face.
		l_face = self.faces[face_relations['_'.join([str(face_idx), 'l'])]]
		l_face_copy = np.copy(l_face)

		# Face below.
		d_face = self.faces[face_relations['_'.join([str(face_idx), 'd'])]]
		d_face_copy = np.copy(d_face)

		# Right face.
		r_face = self.faces[face_relations['_'.join([str(face_idx), 'r'])]]
		r_face_copy = np.copy(r_face)

		### Rotate all affected edges appropriately.
		if face_idx == 0:
			# Rotate edges adjacent to face 0.
			if rotation == 0:
				# Rotate clockwise.
				u_face[-1] = np.flip(l_face_copy.transpose()[-1], axis=[0])
				l_face.transpose()[-1] = d_face_copy[0]
				d_face[0] = np.flip(r_face_copy.transpose()[0], axis=[0])
				r_face.transpose()[0] = u_face_copy[-1]
			else:
				# Rotate counter-clockwise.
				u_face[-1] = r_face_copy.transpose()[0]
				l_face.transpose()[-1] = np.flip(u_face_copy[-1], axis=[0])
				d_face[0] = l_face_copy.transpose()[-1]
				r_face.transpose()[0] = np.flip(d_face_copy[0], axis=[0])

		elif face_idx == 1:
			# Rotate edges adjacent to face 1.
			if rotation == 0:
				# Rotate clockwise.
				u_face[0] = l_face_copy[0]
				l_face[0] = d_face_copy[0]
				d_face[0] = r_face_copy[0]
				r_face[0] = u_face_copy[0]
			else:
				# Rotate counter-clockwise.
				u_face[0] = r_face_copy[0]
				l_face[0] = u_face_copy[0]
				d_face[0] = l_face_copy[0]
				r_face[0] = d_face_copy[0]

		elif face_idx == 2:
			# Rotate edges adjacent to face 2.
			if rotation == 0:
				# Rotate clockwise.
				u_face.transpose()[0] = np.flip(l_face_copy.transpose()[-1], axis=[0])
				l_face.transpose()[-1] = np.flip(d_face_copy.transpose()[0], axis=[0])
				d_face.transpose()[0] = r_face_copy.transpose()[0]
				r_face.transpose()[0] = u_face_copy.transpose()[0]
			else:
				# Rotate counter-clockwise.
				u_face.transpose()[0] = r_face_copy.transpose()[0]
				l_face.transpose()[-1] = np.flip(u_face_copy.transpose()[0], axis=[0])
				d_face.transpose()[0] = np.flip(l_face_copy.transpose()[-1], axis=[0])
				r_face.transpose()[0] = d_face_copy.transpose()[0]
		
		elif face_idx == 3:
			# Rotate edges adjacent to face 3.
			if rotation == 0:
				# Rotate clockwise.
				u_face[-1] = l_face_copy[-1]
				l_face[-1] = d_face_copy[-1]
				d_face[-1] = r_face_copy[-1]
				r_face[-1] = u_face_copy[-1]
			else:
				# Rotate counter-clockwise.
				u_face[-1] = r_face_copy[-1]
				l_face[-1] = u_face_copy[-1]
				d_face[-1] = l_face_copy[-1]
				r_face[-1] = d_face_copy[-1]
		
		elif face_idx == 4:
			# Rotate edges adjacent to face 4.
			if rotation == 0:
				# Rotate clockwise.
				u_face.transpose()[-1] = l_face_copy.transpose()[-1]
				l_face.transpose()[-1] = d_face_copy.transpose()[-1]
				d_face.transpose()[-1] = np.flip(r_face_copy.transpose()[0], axis=[0])
				r_face.transpose()[0] = np.flip(u_face_copy.transpose()[-1], axis=[0])
			else:
				# Rotate counter-clockwise.
				u_face.transpose()[-1] = np.flip(r_face_copy.transpose()[0], axis=[0])
				l_face.transpose()[-1] = u_face_copy.transpose()[-1]
				d_face.transpose()[-1] = l_face_copy.transpose()[-1]
				r_face.transpose()[0] = np.flip(d_face_copy.transpose()[-1], axis=[0])

		elif face_idx == 5:
			# Rotate edges adjacent to face 5.
			if rotation == 0:
				# Rotate clockwise.
				u_face[0] = l_face_copy.transpose()[-1]
				l_face.transpose()[-1] = np.flip(d_face_copy[-1], axis=[0])
				d_face[-1] = r_face_copy.transpose()[0]
				r_face.transpose()[0] = np.flip(u_face_copy[0], axis=[0])
			else:
				# Rotate counter-clockwise.
				u_face[0] = np.flip(r_face_copy.transpose()[0], axis=[0])
				l_face.transpose()[-1] = u_face_copy[0]
				d_face[-1] = np.flip(l_face_copy.transpose()[-1], axis=[0])
				r_face.transpose()[0] = d_face_copy[-1]
				
	def rotate_middle(self, action):
		"""Rotate an inner row/column of the Cube.

		Rotations are taken with face 0 as the front.
		"""

		# Whether to rotate left/right (0) or up/down (1).
		orientation = (action - 12) // ((self.edge_length - 2) * 2)
		# Direction to rotate (0: clockwise; 1: counter-clockwise)
		rotation = action % 2

		### Gather information on the affected faces.
		# Face on the front.
		f_face = self.faces[0]
		f_face_copy = np.copy(f_face)

		# Face on the back.
		b_face = self.faces[5]
		b_face_copy = np.copy(b_face)

		if orientation == 0:
			### Rotate left/right.
			# Left face.
			l_face = self.faces[2]
			l_face_copy = np.copy(l_face)

			# Right face.
			r_face = self.faces[4]
			r_face_copy = np.copy(r_face)

			# Affected row.
			row_idx = 1 + ((action - 12) % ((self.edge_length - 2) * 2)) // 2

			if rotation == 0:
				# Rotate rightwards.
				r_face[row_idx] = f_face_copy[row_idx]
				b_face[row_idx] = r_face_copy[row_idx]
				l_face[row_idx] = b_face_copy[row_idx]
				f_face[row_idx] = l_face_copy[row_idx]
			else:
				# Rotate leftwards.
				r_face[row_idx] = b_face_copy[row_idx]
				b_face[row_idx] = l_face_copy[row_idx]
				l_face[row_idx] = f_face_copy[row_idx]
				f_face[row_idx] = r_face_copy[row_idx]

		else:
			### Rotate up/down.
			# Face above.
			u_face = self.faces[1]
			u_face_copy = np.copy(u_face)

			# Face below.
			d_face = self.faces[3]
			d_face_copy = np.copy(d_face)

			# Affected column.
			col_idx = 1 + ((action - 12) % ((self.edge_length - 2) * 2)) // 2

			if rotation == 0:
				# Rotate upwards.
				u_face.transpose()[col_idx] = f_face_copy.transpose()[col_idx]
				b_face.transpose()[-1-col_idx] = np.flip(u_face_copy.transpose()[col_idx])
				d_face.transpose()[col_idx] = np.flip(b_face_copy.transpose()[-1-col_idx])
				f_face.transpose()[col_idx] = d_face_copy.transpose()[col_idx]
			else:
				# Rotate downwards.
				u_face.transpose()[col_idx] = np.flip(b_face_copy.transpose()[-1-col_idx])
				b_face.transpose()[-1-col_idx] = np.flip(d_face_copy.transpose()[col_idx])
				d_face.transpose()[col_idx] = f_face_copy.transpose()[col_idx]
				f_face.transpose()[col_idx] = u_face_copy.transpose()[col_idx]

	def scramble(self, iterations=scramble_iterations):
		"""Rotate the Cube randomly the specified number of times (iterations)."""
		for _ in range(iterations):
			self.random_rotation()

	def random_rotation(self):
		"""Take a random rotation action."""
		action = random.choice(range(12 + 4*(self.edge_length-2)))
		self.rotate(action)
		return action

	def copy(self):
		"""Returns a deep copy of the Cube."""
		clone_cube = Cube(self.edge_length)
		clone_cube.faces = np.copy(self.faces)
		return clone_cube

	def __repr__(self):
		"""Returns a coloured grid of the flattened Cube."""
		edge_length = self.edge_length

		# Gather the Cube's faces.
		red_face = self.faces[colours['red']]
		white_face = self.faces[colours['white']]
		green_face = self.faces[colours['green']]
		yellow_face = self.faces[colours['yellow']]
		blue_face = self.faces[colours['blue']]
		orange_face = self.faces[colours['orange']]

		# Concatenate the 4 middle faces together for simplicity.
		middle_faces = np.concatenate([green_face, red_face, blue_face, orange_face], axis=1)

		# Flattened lists of coloured blocks for all faces.
		white_face_blocks = [console_colours[x]+console_block+text_colour_white for x in white_face.flatten()]
		middle_faces_blocks = [console_colours[x]+console_block+text_colour_white for x in middle_faces.flatten()]
		yellow_face_blocks = [console_colours[x]+console_block+text_colour_white for x in yellow_face.flatten()]

		cube_str = ''
		# Construct the string containing the formatted coloured blocks.
		for row_idx in range(edge_length * 3):
			if row_idx % edge_length == 0:
				# Add an extra blank line between faces.
				cube_str += '\n '

			for col_idx in range(edge_length * 4):
				if col_idx % edge_length == 0:
					# Add an extra blank column between faces.
					cube_str += ' '

				if (row_idx < edge_length or row_idx > edge_length * 2 - 1) and \
						(col_idx < edge_length or col_idx > edge_length * 2 - 1):
					# Empty space in the grid.
					cube_str += ' '

				elif row_idx < edge_length:
					# Part of the white face.
					position = row_idx*edge_length + col_idx%edge_length
					colour = white_face_blocks[position]
					cube_str += colour

				elif row_idx > edge_length * 2 - 1:
					# Part of the yellow face.
					position = (row_idx%edge_length)*edge_length + col_idx%edge_length
					colour = yellow_face_blocks[position]
					cube_str += colour

				else:
					# Part of middle 4 faces.
					position = (row_idx%edge_length)*edge_length*4 + col_idx%(edge_length*4)
					colour = middle_faces_blocks[position]
					cube_str += colour

			# Start a new row.
			cube_str += '\n '

		return cube_str

	def __eq__(self, other):
		return np.array_equal(self.faces, other.faces)


def main():
	global edge_length

	# Read all command-line arguments.
	try:
		opts, args = getopt.getopt(sys.argv[1:], 's:h', ['size=', 'help'])
		for opt, arg in opts:
			if opt in ('-h', '--help'):
				print('Options:')
				print(' -s N, --size=N      number of squares per Cube edge (default: 3)')
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

	except getopt.GetoptError:
		print('\nError: Unrecognized option provided,'
			+ ' or an option that requires an argument was given none.')
		print('Call with option \'--help\' or \'-h\' for the help page.\n\n')
		quit()

	cube = Cube(edge_length=edge_length)
	print('Original cube:')
	print(cube)

	print('\nScrambling...')
	cube.scramble()
	print('Scrambled cube:')
	print(cube)


if __name__ == '__main__':
	main()