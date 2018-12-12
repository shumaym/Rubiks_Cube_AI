import numpy as np
import random

# Number of squares along each edge of the Cube.
edge_length = 3
# Unicode character used in a Cube's visual representation.
console_block = u'\u25a0'
# Default console text colour.
text_colour_white  = '\033[0m'
# Number of random rotations to make during a Cube's scrambling.
scramble_iterations = 100000

# Colours of Rubik's Cube.
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

# For each face, stores the colour of each adjacent face.
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
	""" Emulates a Rubik's Cube. """
	def __init__(self, edge_length):
		self.edge_length = edge_length
		self.faces = np.zeros([6,self.edge_length,self.edge_length], dtype=np.int32)

		# rotate_cw_take_map specifies how to rearrange a face upon clockwise rotation.
		rotate_cw_take_map = np.arange(self.edge_length * self.edge_length)
		rotate_cw_take_map = rotate_cw_take_map.reshape(self.edge_length, self.edge_length)
		self.rotate_cw_take_map = np.flip(rotate_cw_take_map.transpose(), axis=[1])

		for face_idx in range(self.faces.shape[0]):
			# Init each face with a single colour.
			self.faces[face_idx].fill(face_idx)


	def rotate_face(self, face, clockwise=True):
		"""
		Rotate a single face 90 degrees in the given direction.
		Sides are not accounted for.
		"""
		if clockwise:
			self.faces[face] = np.take(
				a=self.faces[face],
				indices=self.rotate_cw_take_map).reshape(self.edge_length, self.edge_length)
		else:
			self.faces[face] = np.take(
				a=self.faces[face],
				indices=np.flip(
					self.rotate_cw_take_map.flatten(),
					axis=[0])).reshape(self.edge_length, self.edge_length)


	def rotate_sides(self, face, clockwise=True):
		"""
		Rotate the sides of a face 90 degrees in the given direction.
		Rotated face is not accounted for.
		"""
		# Gather information on the face above the rotating face.
		u_face = self.faces[face_relations['_'.join([str(face), 'u'])]]
		u_face_copy = np.copy(u_face)

		# Face to the left.
		l_face = self.faces[face_relations['_'.join([str(face), 'l'])]]
		l_face_copy = np.copy(l_face)

		# Face below.
		d_face = self.faces[face_relations['_'.join([str(face), 'd'])]]
		d_face_copy = np.copy(d_face)

		# Face to the right.
		r_face = self.faces[face_relations['_'.join([str(face), 'r'])]]
		r_face_copy = np.copy(r_face)

		# Rotate all affected sides appropriately.
		if face == 0:
			if clockwise:
				u_face[-1] = np.flip(l_face_copy.transpose()[-1], axis=[0])
				l_face.transpose()[-1] = d_face_copy[0]
				d_face[0] = np.flip(r_face_copy.transpose()[0], axis=[0])
				r_face.transpose()[0] = u_face_copy[-1]
			else:
				u_face[-1] = r_face_copy.transpose()[0]
				l_face.transpose()[-1] = np.flip(u_face_copy[-1], axis=[0])
				d_face[0] = l_face_copy.transpose()[-1]
				r_face.transpose()[0] = np.flip(d_face_copy[0], axis=[0])

		elif face == 1:
			if clockwise:
				u_face[0] = l_face_copy[0]
				l_face[0] = d_face_copy[0]
				d_face[0] = r_face_copy[0]
				r_face[0] = u_face_copy[0]
			else:
				u_face[0] = r_face_copy[0]
				l_face[0] = u_face_copy[0]
				d_face[0] = l_face_copy[0]
				r_face[0] = d_face_copy[0]

		elif face == 2:
			if clockwise:
				u_face.transpose()[0] = l_face_copy.transpose()[-1]
				l_face.transpose()[-1] = d_face_copy.transpose()[0]
				d_face.transpose()[0] = np.flip(r_face_copy.transpose()[0], axis=[0])
				r_face.transpose()[0] = u_face_copy.transpose()[0]
			else:
				u_face.transpose()[0] = np.flip(r_face_copy.transpose()[0], axis=[0])
				l_face.transpose()[-1] = u_face_copy.transpose()[0]
				d_face.transpose()[0] = l_face_copy.transpose()[-1]
				r_face.transpose()[0] = d_face_copy.transpose()[0]
		
		elif face == 3:
			if clockwise:
				u_face[-1] = l_face_copy[-1]
				l_face[-1] = d_face_copy[-1]
				d_face[-1] = r_face_copy[-1]
				r_face[-1] = u_face_copy[-1]
			else:
				u_face[-1] = r_face_copy[-1]
				l_face[-1] = u_face_copy[-1]
				d_face[-1] = l_face_copy[-1]
				r_face[-1] = d_face_copy[-1]
		
		elif face == 4:
			if clockwise:
				u_face.transpose()[-1] = l_face_copy.transpose()[-1]
				l_face.transpose()[-1] = d_face_copy.transpose()[-1]
				d_face.transpose()[-1] = r_face_copy.transpose()[0]
				r_face.transpose()[0] = u_face_copy.transpose()[-1]
			else:
				u_face.transpose()[-1] = r_face_copy.transpose()[0]
				l_face.transpose()[-1] = u_face_copy.transpose()[-1]
				d_face.transpose()[-1] = l_face_copy.transpose()[-1]
				r_face.transpose()[0] = d_face_copy.transpose()[-1]

		elif face == 5:
			if clockwise:
				u_face[0] = np.flip(l_face_copy.transpose()[-1], axis=[0])
				l_face.transpose()[-1] = np.flip(d_face_copy[-1], axis=[0])
				d_face[-1] = np.flip(r_face_copy.transpose()[0], axis=[0])
				r_face.transpose()[0] = np.flip(u_face_copy[0], axis=[0])
			else:
				u_face[0] = np.flip(r_face_copy.transpose()[0], axis=[0])
				l_face.transpose()[-1] = np.flip(u_face_copy[0], axis=[0])
				d_face[-1] = np.flip(l_face_copy.transpose()[-1], axis=[0])
				r_face.transpose()[0] = np.flip(d_face_copy[-1], axis=[0])
				

	def rotate(self, face, clockwise=True):
		""" First rotates the given face in the given direction, then its sides. """
		self.rotate_face(face, clockwise)
		self.rotate_sides(face, clockwise)


	def scramble(self, iterations):
		""" Rotate the cube's faces randomly the specified number of times (iterations). """
		face_numbers = range(6)
		for i in range(iterations):
			face = random.choice(face_numbers)
			rotation = bool(random.getrandbits(1))
			self.rotate(face=face, clockwise=rotation)


	def __repr__(self):
		""" Returns a coloured grid of the flattened cube. """
		red_face = self.faces[colours['red']]
		white_face = self.faces[colours['white']]
		green_face = self.faces[colours['green']]
		yellow_face = self.faces[colours['yellow']]
		blue_face = self.faces[colours['blue']]
		orange_face = self.faces[colours['orange']]

		# Concatenate the 4 middle faces together for simplicity.
		middle_faces = np.concatenate([green_face, red_face, blue_face, orange_face], axis=1)

		# Flattened list of coloured blocks for all faces.
		white_face_blocks = [console_colours[x]+console_block+text_colour_white for x in white_face.flatten()]
		middle_faces_blocks = [console_colours[x]+console_block+text_colour_white for x in middle_faces.flatten()]
		yellow_face_blocks = [console_colours[x]+console_block+text_colour_white for x in yellow_face.flatten()]

		cube_str = '\n '
		# Construct the string containing the formatted coloured blocks.
		for row_idx in range(edge_length * 3):
			if row_idx > 0 and row_idx % edge_length == 0:
				# Add an extra blank line between faces.
				cube_str += '\n '

			for col_idx in range(edge_length * 4):
				if col_idx % edge_length == 0:
					# Add an extra blank column between faces.
					cube_str += ' '

				if (row_idx < edge_length or row_idx > edge_length * 2 - 1) and (
						col_idx < edge_length or col_idx > edge_length * 2 - 1):
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


def main():
	cube = Cube(edge_length=edge_length)
	print('Original cube:')
	print(cube)

	print('\n\nScrambling...')
	cube.scramble(iterations=scramble_iterations)
	print('Scrambled cube:')
	print(cube)

if __name__ == '__main__':
	main()