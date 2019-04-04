import random

import ctypes
from ctypes import *

import itertools


ANN_DLL_PATH = 'network/bin/ann.dll'
_network_ref = ctypes.CDLL(ANN_DLL_PATH)


class NeuralNetwork(object):
	def __init__(self, layer_sizes, batch_size, learning_rate, existing_network=None, output_file=None):
		if existing_network:
			self.network_pointer = NeuralNetwork.network_from_file(existing_network)
			metadata = NeuralNetwork.metadata_from_file(existing_network)

			self.layer_sizes = metadata['layer_sizes']
			self.learning_rate = metadata['learning_rate']
			self.batch_size = metadata['batch_size']
		else:
			create_network = _network_ref.createNetwork
			create_network.argtypes = [POINTER(c_uint), c_uint, c_uint, c_float]
			create_network.restype = c_void_p

			layer_sizes_type = c_uint * len(layer_sizes)
			layer_sizes_input = layer_sizes_type()

			for i, size in enumerate(layer_sizes):
				layer_sizes_input[i] = size

			self.network_pointer = create_network(layer_sizes_input, len(layer_sizes), batch_size, learning_rate)
			self.layer_sizes = layer_sizes
			self.batch_size = batch_size
			self.learning_rate = learning_rate

		self.output_file = output_file

		if not self.network_pointer:
			print('Error initializing neural network!')


	@staticmethod
	def network_from_file(filename):
		load_network = _network_ref.loadNetwork
		load_network.argtypes = [c_char_p, c_size_t]
		load_network.restype = c_void_p

		return load_network(filename.encode('ascii'), len(filename))


	@staticmethod
	def metadata_from_file(filename):
		with open(filename, 'r') as f:
			lines = f.readlines()
			f.close()

		num_layers = int(lines[0])
		layer_sizes = list()

		for i in range(num_layers):
			layer_sizes.append(int(lines[i + 1]))

		learning_rate = float(lines[num_layers + 1])
		batch_size = int(lines[num_layers + 2])

		return {
			'layer_sizes': layer_sizes,
			'learning_rate': learning_rate,
			'batch_size': batch_size
		}


	@staticmethod
	def load_from_file(filename):
		network_pointer = NeuralNetwork.network_from_file(filename)
		metadata = NeuralNetwork.metadata_from_file(filename)
		return NeuralNetwork(metadata['layer_sizes'], metadata['batch_size'], metadata['learning_rate'], network_pointer)


	def train(self, network_input, expected_output):
		if not self.network_pointer:
			print('Neural network has not been initialized')
			return

		if len(network_input) == 0:
			print('Network input length was 0, no training was completed')
			return

		train_network = _network_ref.trainNetwork
		train_network.argtypes = [c_void_p, POINTER(c_float), POINTER(c_float)]

		input_array = (c_float * (len(network_input) * self.layer_sizes[0]))()
		output_array = (c_float * (len(network_input) * self.layer_sizes[-1]))()

		consolidated_input = list(itertools.chain.from_iterable(network_input))
		consolidated_output = list(itertools.chain.from_iterable(expected_output))

		# For some reason, this copies consolidated_* to *_array around 10x faster than using a for loop
		input_array[:] = consolidated_input
		output_array[:] = consolidated_output

		train_network(self.network_pointer, input_array, output_array)


	def output(self, network_input):
		if not self.network_pointer:
			print('Neural network has not been initialized')
			return list()

		assert len(network_input) == self.layer_sizes[0]

		get_output = _network_ref.getNetworkOutputForInput
		get_output.argtypes = [c_void_p, POINTER(c_float), POINTER(c_float)]

		input_array_type = c_float * self.layer_sizes[0]
		output_array_type = c_float * self.layer_sizes[-1]

		input_array = input_array_type()
		output_array = output_array_type()

		for i, obj in enumerate(network_input):
			input_array[i] = obj

		get_output(self.network_pointer, input_array, output_array)

		return list(output_array)


	def save(self):
		if not self.network_pointer:
			print('Neural network has not been initialized')
			return

		if not self.output_file:
			print('No output file specified')
			return

		save_network = _network_ref.saveNetwork
		save_network.argtypes = [c_void_p, c_char_p, c_size_t]

		save_network(self.network_pointer, self.output_file.encode('ascii'), len(self.output_file))



if __name__ == '__main__':
	network = NeuralNetwork([10, 5, 5, 10], 32, 0.1)

	single_input = [0.15, 0.45, 0.78, 0.04, 0.45, 0.73, 0.19, 0.11, 0.01, 0.11]
	single_input_2 = [0.38, 0.92, 0.16, 0.63, 0.82, 0.11, 0.02, 0.73, 0.25, 0.68]

	input_values = [single_input] * 32
	input_values_2 = [single_input_2] * 32

	output_values = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]] * 32
	output_values_2 = [[0.3, 0.9, 0.5, 0.0, 0.7, 0.4, 0.4, 0.1, 0.8, 0.3]] * 32

	for i in range(3200):
		network.train(input_values, output_values)
		network.train(input_values_2, output_values_2)

	output = network.output(single_input)
	for i, item in enumerate(output):
		print('%.1f' % item, output_values[0][i])

	print('-' * 30)

	output = network.output(single_input_2)
	for i, item in enumerate(output):
		print('%.1f' % item, output_values_2[0][i])
