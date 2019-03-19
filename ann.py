import random

import ctypes
from ctypes import *


ANN_DLL_PATH = 'network/bin/ann.dll'
_network_ref = ctypes.CDLL(ANN_DLL_PATH)


class NeuralNetwork(object):
	def __init__(self, num_input_neurons, num_hidden_layers, neurons_per_hidden_layer, num_output_neurons, learning_rate, network_pointer=None):
		if network_pointer:
			self.network_pointer = network_pointer
		else:
			create_network = _network_ref.createNetwork
			create_network.argtypes = [c_int, c_int, c_int, c_int, c_float]
			create_network.restype = c_void_p
			self.network_pointer = create_network(num_input_neurons, num_hidden_layers, neurons_per_hidden_layer, num_output_neurons, learning_rate)

		self.num_input_neurons = num_input_neurons
		self.num_hidden_layers = num_hidden_layers
		self.neurons_per_hidden_layer = neurons_per_hidden_layer
		self.num_output_neurons = num_output_neurons
		self.learning_rate = learning_rate

		if not self.network_pointer:
			print('Error initializing neural network!')


	@staticmethod
	def load_from_file(filename):
		load_network = _network_ref.loadNetwork
		load_network.argtypes = [c_char_p, c_size_t]
		load_network.restype = c_void_p

		network_pointer = load_network(filename.encode('ascii'), len(filename))

		with open(filename, 'r') as f:
			lines = f.readlines()
			f.close()

		num_input_neurons = int(lines[0])
		num_hidden_layers = int(lines[1])
		neurons_per_hidden_layer = int(lines[2])
		num_output_neurons = int(lines[3])
		learning_rate = float(lines[4])

		return NeuralNetwork(num_input_neurons, num_hidden_layers, neurons_per_hidden_layer, num_output_neurons, learning_rate, network_pointer)


	def train(self, network_input, expected_output):
		if not self.network_pointer:
			print('Neural network has not been initialized')
			return

		if len(network_input) == 0:
			print('Network input length was 0, no training was completed')
			return

		train_network = _network_ref.batchTrainNetwork
		train_network.argtypes = [c_void_p, POINTER(c_float), POINTER(c_float), POINTER(c_float), c_uint]

		input_array_type = c_float * (len(network_input) * self.num_input_neurons)
		output_array_type = c_float * (len(network_input) * self.num_output_neurons)
		actual_output_type = c_float * (len(network_input) * self.num_output_neurons)

		input_array = input_array_type()
		output_array = output_array_type()
		actual_output = actual_output_type()

		for i, training in enumerate(network_input):
			for k, neuron in enumerate(training):
				input_array[(i * len(training)) + k] = float(neuron)

		for i, training in enumerate(expected_output):
			for k, neuron in enumerate(training):
				output_array[(i * len(training)) + k] = float(neuron)

		train_network(self.network_pointer, input_array, output_array, actual_output, len(network_input))

		return actual_output


	def output(self, network_input):
		if not self.network_pointer:
			print('Neural network has not been initialized')
			return list()

		assert len(network_input) == self.num_input_neurons

		get_output = _network_ref.getNetworkOutputForInput
		get_output.argtypes = [c_void_p, POINTER(c_float), c_size_t, POINTER(c_float), c_size_t]

		input_array_type = c_float * self.num_input_neurons
		output_array_type = c_float * self.num_output_neurons

		input_array = input_array_type()
		output_array = output_array_type()

		for i, obj in enumerate(network_input):
			input_array[i] = obj

		get_output(self.network_pointer, input_array, self.num_input_neurons, output_array, self.num_output_neurons)

		return list(output_array)


	def save(self, filename):
		if not self.network_pointer:
			print('Neural network has not been initialized')
			return

		save_network = _network_ref.saveNetwork
		save_network.argtypes = [c_void_p, c_char_p, c_size_t]

		save_network(self.network_pointer, filename.encode('ascii'), len(filename))



if __name__ == '__main__':
	network = NeuralNetwork(10, 2, 5, 10, 0.1)

	single_input = [0.15, 0.45, 0.78, 0.04, 0.45, 0.73, 0.19, 0.11, 0.01, 0.11]

	input_values = single_input * 32
	output_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] * 32

	for i in range(3200):
		network.train(input_values, output_values, 32)

	output = network.output(single_input)
	for item in output:
		print('%.2f' % item)
