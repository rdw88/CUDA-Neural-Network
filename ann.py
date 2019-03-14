import random

import ctypes
from ctypes import *


ANN_DLL_PATH = 'network/bin/ann.dll'


class NeuralNetwork(object):
	def __init__(self, num_input_neurons, num_hidden_layers, neurons_per_hidden_layer, num_output_neurons, learning_rate):
		self.network_ref = ctypes.CDLL(ANN_DLL_PATH)

		create_network = self.network_ref.createNetwork
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


	def train(self, network_input, expected_output):
		if not self.network_pointer:
			print('Neural network has not been initialized')
			return

		train_network = self.network_ref.trainNetwork
		train_network.argtypes = [c_void_p, POINTER(c_float), c_size_t, POINTER(c_float), c_size_t]

		input_array_type = c_float * len(network_input)
		output_array_type = c_float * len(expected_output)

		input_array = input_array_type()
		output_array = output_array_type()

		for i, obj in enumerate(network_input):
			input_array[i] = obj

		for i, obj in enumerate(expected_output):
			output_array[i] = obj

		train_network(self.network_pointer, input_array, len(network_input), output_array, len(expected_output))


	def output(self):
		if not self.network_pointer:
			print('Neural network has not been initialized')
			return list()

		get_output = self.network_ref.getNetworkOutput
		get_output.argtypes = [c_void_p, POINTER(c_float), c_size_t]

		output_array_type = c_float * self.num_output_neurons
		output_array = output_array_type()

		get_output(self.network_pointer, output_array, self.num_output_neurons)

		return list(output_array)



if __name__ == '__main__':
	network = NeuralNetwork(3, 2, 6, 10, 0.2)

	input_values = [0.1, 0.6, 0.3]
	output_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

	for i in range(1000):
		network.train(input_values, output_values)

	output = network.output()
	for item in output:
		print('%.2f' % item)
