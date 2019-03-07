import random

import ctypes
from ctypes import *


ANN_DLL_PATH = 'network/bin/ann.dll'


class NeuralNetwork(object):
	def __init__(self, num_input_neurons, num_hidden_layers, neurons_per_hidden_layer, num_output_neurons):
		self.network_ref = ctypes.CDLL(ANN_DLL_PATH)

		create_network = self.network_ref.createNetwork
		create_network.restype = c_void_p

		self.network_pointer = create_network(num_input_neurons, num_hidden_layers, neurons_per_hidden_layer, num_output_neurons)


	def train(self, network_input, expected_output):
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



if __name__ == '__main__':
	network = NeuralNetwork(768, 2, 350, 768)

	random_input = [random.random() for i in range(768)]

	for i in range(100):
		network.train(random_input, random_input)