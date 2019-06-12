import random
import os

from ctypes import CDLL, c_uint, c_int, c_float, c_void_p, c_char_p, c_size_t, c_bool, POINTER, Structure

import itertools


ANN_DLL_PATH = os.path.join(os.environ['ANN_DLL_PATH'], 'ann.dll')
_network_ref = CDLL(ANN_DLL_PATH)



class ActivationType:
	RELU, SIGMOID, SOFTMAX = range(3)



class Activation(Structure):
	_fields_ = [
		('activationType', c_int),
		('maxThreshold', c_float)
	]


	@staticmethod
	def relu(max_threshold=-1):
		max_c_int = (1 << 31) - 1

		activation = Activation()
		activation.activationType = ActivationType.RELU
		activation.maxThreshold = float(max_threshold) if max_threshold != -1 else float(max_c_int)

		return activation


	@staticmethod
	def sigmoid():
		activation = Activation()
		activation.activationType = ActivationType.SIGMOID
		return activation


	@staticmethod
	def softmax():
		activation = Activation()
		activation.activationType = ActivationType.SOFTMAX
		return activation



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

		learning_rate = float(lines[(num_layers * 3) + 1])
		batch_size = int(lines[(num_layers * 3) + 2])

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

		if len(network_input) != self.layer_sizes[0] * self.batch_size:
			print('Training input length must be equal to the input layer size!')
			return

		if len(expected_output) != self.layer_sizes[-1] * self.batch_size:
			print('Training output length must be equal to the output layer size!')
			return

		train_network = _network_ref.trainNetwork
		train_network.argtypes = [c_void_p, POINTER(c_float), POINTER(c_float)]

		input_array = (c_float * len(network_input))()
		output_array = (c_float * len(expected_output))()

		# For some reason, this copies network_* to *_array around 10x faster than using a for loop
		input_array[:] = network_input
		output_array[:] = expected_output

		train_network(self.network_pointer, input_array, output_array)


	def update(self, output_error):
		if not self.network_pointer:
			print('Neural network has not been initialized')
			return

		update_network = _network_ref.updateNetwork
		update_network.argtypes = [c_void_p, POINTER(c_float), c_uint]

		output_error_array = (c_float * len(output_error))()
		output_error_array[:] = output_error

		update_network(self.network_pointer, output_error_array, len(output_error))


	def output(self, network_input):
		if not self.network_pointer:
			print('Neural network has not been initialized')
			return list()

		if len(network_input) % self.layer_sizes[0] != 0:
			print('Input length must be equal to the input layer size!')
			return

		get_output = _network_ref.getNetworkOutputForInput
		get_output.argtypes = [c_void_p, POINTER(c_float), c_uint, POINTER(c_float), c_uint]

		output_size = int(len(network_input) / self.layer_sizes[0]) * self.layer_sizes[-1]

		input_array = (c_float * len(network_input))()
		output_array = (c_float * output_size)()

		input_array[:] = network_input

		get_output(self.network_pointer, input_array, len(network_input), output_array, output_size)

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


	def set_learning_rate(self, learning_rate):
		self.learning_rate = learning_rate
		
		set_lrate = _network_ref.setLearningRate
		set_lrate.argtypes = [c_void_p, c_float]

		set_lrate(self.network_pointer, learning_rate)


	def set_layer_activations(self, activations):
		set_activations = _network_ref.setLayerActivations
		set_activations.argtypes = [c_void_p, POINTER(Activation), c_uint]

		input_activations = (Activation * len(activations))(*activations)

		set_activations(self.network_pointer, input_activations, len(activations))


	def set_synapse_matrix(self, layer, matrix):
		set_matrix = _network_ref.setSynapseMatrix
		set_matrix.argtypes = [c_void_p, c_uint, POINTER(c_float), c_uint]

		input_matrix = (c_float * len(matrix))()
		input_matrix[:] = matrix

		set_matrix(self.network_pointer, layer, input_matrix, len(matrix))


	def set_bias_vector(self, layer, vector):
		set_bias = _network_ref.setBiasVector
		set_bias.argtypes = [c_void_p, c_uint, POINTER(c_float), c_uint]

		input_vector = (c_float * len(vector))()
		input_vector[:] = vector

		set_bias(self.network_pointer, layer, input_vector, len(vector))


	def set_calc_input_error(self, calculate):
		set_calc = _network_ref.setCalcInputLayerError
		set_calc.argtypes = [c_void_p, c_bool]

		set_calc(self.network_pointer, calculate)


	def get_synapse_matrix(self, layer):
		if layer == 0 or layer == len(self.layer_sizes) * -1:
			print('WARNING: Attempted to get the synapse matrix of the input layer')
			return None

		get_matrix = _network_ref.getSynapseMatrix
		get_matrix.argtypes = [c_void_p, c_uint, POINTER(c_float)]

		output_size = self.layer_sizes[layer] * self.layer_sizes[layer - 1]
		matrix = (c_float * output_size)()

		get_matrix(self.network_pointer, layer, matrix)

		return list(matrix)


	def get_bias_vector(self, layer):
		if layer == 0 or layer == len(self.layer_sizes) * -1:
			print('WARNING: Attempted to get the bias vector of the input layer')
			return None

		get_bias = _network_ref.getBiasVector
		get_bias.argtypes = [c_void_p, c_uint, POINTER(c_float)]

		vector = (c_float * self.layer_sizes[layer])()

		get_bias(self.network_pointer, layer, vector)

		return list(vector)


	def get_error_vector(self, layer):
		get_error = _network_ref.getErrorVector
		get_error.argtypes = [c_void_p, c_uint, POINTER(c_float)]

		vector = (c_float * self.layer_sizes[layer])()

		get_error(self.network_pointer, layer, vector)

		return list(vector)


	def get_layer_count(self):
		return len(self.layer_sizes)



if __name__ == '__main__':
	network = NeuralNetwork([10, 5, 5, 10], 32, 0.1)
	network.set_layer_activations([Activation.relu(max_threshold=5), Activation.relu(max_threshold=10), Activation.relu(max_threshold=15), Activation.sigmoid()])

	single_input = [0.15, 0.45, 0.78, 0.04, 0.45, 0.73, 0.19, 0.11, 0.01, 0.11]
	single_input_2 = [0.38, 0.92, 0.16, 0.63, 0.82, 0.11, 0.02, 0.73, 0.25, 0.68]

	input_values = single_input * 32
	input_values_2 = single_input_2 * 32

	output_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] * 32
	output_values_2 = [0.3, 0.9, 0.5, 0.0, 0.7, 0.4, 0.4, 0.1, 0.8, 0.3] * 32

	for i in range(3200):
		network.train(input_values, output_values)
		network.train(input_values_2, output_values_2)

	output = network.output(single_input)
	for i, item in enumerate(output):
		print('%.1f' % item, output_values[i])

	print('-' * 30)

	output = network.output(single_input_2)
	for i, item in enumerate(output):
		print('%.1f' % item, output_values_2[i])

	print('-' * 30)

	batched_input = single_input + single_input_2
	batched_output = network.output(batched_input)
	expected_batched_output = output_values[:10] + output_values_2[:10]

	for i, item in enumerate(batched_output):
		print('%.1f' % item, expected_batched_output[i])