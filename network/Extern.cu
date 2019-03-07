#include "Extern.h"
#include "NeuralNetwork.h"

#include <iostream>
#include <vector>


extern "C" __declspec(dllexport) void *createNetwork(int numInputNeurons, int numHiddenLayers, int neuronsPerHiddenLayer, int numOutputNeurons) {
	std::cout << "Creating network with the following paramters: " << numInputNeurons << " " << numHiddenLayers << " " << neuronsPerHiddenLayer << " " << numOutputNeurons << std::endl;
	NeuralNetwork *network = new NeuralNetwork(numInputNeurons, numHiddenLayers, neuronsPerHiddenLayer, numOutputNeurons);
	return (void *) network;
}



extern "C" __declspec(dllexport) void trainNetwork(void *_network, float *input, size_t inputSize, float *expectedOutput, size_t outputSize) {
	if (_network == NULL) {
		return;
	}

	NeuralNetwork *network = (NeuralNetwork *) _network;

	std::vector<float> networkInput;
	for (int i = 0; i < inputSize; i++) {
		networkInput.push_back(input[i]);
	}

	std::vector<float> networkOutput;
	for (int i = 0; i < outputSize; i++) {
		networkOutput.push_back(expectedOutput[i]);
	}

	network->train(&networkInput, &networkOutput);
}