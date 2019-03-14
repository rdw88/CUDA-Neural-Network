#include "Extern.h"
#include "NeuralNetwork.h"

#include <iostream>
#include <vector>


NeuralNetwork *getNetworkPointer(void *pointer) {
	if (pointer == NULL) {
		return NULL;
	}

	return (NeuralNetwork *) pointer;
}


extern "C" __declspec(dllexport) void *createNetwork(int numInputNeurons, int numHiddenLayers, int neuronsPerHiddenLayer, int numOutputNeurons, float learningRate) {
	std::cout << "Creating network with the following paramters: " << numInputNeurons << " " << numHiddenLayers << " " << neuronsPerHiddenLayer << " " << numOutputNeurons << " " << learningRate << std::endl;
	NeuralNetwork *network = new NeuralNetwork(numInputNeurons, numHiddenLayers, neuronsPerHiddenLayer, numOutputNeurons, learningRate);
	return (void *) network;
}



extern "C" __declspec(dllexport) void trainNetwork(void *_network, float *input, size_t inputSize, float *expectedOutput, size_t outputSize) {
	NeuralNetwork *network = getNetworkPointer(_network);

	std::vector<float> networkInput;
	for (int i = 0; i < inputSize; i++) {
		networkInput.push_back(input[i]);
	}

	std::vector<float> networkOutput;
	for (int i = 0; i < outputSize; i++) {
		networkOutput.push_back(expectedOutput[i]);
	}

	//network->train(&networkInput, &networkOutput);
}



extern "C" __declspec(dllexport) void getNetworkOutput(void *_network, float *output, size_t outputSize) {
	NeuralNetwork *network = getNetworkPointer(_network);
/*
	std::vector<float> networkOutput = network->getOutput();
	if (networkOutput.size() != outputSize) {
		std::cout << "ERROR: Expected output size of " << networkOutput.size() << " but got " << outputSize << std::endl;
		return;
	}

	for (int i = 0; i < networkOutput.size(); i++) {
		output[i] = networkOutput[i];
	}
	*/
}