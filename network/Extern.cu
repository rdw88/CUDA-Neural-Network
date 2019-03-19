#include "Extern.h"
#include "NeuralNetwork.h"

#include <iostream>
#include <vector>
#include <string>
#include <stdlib.h>
#include <chrono>


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



extern "C" __declspec(dllexport) void batchTrainNetwork(void *_network, float *input, float *expectedOutput, float *actualOutput, unsigned int numTrainings) {
	NeuralNetwork *network = getNetworkPointer(_network);

	unsigned int inputLayerSize = network->getInputLayer()->getLayerSize();
	unsigned int outputLayerSize = network->getOutputLayer()->getLayerSize();

	std::vector<std::vector<float>> networkInput;
	std::vector<std::vector<float>> networkOutput;

	for (int i = 0; i < numTrainings; i++) {
		std::vector<float> training;

		for (int k = 0; k < inputLayerSize; k++) {
			training.push_back(input[(i * inputLayerSize) + k]);
		}

		networkInput.push_back(training);
	}

	for (int i = 0; i < numTrainings; i++) {
		std::vector<float> training;

		for (int k = 0; k < outputLayerSize; k++) {
			training.push_back(expectedOutput[(i * outputLayerSize) + k]);
		}

		networkOutput.push_back(training);
	}

	std::vector<std::vector<float>> actual = network->train(networkInput, networkOutput);

	for (int i = 0; i < actual.size(); i++) {
		memcpy(&actualOutput[i * outputLayerSize], actual[i].data(), outputLayerSize * sizeof(float));
	}
}



extern "C" __declspec(dllexport) void getNetworkOutputForInput(void *_network, float *input, size_t inputSize, float *output, size_t outputSize) {
	NeuralNetwork *network = getNetworkPointer(_network);

	std::vector<float> networkInput;
	for (int i = 0; i < inputSize; i++) {
		networkInput.push_back(input[i]);
	}

	std::vector<float> networkOutput = network->getOutputForInput(&networkInput);
	if (networkOutput.size() != outputSize) {
		std::cout << "ERROR: Expected output size of " << networkOutput.size() << " but got " << outputSize << std::endl;
		return;
	}

	for (int i = 0; i < networkOutput.size(); i++) {
		output[i] = networkOutput[i];
	}
}



extern "C" __declspec(dllexport) void saveNetwork(void *_network, char *filename, size_t filenameSize) {
	NeuralNetwork *network = getNetworkPointer(_network);

	if (filename == NULL) {
		std::cout << "ERROR: Filename is null!" << std::endl;
		return;
	}

	std::string file(filename, filenameSize);
	network->save(file);
}



extern "C" __declspec(dllexport) void *loadNetwork(char *filename, size_t filenameSize) {
	if (filename == NULL) {
		std::cout << "ERROR: Filename is null!" << std::endl;
		return NULL;
	}

	std::string file(filename, filenameSize);
	
	NeuralNetwork *network = networkFromFile(file);
	if (network == NULL) {
		std::cout << "Failed to load network from file!" << std::endl;
		return NULL;
	}

	return (void *) network;
}