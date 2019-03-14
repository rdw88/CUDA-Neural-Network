#include "Extern.h"
#include "NeuralNetwork.h"

#include <iostream>
#include <vector>
#include <string>


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



extern "C" __declspec(dllexport) void batchTrainNetwork(void *_network, float *input, size_t inputSize, float *expectedOutput, size_t outputSize, size_t trainingsPerBatch) {
	NeuralNetwork *network = getNetworkPointer(_network);

	std::vector<float> networkInput;
	for (int i = 0; i < inputSize; i++) {
		networkInput.push_back(input[i]);
	}

	std::vector<float> networkOutput;
	for (int i = 0; i < outputSize; i++) {
		networkOutput.push_back(expectedOutput[i]);
	}

	network->batchTrain(&networkInput, &networkOutput, trainingsPerBatch);
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