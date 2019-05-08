/**
 * Extern.cu
 * April 4, 2019
 * Ryan Wise
 * 
 * An external interface to the neural network that can be compiled into a DLL and invoked from another programming language.
 * 
 */


#include "Extern.h"
#include "NeuralNetwork.h"

#include <iostream>
#include <vector>
#include <string>
#include <stdlib.h>


using namespace std;


NeuralNetwork *getNetworkPointer(void *pointer) {
	if (pointer == NULL) {
		return NULL;
	}

	return (NeuralNetwork *) pointer;
}



extern "C" __declspec(dllexport) void *createNetwork(unsigned int *layerSizes, unsigned int numLayers, unsigned int batchSize, float learningRate) {
	vector<unsigned int> layerSizeVector;

	for (int i = 0; i < numLayers; i++) {
		layerSizeVector.push_back(layerSizes[i]);
	}

	NeuralNetwork *network = new NeuralNetwork(layerSizeVector, batchSize, learningRate);
	return (void *) network;
}



extern "C" __declspec(dllexport) void trainNetwork(void *_network, float *input, float *expectedOutput) {
	NeuralNetwork *network = getNetworkPointer(_network);

	int inputLayerSize = network->getInputSize() * network->getBatchSize();
	int outputLayerSize = network->getOutputSize() * network->getBatchSize();

	vector<float> networkInput { input, input + inputLayerSize };
	vector<float> networkOutput { expectedOutput, expectedOutput + outputLayerSize };

	network->train(networkInput, networkOutput);
}



extern "C" __declspec(dllexport) void getNetworkOutputForInput(void *_network, float *input, unsigned int inputSize, float *output, unsigned int outputSize) {
	NeuralNetwork *network = getNetworkPointer(_network);

	vector<float> networkInput { input, input + inputSize };
	vector<float> networkOutput = network->getOutputForInput(networkInput);

	if (networkOutput.size() != outputSize) {
		cout << "ERROR: Expected output size of " << networkOutput.size() << " but got " << outputSize << endl;
		return;
	}

	memcpy(output, networkOutput.data(), networkOutput.size() * sizeof(float));
}



extern "C" __declspec(dllexport) void saveNetwork(void *_network, char *filename, size_t filenameSize) {
	NeuralNetwork *network = getNetworkPointer(_network);

	if (filename == NULL) {
		cout << "ERROR: Filename is null!" << endl;
		return;
	}

	string file(filename, filenameSize);
	network->save(file);
}



extern "C" __declspec(dllexport) void *loadNetwork(char *filename, size_t filenameSize) {
	if (filename == NULL) {
		cout << "ERROR: Filename is null!" << endl;
		return NULL;
	}

	string file(filename, filenameSize);
	
	NeuralNetwork *network = networkFromFile(file);
	if (network == NULL) {
		cout << "Failed to load network from file!" << endl;
		return NULL;
	}

	return (void *) network;
}



extern "C" __declspec(dllexport) void setLearningRate(void *_network, float learningRate) {
	NeuralNetwork *network = getNetworkPointer(_network);
	network->setLearningRate(learningRate);
}



extern "C" __declspec(dllexport) void setLayerActivations(void *_network, int *activations, unsigned int activationSize) {
	NeuralNetwork *network = getNetworkPointer(_network);

	vector<Activation> layerActivations;
	for (int i = 0; i < activationSize; i++) {
		layerActivations.push_back(static_cast<Activation>(activations[i]));
	}

	network->setLayerActivations(layerActivations);
}