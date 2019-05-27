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
#include <algorithm>


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
	NeuralNetwork *network = (NeuralNetwork *) _network;

	int inputLayerSize = network->getInputSize() * network->getBatchSize();
	int outputLayerSize = network->getOutputSize() * network->getBatchSize();

	vector<float> networkInput(inputLayerSize);
	vector<float> networkOutput(outputLayerSize);

	memcpy(&networkInput[0], input, inputLayerSize * sizeof(float));
	memcpy(&networkOutput[0], expectedOutput, outputLayerSize * sizeof(float));

	network->train(networkInput, networkOutput);
}



extern "C" __declspec(dllexport) void updateNetwork(void *_network, float *outputError, unsigned int outputErrorSize) {
	NeuralNetwork *network = (NeuralNetwork *) _network;

	if (outputErrorSize != network->getOutputSize()) {
		cout << "ERROR: Output error size does not equal the output layer size of the network." << endl;
		return;
	}

	vector<float> networkError(outputErrorSize);

	memcpy(&networkError[0], outputError, outputErrorSize * sizeof(float));

	network->updateNetwork(networkError);
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



extern "C" __declspec(dllexport) void setSynapseMatrix(void *_network, unsigned int layer, float *synapseMatrix, unsigned int matrixLength) {
	NeuralNetwork *network = getNetworkPointer(_network);

	vector<float> matrix { synapseMatrix, synapseMatrix + matrixLength };

	network->setSynapseMatrix(layer, matrix);
}



extern "C" __declspec(dllexport) void setBiasVector(void *_network, unsigned int layer, float *biasVectorRef, unsigned int vectorLength) {
	NeuralNetwork *network = getNetworkPointer(_network);

	vector<float> biasVector { biasVectorRef, biasVectorRef + vectorLength };

	network->setBiasVector(layer, biasVector);
}



extern "C" __declspec(dllexport) void setLearningRate(void *_network, float learningRate) {
	NeuralNetwork *network = getNetworkPointer(_network);
	network->setLearningRate(learningRate);
}



extern "C" __declspec(dllexport) void setLayerActivations(void *_network, Activation *activations, unsigned int activationSize) {
	NeuralNetwork *network = getNetworkPointer(_network);

	vector<Activation> layerActivations { activations, activations + activationSize };

	network->setLayerActivations(layerActivations);
}



extern "C" __declspec(dllexport) void setCalcInputLayerError(void *_network, bool calculate) {
	NeuralNetwork *network = getNetworkPointer(_network);
	network->setCalcInputLayerError(calculate);
}



extern "C" __declspec(dllexport) void getSynapseMatrix(void *_network, unsigned int layer, float *synapseMatrix) {
	NeuralNetwork *network = getNetworkPointer(_network);

	vector<float> matrix = network->getSynapseMatrixValues(layer);
	if (matrix.size() > 0) {
		memcpy(synapseMatrix, &matrix[0], matrix.size() * sizeof(float));
	}
}



extern "C" __declspec(dllexport) void getBiasVector(void *_network, unsigned int layer, float *biasVector) {
	NeuralNetwork *network = getNetworkPointer(_network);

	vector<float> biasVectorValues = network->getBiasVectorValues(layer);
	if (biasVectorValues.size() > 0) {
		memcpy(biasVector, &biasVectorValues[0], biasVectorValues.size() * sizeof(float));
	}
}