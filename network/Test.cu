/**
	Test.cu
	April 4, 2019
	Ryan Wise

	A collection of unit tests for the neural network.
	
*/


#include "NeuralNetwork.h"
#include "GPU.h"

#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <map>
#include <math.h>
#include <cfloat>


using namespace std;


float test_sigmoid(float x) {
	return 1 / (1 + exp(-x));
}


float test_sigmoidDerivative(float x) {
	return x * (1 - x);
}


float test_softmax(float value, float sum) {
	return exp(value) / sum;
}


bool fequalf(float x, float y) {
	return abs(x - y) < FLT_EPSILON;
}


float squaredError(float x, float y) {
	return powf(x - y, 2.0);
}


static vector<unsigned int> defaultNetworkNeuronsPerLayer = {5, 3, 2, 4};
static Activation activationSigmoid = newActivation(SIGMOID);
static Activation activationRelu = newActivation(RELU);
static Activation activationSoftmax = newActivation(SOFTMAX);
static vector<Activation> defaultNetworkActivations = {activationSigmoid, activationSigmoid, activationSigmoid, activationSigmoid};
static unsigned int defaultNetworkBatchSize = 32;
static float defaultNetworkLearningRate = 0.1f;

NeuralNetwork *newDefaultNetwork() {
	NeuralNetwork *network = new NeuralNetwork(defaultNetworkNeuronsPerLayer, defaultNetworkBatchSize, defaultNetworkLearningRate);
	network->setLayerActivations(defaultNetworkActivations);
	return network;
}


static float testNetworkLearningRate = 0.1f;

static vector<float> layer1Matrix = {
	0.5, 1.0, 0.2,
	0.8, 0.1, 0.3
};

static vector<float> layer2Matrix = {
	0.1, 0.6,
	0.7, 0.2,
	0.9, 0.3
};

static vector<float> layer1Bias = {
	0.6,
	0.3
};

static vector<float> layer2Bias = {
	0.9,
	0.2,
	0.1
};

static vector<float> inputValues = {
	0.1,
	0.3,
	0.5,

	0.8,
	0.9,
	0.4
};

static vector<float> expectedOutput = {
	0.8,
	0.4,
	0.1,

	0.5,
	0.2,
	0.7
};


static vector<float> hiddenLayerValues = {
	test_sigmoid(((inputValues[0] * layer1Matrix[0]) + (inputValues[1] * layer1Matrix[1]) + (inputValues[2] * layer1Matrix[2])) + layer1Bias[0]),
	test_sigmoid(((inputValues[0] * layer1Matrix[3]) + (inputValues[1] * layer1Matrix[4]) + (inputValues[2] * layer1Matrix[5])) + layer1Bias[1]),

	test_sigmoid(((inputValues[3] * layer1Matrix[0]) + (inputValues[4] * layer1Matrix[1]) + (inputValues[5] * layer1Matrix[2])) + layer1Bias[0]),
	test_sigmoid(((inputValues[3] * layer1Matrix[3]) + (inputValues[4] * layer1Matrix[4]) + (inputValues[5] * layer1Matrix[5])) + layer1Bias[1])
};

static vector<float> outputLayerValues = {
	test_sigmoid(((hiddenLayerValues[0] * layer2Matrix[0]) + (hiddenLayerValues[1] * layer2Matrix[1])) + layer2Bias[0]),
	test_sigmoid(((hiddenLayerValues[0] * layer2Matrix[2]) + (hiddenLayerValues[1] * layer2Matrix[3])) + layer2Bias[1]),
	test_sigmoid(((hiddenLayerValues[0] * layer2Matrix[4]) + (hiddenLayerValues[1] * layer2Matrix[5])) + layer2Bias[2]),

	test_sigmoid(((hiddenLayerValues[2] * layer2Matrix[0]) + (hiddenLayerValues[3] * layer2Matrix[1])) + layer2Bias[0]),
	test_sigmoid(((hiddenLayerValues[2] * layer2Matrix[2]) + (hiddenLayerValues[3] * layer2Matrix[3])) + layer2Bias[1]),
	test_sigmoid(((hiddenLayerValues[2] * layer2Matrix[4]) + (hiddenLayerValues[3] * layer2Matrix[5])) + layer2Bias[2])
};

static vector<float> expectedTotalError = {
	(squaredError(outputLayerValues[0], expectedOutput[0]) + squaredError(outputLayerValues[1], expectedOutput[1]) + squaredError(outputLayerValues[2], expectedOutput[2])) / 3.0f,
	(squaredError(outputLayerValues[3], expectedOutput[3]) + squaredError(outputLayerValues[4], expectedOutput[4]) + squaredError(outputLayerValues[5], expectedOutput[5])) / 3.0f
};

static vector<float> softmaxOutputValues1 = {
	((hiddenLayerValues[0] * layer2Matrix[0]) + (hiddenLayerValues[1] * layer2Matrix[1])) + layer2Bias[0],
	((hiddenLayerValues[0] * layer2Matrix[2]) + (hiddenLayerValues[1] * layer2Matrix[3])) + layer2Bias[1],
	((hiddenLayerValues[0] * layer2Matrix[4]) + (hiddenLayerValues[1] * layer2Matrix[5])) + layer2Bias[2]
};


static vector<float> softmaxOutputValues2 = {
	((hiddenLayerValues[2] * layer2Matrix[0]) + (hiddenLayerValues[3] * layer2Matrix[1])) + layer2Bias[0],
	((hiddenLayerValues[2] * layer2Matrix[2]) + (hiddenLayerValues[3] * layer2Matrix[3])) + layer2Bias[1],
	((hiddenLayerValues[2] * layer2Matrix[4]) + (hiddenLayerValues[3] * layer2Matrix[5])) + layer2Bias[2]
};


static vector<float> expectedError = {
	(((outputLayerValues[0] - expectedOutput[0]) * test_sigmoidDerivative(outputLayerValues[0])) + ((outputLayerValues[3] - expectedOutput[3]) * test_sigmoidDerivative(outputLayerValues[3]))) / 2.0f,
	(((outputLayerValues[1] - expectedOutput[1]) * test_sigmoidDerivative(outputLayerValues[1])) + ((outputLayerValues[4] - expectedOutput[4]) * test_sigmoidDerivative(outputLayerValues[4]))) / 2.0f,
	(((outputLayerValues[2] - expectedOutput[2]) * test_sigmoidDerivative(outputLayerValues[2])) + ((outputLayerValues[5] - expectedOutput[5]) * test_sigmoidDerivative(outputLayerValues[5]))) / 2.0f
};


static vector<float> hiddenLayerExpectedError = {
	(expectedError[0] * layer2Matrix[0] * test_sigmoidDerivative((hiddenLayerValues[0] + hiddenLayerValues[2]) / 2.0f)) + (expectedError[1] * layer2Matrix[2] * test_sigmoidDerivative((hiddenLayerValues[0] + hiddenLayerValues[2]) / 2.0f)) + (expectedError[2] * layer2Matrix[4] * test_sigmoidDerivative((hiddenLayerValues[0] + hiddenLayerValues[2]) / 2.0f)),
	(expectedError[0] * layer2Matrix[1] * test_sigmoidDerivative((hiddenLayerValues[1] + hiddenLayerValues[3]) / 2.0f)) + (expectedError[1] * layer2Matrix[3] * test_sigmoidDerivative((hiddenLayerValues[1] + hiddenLayerValues[3]) / 2.0f)) + (expectedError[2] * layer2Matrix[5] * test_sigmoidDerivative((hiddenLayerValues[1] + hiddenLayerValues[3]) / 2.0f))
};


static vector<float> inputLayerExpectedError = {
	(hiddenLayerExpectedError[0] * layer1Matrix[0] * test_sigmoidDerivative((inputValues[0] + inputValues[3]) / 2.0f)) + (hiddenLayerExpectedError[1] * layer1Matrix[3] * test_sigmoidDerivative((inputValues[0] + inputValues[3]) / 2.0f)),
	(hiddenLayerExpectedError[0] * layer1Matrix[1] * test_sigmoidDerivative((inputValues[1] + inputValues[4]) / 2.0f)) + (hiddenLayerExpectedError[1] * layer1Matrix[4] * test_sigmoidDerivative((inputValues[1] + inputValues[4]) / 2.0f)),
	(hiddenLayerExpectedError[0] * layer1Matrix[2] * test_sigmoidDerivative((inputValues[2] + inputValues[5]) / 2.0f)) + (hiddenLayerExpectedError[1] * layer1Matrix[5] * test_sigmoidDerivative((inputValues[2] + inputValues[5]) / 2.0f))
};


static vector<float> newLayer1Matrix = {
	(layer1Matrix[0] - (testNetworkLearningRate * hiddenLayerExpectedError[0] * ((inputValues[0] + inputValues[3]) / 2.0f))),
	(layer1Matrix[1] - (testNetworkLearningRate * hiddenLayerExpectedError[0] * ((inputValues[1] + inputValues[4]) / 2.0f))),
	(layer1Matrix[2] - (testNetworkLearningRate * hiddenLayerExpectedError[0] * ((inputValues[2] + inputValues[5]) / 2.0f))),

	(layer1Matrix[3] - (testNetworkLearningRate * hiddenLayerExpectedError[1] * ((inputValues[0] + inputValues[3]) / 2.0f))),
	(layer1Matrix[4] - (testNetworkLearningRate * hiddenLayerExpectedError[1] * ((inputValues[1] + inputValues[4]) / 2.0f))),
	(layer1Matrix[5] - (testNetworkLearningRate * hiddenLayerExpectedError[1] * ((inputValues[2] + inputValues[5]) / 2.0f)))
};


static vector<float> newLayer2Matrix = {
	(layer2Matrix[0] - (testNetworkLearningRate * expectedError[0] * ((hiddenLayerValues[0] + hiddenLayerValues[2]) / 2.0f))),
	(layer2Matrix[1] - (testNetworkLearningRate * expectedError[0] * ((hiddenLayerValues[1] + hiddenLayerValues[3]) / 2.0f))),

	(layer2Matrix[2] - (testNetworkLearningRate * expectedError[1] * ((hiddenLayerValues[0] + hiddenLayerValues[2]) / 2.0f))),
	(layer2Matrix[3] - (testNetworkLearningRate * expectedError[1] * ((hiddenLayerValues[1] + hiddenLayerValues[3]) / 2.0f))),

	(layer2Matrix[4] - (testNetworkLearningRate * expectedError[2] * ((hiddenLayerValues[0] + hiddenLayerValues[2]) / 2.0f))),
	(layer2Matrix[5] - (testNetworkLearningRate * expectedError[2] * ((hiddenLayerValues[1] + hiddenLayerValues[3]) / 2.0f)))
};

NeuralNetwork *newTestNetwork() {
	vector<unsigned int> layerSizes = {3, 2, 3};
	vector<Activation> activations = {activationSigmoid, activationSigmoid, activationSigmoid};

	NeuralNetwork *network = new NeuralNetwork(layerSizes, 2, testNetworkLearningRate);

	network->setLayerActivations(activations);
	network->setSynapseMatrix(1, layer1Matrix);
	network->setSynapseMatrix(2, layer2Matrix);
	network->setBiasVector(1, layer1Bias);
	network->setBiasVector(2, layer2Bias);
	network->loadInput(inputValues);
	network->loadExpectedOutput(expectedOutput);

	return network;
}


void test_networkInitialization() {
	NeuralNetwork *network = newDefaultNetwork();

	vector<float *> biasVectors = network->getBiasVectors();
	vector<float *> errorVectors = network->getErrorVectors();
	vector<float **> synapseMatrices = network->getSynapseMatrices();
	vector<float **> valueVectors = network->getValueVectors();
	vector<unsigned int> layerSizes = network->getLayerSizes();
	unsigned int networkBatchSize = network->getBatchSize();
	float networkLearningRate = network->getLearningRate();

	assert(biasVectors.size() == defaultNetworkNeuronsPerLayer.size());
	assert(errorVectors.size() == defaultNetworkNeuronsPerLayer.size());
	assert(synapseMatrices.size() == defaultNetworkNeuronsPerLayer.size());
	assert(valueVectors.size() == defaultNetworkNeuronsPerLayer.size());
	assert(layerSizes.size() == defaultNetworkNeuronsPerLayer.size());
	assert(networkBatchSize == defaultNetworkBatchSize);
	assert(networkLearningRate == defaultNetworkLearningRate);

	assert(biasVectors[0] == NULL);
	assert(synapseMatrices[0] == NULL);
	assert(errorVectors[0] != NULL);

	for (int i = 0; i < defaultNetworkNeuronsPerLayer.size(); i++) {
		assert(defaultNetworkNeuronsPerLayer[i] == layerSizes[i]);
	}

	delete network;
}


void test_loadInput() {
	NeuralNetwork *network = newDefaultNetwork();

	vector<float> input = {0.2, 0.5, 0.3, 1.0, 0.7};
	vector<float> batchedInput;

	for (int i = 0; i < network->getBatchSize(); i++) {
		for (int k = 0; k < input.size(); k++) {
			batchedInput.push_back(input[k]);
		}
	}

	network->loadInput(batchedInput);

	float **batchedGPUInputVectors = network->getValueVectors()[0];

	float **batchedCPUInputVectors = (float **) allocPinnedMemory(network->getBatchSize() * sizeof(float *));
	gpu_copyMemory(batchedCPUInputVectors, batchedGPUInputVectors, network->getBatchSize() * sizeof(float *));

	for (int i = 0; i < network->getBatchSize(); i++) {
		float *gpuInputVector = batchedCPUInputVectors[i];
		float *cpuInputVector = (float *) allocPinnedMemory(network->getInputSize() * sizeof(float));
		gpu_copyMemory(cpuInputVector, gpuInputVector, network->getInputSize() * sizeof(float));

		for (int k = 0; k < network->getInputSize(); k++) {
			assert(fequalf(cpuInputVector[k], input[k]));
		}

		freePinnedMemory(cpuInputVector);
	}

	freePinnedMemory(batchedCPUInputVectors);
}


void test_loadExpectedOutput() {
	NeuralNetwork *network = newDefaultNetwork();

	vector<float> output = {0.2, 0.7, 0.5, 0.3};
	vector<float> batchedOutput;

	for (int i = 0; i < network->getBatchSize(); i++) {
		for (int k = 0; k < output.size(); k++) {
			batchedOutput.push_back(output[k]);
		}
	}

	network->loadExpectedOutput(batchedOutput);

	float *batchedGPUExpectedOutput = network->getExpectedOutput();
	float *cpuBatchedOutput = (float *) allocPinnedMemory(network->getOutputSize() * network->getBatchSize() * sizeof(float));

	gpu_copyMemory(cpuBatchedOutput, batchedGPUExpectedOutput, network->getOutputSize() * network->getBatchSize() * sizeof(float));

	for (int i = 0; i < network->getBatchSize(); i++) {
		for (int k = 0; k < output.size(); k++) {
			assert(fequalf(output[k], cpuBatchedOutput[(i * output.size()) + k]));
		}
	}

	freePinnedMemory(cpuBatchedOutput);
}


void verify_outputValues(NeuralNetwork *network, vector<float> expectedOutput) {
	network->feedForward();

	float **batchedOutputLayer = network->getOutputLayer();
	float **cpuBatchedOutputLayer = (float **) allocPinnedMemory(network->getBatchSize() * sizeof(float *));
	gpu_copyMemory(cpuBatchedOutputLayer, batchedOutputLayer, network->getBatchSize() * sizeof(float *));

	for (int i = 0; i < network->getBatchSize(); i++) {
		float *cpuOutputLayer = (float *) allocPinnedMemory(network->getOutputSize() * sizeof(float));
		gpu_copyMemory(cpuOutputLayer, cpuBatchedOutputLayer[i], network->getOutputSize() * sizeof(float));

		for (int k = 0; k < network->getOutputSize(); k++) {
			assert(fequalf(cpuOutputLayer[k], expectedOutput[(i * network->getOutputSize()) + k]));
		}

		freePinnedMemory(cpuOutputLayer);
	}

	freePinnedMemory(cpuBatchedOutputLayer);
}


void test_feedForward() {
	NeuralNetwork *network = newTestNetwork();
	verify_outputValues(network, outputLayerValues);	
}


void test_calculateError() {
	NeuralNetwork *network = newTestNetwork();

	network->feedForward();
	network->calculateError();

	vector<float> networkError = network->getErrorVectorForLayer(network->getLayerCount() - 1);
	for (int i = 0; i < networkError.size(); i++) {
		assert(fequalf(networkError[i], expectedError[i]));
	}
}


void test_gpu_softmax() {
	NeuralNetwork *network = newTestNetwork();
	vector<Activation> activations = {activationSigmoid, activationSigmoid, activationSoftmax};

	network->setLayerActivations(activations);

	vector<float> outputValues = softmaxOutputValues1;
	for (int i = 0; i < softmaxOutputValues2.size(); i++) {
		outputValues.push_back(softmaxOutputValues2[i]);
	}

	verify_outputValues(network, outputValues);
}


void test_backpropogate() {
	NeuralNetwork *network = newTestNetwork();

	network->feedForward();
	network->calculateError();
	network->backpropogate();

	/* Verify that the input layer error is not calculated since we have not set the flag */
	vector<float> networkError = network->getErrorVectorForLayer(0);
	for (int i = 0; i < networkError.size(); i++) {
		assert(networkError[i] == 0.0f);
	}

	networkError = network->getErrorVectorForLayer(network->getLayerCount() - 2);
	for (int i = 0; i < networkError.size(); i++) {
		assert(fequalf(networkError[i], hiddenLayerExpectedError[i]));
	}
}


void test_backpropogateWithInputLayerError() {
	NeuralNetwork *network = newTestNetwork();

	network->setCalcInputLayerError(true);
	network->feedForward();
	network->calculateError();
	network->backpropogate();

	float *gpuInputLayerError = network->getErrorVectors()[0];
	assert(gpuInputLayerError != NULL);

	vector<float> networkError = network->getErrorVectorForLayer(0);
	for (int i = 0; i < networkError.size(); i++) {
		assert(fequalf(networkError[i], inputLayerExpectedError[i]));
	}
}


void verify_applyWeights(NeuralNetwork *network) {
	vector<vector<float>> expectedMatrices = { newLayer1Matrix, newLayer2Matrix };

	for (int i = 1; i < network->getSynapseMatrices().size(); i++) {
		unsigned int matrixSize = network->getLayerSizes()[i] * network->getLayerSizes()[i - 1];
		vector<float> synapseMatrix = network->getSynapseMatrixValues(i);

		assert(matrixSize == synapseMatrix.size());

		for (int k = 0; k < matrixSize; k++) {
			assert(fequalf(synapseMatrix[k], expectedMatrices[i - 1][k]));
		}
		
		/* Verify the copies of each synapse matrix used for each batch also receive the updated weights */
		float *gpuCopiedSynapseMatrix = network->getCPUSynapseMatrices()[i][1];
		float *cpuCopiedSynapseMatrix = (float *) allocPinnedMemory(matrixSize * sizeof(float));
		gpu_copyMemory(cpuCopiedSynapseMatrix, gpuCopiedSynapseMatrix, matrixSize * sizeof(float));

		for (int k = 0; k < matrixSize; k++) {
			assert(fequalf(synapseMatrix[k], cpuCopiedSynapseMatrix[k]));
		}

		freePinnedMemory(cpuCopiedSynapseMatrix);
	}
}


void test_updateNetwork() {
	NeuralNetwork *network = newTestNetwork();
	
	network->feedForward();
	network->updateNetwork(expectedError);

	verify_applyWeights(network);
}


void test_applyWeights() {
	NeuralNetwork *network = newTestNetwork();

	network->feedForward();
	network->calculateError();
	network->backpropogate();
	network->applyWeights();

	verify_applyWeights(network);
}


void test_mse_loss() {
	NeuralNetwork *network = newTestNetwork();

	network->setLossFunction(MEAN_SQUARED_ERROR);
	network->feedForward();
	network->calculateError();

	vector<float> totalError = network->getTotalError();

	assert(totalError.size() == network->getBatchSize());
	assert(totalError.size() == expectedTotalError.size());

	for (int i = 0; i < totalError.size(); i++) {
		assert(fequalf(totalError[i], expectedTotalError[i]));
	}
}


void test_train() {
	vector<unsigned int> trainNetworkLayerSizes = { 784, 784, 500, 1000, 10 };
	vector<Activation> layerActivations = { activationRelu, activationRelu, activationRelu, activationRelu, activationSigmoid };

	NeuralNetwork network(trainNetworkLayerSizes, 32, 0.1);
	network.setLayerActivations(layerActivations);

	vector<float> input;
	vector<float> expectedOutput;

	for (int i = 0; i < network.getInputSize(); i++) {
		input.push_back(i / 784.0);
	}

	for (int i = 0; i < network.getOutputSize(); i++) {
		expectedOutput.push_back(i / 10.0);
	}

	vector<float> actualInput;
	vector<float> actualOutput;

	for (int i = 0; i < network.getBatchSize(); i++) {
		for (int k = 0; k < network.getInputSize(); k++) {
			actualInput.push_back(input[k]);
		}
	}

	for (int i = 0; i < network.getBatchSize(); i++) {
		for (int k = 0; k < network.getOutputSize(); k++) {
			actualOutput.push_back(expectedOutput[k]);
		}
	}

	for (int i = 0; i < 1875; i++) {
		network.train(actualInput, actualOutput);
	}

	vector<float> networkOutput = network.getOutputForInput(input);
	for (int i = 0; i < network.getOutputSize(); i++) {
		float roundedOutput = roundf(networkOutput[i] * 10) / 10;
		assert(fequalf(roundedOutput, expectedOutput[i]));
	}
}


void test_networkFromFile() {
	NeuralNetwork *network = newTestNetwork();

	Activation relu = newActivation(RELU);
	relu.maxThreshold = 10;

	Activation sigmoid = newActivation(SIGMOID);
	sigmoid.maxThreshold = 5;
	
	vector<Activation> activations = { relu, relu, sigmoid };
	network->setLayerActivations(activations);

	network->save("test_networkFromFile.csv");

	NeuralNetwork *loadedNetwork = networkFromFile("test_networkFromFile.csv");

	assert(loadedNetwork->getBiasVectors().size() == network->getBiasVectors().size());
	assert(loadedNetwork->getErrorVectors().size() == network->getErrorVectors().size());
	assert(loadedNetwork->getSynapseMatrices().size() == network->getSynapseMatrices().size());
	assert(loadedNetwork->getValueVectors().size() == network->getValueVectors().size());
	assert(loadedNetwork->getLayerSizes().size() == network->getLayerSizes().size());
	assert(loadedNetwork->getLayerActivations().size() == network->getLayerSizes().size());
	assert(loadedNetwork->getBatchSize() == network->getBatchSize());
	assert(loadedNetwork->getLearningRate() == network->getLearningRate());
	assert(loadedNetwork->getBiasVectors()[0] == NULL);
	assert(loadedNetwork->getSynapseMatrices()[0] == NULL);
	assert(loadedNetwork->getErrorVectors()[0] != NULL);

	for (int i = 0; i < loadedNetwork->getLayerSizes().size(); i++) {
		assert(loadedNetwork->getLayerSizes()[i] == network->getLayerSizes()[i]);

		vector<Activation> layerActivations = loadedNetwork->getLayerActivations();
		assert(layerActivations[i].activationType == activations[i].activationType);
		assert(fequalf(layerActivations[i].maxThreshold, activations[i].maxThreshold));
	}

	for (int i = 1; i < loadedNetwork->getBiasVectors().size(); i++) {
		vector<float> loadedNetworkBias = loadedNetwork->getBiasVectorValues(i);
		vector<float> networkBias = network->getBiasVectorValues(i);

		vector<float> loadedNetworkMatrix = loadedNetwork->getSynapseMatrixValues(i);
		vector<float> networkMatrix = network->getSynapseMatrixValues(i);

		assert(loadedNetworkBias.size() == networkBias.size());
		assert(loadedNetworkMatrix.size() == networkMatrix.size());

		for (int k = 0; k < loadedNetworkBias.size(); k++) {
			assert(fequalf(loadedNetworkBias[k], networkBias[k]));
		}

		for (int k = 0; k < loadedNetworkMatrix.size(); k++) {
			assert(fequalf(loadedNetworkMatrix[k], networkMatrix[k]));
		}
	}

	remove("test_networkFromFile.csv");
}


void initSetupData() {
	float sum1 = 0;
	for (int i = 0; i < softmaxOutputValues1.size(); i++) {
		sum1 += exp(softmaxOutputValues1[i]);
	}

	float sum2 = 0;
	for (int i = 0; i < softmaxOutputValues2.size(); i++) {
		sum2 += exp(softmaxOutputValues2[i]);
	}

	softmaxOutputValues1[0] = test_softmax(softmaxOutputValues1[0], sum1);
	softmaxOutputValues1[1] = test_softmax(softmaxOutputValues1[1], sum1);
	softmaxOutputValues1[2] = test_softmax(softmaxOutputValues1[2], sum1);

	softmaxOutputValues2[0] = test_softmax(softmaxOutputValues2[0], sum2);
	softmaxOutputValues2[1] = test_softmax(softmaxOutputValues2[1], sum2);
	softmaxOutputValues2[2] = test_softmax(softmaxOutputValues2[2], sum2);
}


int main(void) {
	initSetupData();

	map<string, void (*)()> tests;

	tests["Test 00: test_networkInitialization"] = &test_networkInitialization;
	tests["Test 01: test_loadInput"] = &test_loadInput;
	tests["Test 02: test_loadExpectedOutput"] = &test_loadExpectedOutput;
	tests["Test 03: test_feedForward"] = &test_feedForward;
	tests["Test 04: test_calculateError"] = &test_calculateError;
	tests["Test 05: test_backpropogate"] = &test_backpropogate;
	tests["Test 06: test_backpropogateWithInputLayerError"] = &test_backpropogateWithInputLayerError;
	tests["Test 07: test_applyWeights"] = &test_applyWeights;
	tests["Test 08: test_updateNetwork"] = &test_updateNetwork;
	tests["Test 09: test_networkFromFile"] = &test_networkFromFile;
	tests["Test 10: test_gpu_softmax"] = &test_gpu_softmax;
	tests["Test 11: test_mse_loss"] = &test_mse_loss;
	tests["Test 12: test_train"] = &test_train;

	for (auto const& x : tests) {
		cout << x.first << "() ... ";

		x.second();

		cout << "pass" << endl;
	}

	return 0;
}