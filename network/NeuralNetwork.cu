#include "NeuralNetwork.h"
#include "Util.h"
#include "GPU.h"

#include <chrono>
#include <iostream>
#include <fstream>
#include <stdlib.h>


using namespace std;


NeuralNetwork::NeuralNetwork() {
}


NeuralNetwork::NeuralNetwork(int numInputNeurons, int numHiddenLayers, int neuronsPerHiddenLayer, int numOutputNeurons, float learningRate) {
	createCublasContext();

	Layer *inputLayer = new Layer(this, numInputNeurons);
	Layer *previousLayer = inputLayer;

	for (int i = 0; i < numHiddenLayers; i++) {
		Layer *hiddenLayer = new Layer(this, neuronsPerHiddenLayer);
		previousLayer->setOutputLayer(hiddenLayer);
		previousLayer = hiddenLayer;
	}

	Layer *outputLayer = new Layer(this, numOutputNeurons);
	previousLayer->setOutputLayer(outputLayer);

	m_InputLayer = inputLayer;
	m_LearningRate = learningRate;
	m_NumHiddenLayers = numHiddenLayers;
}


Layer *NeuralNetwork::getInputLayer() {
	return m_InputLayer;
}


Layer *NeuralNetwork::getOutputLayer() {
	Layer *layer = m_InputLayer;

	while (layer->getNextLayer() != NULL) {
		layer = layer->getNextLayer();
	}

	return layer;
}


std::vector<float *> NeuralNetwork::loadSynapseMatricesIntoGPU() {
	std::vector<float *> synapseMatrices;
	Layer *layer = m_InputLayer->getNextLayer();

	while (layer != NULL) {
		std::vector<float> synapseMatrix = layer->getInputSynapseMatrix();

		synapseMatrices.push_back(gpu_loadVector(&synapseMatrix));

		layer = layer->getNextLayer();
	}
	
	return synapseMatrices;
}


std::vector<float *> NeuralNetwork::loadBatchIntoGPU(std::vector<float> *batch, unsigned int trainingsPerBatch) {
	std::vector<float *> batchVector;

	for (int i = 0; i < trainingsPerBatch; i++) {
		std::vector<float>::const_iterator start = batch->begin() + (i * m_InputLayer->getLayerSize());
		std::vector<float>::const_iterator end = batch->begin() + ((i + 1) * m_InputLayer->getLayerSize());

		std::vector<float> training(start, end);

		batchVector.push_back(gpu_loadVector(&training));
	}

	return batchVector;
}


std::vector<float *> NeuralNetwork::loadBatchIntoCPU(std::vector<float> *batch, unsigned int trainingsPerBatch) {
	std::vector<float *> batchVector;

	for (int i = 0; i < trainingsPerBatch; i++) {
		std::vector<float>::const_iterator start = batch->begin() + (i * m_InputLayer->getLayerSize());
		std::vector<float>::const_iterator end = batch->begin() + ((i + 1) * m_InputLayer->getLayerSize());

		std::vector<float> training(start, end);

		float *cpuVector = (float *) malloc(training.size() * sizeof(float));
		memcpy(cpuVector, training.data(), training.size() * sizeof(float));

		batchVector.push_back(cpuVector);
	}

	return batchVector;
}


std::vector<float *> NeuralNetwork::activateLayerResults(std::vector<float *> *layerResult, size_t layerSize) {
	std::vector<float *> cpuActivatedLayerResults;

	for (int i = 0; i < layerResult->size(); i++) {
		float *resultVector = (* layerResult)[i];
		float *cpuVector = gpu_unloadVector(resultVector, layerSize);

		gpu_freeMemory(resultVector);

		for (int k = 0; k < layerSize; k++) {
			float neuronValue = sigmoid(cpuVector[k]);
			cpuVector[k] = neuronValue;
		}

		float *gpuVector = gpu_loadVectorFromPointer(cpuVector, layerSize);
		(* layerResult)[i] = gpuVector;
		cpuActivatedLayerResults.push_back(cpuVector);
	}

	return cpuActivatedLayerResults;
}


std::vector<float *> NeuralNetwork::runTraining(std::vector<float> *batch, std::vector<std::vector<float *>> *layerResults, unsigned int trainingsPerBatch) {
	layerResults->push_back(loadBatchIntoCPU(batch, trainingsPerBatch));

	std::vector<float *> synapseMatrices = loadSynapseMatricesIntoGPU();
	std::vector<float *> batchedValueVector = loadBatchIntoGPU(batch, trainingsPerBatch);

	Layer *layer = m_InputLayer->getNextLayer();

	for (int i = 0; layer != NULL; i++) {
		std::vector<float *> batchedSynapseMatrix(trainingsPerBatch, synapseMatrices[i]);

		std::vector<float> layerBiasVector = layer->getBiasVector();
		std::vector<float *> batchedBiasVectors;

		for (int k = 0; k < trainingsPerBatch; k++) {
			batchedBiasVectors.push_back(gpu_loadVector(&layerBiasVector));
		}

		gpu_batchVectorMatrixMultiply(&batchedSynapseMatrix, &batchedValueVector, &batchedBiasVectors, layer->getPreviousLayer()->getLayerSize(), layer->getLayerSize(), trainingsPerBatch);

		std::vector<float *> cpuActivatedLayerResults = activateLayerResults(&batchedBiasVectors, layer->getLayerSize());
		layerResults->push_back(cpuActivatedLayerResults);

		for (int k = 0; k < batchedValueVector.size(); k++) {
			gpu_freeMemory(batchedValueVector[k]);
		}

		batchedValueVector.swap(batchedBiasVectors);

		layer = layer->getNextLayer();
	}

	for (int i = 0; i < synapseMatrices.size(); i++) {
		gpu_freeMemory(synapseMatrices[i]);
	}

	for (int i = 0; i < batchedValueVector.size(); i++) {
		float *cpuVector = gpu_unloadVector(batchedValueVector[i], getOutputLayer()->getLayerSize());
		gpu_freeMemory(batchedValueVector[i]);
		batchedValueVector[i] = cpuVector;
	}

	return batchedValueVector;
}


std::vector<float> NeuralNetwork::averageBatchedInputValues(std::vector<float *> *batch, unsigned int trainingsPerBatch, unsigned int layerSize) {
	std::vector<float> averageInputs;

	for (int i = 0; i < layerSize; i++) {
		float averageInput = 0;

		for (int k = 0; k < trainingsPerBatch; k++) {
			averageInput += (* batch)[k][i];
		}

		averageInputs.push_back(averageInput / trainingsPerBatch);
	}

	return averageInputs;
}


void NeuralNetwork::batchTrain(std::vector<float> *batch, std::vector<float> *expectedOutputs, std::vector<float> *actualOutputs, unsigned int trainingsPerBatch) {
	if (batch->size() != m_InputLayer->getLayerSize() * trainingsPerBatch) {
		std::cout << "Input data size not equal to (numInputNeurons * trainingsPerBatch)" << std::endl;
		return;
	}

	std::vector<std::vector<float *>> layerResults;
	
	std::vector<float *> trainingResults = runTraining(batch, &layerResults, trainingsPerBatch);
	for (int i = 0; i < trainingsPerBatch; i++) {
		for (int k = 0; k < getOutputLayer()->getLayerSize(); k++) {
			(* actualOutputs)[(i * getOutputLayer()->getLayerSize()) + k] = trainingResults[i][k];
		}
	}

	unsigned int outputLayerSize = getOutputLayer()->getLayerSize();
	std::vector<float> expected = (* expectedOutputs);
	for (int i = 0; i < outputLayerSize; i++) {
		float error = 0;

		for (int k = 0; k < trainingsPerBatch; k++) {
			float actualValue = trainingResults[k][i];
			float expectedValue = expected[(k * outputLayerSize) + i];
			error += ((actualValue - expectedValue) * sigmoidDerivative(actualValue)); // swapped here 
			//std::cout << "Error: " << error << ", ActualValue: " << actualValue << ", Expected: " << expectedValue << "sigmoid: " << sigmoidDerivative(actualValue) << std::endl;
		}

		//std::cout << "Final: " << (error / trainingsPerBatch) << std::endl;

		getOutputLayer()->setError(i, error / trainingsPerBatch);
	}

	getOutputLayer()->getPreviousLayer()->updateError(NULL);

	for (int i = 0; i < trainingResults.size(); i++) {
		free(trainingResults[i]);
	}

	Layer *layer = m_InputLayer->getNextLayer();
	for (int i = 0; layer != NULL; i++) {
		std::vector<float> averageInputs = averageBatchedInputValues(&layerResults[i], trainingsPerBatch, layer->getPreviousLayer()->getLayerSize());
		layer->applyWeights(&averageInputs);
		layer = layer->getNextLayer();
	}

	for (int i = 0; i < layerResults.size(); i++) {
		for (int k = 0; k < layerResults[i].size(); k++) {
			free(layerResults[i][k]);
		}
	}
}


std::vector<std::vector<float>> NeuralNetwork::train(std::vector<std::vector<float>> batch, std::vector<std::vector<float>> expectedOutputs) {
	if (batch.size() != expectedOutputs.size()) {
		std::cout << "Input and expected output vector sizes should be equal" << std::endl;
		return std::vector<std::vector<float>>();
	}

	unsigned int outputNeuronCount = getOutputLayer()->getLayerSize();

	std::vector<float> consolidatedBatch;
	std::vector<float> consolidatedExpected;
	std::vector<float> consolidatedActual(batch.size() * outputNeuronCount);

	for (int i = 0; i < batch.size(); i++) {
		for (int k = 0; k < batch[i].size(); k++) {
			consolidatedBatch.push_back(batch[i][k]);
		}
	}

	for (int i = 0; i < expectedOutputs.size(); i++) {
		for (int k = 0; k < expectedOutputs[i].size(); k++) {
			consolidatedExpected.push_back(expectedOutputs[i][k]);
		}
	}

	batchTrain(&consolidatedBatch, &consolidatedExpected, &consolidatedActual, batch.size());

	std::vector<std::vector<float>> actualOutputs;
	for (int i = 0; i < batch.size(); i++) {
		std::vector<float> trainingOutput;

		for (int k = 0; k < outputNeuronCount; k++) {
			trainingOutput.push_back(consolidatedActual[(i * outputNeuronCount) + k]);
		}

		actualOutputs.push_back(trainingOutput);
	}

	return actualOutputs;
}


std::vector<float> NeuralNetwork::getOutputForInput(std::vector<float> *input) {
	std::vector<std::vector<float *>> layerResults;
	runTraining(input, &layerResults, 1);

	unsigned int outputLayerSize = getOutputLayer()->getLayerSize();
	std::vector<float> outputValues;

	for (int i = 0; i < outputLayerSize; i++) {
		outputValues.push_back(layerResults[layerResults.size() - 1][0][i]);
	}

	for (int i = 0; i < layerResults.size(); i++) {
		for (int k = 0; k < layerResults[i].size(); k++) {
			free(layerResults[i][k]);
		}
	}

	return outputValues;
}


void NeuralNetwork::save(string filename) {
	fstream outputFile;
	outputFile.open(filename, fstream::out | fstream::trunc);

	if (!outputFile.is_open()) {
		cout << "Failed to open file " << filename << endl;
		return;
	}

	outputFile << m_InputLayer->getLayerSize() << endl;
	outputFile << m_NumHiddenLayers << endl;
	outputFile << m_InputLayer->getNextLayer()->getLayerSize() << endl;
	outputFile << getOutputLayer()->getLayerSize() << endl;
	outputFile << m_LearningRate << endl;

	Layer *layer = m_InputLayer->getNextLayer();
	while (layer != NULL) {
		std::vector<float> synapses = layer->getInputSynapseMatrix();
		std::vector<float> biases = layer->getBiasVector();

		string synapseString("");
		for (int i = 0; i < synapses.size(); i++) {
			synapseString += to_string(synapses[i]) + ",";
		}

		string biasString("");
		for (int i = 0; i < biases.size(); i++) {
			biasString += to_string(biases[i]) + ",";
		}

		outputFile << synapseString.substr(0, synapseString.size() - 1) << endl;
		outputFile << biasString.substr(0, biasString.size() - 1) << endl;

		layer = layer->getNextLayer();
	}

	outputFile.close();
}


NeuralNetwork *networkFromFile(string filename) {
	ifstream inputFile(filename);

	if (!inputFile.is_open()) {
		cout << "Failed to open file " << filename << endl;
		return NULL;
	}

	int numInputNeurons;
	int numHiddenLayers;
	int neuronsPerHiddenLayer;
	int numOutputNeurons;
	float learningRate;

	inputFile >> numInputNeurons;
	inputFile >> numHiddenLayers;
	inputFile >> neuronsPerHiddenLayer;
	inputFile >> numOutputNeurons;
	inputFile >> learningRate;

	string dummy;
	getline(inputFile, dummy);

	NeuralNetwork *network = new NeuralNetwork(numInputNeurons, numHiddenLayers, neuronsPerHiddenLayer, numOutputNeurons, learningRate);

	Layer *layer = network->getInputLayer()->getNextLayer();
	while (layer != NULL) {
		std::vector<float> matrix;
		std::vector<float> vector;

		string synapseMatrix;
		getline(inputFile, synapseMatrix);
		size_t position = 0;
		string weight;

		while ((position = synapseMatrix.find(",")) != string::npos) {
			weight = synapseMatrix.substr(0, position);
			matrix.push_back(stof(weight));
			synapseMatrix.erase(0, position + 1);
		}

		matrix.push_back(stof(synapseMatrix));

		string biasVector;
		getline(inputFile, biasVector);
		position = 0;
		string bias;

		while ((position = biasVector.find(",")) != string::npos) {
			bias = biasVector.substr(0, position);
			vector.push_back(stof(bias));
			biasVector.erase(0, position + 1);
		}

		vector.push_back(stof(biasVector));

		layer->setSynapseMatrix(matrix);
		layer->setBiasVector(vector);

		layer = layer->getNextLayer();
	}

	inputFile.close();

	return network;
}


float NeuralNetwork::getLearningRate() {
	return m_LearningRate;
}


/*int main(void) {
	createCublasContext();

	unsigned int numBatches = 3200;
	unsigned int trainingsPerBatch = 32;
	unsigned int numInputNeurons = 10;

	//NeuralNetwork network(numInputNeurons, 2, 5, 10, 0.1f);
	
	vector<vector<float>> inputBatches;
	vector<vector<float>> outputBatches;
	for (int i = 0; i < numBatches; i++) {
		vector<float> inputBatch;
		vector<float> outputBatch;

		for (int k = 0; k < trainingsPerBatch; k++) {
			inputBatch.push_back(0.15);
			inputBatch.push_back(0.45);
			inputBatch.push_back(0.78);
			inputBatch.push_back(0.04);
			inputBatch.push_back(0.45);
			inputBatch.push_back(0.73);
			inputBatch.push_back(0.19);
			inputBatch.push_back(0.11);
			inputBatch.push_back(0.01);
			inputBatch.push_back(0.11);

			outputBatch.push_back(0.3);
			outputBatch.push_back(0.7);
			outputBatch.push_back(0.1);
			outputBatch.push_back(0.0);
			outputBatch.push_back(0.9);
			outputBatch.push_back(0.9);
			outputBatch.push_back(0.4);
			outputBatch.push_back(0.6);
			outputBatch.push_back(0.2);
			outputBatch.push_back(0.1);
		}

		inputBatches.push_back(inputBatch);
		outputBatches.push_back(outputBatch);
	}

	auto start = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < numBatches; i++) {
		network.batchTrain(&inputBatches[i], &outputBatches[i], trainingsPerBatch);	
	}

	auto end = std::chrono::high_resolution_clock::now();
	cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << endl;

	NeuralNetwork *network = networkFromFile("archive/test.csv");

	std::vector<float> inputVector = inputBatches[0];

	std::vector<float> output = network->getOutputForInput(&inputVector);
	cout << "Output: " << endl;

	for (int i = 0; i < output.size(); i++) {
		cout << output[i] << endl;
	}

	//network.save("archive/test.csv");

	destroyCublasContext();

  	return 0;
}*/