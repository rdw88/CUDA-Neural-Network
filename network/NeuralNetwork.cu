/**
 * NeuralNetwork.cu
 * April 4, 2019
 * Ryan Wise
 * 
 * An implementation of a neural network that runs on a CUDA-enabled NVIDIA GPU.
 * 
 */


#include "NeuralNetwork.h"
#include "Util.h"
#include "GPU.h"

#include <iostream>
#include <fstream>
#include <stdlib.h>


using namespace std;


/**
	Do not use.

	The default constructor for the neural network.
*/
NeuralNetwork::NeuralNetwork() {
}


/**
	Constructs a new neural network.

	@param neuronsPerLayer The number of neurons per layer.
	@param batchSize The batch size to be used in training.
	@param learningRate The learning rate of the network.
*/
NeuralNetwork::NeuralNetwork(vector<unsigned int> neuronsPerLayer, unsigned int batchSize, float learningRate) {
	createCublasContext();

	for (int i = 0; i < neuronsPerLayer.size(); i++)
		m_LayerSizes.push_back(neuronsPerLayer[i]);

	m_BatchSize = batchSize;
	m_LearningRate = learningRate;

	/* Initialize vector and matrix memory in GPU */

	m_SynapseMatrices.push_back(NULL); // Input layer has no synapse matrix (no incoming synapses).
	m_CpuSynapseMatrices.push_back(NULL);
	m_BiasVectors.push_back(NULL); // Input layer has no bias.
	m_ErrorVectors.push_back(NULL); // Input layer does not have error.

	float **gpuInputValueVectors = (float **) gpu_allocMemory(m_BatchSize * sizeof(float *));
	float **inputValueVectors = (float **) allocPinnedMemory(m_BatchSize * sizeof(float *)); // Input layer has values (network input)

	for (int i = 0; i < m_BatchSize; i++) {
		inputValueVectors[i] = (float *) gpu_allocMemory(m_LayerSizes[0] * sizeof(float));
	}

	gpu_copyMemory(gpuInputValueVectors, inputValueVectors, m_BatchSize * sizeof(float *));

	m_ValueVectors.push_back(gpuInputValueVectors);
	m_CpuValueVectors.push_back(inputValueVectors);

	for (int i = 1; i < m_LayerSizes.size(); i++) {
		float **valueVectors = (float **) allocPinnedMemory(m_BatchSize * sizeof(float *));

		for (int k = 0; k < m_BatchSize; k++) {
			valueVectors[k] = (float *) gpu_allocMemory(m_LayerSizes[i] * sizeof(float));
		}

		float *biasVector = (float *) gpu_allocMemory(m_LayerSizes[i] * sizeof(float));
		float *errorVector = (float *) gpu_allocMemory(m_LayerSizes[i] * sizeof(float));

		vector<float> initialBiases(m_LayerSizes[i]);
		for (int k = 0; k < m_LayerSizes[i]; k++) {
			initialBiases[k] = randomWeight(m_LayerSizes[i - 1]);
		}

		gpu_copyMemory(biasVector, &(initialBiases[0]), m_LayerSizes[i] * sizeof(float));

		unsigned int matrixSize = m_LayerSizes[i] * m_LayerSizes[i - 1];
		float **synapseMatrices = (float **) allocPinnedMemory(m_BatchSize * sizeof(float *));

		for (int k = 0; k < m_BatchSize; k++) {
			synapseMatrices[k] = (float *) gpu_allocMemory(matrixSize * sizeof(float));	
		}

		vector<float> initialWeights(matrixSize);
		for (int k = 0; k < matrixSize; k++) {
			initialWeights[k] = randomWeight(m_LayerSizes[i - 1]);
		}

		for (int k = 0; k < m_BatchSize; k++) {
			gpu_copyMemory(synapseMatrices[k], &(initialWeights[0]), matrixSize * sizeof(float));	
		}

		float **gpuValueVectors = (float **) gpu_allocMemory(m_BatchSize * sizeof(float *));
		float **gpuSynapseMatrices = (float **) gpu_allocMemory(m_BatchSize * sizeof(float *));

		gpu_copyMemory(gpuValueVectors, valueVectors, m_BatchSize * sizeof(float *));
		gpu_copyMemory(gpuSynapseMatrices, synapseMatrices, m_BatchSize * sizeof(float *));

		m_CpuValueVectors.push_back(valueVectors);
		m_CpuSynapseMatrices.push_back(synapseMatrices);

		m_ValueVectors.push_back(gpuValueVectors);
		m_SynapseMatrices.push_back(gpuSynapseMatrices);

		m_BiasVectors.push_back(biasVector);
		m_ErrorVectors.push_back(errorVector);
	}

	m_ExpectedOutput = (float *) gpu_allocMemory(m_BatchSize * getOutputSize() * sizeof(float));
}


/**
	Deallocates all resources created by the network, including memory allocated in GPU memory.
*/
NeuralNetwork::~NeuralNetwork() {
	for (int i = 1; i < m_SynapseMatrices.size(); i++) {
		for (int k = 0; k < m_BatchSize; k++) {
			gpu_freeMemory(m_CpuSynapseMatrices[i][k]);
		}

		gpu_freeMemory(m_SynapseMatrices[i]);
		freePinnedMemory(m_CpuSynapseMatrices[i]);
	}

	for (int i = 1; i < m_BiasVectors.size(); i++) {
		gpu_freeMemory(m_BiasVectors[i]);
	}

	for (int i = 1; i < m_ErrorVectors.size(); i++) {
		gpu_freeMemory(m_ErrorVectors[i]);
	}

	for (int i = 0; i < m_ValueVectors.size(); i++) {
		for (int k = 0; k < m_BatchSize; k++) {
			gpu_freeMemory(m_CpuValueVectors[i][k]);
		}

		gpu_freeMemory(m_ValueVectors[i]);
		freePinnedMemory(m_CpuValueVectors[i]);
	}

	gpu_freeMemory(m_ExpectedOutput);

	destroyCublasContext();
}


/**
	Train the neural network. The network makes predictions based on the training examples provided as input
	and corrects itself through backpropogation based on the expected outputs.

	@param batch A vector containing the input batch. The size of this vector should be the batch size of the 
				 network multiplied by the number of input neurons.
	@param expectedOutput The expected output values for the input. The size of this vector should be the batch
						  size of the network multiplied by the number of output neurons.
*/
void NeuralNetwork::train(vector<float> batch, vector<float> expectedOutput) {
	if (batch.size() / m_BatchSize != getInputSize()) {
		cout << "Batch size should be equal to batchSize * inputLayerSize" << endl;
		return;
	}

	if (expectedOutput.size() / m_BatchSize != getOutputSize()) {
		cout << "Expected output size should be equal to batchSize * outputLayerSize" << endl;
		return;
	}

	loadInput(batch);
	loadExpectedOutput(expectedOutput);
	feedForward();
	updateNetwork();
}


/**
	Load the input vector into the network.

	@param input A vector containing input values for the network.
*/
void NeuralNetwork::loadInput(vector<float> input) {
	if (input.size() != m_BatchSize * getInputSize()) {
		cout << "Input size must be batchSize * inputSize" << endl;
		return;
	}

	float *pinnedInputMemory = (float *) allocPinnedMemory(input.size() * sizeof(float));

	gpu_copyMemory(pinnedInputMemory, input.data(), input.size() * sizeof(float));

	/* Load input data into GPU memory */
	for (int i = 0; i < m_BatchSize; i++) {
		float *inputLayerBatchPointer = m_CpuValueVectors[0][i];
		float *trainingSubset = &pinnedInputMemory[i * getInputSize()];

		gpu_copyMemory(inputLayerBatchPointer, trainingSubset, getInputSize() * sizeof(float));
	}

	freePinnedMemory(pinnedInputMemory);
}


/**
	Load the expected output vector.

	@param expectedOutput A vector containing the expected output of the network.
*/
void NeuralNetwork::loadExpectedOutput(vector<float> expectedOutput) {
	if (expectedOutput.size() != m_BatchSize * getOutputSize()) {
		cout << "Expected output size must be batchSize * outputSize" << endl;
		return;
	}

	float *pinnedOutputMemory = (float *) allocPinnedMemory(expectedOutput.size() * sizeof(float));

	gpu_copyMemory(pinnedOutputMemory, expectedOutput.data(), expectedOutput.size() * sizeof(float));
	gpu_copyMemory(m_ExpectedOutput, pinnedOutputMemory, expectedOutput.size() * sizeof(float));

	freePinnedMemory(pinnedOutputMemory);
}


/**
	Get the output of the network given an input.

	@param input The input values to be assigned to the input neurons.
	@return A vector containing the values of the output neurons derived from the supplied input.
*/
vector<float> NeuralNetwork::getOutputForInput(vector<float> input) {
	if (input.size() != getInputSize()) {
		cout << "Input size must be equal to number of input neurons!" << endl;
		return vector<float>();
	}

	if (m_BatchSize > 1) {
		unsigned int inputSize = input.size();

		for (int i = 0; i < m_BatchSize - 1; i++) {
			for (int k = 0; k < inputSize; k++) {
				input.push_back(input[k]);
			}
		}
	}

	loadInput(input);
	feedForward();

	return getCurrentOutput();
}


/**
	Perform the feed forward step of training. This step takes the current input set on the network's input neurons
	and updates the values of the hidden and output layers of the network with their corresponding values according
	to the input, weights, and biases.
*/
void NeuralNetwork::feedForward() {
	for (int i = 1; i < getLayerCount(); i++) {
		for (int k = 0; k < m_BatchSize; k++) {
			gpu_copyMemory(m_CpuValueVectors[i][k], m_BiasVectors[i], m_LayerSizes[i] * sizeof(float));
		}
		
		gpu_batchVectorMatrixMultiply(m_SynapseMatrices[i], m_ValueVectors[i - 1], m_ValueVectors[i], m_LayerSizes[i - 1], m_LayerSizes[i], m_BatchSize);
		gpu_sigmoid(m_ValueVectors[i], m_BatchSize, m_LayerSizes[i]);
	}
}


/**
	Calculate the error of the network based on the loaded input and expected ouputs.
*/
void NeuralNetwork::calculateError() {
	gpu_calculateError(getOutputLayer(), m_ExpectedOutput, m_ErrorVectors[m_ErrorVectors.size() - 1], m_BatchSize, getOutputSize());
}


/**
	Based on the calculated error of the output layer, backpropogate the error of the network to its hidden layers.
*/
void NeuralNetwork::backpropogate() {
	for (int i = getLayerCount() - 1; i > 1; i--) {
		gpu_backpropogate(m_CpuSynapseMatrices[i][0], m_ErrorVectors[i], m_ErrorVectors[i - 1], m_ValueVectors[i - 1], m_LayerSizes[i], m_LayerSizes[i - 1], m_BatchSize);
	}
}


/**
	Update the weights and biases of the network based on the error calculated from backpropogation.
*/
void NeuralNetwork::applyWeights() {
	for (int i = 1; i < getLayerCount(); i++) {
		gpu_updateLayer(m_CpuSynapseMatrices[i][0], m_ValueVectors[i - 1], m_ErrorVectors[i], m_BiasVectors[i], m_LayerSizes[i], m_LayerSizes[i - 1], m_BatchSize, m_LearningRate);
	}
	
	for (int i = 1; i < m_SynapseMatrices.size(); i++) {
		for (int k = 1; k < m_BatchSize; k++) {
			gpu_copyMemory(m_CpuSynapseMatrices[i][k], m_CpuSynapseMatrices[i][0], m_LayerSizes[i] * m_LayerSizes[i - 1] * sizeof(float));
		}
	}

	for (int i = 1; i < m_ErrorVectors.size(); i++) {
		gpu_clearMemory(m_ErrorVectors[i], getLayerSizes()[i] * sizeof(float));
	}
}


/**
	A wrapper method that calls *calculateError()*, *backpropogate()*, and *applyWeights()*
*/
void NeuralNetwork::updateNetwork() {
	calculateError();
	backpropogate();
	applyWeights();
}


/**
	The output values of the network. This is different from *getOutputLayer()* by returning by value instead of by reference.
	*getOutputLayer()* also returns GPU pointers that cannot be dereferenced until copied back to CPU memory.

	@return A vector containing the current values of the output neurons.
*/
vector<float> NeuralNetwork::getCurrentOutput() {
	vector<float> cpuOutput(getOutputSize());
	float *gpuOutput = m_CpuValueVectors[m_CpuValueVectors.size() - 1][0];

	gpu_copyMemory(&(cpuOutput[0]), gpuOutput, getOutputSize() * sizeof(float));

	return cpuOutput;
}


/**
	Save the current state of the neural network to a file. Saves network size, learning rate, batch size, 
	weights, and biases. To load a network saved with this method, use *networkFromFile(std::string filename)*.

	@param filename The path to save the network to on disk.
*/
void NeuralNetwork::save(string filename) {
	fstream outputFile;
	outputFile.open(filename, fstream::out | fstream::trunc);

	if (!outputFile.is_open()) {
		cout << "Failed to open file " << filename << endl;
		return;
	}

	outputFile << getLayerCount() << endl;

	for (int i = 0; i < getLayerCount(); i++)
		outputFile << m_LayerSizes[i] << endl;

	outputFile << m_LearningRate << endl;
	outputFile << m_BatchSize << endl;

	for (int i = 1; i < getLayerCount(); i++) {
		vector<float> synapses(m_LayerSizes[i] * m_LayerSizes[i - 1]);
		vector<float> biases(m_LayerSizes[i]);

		gpu_copyMemory(&(synapses[0]), m_CpuSynapseMatrices[i][0], m_LayerSizes[i] * m_LayerSizes[i - 1] * sizeof(float));
		gpu_copyMemory(&(biases[0]), m_BiasVectors[i], m_LayerSizes[i] * sizeof(float));

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
	}

	outputFile.close();
}


/**
	Sets the synapse matrix for a given layer.

	@param layer The layer index. Index cannot be 0 (the input layer).
	@param matrix The synapse matrix. Vector length must be the size of @layer multiplied by size of (@layer - 1).
*/
void NeuralNetwork::setSynapseMatrix(unsigned int layer, vector<float> matrix) {
	if (layer == 0) {
		cout << "Cannot set the synapse matrix of layer 0!" << endl;
		return;
	}

	if (matrix.size() != m_LayerSizes[layer] * m_LayerSizes[layer - 1]) {
		cout << "Synapse matrix size must be equal to previousLayerSize * thisLayerSize" << endl;
		return;
	}

	for (int i = 0; i < m_BatchSize; i++) {
		gpu_copyMemory(m_CpuSynapseMatrices[layer][i], &(matrix[0]), matrix.size() * sizeof(float));
	}
}


/**
	Sets the bias vector for a given layer.

	@param layer The layer index. Index cannot be 0 (the input layer).
	@param vector The bias vector. Vector length must be equal to size of @layer.
*/
void NeuralNetwork::setBiasVector(unsigned int layer, vector<float> vector) {
	if (layer == 0) {
		cout << "Cannot set the bias vector of layer 0!" << endl;
		return;
	}

	if (vector.size() != m_LayerSizes[layer]) {
		cout << "Bias vector size must be equal to thisLayerSize" << endl;
		return;
	}

	gpu_copyMemory(m_BiasVectors[layer], &(vector[0]), vector.size() * sizeof(float));
}


/**
	The output values of the network.

	@return An array of vectors containing the output values of the network for each training example provided as input.
*/
float **NeuralNetwork::getOutputLayer() {
	return m_ValueVectors[m_ValueVectors.size() - 1];
}


/**
	The bias vectors for the network.

	@return A vector of bias vectors for each layer. The first vector will always be null (the input layer).
*/
vector<float *> NeuralNetwork::getBiasVectors() {
	return m_BiasVectors;
}


/**
	The error vectors for the network.

	@return A vector of error vectors for each layer. The first vector will always be null (the input layer).
*/
vector<float *> NeuralNetwork::getErrorVectors() {
	return m_ErrorVectors;
}


/**
	The synapse matrices for the network.

	@return A vector of synapse matrices for each layer. The first matrix will always be null (the input layer).
*/
vector<float **> NeuralNetwork::getSynapseMatrices() {
	return m_SynapseMatrices;
}


/**
	The value vectors for the network.

	@return A vector of vectors with the current values of the neurons in the network.
*/
vector<float **> NeuralNetwork::getValueVectors() {
	return m_ValueVectors;
}


/**
	The layer sizes of the network.

	@return A vector containing the sizes of each layer in the network.
*/
vector<unsigned int> NeuralNetwork::getLayerSizes() {
	return m_LayerSizes;
}


/**
	The expected output loaded in the network.

	@return A pointer to GPU memory that contains the current expected output vector.
*/
float *NeuralNetwork::getExpectedOutput() {
	return m_ExpectedOutput;
}


/**
	The learning rate of the network.

	@return The learning rate.
*/
float NeuralNetwork::getLearningRate() {
	return m_LearningRate;
}


/**
	The batch size the network uses to train.

	@return The batch size of the network.
*/
unsigned int NeuralNetwork::getBatchSize() {
	return m_BatchSize;
}


/**
	The input size of the network.

	@return The number of neurons in the input layer.
*/
unsigned int NeuralNetwork::getInputSize() {
	return m_LayerSizes[0];
}


/**
	The output size of the network.

	@return The number of neurons in the output layer.
*/
unsigned int NeuralNetwork::getOutputSize() {
	return m_LayerSizes[m_LayerSizes.size() - 1];
}


/**
	The number of layers in the network, including the input and output layers.

	@return The number of layers in the network.
*/
unsigned int NeuralNetwork::getLayerCount() {
	return m_LayerSizes.size();
}


/**
	Load a neural network from the file specified in @filename. The format expected is the same format used
	to save a neural network in the *save()* method.

	@param filename A path to a saved neural network.
	@return The loaded neural network.
*/
NeuralNetwork *networkFromFile(string filename) {
	ifstream inputFile(filename);

	if (!inputFile.is_open()) {
		cout << "Failed to open file " << filename << endl;
		return NULL;
	}

	int numLayers;
	inputFile >> numLayers;

	vector<unsigned int> layerSizes;

	for (int i = 0; i < numLayers; i++) {
		unsigned int layerSize;
		inputFile >> layerSize;
		layerSizes.push_back(layerSize);
	}

	float learningRate;
	inputFile >> learningRate;

	unsigned int batchSize;
	inputFile >> batchSize;

	string dummy;
	getline(inputFile, dummy);

	NeuralNetwork *network = new NeuralNetwork(layerSizes, batchSize, learningRate);

	for (int i = 1; i < numLayers; i++) {
		vector<float> matrix;
		vector<float> vector;

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

		network->setSynapseMatrix(i, matrix);
		network->setBiasVector(i, vector);
	}

	inputFile.close();

	return network;
}