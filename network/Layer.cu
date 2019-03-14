#include "Layer.h"
#include "NeuralNetwork.h"
#include "Util.h"

#include <iostream>
#include <chrono>
#include <stdlib.h>



Layer::Layer() {
}


Layer::Layer(NeuralNetwork *network, int numNeurons) {
	m_Network = network;

	for (int i = 0; i < numNeurons; i++) {
		m_BiasVector.push_back(standardNormalRandom());
	}

	m_Neurons = std::vector<float>(numNeurons, 0);
	m_ErrorVector = std::vector<float>(numNeurons, 0);
	m_LayerSize = numNeurons;
}


void Layer::setOutputLayer(Layer *outputLayer) {
	m_NextLayer = outputLayer;
	outputLayer->setPreviousLayer(this);
}


void Layer::setPreviousLayer(Layer *previous) {
	m_PreviousLayer = previous;
	m_InputSynapseMatrix = std::vector<float>(previous->getLayerSize() * m_LayerSize);

	for (int i = 0; i < m_InputSynapseMatrix.size(); i++) {
		m_InputSynapseMatrix[i] = standardNormalRandom();
	}
}


void Layer::setInput(std::vector<float> *input) {
	std::vector<float> inputValues = *input;

	for (int i = 0; i < m_LayerSize; i++) {
		m_Neurons[i] = inputValues[i];
	}
}


void Layer::updateError(std::vector<float> *expectedValues) {
	if (isOutputLayer())
		return;

	for (int i = 0; i < m_LayerSize; i++) {
		setError(i, 0);
	}

	if (m_PreviousLayer != NULL) {
		m_PreviousLayer->updateError(expectedValues);
	}
}


void Layer::setError(int neuron, float error) {
	if (isOutputLayer()) {
		m_ErrorVector[neuron] = error;
		return;
	}

	error = 0.0f;

	for (int i = 0; i < getNextLayer()->getLayerSize(); i++) {
		float synapseWeight = getNextLayer()->getSynapseWeight(neuron, i);
		float outputNeuronError = getNextLayer()->getErrorVector()[i];

		error += (synapseWeight * outputNeuronError);
	}

	m_ErrorVector[neuron] = error * sigmoidDerivative(m_Neurons[neuron]);
}


void Layer::applyWeights(std::vector<float> *inputValues) {
	for (int i = 0; i < m_LayerSize; i++) {
		for (int k = 0; k < m_PreviousLayer->getLayerSize(); k++) {
			float newWeight = getSynapseWeight(k, i) + (m_Network->getLearningRate() * m_ErrorVector[i] * (* inputValues)[k]);
			setSynapseMatrixValue(newWeight, k, i);
		}

		m_BiasVector[i] += (m_Network->getLearningRate() * m_ErrorVector[i]);
	}
}


void Layer::setSynapseMatrixValue(float value, int inputNeuron, int outputNeuron) {
	m_InputSynapseMatrix[(outputNeuron * getPreviousLayer()->getLayerSize()) + inputNeuron] = value;
}


float Layer::getSynapseWeight(int inputNeuron, int outputNeuron) {
	return m_InputSynapseMatrix[(outputNeuron * getPreviousLayer()->getLayerSize()) + inputNeuron];
}


void Layer::setSynapseMatrix(std::vector<float> matrix) {
	m_InputSynapseMatrix.swap(matrix);
}


void Layer::setBiasVector(std::vector<float> vector) {
	m_BiasVector.swap(vector);
}


NeuralNetwork *Layer::getNeuralNetwork() {
	return m_Network;
}


std::vector<float> Layer::getNeurons() {
	return m_Neurons;
}


void Layer::setNeuron(int index, float value) {
	m_Neurons[index] = value;
}


Layer *Layer::getNextLayer() {
	return m_NextLayer;
}


Layer *Layer::getPreviousLayer() {
	return m_PreviousLayer;
}


bool Layer::isOutputLayer() {
	return m_NextLayer == NULL;
}


bool Layer::isInputLayer() {
	return m_PreviousLayer == NULL;
}


std::vector<float> Layer::getInputSynapseMatrix() {
	return m_InputSynapseMatrix;
}


std::vector<float> Layer::getBiasVector() {
	return m_BiasVector;
}


std::vector<float> Layer::getErrorVector() {
	return m_ErrorVector;
}


unsigned int Layer::getLayerSize() {
	return m_LayerSize;
}