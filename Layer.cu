#include "Layer.h"
#include "NeuralNetwork.h"

#include <iostream>


Layer::Layer() {
}


Layer::Layer(NeuralNetwork *network, int numNeurons) {
	m_Network = network;

	for (int i = 0; i < numNeurons; i++) {
		Neuron *neuron = new Neuron(this);
		m_Neurons.push_back(neuron);
	}
}


void Layer::setOutputLayer(Layer *outputLayer) {
	for (int i = 0; i < m_Neurons.size(); i++) {
		m_Neurons[i]->setOutputLayer(outputLayer);
	}

	m_NextLayer = outputLayer;
	outputLayer->setPreviousLayer(this);
}


void Layer::updateError(std::vector<float> *expectedValues) {
	for (int i = 0; i < m_Neurons.size(); i++) {
		if (isOutputLayer())
			m_Neurons[i]->updateError((* expectedValues)[i]);
		else
			m_Neurons[i]->updateError(0.0f);
	}

	if (m_PreviousLayer != NULL) {
		m_PreviousLayer->updateError(expectedValues);
	}
}


void Layer::applyWeights(std::vector<float> *input) {
	for (int i = 0; i < m_Neurons.size(); i++) {
		m_Neurons[i]->applyWeights(input);
	}

	if (m_NextLayer != NULL) {
		std::vector<float> neuronValues;

		for (int i = 0; i < m_Neurons.size(); i++) {
			neuronValues.push_back(m_Neurons[i]->getValue());
		}

		m_NextLayer->applyWeights(&neuronValues);
	}
}


std::vector<Neuron *> Layer::getNeurons() {
	return m_Neurons;
}


Layer *Layer::getNextLayer() {
	return m_NextLayer;
}


NeuralNetwork *Layer::getNeuralNetwork() {
	return m_Network;
}


bool Layer::isOutputLayer() {
	return m_NextLayer == NULL;
}


void Layer::setPreviousLayer(Layer *previous) {
	m_PreviousLayer = previous;
}


Layer *Layer::getPreviousLayer() {
	return m_PreviousLayer;
}