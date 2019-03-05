#include "Layer.h"
#include "NeuralNetwork.h"

#include <iostream>


Layer::Layer() {
}


Layer::Layer(NeuralNetwork *network, int numNeurons) {
	m_Network = network;
	m_Neurons = std::vector<Neuron *>();

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
}


void Layer::invalidateMemos() {
	Layer *layer = m_Network->getInputLayer();

	while (layer != NULL) {
		std::vector<Neuron *> neurons = layer->getNeurons();

		for (int i = 0; i < neurons.size(); i++) {
			neurons[i]->resetMemo();
		}

		layer = layer->getNextLayer();
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