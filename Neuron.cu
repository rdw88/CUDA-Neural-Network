#include "Neuron.h"
#include "Layer.h"
#include "Util.h"
#include "NeuralNetwork.h"

#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/inner_product.h>



Neuron::Neuron() {
}



Neuron::Neuron(Layer *layer) {
	m_Bias = standardNormalRandom();
	m_InputSynapses = std::vector<Synapse *>();
	m_OutputSynapses = std::vector<Synapse *>();
	m_Layer = layer;
}



void Neuron::setOutputLayer(Layer *layer) {
	std::vector<Neuron *> neurons = layer->getNeurons();

	for (int i = 0; i < neurons.size(); i++) {
		Neuron *inputNeuron = this;
		Neuron *outputNeuron = neurons[i];
		Synapse *synapse = new Synapse(inputNeuron, outputNeuron);
		m_OutputSynapses.push_back(synapse);
		outputNeuron->m_InputSynapses.push_back(synapse);
	}
}



float Neuron::getValue() {
	if (this->isInputNeuron()) {
		return m_Value;
	}

	if (m_Memo != -1) {
		return m_Memo;
	}

	thrust::host_vector<float> hostWeights;
	thrust::host_vector<float> hostValues;

	for (int i = 0; i < m_InputSynapses.size(); i++) {
		hostWeights.push_back(m_InputSynapses[i]->getWeight());
		hostValues.push_back(m_InputSynapses[i]->getInputNeuron()->getValue());
	}

	thrust::device_vector<float> deviceWeights = hostWeights;
	thrust::device_vector<float> deviceValues = hostValues;

	float value = thrust::inner_product(deviceValues.begin(), deviceValues.end(), deviceWeights.begin(), 0.0f) + m_Bias;
	m_Memo = value;

	return value;
}


void Neuron::setValue(float value) {
	m_Value = value;
	m_Layer->invalidateMemos();
}


bool Neuron::isInputNeuron() {
	return m_InputSynapses.size() == 0;//m_Layer->getNeuralNetwork()->getInputLayer() == m_Layer;
}


void Neuron::resetMemo() {
	m_Memo = -1;
}