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

	float value = m_Bias;

	for (int i = 0; i < m_InputSynapses.size(); i++) {
		value += (m_InputSynapses[i]->getWeight() * m_InputSynapses[i]->getInputNeuron()->getValue());
	}

	value = sigmoid(value);
	m_Memo = value;
	return value;
}


void Neuron::updateError(float expectedValue) {
	float value = getValue();

	if (isOutputNeuron()) {
		m_Error = (expectedValue - value) * sigmoidDerivative(value);
		return;
	}

	float error = 0.0f;
	for (int i = 0; i < m_OutputSynapses.size(); i++) {
		error += (m_OutputSynapses[i]->getWeight() * m_OutputSynapses[i]->getOutputNeuron()->getError());
	}

	m_Error = error * sigmoidDerivative(value);
}


void Neuron::applyWeights(std::vector<float> *input) {
	for (int i = 0; i < m_InputSynapses.size(); i++) {
		float newWeight = m_InputSynapses[i]->getWeight() + (LEARNING_RATE * m_Error * (* input)[i]);
		m_InputSynapses[i]->setWeight(newWeight);
	}

	m_Bias += (LEARNING_RATE * m_Error);
	resetMemo();
}


void Neuron::setValue(float value) {
	m_Value = value;
}


bool Neuron::isInputNeuron() {
	return m_InputSynapses.size() == 0;
}


bool Neuron::isOutputNeuron() {
	return m_OutputSynapses.size() == 0;
}


void Neuron::resetMemo() {
	m_Memo = -1;
}


float Neuron::getError() {
	return m_Error;
}