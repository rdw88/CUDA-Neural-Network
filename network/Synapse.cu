#include "Synapse.h"
#include "Neuron.h"
#include "Util.h"

#include <iostream>


Synapse::Synapse() {
}



Synapse::Synapse(Neuron *inputNeuron, Neuron *outputNeuron) {
	m_Weight = standardNormalRandom();
	m_InputNeuron = inputNeuron;
	m_OutputNeuron = outputNeuron;
}


float Synapse::getWeight() {
	return m_Weight;
}


void Synapse::setWeight(float weight) {
	m_Weight = weight;
}


Neuron *Synapse::getInputNeuron() {
	return m_InputNeuron;
}


Neuron *Synapse::getOutputNeuron() {
	return m_OutputNeuron;
}