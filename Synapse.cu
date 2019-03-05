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


Neuron *Synapse::getInputNeuron() {
	return m_InputNeuron;
}