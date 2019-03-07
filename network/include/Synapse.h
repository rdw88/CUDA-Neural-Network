#ifndef SYNAPSE_H__
#define SYNAPSE_H__

class Neuron;

class Synapse {
	private:
		float m_Weight;
		Neuron *m_InputNeuron;
		Neuron *m_OutputNeuron;

	public:
		Synapse();
		Synapse(Neuron *inputNeuron, Neuron *outputNeuron);
		float getWeight();
		void setWeight(float weight);
		Neuron *getInputNeuron();
		Neuron *getOutputNeuron();
};



#endif