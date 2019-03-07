#ifndef NEURON_H__
#define NEURON_H__


#include <vector>

#include "Synapse.h"


class Layer;

class Neuron {
	private:
		float m_Value;
		float m_Bias;
		float m_Error;
		float m_Memo = -1;

		Layer *m_Layer;

		std::vector<Synapse *> m_OutputSynapses;
		std::vector<Synapse *> m_InputSynapses;

	public:
		Neuron();
		Neuron(Layer *layer);
		void setOutputLayer(Layer *layer);
		float getValue();
		void setValue(float value);
		void resetMemo();
		bool isInputNeuron();
		bool isOutputNeuron();
		void updateError(float expectedValue);
		void applyWeights(std::vector<float> *input);
		float getError();
};



#endif