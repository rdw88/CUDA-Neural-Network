#ifndef LAYER_H__
#define LAYER_H__


#include <vector>
#include "Neuron.h"


class NeuralNetwork;


class Layer {
	private:
		std::vector<Neuron *> m_Neurons;
		NeuralNetwork *m_Network;
		Layer *m_NextLayer;

	public:
		Layer();
		Layer(NeuralNetwork *network, int numNeurons);
		void setOutputLayer(Layer *layer);
		void invalidateMemos();
		std::vector<Neuron *> getNeurons();
		Layer *getNextLayer();
		NeuralNetwork *getNeuralNetwork();
};



#endif