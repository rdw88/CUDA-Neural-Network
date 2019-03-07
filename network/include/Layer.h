#ifndef LAYER_H__
#define LAYER_H__


#include <vector>
#include "Neuron.h"


class NeuralNetwork;


class Layer {
	private:
		std::vector<Neuron *> m_Neurons;
		NeuralNetwork *m_Network;
		Layer *m_NextLayer = NULL;
		Layer *m_PreviousLayer = NULL;

	public:
		Layer();
		Layer(NeuralNetwork *network, int numNeurons);
		void setOutputLayer(Layer *layer);
		bool isOutputLayer();
		void setPreviousLayer(Layer *previous);
		Layer *getPreviousLayer();
		std::vector<Neuron *> getNeurons();
		Layer *getNextLayer();
		NeuralNetwork *getNeuralNetwork();
		void updateError(std::vector<float> *expectedValues);
		void applyWeights(std::vector<float> *input);
};



#endif