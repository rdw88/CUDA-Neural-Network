#ifndef ANN_H__
#define ANN_H__


#include "Layer.h"


class NeuralNetwork {
	private:
		Layer *m_InputLayer;

	public:
		NeuralNetwork();
		NeuralNetwork(int numInputNeurons, int numHiddenLayers, int neuronsPerHiddenLayer, int numOutputNeurons);
		Layer *getInputLayer();
		Layer *getOutputLayer();
		void setInput(float input[]);
		
};


#endif