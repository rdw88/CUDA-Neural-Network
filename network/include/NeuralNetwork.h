#ifndef ANN_H__
#define ANN_H__


#include "Layer.h"


#define LEARNING_RATE 0.1f


class NeuralNetwork {
	private:
		Layer *m_InputLayer;

	public:
		NeuralNetwork();
		NeuralNetwork(int numInputNeurons, int numHiddenLayers, int neuronsPerHiddenLayer, int numOutputNeurons);
		Layer *getInputLayer();
		Layer *getOutputLayer();
		void invalidateMemos();
		void setInput(std::vector<float> *input);
		std::vector<float> getOutput();
		void train(std::vector<float> *input, std::vector<float> *expectedOutput);
};


#endif