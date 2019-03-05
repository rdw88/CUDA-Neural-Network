#include "NeuralNetwork.h"
#include "Util.h"

#include <iostream>


using namespace std;


NeuralNetwork::NeuralNetwork() {
}


NeuralNetwork::NeuralNetwork(int numInputNeurons, int numHiddenLayers, int neuronsPerHiddenLayer, int numOutputNeurons) {
	Layer *inputLayer = new Layer(this, numInputNeurons);
	Layer *previousLayer = inputLayer;

	for (int i = 0; i < numHiddenLayers; i++) {
		Layer *hiddenLayer = new Layer(this, neuronsPerHiddenLayer);
		previousLayer->setOutputLayer(hiddenLayer);
		previousLayer = hiddenLayer;
	}

	Layer *outputLayer = new Layer(this, numOutputNeurons);
	previousLayer->setOutputLayer(outputLayer);

	m_InputLayer = inputLayer;
}


Layer *NeuralNetwork::getInputLayer() {
	return m_InputLayer;
}


Layer *NeuralNetwork::getOutputLayer() {
	Layer *layer = m_InputLayer;

	while (layer->getNextLayer() != NULL) {
		layer = layer->getNextLayer();
	}

	return layer;
}


void NeuralNetwork::setInput(float input[]) {
	vector<Neuron *> neurons = m_InputLayer->getNeurons();

	for (int i = 0; i < neurons.size(); i++) {
		neurons[i]->setValue(input[i]);
	}
}


int main(void) {
	NeuralNetwork network(2, 2, 2, 2);

	float input[] = { 0.2, 0.5 };
	network.setInput(input);

	vector<Neuron *> neurons = network.getOutputLayer()->getNeurons();
	for (int i = 0; i < neurons.size(); i++) {
		cout << "Output Neuron value: " << neurons[i]->getValue() << endl;
	}

  	return 0;
}