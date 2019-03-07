#include "NeuralNetwork.h"
#include "Util.h"

#include <iostream>


using namespace std;


#define RUNS_PER_BATCH 9


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


void NeuralNetwork::invalidateMemos() {
	Layer *layer = m_InputLayer;

	while (layer != NULL) {
		std::vector<Neuron *> neurons = layer->getNeurons();

		for (int i = 0; i < neurons.size(); i++) {
			neurons[i]->resetMemo();
		}

		layer = layer->getNextLayer();
	}
}


void NeuralNetwork::setInput(std::vector<float> *input) {
	vector<Neuron *> neurons = m_InputLayer->getNeurons();
	vector<float> inputValues = *input;

	for (int i = 0; i < neurons.size(); i++) {
		neurons[i]->setValue(inputValues[i]);
	}

	invalidateMemos();
}


vector<float> NeuralNetwork::getOutput() {
	vector<Neuron *> neurons = getOutputLayer()->getNeurons();
	vector<float> outputValues;

	for (int i = 0; i < neurons.size(); i++) {
		outputValues.push_back(neurons[i]->getValue());
	}

	return outputValues;
}


void NeuralNetwork::train(std::vector<float> *input, std::vector<float> *expectedOutput) {
	setInput(input);
	getOutputLayer()->updateError(expectedOutput);
	getInputLayer()->applyWeights(input);
}


int main(void) {
	NeuralNetwork network(768, 2, 350, 768);

	vector<float> in(768);
	for (int i = 0; i < 768; i++)
		in[i] = sigmoid(standardNormalRandom());

	vector<float> out;
	out = in;

	for (int i = 0; i < 100; i++) {
		network.train(&in, &out);
	}

	vector<float> actualOutput = network.getOutput();

	for (int i = 0; i < actualOutput.size(); i++) {
		cout << in[i] << endl;
		cout << actualOutput[i] << endl;
	}

  	return 0;
}