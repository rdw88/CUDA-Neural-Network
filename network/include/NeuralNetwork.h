#ifndef ANN_H__
#define ANN_H__


#include "Layer.h"
#include <string>


class NeuralNetwork {
	private:
		Layer *m_InputLayer = NULL;
		float m_LearningRate;
		unsigned int m_NumHiddenLayers;

	public:
		NeuralNetwork();
		NeuralNetwork(int numInputNeurons, int numHiddenLayers, int neuronsPerHiddenLayer, int numOutputNeurons, float learningRate);
		void batchTrain(std::vector<float> *batch, std::vector<float> *expectedOutputs, unsigned int trainingsPerBatch);
		std::vector<float *> loadSynapseMatricesIntoGPU();
		std::vector<float *> loadBatchIntoGPU(std::vector<float> *batch, unsigned int trainingsPerBatch);
		std::vector<float *> loadBatchIntoCPU(std::vector<float> *batch, unsigned int trainingsPerBatch);
		std::vector<float *> activateLayerResults(std::vector<float *> *layerResult, size_t layerSize);
		std::vector<float> averageBatchedInputValues(std::vector<float *> *batch, unsigned int trainingsPerBatch, unsigned int layerSize);
		std::vector<float *> runTraining(std::vector<float> *batch, std::vector<std::vector<float *>> *layerResults, unsigned int trainingsPerBatch);
		std::vector<float> getOutputForInput(std::vector<float> *input);
		void save(std::string filename);
		NeuralNetwork *networkFromFile(std::string filename);
		float getLearningRate();
		Layer *getInputLayer();
		Layer *getOutputLayer();
};


#endif