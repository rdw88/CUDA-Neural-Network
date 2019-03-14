#ifndef LAYER_H__
#define LAYER_H__


#include <vector>


class NeuralNetwork;


class Layer {
	private:
		NeuralNetwork *m_Network;
		Layer *m_NextLayer = NULL;
		Layer *m_PreviousLayer = NULL;
		unsigned int m_LayerSize;

		std::vector<float> m_Neurons;
		std::vector<float> m_InputSynapseMatrix;
		std::vector<float> m_BiasVector;
		std::vector<float> m_ErrorVector;

	public:
		Layer();
		Layer(NeuralNetwork *network, int numNeurons);
		void setOutputLayer(Layer *layer);
		void setPreviousLayer(Layer *previous);
		void setInput(std::vector<float> *input);
		void updateError(std::vector<float> *expectedValues);
		void setError(int neuron, float expectedValue);
		void applyWeights(std::vector<float> *inputValues);
		void setSynapseMatrixValue(float value, int inputNeuron, int outputNeuron);
		void setSynapseMatrix(std::vector<float> matrix);
		void setBiasVector(std::vector<float> vector);
		float getSynapseWeight(int inputNeuron, int outputNeuron);
		void setNeuron(int index, float value);
		std::vector<float> getNeurons();
		std::vector<float> getInputSynapseMatrix();
		std::vector<float> getBiasVector();
		std::vector<float> getErrorVector();
		unsigned int getLayerSize();
		bool isOutputLayer();
		bool isInputLayer();
		Layer *getPreviousLayer();
		Layer *getNextLayer();
		NeuralNetwork *getNeuralNetwork();
};



#endif