#ifndef EXTERN_H__
#define EXTERN_H__


extern "C" __declspec(dllexport) void *createNetwork(int numInputNeurons, int numHiddenLayers, int neuronsPerHiddenLayer, int numOutputNeurons, float learningRate);

extern "C" __declspec(dllexport) void trainNetwork(void *_network, float *input, size_t inputSize, float *expectedOutput, size_t outputSize);

extern "C" __declspec(dllexport) void getNetworkOutput(void *_network, float *output, size_t outputSize);


#endif