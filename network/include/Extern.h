#ifndef EXTERN_H__
#define EXTERN_H__


extern "C" __declspec(dllexport) void *createNetwork(int numInputNeurons, int numHiddenLayers, int neuronsPerHiddenLayer, int numOutputNeurons);

extern "C" __declspec(dllexport) void trainNetwork(void *network, float *input, size_t inputSize, float *expectedOutput, size_t outputSize);


#endif