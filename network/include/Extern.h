/**
 * Extern.h
 * April 4, 2019
 * Ryan Wise
 * 
 * An external interface to the neural network that can be compiled into a DLL and invoked from another programming language.
 * 
 */


#ifndef EXTERN_H__
#define EXTERN_H__


extern "C" __declspec(dllexport) void *createNetwork(unsigned int *layerSizes, unsigned int numLayers, unsigned int batchSize, float learningRate);

extern "C" __declspec(dllexport) void trainNetwork(void *_network, float *input, float *expectedOutput);

extern "C" __declspec(dllexport) void getNetworkOutputForInput(void *_network, float *input, unsigned int inputSize, float *output, unsigned int outputSize);

extern "C" __declspec(dllexport) void saveNetwork(void *_network, char *filename, size_t filenameSize);

extern "C" __declspec(dllexport) void *loadNetwork(char *filename, size_t filenameSize);

extern "C" __declspec(dllexport) void setLearningRate(void *_network, float learningRate);

extern "C" __declspec(dllexport) void setLayerActivations(void *_network, int *activations, unsigned int activationSize);


#endif