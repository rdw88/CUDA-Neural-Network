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


#include "Activation.h"


extern "C" __declspec(dllexport) void *createNetwork(unsigned int *layerSizes, unsigned int numLayers, unsigned int batchSize, float learningRate);

extern "C" __declspec(dllexport) void trainNetwork(void *_network, float *input, float *expectedOutput);

extern "C" __declspec(dllexport) void getNetworkOutputForInput(void *_network, float *input, unsigned int inputSize, float *output, unsigned int outputSize);

extern "C" __declspec(dllexport) void saveNetwork(void *_network, char *filename, size_t filenameSize);

extern "C" __declspec(dllexport) void *loadNetwork(char *filename, size_t filenameSize);

extern "C" __declspec(dllexport) void setSynapseMatrix(void *_network, unsigned int layer, float *synapseMatrix, unsigned int matrixLength);

extern "C" __declspec(dllexport) void setBiasVector(void *_network, unsigned int layer, float *biasVectorRef, unsigned int vectorLength);

extern "C" __declspec(dllexport) void setLearningRate(void *_network, float learningRate);

extern "C" __declspec(dllexport) void setLayerActivations(void *_network, Activation *activations, unsigned int activationSize);

extern "C" __declspec(dllexport) void getSynapseMatrix(void *_network, unsigned int layer, float *synapseMatrix);

extern "C" __declspec(dllexport) void getBiasVector(void *_network, unsigned int layer, float *biasVector);


#endif