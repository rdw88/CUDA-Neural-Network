#ifndef EXTERN_H__
#define EXTERN_H__


extern "C" __declspec(dllexport) void *createNetwork(int numInputNeurons, int numHiddenLayers, int neuronsPerHiddenLayer, int numOutputNeurons, float learningRate);

extern "C" __declspec(dllexport) void batchTrainNetwork(void *_network, float *input, size_t inputSize, float *expectedOutput, size_t outputSize, size_t trainingsPerBatch);

extern "C" __declspec(dllexport) void getNetworkOutputForInput(void *_network, float *input, size_t inputSize, float *output, size_t outputSize);

extern "C" __declspec(dllexport) void saveNetwork(void *_network, char *filename, size_t filenameSize);

extern "C" __declspec(dllexport) void *loadNetwork(char *filename, size_t filenameSize);


#endif