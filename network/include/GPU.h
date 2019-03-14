#ifndef GPU_H__
#define GPU_H__


#include <vector>


void createCublasContext();

void destroyCublasContext();

float *gpu_loadVector(std::vector<float> *vector);

float *gpu_loadVectorFromPointer(float *cpuPointer, size_t numFloats);

float *gpu_unloadVector(float *gpuPointer, size_t numFloats);

void gpu_freeMemory(float *gpuPointer);

void gpu_batchVectorMatrixMultiply(std::vector<float *> *matrices, std::vector<float *> *vectors, std::vector<float *> *results, int numColumns, int numRows, int batches);


#endif