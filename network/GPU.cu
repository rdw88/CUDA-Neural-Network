#include "GPU.h"

#include <iostream>

#include <cublas_v2.h>
#include <cuda.h>
#include <stdlib.h>


#pragma comment(lib, "cublas.lib")


static cublasHandle_t cublasContext;



void createCublasContext() {
	cublasCreate(&cublasContext);
}



void destroyCublasContext() {
	cublasDestroy(cublasContext);
}



float *gpu_loadVector(std::vector<float> *vector) {
	unsigned int size = vector->size();
	float *pointer;

	cudaMalloc(&pointer, size * sizeof(float));
	cudaMemcpy(pointer, vector->data(), size * sizeof(float), cudaMemcpyHostToDevice);
	
	return pointer;
}



float *gpu_loadVectorFromPointer(float *cpuPointer, size_t numFloats) {
	float *gpuPointer;

	cudaError_t error = cudaMalloc(&gpuPointer, numFloats * sizeof(float));
	error = cudaMemcpy(gpuPointer, cpuPointer, numFloats * sizeof(float), cudaMemcpyHostToDevice);

	return gpuPointer;
}



float *gpu_unloadVector(float *gpuPointer, size_t numFloats) {
	float *cpuPointer = (float *) malloc(numFloats * sizeof(float));

	cudaMemcpy(cpuPointer, gpuPointer, numFloats * sizeof(float), cudaMemcpyDeviceToHost);

	return cpuPointer;
}



void gpu_freeMemory(float *gpuPointer) {
	cudaFree(gpuPointer);
}



void gpu_batchVectorMatrixMultiply(std::vector<float *> *matrices, std::vector<float *> *vectors, std::vector<float *> *results, int numColumns, int numRows, int batches) {
	float **gpuMatrixPointers;
	float **gpuVectorPointers;
	float **gpuResultPointers;

	cudaMalloc(&gpuMatrixPointers, matrices->size() * sizeof(float *));
	cudaMalloc(&gpuVectorPointers, vectors->size() * sizeof(float *));
	cudaMalloc(&gpuResultPointers, results->size() * sizeof(float *));

	cudaMemcpy(gpuMatrixPointers, &((* matrices)[0]), matrices->size() * sizeof(float *), cudaMemcpyHostToDevice);
	cudaMemcpy(gpuVectorPointers, &((* vectors)[0]), vectors->size() * sizeof(float *), cudaMemcpyHostToDevice);
	cudaMemcpy(gpuResultPointers, &((* results)[0]), results->size() * sizeof(float *), cudaMemcpyHostToDevice);

	const float alpha = 1;
	const float beta = 1;
	const float *alphaRef = &alpha;
	const float *betaRef = &beta;

	int lda = numRows;
	int ldb = numColumns;
	int ldc = numRows;

	cublasSgemmBatched(cublasContext, CUBLAS_OP_N, CUBLAS_OP_N, numRows, 1, numColumns, alphaRef, gpuMatrixPointers, lda, gpuVectorPointers, ldb, betaRef, gpuResultPointers, ldc, batches);

	cudaFree(gpuMatrixPointers);
	cudaFree(gpuVectorPointers);
	cudaFree(gpuResultPointers);
}