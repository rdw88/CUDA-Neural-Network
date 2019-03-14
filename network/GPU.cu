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


/*
int main(void) {
	createCublasContext();

	float matrix[] = {
		1.0, 2.0, 7.0, 2.0, 5.0, 3.0
	};

	float vector1[] = {
		1.0,
		2.0
	};

	float vector2[] = {
		3.0,
		4.0
	};

	float bias[] {
		5.0,
		2.0,
		3.0
	};

	float *cpuMatrix = (float *) malloc(6 * sizeof(float));
	float *cpuVector1 = (float *) malloc(2 * sizeof(float));
	float *cpuVector2 = (float *) malloc(2 * sizeof(float));
	float *cpuBias = (float *) malloc(3 * sizeof(float));

	memcpy(cpuMatrix, matrix, sizeof(float) * 6);
	memcpy(cpuVector1, vector1, sizeof(float) * 2);
	memcpy(cpuVector2, vector2, sizeof(float) * 2);
	memcpy(cpuBias, bias, sizeof(float) * 3);

	float *gpuMatrix = gpu_loadVectorFromPointer(cpuMatrix, 6);
	float *gpuVector1 = gpu_loadVectorFromPointer(cpuVector1, 2);
	float *gpuVector2 = gpu_loadVectorFromPointer(cpuVector2, 2);
	float *gpuBias1 = gpu_loadVectorFromPointer(cpuBias, 3);
	float *gpuBias2 = gpu_loadVectorFromPointer(cpuBias, 3);

	std::vector<float *> matrices(2);
	std::vector<float *> vectors(2);
	std::vector<float *> results(2);

	matrices[0] = gpuMatrix;
	matrices[1] = gpuMatrix;
	vectors[0] = gpuVector1;
	vectors[1] = gpuVector2;
	results[0] = gpuBias1;
	results[1] = gpuBias2;

	gpu_batchVectorMatrixMultiply(&matrices, &vectors, &results, 2, 3, 2);

	float expected1[] = {
		(matrix[0] * vector1[0]) + (matrix[3] * vector1[1]) + bias[0],
		(matrix[1] * vector1[0]) + (matrix[4] * vector1[1]) + bias[1],
		(matrix[2] * vector1[0]) + (matrix[5] * vector1[1]) + bias[2]
	};

	float expected2[] = {
		(matrix[0] * vector2[0]) + (matrix[3] * vector2[1]) + bias[0],
		(matrix[1] * vector2[0]) + (matrix[4] * vector2[1]) + bias[1],
		(matrix[2] * vector2[0]) + (matrix[5] * vector2[1]) + bias[2]
	};

	float *cpuResult1 = gpu_unloadVector(gpuBias1, 3);
	float *cpuResult2 = gpu_unloadVector(gpuBias2, 3);

	std::cout << "verifying" << std::endl;

	for (int i = 0; i < 3; i++) {
		if (expected1[i] != cpuResult1[i]) {
			std::cout << "Result 1: " << "(Expected: " << expected1[i] << ", Result: " << cpuResult1[i] << std::endl;
		}
	}

	for (int i = 0; i < 3; i++) {
		if (expected2[i] != cpuResult2[i]) {
			std::cout << "Result 2: " << "(Expected: " << expected2[i] << ", Result: " << cpuResult2[i] << std::endl;
		}
	}

	std::cout << "done!" << std::endl;

	destroyCublasContext();

	return 0;
}*/