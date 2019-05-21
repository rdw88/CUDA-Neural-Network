/**
 * GPU.cu
 * April 4, 2019
 * Ryan Wise
 * 
 * An interface to a CUDA-enabled NVIDIA GPU.
 * 
 */


#include "GPU.h"

#include <cublas_v2.h>
#include <cuda.h>
#include <stdlib.h>
#include <math.h>


#pragma comment(lib, "cublas.lib")

using namespace std;


static cublasHandle_t cublasContext;


/*
	A device function pointer type. Used for specifying an activation function.
*/
typedef float (*Op)(float, Activation *);


/**
	Create the cuBLAS context for the neural network. This creates a global context that is used
	to perform basic linear algebra operations on the GPU.
*/
void createCublasContext() {
	cublasCreate(&cublasContext);
}


/**
	Destroy the cuBLAS context.
*/
void destroyCublasContext() {
	cublasDestroy(cublasContext);
}


/**
	Allocate pinned CPU memory. CPU memory is pageable by default and the GPU cannot access data directly in paged CPU
	memory. Using pinned memory removes pagination and therefore removes an extra step from moving data from CPU to GPU
	memory, increasing performance.

	@param numBytes The number of bytes of pinned memory to allocate.
	@return A pointer to the beginning of the allocated memory block.
*/
void *allocPinnedMemory(size_t numBytes) {
	void *pointer;

	cudaMallocHost(&pointer, numBytes);

	return pointer;
}


/**
	Frees pinned memory.

	@param pointer A pointer to pinned memory allocated with *allocPinnedMemory()*.
*/
void freePinnedMemory(void *pointer) {
	cudaFreeHost(pointer);
}


/**
 * Allocates memory on the GPU.
 * 
 * @param size The number of bytes to allocate on the GPU.
 * @return A pointer to the beginning of the allocated memory block.
 */
void *gpu_allocMemory(size_t size) {
	void *gpuPointer;

	cudaMalloc(&gpuPointer, size);

	return gpuPointer;
}


/**
 * Copies memory to and from the GPU. Supports any permutation of copying memory between the CPU and GPU.
 * 
 * @param toPointer A pointer to CPU or GPU memory that will be copied to.
 * @param fromPointer A pointer to CPU or GPU memory that will be copied from.
 * @param numBytes The number of bytes to copy.
 */
void gpu_copyMemory(void *toPointer, void *fromPointer, size_t numBytes) {
	cudaMemcpy(toPointer, fromPointer, numBytes, cudaMemcpyDefault);
}


/**
 * Clear memory in the GPU (set to zero).
 * 
 * @param gpuPointer Pointer to the memory in the GPU to clear.
 * @param numBytes The number of bytes to zero out.
 */
void gpu_clearMemory(void *gpuPointer, size_t numBytes) {
	cudaMemset(gpuPointer, 0, numBytes);
}


/**
 * Free memory in the GPU.
 * 
 * @param gpuPointer Pointer to GPU memory to free.
 */
void gpu_freeMemory(void *gpuPointer) {
	cudaFree(gpuPointer);
}


/**
 * Perform matrix-vector multiplication for a batch of matrices and vectors and store the result vectors. The matrix-vector
 * multiplications are performed in parallel on the GPU and therefore all pointers are expected to be pointers to GPU memory.
 * 
 * @param matrices An array of matrices of size @batches. Each matrix is a one-dimensional array where the matrix is stored in row-major order.
 * @param vectors An array of vectors of size @batches.
 * @param results An array of vectors of size @batches. The results of the matrix-vector multiplications are stored in each vector allocated here.
 * @param numColumns The number of columns in each matrix.
 * @param numRows The number of rows in each matrix.
 * @param batches The number of batches to process.
 */
void gpu_batchVectorMatrixMultiply(float **matrices, float **vectors, float **results, unsigned int numColumns, unsigned int numRows, unsigned int batches) {
	const float alpha = 1;
	const float beta = 1;

	int lda = numColumns;
	int ldb = numColumns;
	int ldc = numRows;

	cublasSgemmBatched(cublasContext, CUBLAS_OP_T, CUBLAS_OP_N, numRows, 1, numColumns, &alpha, matrices, lda, vectors, ldb, &beta, results, ldc, batches);
}


/**
	Sigmoid implementation running on the GPU.

	@param input The input value to perform the sigmoid function on.
	@param activation The activation hyperparameters.
	@return Sigmoid of the input value.
*/
__device__ __forceinline__ float sigmoid(float input, Activation *activation) {
	return 1.0f / (1.0f + exp(-input));
}


/**
	The derivative of sigmoid implemented for the GPU.

	@param input The input value to perform the sigmoid derivative function on.
	@param activation The activation hyperparameters.
	@return The sigmoid derivative of the input value.
*/
__device__ __forceinline__ float sigmoidDerivative(float input, Activation *activation) {
	return input * (1 - input);
}


/**
	ReLU implementation running on the GPU.

	@param input The input value to perform the ReLU function on.
	@param activation The activation hyperparameters.
	@return ReLU of the input.
*/
__device__ __forceinline__ float relu(float input, Activation *activation) {
	if (input > activation->maxThreshold)
		return activation->maxThreshold;

	return fmaxf(0.0f, input);
}


/**
	ReLU derivative implementation running on the GPU.

	@param input The input value to perform the ReLU derivative function on.
	@param activation The activation hyperparameters.
	@return The ReLU derivative of the input.
*/
__device__ __forceinline__ float reluDerivative(float input, Activation *activation) {
	if (input > 0 && input < activation->maxThreshold)
		return 1;

	return 0;
}


/**
	Must maintain same order as the ActivationType enum defined in Activation.h

	Indexed activation functions for fast lookup of which activation function to use during
	feed forward and backpropogation operations. 
*/
__device__ Op activationOps[] = {
	&relu,
	&sigmoid
};

__device__ Op activationDerivativeOps[] = {
	&reluDerivative,
	&sigmoidDerivative
};


/**
	CUDA kernel that performs the activation function in parallel for the elements in the passed vectors.

	@param vectors An array of vectors to perform the activation function on.
	@param activation The activation function to use along with any hyperparameters for the activation function.
*/
__global__ void activation_gpu_kernel(float ** __restrict__ vectors, Activation *activation) {
	unsigned int vectorIndex = blockIdx.x;
	unsigned int vectorSubindex = threadIdx.x;
	unsigned int activationOperation = (unsigned int) activation->activationType;

	vectors[vectorIndex][vectorSubindex] = activationOps[activationOperation](vectors[vectorIndex][vectorSubindex], activation);
}


/**
	CUDA kernel that calculates the error of a network's output neurons in parallel.

	@param resultVectors An array of vectors that represent the output neuron values for each training example in a batch.
	@param expectedVector An array of vectors that represent the expected output for the output neurons for each training example in a batch.
	@param errorVector A vector with length equal to the number of output neurons that stores the resultant errors of each neuron.
	@param numVectors The number of vectors in the training batch.
	@param vectorLength The length of each vector, should be equal to the number of output neurons.
	@param activation The activation function to use along with any hyperparameters for the activation function.
*/
__global__ void calculateError_gpu_kernel(float ** __restrict__ resultVectors, float * __restrict__ expectedVector, float * __restrict__ errorVector,
	unsigned int numVectors, unsigned int vectorLength, Activation *activation) {

	unsigned int vectorIndex = blockIdx.x;
	unsigned int vectorSubindex = threadIdx.x;
	unsigned int expectedVectorIndex = (vectorIndex * vectorLength) + vectorSubindex;
	unsigned int activationOperation = (unsigned int) activation->activationType;

	float batches = (float) numVectors;
	float activationDerivative = activationDerivativeOps[activationOperation](resultVectors[vectorIndex][vectorSubindex], activation);
	float error = ((resultVectors[vectorIndex][vectorSubindex] - expectedVector[expectedVectorIndex]) * activationDerivative) / batches;

	atomicAdd(&errorVector[vectorSubindex], error);
}


/**
	CUDA kernel that calculates the error of each neuron in a hidden layer in parallel for a given training example in a batch.

	@param synapseMatrix The synapse matrix for which the target layer is an input to.
	@param errorVector The error vector of the next layer.
	@param destinationErrorVector The error vector of the target layer. The results of this function call will be stored here.
	@param destinationValueVector An array of value vectors of the target layer. Each vector in the array represents the values of the neurons
	for a training example in a batch.
	@param errorVectorSize The size of @errorVector.
	@param destinationErrorSize The size of @destinationErrorVector.
	@param batchSize The number of training examples in a batch.
	@param activation The activation function to use along with any hyperparameters for the activation function.
*/
__global__ void backpropogate_gpu_kernel(float * __restrict__ synapseMatrix, float * __restrict__ errorVector, float * __restrict__ destinationErrorVector, float ** __restrict__ destinationValueVector,
	unsigned int errorVectorSize, unsigned int destinationErrorSize, unsigned int batchSize, Activation *activation) {

	unsigned int sourceIndex = threadIdx.x;
	unsigned int destinationIndex = blockIdx.x;
	unsigned int activationOperation = (unsigned int) activation->activationType;

	__shared__ float averageDestinationValue;

	if (sourceIndex == 0) {
		averageDestinationValue = 0.0f;
		
		for (int i = 0; i < batchSize; i++) {
			averageDestinationValue += destinationValueVector[i][destinationIndex];
		}

		averageDestinationValue = averageDestinationValue / ((float) batchSize);
	}

	__syncthreads();

	float activationDerivative = activationDerivativeOps[activationOperation](averageDestinationValue, activation);
	float error = errorVector[sourceIndex] * synapseMatrix[(sourceIndex * destinationErrorSize) + destinationIndex] * activationDerivative;

	atomicAdd(&destinationErrorVector[destinationIndex], error);
}


/**
	CUDA kernel that updates the weights and biases of a given layer in the neural network. Each input synapse's new weight for the given layer
	is calculated in parallel.

	@param synapseMatrix The input synapse matrix for the target layer.
	@param valueVectors An array of vectors that represent the values of the neurons on the previous layer.
	@param errorVector The error vector for the target layer.
	@param biasVector The bias vector for the target layer.
	@param layerSize The size of the target layer.
	@param previousLayerSize The size of the previous layer.
	@param batchSize The number of training examples in a batch.
	@param learningRate The learning rate of the network.
*/
__global__ void updateLayer_gpu_kernel(float * __restrict__ synapseMatrix, float ** __restrict__ valueVectors, float * __restrict__ errorVector, float * __restrict__ biasVector,
	unsigned int layerSize, unsigned int previousLayerSize, unsigned int batchSize, float learningRate) {

	unsigned int layerIndex = blockIdx.x;
	unsigned int previousLayerIndex = threadIdx.x;

	float currentWeight = synapseMatrix[(layerIndex * previousLayerSize) + previousLayerIndex];
	float averageInputValue = 0.0f;

	for (int i = 0; i < batchSize; i++) {
		averageInputValue += valueVectors[i][previousLayerIndex];
	}

	averageInputValue = averageInputValue / ((float) batchSize);

	float newWeight = currentWeight - (learningRate * errorVector[layerIndex] * averageInputValue);

	synapseMatrix[(layerIndex * previousLayerSize) + previousLayerIndex] = newWeight;

	if (previousLayerIndex == 0) {
		atomicAdd(&biasVector[layerIndex], (learningRate * errorVector[layerIndex]));
	}
}


/**
 * Perform the sigmoid function on each element in each vector in-place. The sigmoid is calculated in parallel on the GPU
 * for each element.
 * 
 * @param vectors An array of vectors allocated on the GPU.
 * @param numVectors The number of vectors in @vectors.
 * @param vectorLength The length of each vector in @vectors.
 * @param activation The activation function to use.
 */
void gpu_activate(float **vectors, unsigned int numVectors, unsigned int vectorLength, Activation *activation) {
	unsigned int numThreadBlocks = numVectors;
	unsigned int threadsPerBlock = vectorLength;

	/* Running on a Nvidia GeForce GTX 1080 which has a max thread per block count of 1024. May need to configure this constant for other GPUs. */

	if (threadsPerBlock > 1024)
		threadsPerBlock = 1024;

	activation_gpu_kernel<<<numThreadBlocks, threadsPerBlock>>>(vectors, activation);
}


/**
 * Runs error calculation for a neural network's output layer on the GPU. Calculates in parallel the error of each vector that represents
 * the values of the output neurons in a network and then averages the error for each neuron.
 * 
 * @param outputVectors An array of vectors that represents the values of neurons on a network's output layer.
 * @param expectedVector An array of vectors that represents the expected values of each of the neurons on an output layer.
 * @param errorVector A allocated vector on the GPU that will hold the result vector.
 * @param numVectors The number of vectors in @outputVectors.
 * @param vectorLength The length of each vector in @outputVectors, @expectedVector, and @errorVector.
 * @param activation The activation function to use.
 */
void gpu_calculateError(float **outputVectors, float *expectedVector, float *errorVector, unsigned int numVectors, unsigned int vectorLength, Activation *activation) {
	unsigned int numThreadBlocks = numVectors;
	unsigned int threadsPerBlock = vectorLength;

	if (threadsPerBlock > 1024)
		threadsPerBlock = 1024;

	calculateError_gpu_kernel<<<numThreadBlocks, threadsPerBlock>>>(outputVectors, expectedVector, errorVector, numVectors, vectorLength, activation);
}


/**
 * Runs backpropogation for one hidden layer of a neural network on the GPU. The error calculation for each neuron on a hidden layer is based on
 * the average of the neuron's errors for each training example in a batch. The calculation of each error for each training example is calculated in parallel
 * here and then once completed, the results are averaged to give the final error for the neuron.
 * 
 * @param synapseMatrix The synpase matrix that contains the output synapse weights for the target layer we are calculating error for. For example, if we are calculating error
 * for layer n, this synapse matrix will be for the synapses that connect layer n-1 to layer n.
 * @param errorVector The error vector for the layer backpropogation previously calculated. For instance, if we are calculating error for layer n, this will be the error vector
 * for layer n+1.
 * @param destinationErrorVector The error vector for the layer backpropogation is currently calculating. This is where the results of backpropogation will be stored.
 * @param destinationValueVector An array of value vectors for the layer backpropogation is currently calculating.
 * @param errorVectorSize The size of @errorVector.
 * @param destinationErrorSize The size of @destinationErrorVector.
 * @param batchSize The number of training examples in the current training batch. This will also be the size of @destinationValueVector.
 * @param activation The activation function to use.
 */
void gpu_backpropogate(float *synapseMatrix, float *errorVector, float *destinationErrorVector, float **destinationValueVector,
	unsigned int errorVectorSize, unsigned int destinationErrorSize, unsigned int batchSize, Activation *activation) {

	unsigned int numThreadBlocks = destinationErrorSize;
	unsigned int numThreads = errorVectorSize;

	if (numThreads > 1024)
		numThreads = 1024;

	backpropogate_gpu_kernel<<<numThreadBlocks, numThreads>>>(synapseMatrix, errorVector, destinationErrorVector, destinationValueVector, errorVectorSize, destinationErrorSize, batchSize, activation);
}


/**
 * Updates the weights and biases for a given layer based on the error calculated during backpropogation on the GPU. Each synapse connecting the target
 * layer with its previous layer is updated in parallel on the GPU.
 * 
 * @param synapseMatrix The synapse matrix that has the target layer as output.
 * @param valueVectors The values of the previous layer.
 * @param errorVector The error vector for the target layer.
 * @param biasVector The bias vector for the target layer.
 * @param layerSize The size of the target layer.
 * @param previousLayerSize The size of the previous layer.
 * @param batchSize The count of training examples in each training batch.
 * @param learningRate The learning rate of the network.
 */
void gpu_updateLayer(float *synapseMatrix, float **valueVectors, float *errorVector, float *biasVector, unsigned int layerSize, unsigned int previousLayerSize, unsigned int batchSize, float learningRate) {
	unsigned int numThreadBlocks = layerSize;
	unsigned int numThreads = previousLayerSize;

	if (numThreads > 1024)
		numThreads = 1024;

	updateLayer_gpu_kernel<<<numThreadBlocks, numThreads>>>(synapseMatrix, valueVectors, errorVector, biasVector, layerSize, previousLayerSize, batchSize, learningRate);
}