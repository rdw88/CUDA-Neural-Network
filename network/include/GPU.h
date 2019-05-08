/**
 * GPU.h
 * April 4, 2019
 * Ryan Wise
 * 
 * An interface to a CUDA-enabled NVIDIA GPU.
 * 
 */


#ifndef GPU_H__
#define GPU_H__


/**
 * The supported activation functions for a layer in a neural network.
 */
enum Activation {
	RELU = 0,
	SIGMOID = 1
};


/**
	Create the cuBLAS context for the neural network. This creates a global context that is used
	to perform basic linear algebra operations on the GPU.
*/
void createCublasContext();


/**
	Destroy the cuBLAS context.
*/
void destroyCublasContext();


/**
	Allocate pinned CPU memory. CPU memory is pageable by default and the GPU cannot access data directly in paged CPU
	memory. Using pinned memory removes pagination and therefore removes an extra step from moving data from CPU to GPU
	memory, increasing performance.

	@param numBytes The number of bytes of pinned memory to allocate.
	@return A pointer to the beginning of the allocated memory block.
*/
void *allocPinnedMemory(size_t numBytes);


/**
	Frees pinned memory.

	@param pointer A pointer to pinned memory allocated with *allocPinnedMemory()*.
*/
void freePinnedMemory(void *pointer);


/**
 * Allocates memory on the GPU.
 * 
 * @param size The number of bytes to allocate on the GPU.
 * @return A pointer to the beginning of the allocated memory block.
 */
void *gpu_allocMemory(size_t size);


/**
 * Copies memory to and from the GPU. Supports any permutation of copying memory between the CPU and GPU.
 * 
 * @param toPointer A pointer to CPU or GPU memory that will be copied to.
 * @param fromPointer A pointer to CPU or GPU memory that will be copied from.
 * @param numBytes The number of bytes to copy.
 */
void gpu_copyMemory(void *toPointer, void *fromPointer, size_t numBytes);


/**
 * Clear memory in the GPU (set to zero).
 * 
 * @param gpuPointer Pointer to the memory in the GPU to clear.
 * @param numBytes The number of bytes to zero out.
 */
void gpu_clearMemory(void *gpuPointer, size_t numBytes);


/**
 * Free memory in the GPU.
 * 
 * @param gpuPointer Pointer to GPU memory to free.
 */
void gpu_freeMemory(void *gpuPointer);


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
void gpu_batchVectorMatrixMultiply(float **matrices, float **vectors, float **results, unsigned int numColumns, unsigned int numRows, unsigned int batches);


/**
 * Perform the sigmoid function on each element in each vector in-place. The sigmoid is calculated in parallel on the GPU
 * for each element.
 * 
 * @param vectors An array of vectors allocated on the GPU.
 * @param numVectors The number of vectors in @vectors.
 * @param vectorLength The length of each vector in @vectors.
 * @param activation The activation function to use.
 */
void gpu_activate(float **vectors, unsigned int numVectors, unsigned int vectorLength, Activation activation);


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
void gpu_calculateError(float **outputVectors, float *expectedVector, float *errorVector, unsigned int numVectors, unsigned int vectorLength, Activation activation);


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
void gpu_backpropogate(float *synapseMatrix, float *errorVector, float *destinationErrorVector, float **destinationValueVector, unsigned int errorVectorSize, unsigned int destinationErrorSize, unsigned int batchSize, Activation activation);


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
void gpu_updateLayer(float *synapseMatrix, float **valueVectors, float *errorVector, float *biasVector, unsigned int layerSize, unsigned int previousLayerSize, unsigned int batchSize, float learningRate);


#endif