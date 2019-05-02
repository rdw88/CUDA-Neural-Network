# CUDA-Enabled Neural Network

An implementation of a fully connected neural network written in C++ using CUDA kernels.

#### Features
1. Supports any number of hidden layers.
2. Mini-batch training
3. Parallelization of training batches
4. Batched output.
4. Save networks to disk.
5. Load networks from disk.
6. Python interface

#### Requirements

* Nvidia CUDA Toolkit (NVCC)
* Nvidia cuBLAS library

#### Building

##### DLL Export for Python

```
cd network/
nvcc -O3 -shared -o bin/ann.dll -Iinclude/ NeuralNetwork.cu Util.cu GPU.cu Extern.cu
```

##### Standalone

```
cd network/
nvcc -O3 -o bin/ann.exe -Iinclude/ NeuralNetwork.cu Util.cu GPU.cu Extern.cu
```