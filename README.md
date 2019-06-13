# CUDA-Enabled Neural Network

An implementation of a fully connected neural network written in C++ using CUDA kernels.

#### Features
1. Supports any number of hidden layers
2. Mini-batch training
3. Parallelization of training batches
4. Batched output
5. Supports assignment of activation functions for each layer
    * ReLU (with max-threshold hyperparameter)
    * Leaky ReLU
    * Sigmoid
    * Softmax (only supported on the output layer)
6. Track the error of the network during training to validate convergence
7. Save networks to disk
8. Load networks from disk
9. Python interface

#### Requirements

* Nvidia CUDA Toolkit (NVCC)
* Nvidia cuBLAS library
* Python 3.6+ (for python interface)

#### Building

##### DLL Export for Python Interface

```
./build.bat app
```

##### Standalone Unit Tests

```
./build.bat test
```

#### Example

```python
from ann import NeuralNetwork, Activation, LossFunction

# Create a new neural network with 1 hidden layer. Input layer size = 3, Hidden layer size = 2, Output layer size = 1
# Set the batch size of the network to 2
# Set the learning rate of the network to 0.4
# Provide a path to save the network to disk.
network = NeuralNetwork([3, 2, 1], batch_size=2, learning_rate=0.4, output_file='file.csv')

# Set activation functions for each layer
network.set_layer_activations([Activation.relu(max_threshold=100), Activation.relu(max_threshold=100), Activation.sigmoid()])

# Set the loss function to use when calculating the error of the network
network.set_loss_function(LossFunction.MEAN_SQUARED_ERROR)

# Train the network in batches (batch size set to 2)
network.train(
    [
        [0.5, 0.3, 0.2],
        [0.1, 0.9, 0.3]
    ],

    [
        [1.0],
        [0.2]
    ]
)

# Get the output of the network for the provided input
output = network.output([0.1, 0.9, 0.3])

# Save the network to disk
network.save()
```