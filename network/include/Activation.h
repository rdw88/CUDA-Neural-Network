#ifndef ACTIVATION_H__
#define ACTIVATION_H__


/**
 * The supported activation functions for a layer in a neural network.
 */
enum ActivationType {
	RELU = 0,
	SIGMOID = 1,
    SOFTMAX = 2
};


/**
 * An activation function including hyperparameters that can be optionally defined for the activation function.
 */
typedef struct Activation {
    ActivationType activationType;
    float maxThreshold;
    float leakyReluGradient;
} Activation;


/**
 * Create a new Activation with the provided ActivationType.
 * 
 * @param activationType The ActivationType to use.
 * @return A new Activation.
 */
Activation newActivation(ActivationType activationType);

#endif