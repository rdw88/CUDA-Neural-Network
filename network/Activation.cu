#include "Activation.h"
#include <limits>


/**
 * Create a new Activation with the provided ActivationType.
 * 
 * @param activationType The ActivationType to use.
 * @return A new Activation.
 */
Activation newActivation(ActivationType activationType) {
    Activation activation;
    activation.activationType = activationType;
    activation.maxThreshold = std::numeric_limits<float>::max();
    activation.leakyReluGradient = 0;

    return activation;
}