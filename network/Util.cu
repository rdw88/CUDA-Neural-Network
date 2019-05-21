/**
 * Util.cu
 * April 4, 2019
 * Ryan Wise
 * 
 * Basic utility functions used by the neural network.
 * 
 */


#include "Util.h"
#include <random>
#include <math.h>


std::random_device device;
std::default_random_engine generator(device());
std::normal_distribution<float> stdNormalDist(0.0, 1.0);


float standardNormalRandom() {
	return stdNormalDist(generator);
}


float randomWeight(unsigned int previousLayerSize) {
	return standardNormalRandom() * sqrtf(2 / ((float) previousLayerSize)); // He et al. (2015) initialization
}