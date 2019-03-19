//#include "Util.h"
#include <random>
#include <math.h>
#include <iostream>


std::random_device device;
std::default_random_engine generator(device());
std::normal_distribution<float> stdNormalDist(0.0, 1.0);


float standardNormalRandom() {
	return stdNormalDist(generator);
}


float randomWeight(unsigned int previousLayerSize) {
	return standardNormalRandom() * sqrtf(2 / ((float) previousLayerSize)); // He et al. (2015) initialization
}


float sigmoid(float x) {
	return 1 / (1 + exp(-x));
}


float sigmoidDerivative(float x) {
	return x * (1 - x);
}