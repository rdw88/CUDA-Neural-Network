#ifndef UTIL_H__
#define UTIL_H__

float standardNormalRandom();

float randomWeight(unsigned int previousLayerSize);

float sigmoid(float x);

float sigmoidDerivative(float x);

#endif