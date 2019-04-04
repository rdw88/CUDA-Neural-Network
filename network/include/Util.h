/**
 * Util.h
 * April 4, 2019
 * Ryan Wise
 * 
 * Basic utility functions used by the neural network.
 * 
 */


#ifndef UTIL_H__
#define UTIL_H__

float standardNormalRandom();

float randomWeight(unsigned int previousLayerSize);

float sigmoid(float x);

float sigmoidDerivative(float x);

#endif