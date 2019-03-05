#include "Util.h"
#include <random>



float standardNormalRandom() {
	std::random_device device;
	std::default_random_engine generator(device());
	std::normal_distribution<float> stdNormalDist(0.0, 1.0);
	return stdNormalDist(generator);
}