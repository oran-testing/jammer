#ifndef NOISE_H
#define NOISE_H
#include "args.h"
#include <cmath>
#include <complex>
#include <iostream>
#include <math.h>
#include <random>
#include <uhd/error.h>
#include <uhd/stream.hpp>
#include <uhd/types/device_addr.hpp>
#include <uhd/usrp/multi_usrp.hpp>
#include <uhd/usrp/usrp.h>
#include <vector>

const float PI = std::acos(-1.0);

std::vector<std::complex<float>> generateComplexSineWave(const all_args_t args);

void transmission(uhd::usrp::multi_usrp::sptr usrp, const all_args_t args);

#endif
