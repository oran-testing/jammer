#ifndef NOISE_H
#define NOISE_H
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

std::vector<std::complex<float>>
generateComplexSineWave(float amplitude, float amplitude_width,
                        float center_frequency, float bandwidth,
                        float initial_phase, float sampling_freq,
                        size_t num_samples);

void transmission(uhd::usrp::multi_usrp::sptr usrp, float amplitude,
                  float amplitude_width, float center_frequency,
                  float bandwidth, float sampling_freq, size_t buffer_size, size_t num_samples, float initial_phase);
#endif
