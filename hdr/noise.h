#ifndef NOISE_H
#define NOISE_H
#include <cmath>
#include <complex>
#include <math.h>
#include <vector>

const double PI = std::acos(-1.0);

std::vector<std::complex<double>>
generateComplexSineWave(double amplitude, double initial_frequency,
                        double frequency_change_rate, double initial_phase,
                        int num_samples, double sample_rate);
#endif


