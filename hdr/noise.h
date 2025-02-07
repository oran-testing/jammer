#include <complex>
#include <math.h>

std::complex<float> srsran_vec_gen_sine_simd(std::complex<float> amplitude,
                                             float freq, std::complex<float> *z,
                                             int len);
