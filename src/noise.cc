#include "noise.h"

std::complex<float> srsran_vec_gen_sine_simd(std::complex<float> amplitude,
                                             float freq, std::complex<float> *z,
                                             int len) {
  const float TWOPI = 2.0f * (float)M_PI;
  std::complex<float> osc = std::polar(1.0f, TWOPI * freq);
  std::complex<float> phase = 1.0f;
  int i = 0;

  for (; i < len; i++) {
    z[i] = amplitude * phase;

    phase *= osc;
  }
  return phase;
}
