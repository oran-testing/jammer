#include "noise.h"

std::complex<float> srsran_vec_gen_sine_simd(std::complex<float> amplitude,
                                             float freq, std::complex<float> *z,
                                             int len) {
  const float TWOPI = 2.0f * (float)M_PI;
  std::complex<float> osc = std::polar(1.0f, TWOPI * freq); //Angular Velocity Calculation (W)(2(pi)(f))
  std::complex<float> phase = 1.0f;
  int i = 0;

  for (; i < len; i++) {
    z[i] = amplitude * phase; //Stores the evolution of the wave. This is giving the rotation instances around the circumfence of a circle with radius = amplitude; TLDR, not a sin wave.

    phase *= osc; // phase = phase * angular velocity This is just generating a simple rotation of a complex number in polar form
  } //TLDR; Not a Sin wave generator
  return phase;
}
// s(t) = Amplitude * sin(wt + phase)

//z1 = r1e^(i(theta)) Complex number representation
//z = r(cos(angle)+(i)sin(angle)) = r (e^(i(angle))) Another Representation of a compelx numeber