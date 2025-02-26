#include "noise.h"

std::vector<std::complex<double>>
generateComplexSineWave(double amplitude, double initial_frequency,
                        double frequency_change_rate, double initial_phase,
                        int num_samples, double sample_rate) {

  std::vector<std::complex<double>> samples(num_samples);
  const double delta_t =
      1.0 /
      sample_rate; // Calculates the time between each sample; Sample interval
  double current_frequency = initial_frequency;
  double phase = initial_phase;

  for (int i = 0; i < num_samples; ++i) {

    // Generate complex sample using polar coordinates
    samples[i] = std::polar(amplitude, phase);
    // Computes z = amplitude * cos(phase) {real part} + amplitude * (i)
    // sin(phase) {imaginary part Do note that imaginary part is the coff. of i
    // so sin(phase)} If the case of converting to cartesian plane arises the
    // angle (phase) can be found by
    //  arcsin(imaginary/amplitude) or arccos(real/amp.).

    // Update phase for next sample (correct frequency ramp integration)
    phase += 2 * PI * current_frequency *
             delta_t; // 2pif = angular velocity, 2pif(deltat) = change in angle
                      // per sec * delta_t =
    // change in angle in delta_t time.

    // Keep phase wrapped to [0, 2π) to prevent precision loss
    phase = fmod(phase, 2 * PI);
    if (phase < 0.0) {
      phase += 2 * PI;
    }

    // Update frequency for next sample
    current_frequency += frequency_change_rate * delta_t; // updated the mistake
  }

  return samples;
}
