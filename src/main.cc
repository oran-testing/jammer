#include <iostream>

#include "noise.h"
#include "rf.h"

int main(int argc, char *argv[]) {

  // Parameters
  const double amplitude = 1.0;          // Constant amplitude
  const double initial_frequency = 10.0; // Starting frequency (Hz)
  const double freq_change_rate =
      0.1;                           // Frequency change per sample (Hz/sample)
  const double initial_phase = 0.0;  // Starting phase (radians)
  const int num_samples = 1000;      // Number of samples to generate
  const double sample_rate = 1000.0; // Sampling rate (Hz)

  // Generate waveform
  auto signal =
      generateComplexSineWave(amplitude, initial_frequency, freq_change_rate,
                              initial_phase, num_samples, sample_rate);

  // Output samples
  for (const auto &sample : signal) {
    std::cout << "Real: " << sample.real() << " Imag: " << sample.imag()
              << std::endl;
  }

  return 0;
}
