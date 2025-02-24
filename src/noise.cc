#include <iostream>
#include <complex>
#include <cmath>
#include <vector>

const double PI = std::acos(-1.0);

std::vector<std::complex<double>> generateComplexSineWave(double amplitude,
                                                         double initial_frequency,
                                                         double frequency_change_rate,
                                                         double initial_phase,
                                                         int num_samples,
                                                         double sample_rate) {
    std::vector<std::complex<double>> samples(num_samples);
    const double delta_t = 1.0 / sample_rate; // Calculates the time between each sample; Sample interval
    double current_frequency = initial_frequency;
    double phase = initial_phase;

    for (int i = 0; i < num_samples; ++i) {
        // Generate complex sample using polar coordinates
        samples[i] = std::polar(amplitude, phase);
        
        // Update phase for next sample (correct frequency ramp integration)
        phase += 2 * PI * current_frequency * delta_t; // 2pif = angular velocity, 2pif(deltat) = change in angle
        
        // Keep phase wrapped to [0, 2π) to prevent precision loss
        phase = fmod(phase, 2 * PI);
        
        // Update frequency for next sample
        current_frequency += frequency_change_rate;
    }

    return samples;
}

int main() {
    // Parameters
    const double amplitude = 1.0;            // Constant amplitude
    const double initial_frequency = 10.0;    // Starting frequency (Hz)
    const double freq_change_rate = 0.1;      // Frequency change per sample (Hz/sample)
    const double initial_phase = 0.0;         // Starting phase (radians)
    const int num_samples = 1000;             // Number of samples to generate
    const double sample_rate = 1000.0;        // Sampling rate (Hz)

    // Generate waveform
    auto signal = generateComplexSineWave(amplitude,
                                        initial_frequency,
                                        freq_change_rate,
                                        initial_phase,
                                        num_samples,
                                        sample_rate);

    // Output samples
    for (const auto& sample : signal) {
        std::cout << "Real: " << sample.real() 
                  << " Imag: " << sample.imag() 
                  << std::endl;
    }

    return 0;
}