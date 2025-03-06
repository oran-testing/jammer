#include "noise.h"
#include <random>
#include<iostream>



std::vector<std::complex<double>>
generateComplexSineWave(double amplitude, double amplitude_width, double center_frequency, double bandwidth,
                        double initial_phase,
                        double sampling_freq, size_t num_samples) {

  std::vector<std::complex<double>> samples;
  samples.reserve(num_samples);
  
  const double delta_t = 1.0 /sampling_freq; // Calculates the time between each sample; Sample interval
  double halfBandwidth = bandwidth/(2.0*1000.0);
  double phase = initial_phase;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> freq_dist(center_frequency - halfBandwidth, center_frequency + halfBandwidth);
  std::uniform_real_distribution<double> ampl(amplitude - amplitude_width/2.0, amplitude + amplitude_width/2.0 );
  //generates random frequencies in the range [center-halfband, center+halfband]

  for(size_t i = 0; i < num_samples; i++) { //infinite no of samples generated

    double current_freq = freq_dist(gen);
    double current_ampl = ampl(gen);
    
    // Generate complex sample using polar coordinates
    /*Computes z = amplitude * cos(phase) {real part} + amplitude * (i)
    sin(phase) {imaginary part Do note that imaginary part is the coff. of i
    so sin(phase)} If the case of converting to cartesian plane arises the
    angle (phase) can be found by
    arcsin(imaginary/amplitude) or arccos(real/amp.)*/
    samples.push_back(std::polar(current_ampl, phase));

    
    // Update phase for next sample (correct frequency ramp integration)
    phase += 2 * PI * current_freq *
             delta_t; 
             
    // 2pif = angular velocity, 2pif(deltat) = change in angle
    // per sec * delta_t =
    // change in angle in delta_t time.

    // Keep phase wrapped to [0, 2π) to prevent precision loss
    
    phase = fmod(phase, 2 * PI); // remander angle after subtracting the multiples of 2pi 
    //doesn't change the value of cos or sin and also returns it to the principle value from 0 to 2pi. 
    if (phase < 0.0) {
      phase += 2 * PI; 
    }
  }

  return samples;
}

void transmission(uhd::usrp::single_usrp::sptr usrp, double amplitude, double amplitude_width, 
  double center_frequency, double bandwidth, double sampling_freq, 
  size_t buffer_size) {

// Configure the USRP transmission stream
uhd::stream_args_t stream_args("fc32", "sc16");  // Complex float to short conversion
uhd::tx_streamer::sptr tx_stream = usrp->get_tx_stream(stream_args);

uhd::tx_metadata_t metadata;
metadata.start_of_burst = true;  // First packet should have start_of_burst = true
metadata.end_of_burst = false;
metadata.has_time_spec = false;

while (true) {
// Generate `buffer_size` samples per iteration
std::vector<std::complex<double>> = generateComplexSineWave(amplitude, amplitude_width, center_frequency, 
                               bandwidth, 0.0, sampling_freq);

// Copy and convert samples to float
std::vector<std::complex<float>> tx_buffer;
tx_buffer.reserve(samples.size());

for (size_t i = 0; i < samples.size(); i++) {
  tx_buffer.push_back(std::complex<float>(static_cast<float>(samples[i].real()),
                                          static_cast<float>(samples[i].imag())));
}

// Transmit samples
tx_stream->send(buffer.data(), buffer.size(), metadata);

// After the first packet, set `start_of_burst = false`
metadata.start_of_burst = false;
}

// We will never reach this point unless we manually break the loop
}