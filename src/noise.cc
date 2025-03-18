#include "noise.h"

std::vector<std::complex<float>>
generateComplexSineWave(const all_args_t args) {

  std::vector<std::complex<float>> samples;
  samples.reserve(args.num_samples);

  const float delta_t =
      1.0f / args.sampling_freq; // Calculates the time between each sample;
                                // Sample interval
  float halfBandwidth = args.bandwidth / (2.0f);
  float phase = args.initial_phase;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> freq_dist(
      args.center_frequency - halfBandwidth,
      args.center_frequency + halfBandwidth);
  std::uniform_real_distribution<float> ampl(
      args.amplitude - args.amplitude_width / 2.0f,
      args.amplitude + args.amplitude_width / 2.0f);
  // generates random frequencies in the range [center-halfband,
  // center+halfband]

  for (size_t i = 0; i < args.num_samples;
       i++) { // infinite no of samples generated

    float current_freq = freq_dist(gen);
    float current_ampl = ampl(gen);

    // Generate complex sample using polar coordinates
    /*Computes z = args.amplitude * cos(phase) {real part} + args.amplitude *
    (i) sin(phase) {imaginary part Do note that imaginary part is the coff. of i
    so sin(phase)} If the case of converting to cartesian plane arises the
    angle (phase) can be found by
    arcsin(imaginary/args.amplitude) or arccos(real/amp.)*/
    samples.push_back(std::polar(current_ampl, phase));

    // Update phase for next sample (correct frequency ramp integration)
    phase += 2 * PI * current_freq * delta_t;

    // 2pif = angular velocity, 2pif(deltat) = change in angle
    // per sec * delta_t =
    // change in angle in delta_t time.

    // Keep phase wrapped to [0, 2π) to prevent precision loss

    phase = fmod(
        phase, 2 * PI); // remander angle after subtracting the multiples of 2pi
    // doesn't change the value of cos or sin and also returns it to the
    // principle value from 0 to 2pi.
    if (phase < 0.0) {
      phase += 2 * PI;
    }
  }

  return samples;
}

void transmission(uhd::usrp::multi_usrp::sptr usrp, const all_args_t args) {

  // Configure the USRP transmission stream
  uhd::stream_args_t stream_args("fc32",
                                 "sc16"); // Complex float to short conversion
  uhd::tx_streamer::sptr tx_stream = usrp->get_tx_stream(stream_args);

  uhd::tx_metadata_t metadata;
  metadata.start_of_burst =
      true; // First packet should have start_of_burst = true
  metadata.end_of_burst = false;
  metadata.has_time_spec = false;

  std::vector<std::complex<float>> samples = generateComplexSineWave(args);
  
  while (true) {

    // Transmit samples
    tx_stream->send(samples.data(), samples.size(), metadata);
    std::cout << "Transmitting...." << std::endl;

    // After the first packet, set `start_of_burst = false`
    metadata.start_of_burst = false;
  }

  // We will never reach this point unless we manually break the loop
}
