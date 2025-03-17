#include "noise.h"

void transmission(uhd::usrp::multi_usrp::sptr usrp, const all_args_t args) {

  // Configure the USRP transmission stream
  uhd::stream_args_t stream_args("fc32", "sc16"); // Complex float to short conversion
  uhd::tx_streamer::sptr tx_stream = usrp->get_tx_stream(stream_args);

  uhd::tx_metadata_t metadata;
  metadata.start_of_burst = true;   // First packet should have start_of_burst = true
  metadata.end_of_burst = false;
  metadata.has_time_spec = false;

  // Preallocate the transmit buffer once with the required number of samples.
  std::vector<std::complex<float>> tx_buffer(args.num_samples);

  // Calculate the time step between samples.
  const float delta_t = 1.0f / args.sampling_freq;
  float halfBandwidth = args.bandwidth / 2.0f;

  // Initialize phase from the given initial phase.
  float phase = args.initial_phase;

  // Set up the random generators (frequency and amplitude variations)
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> freq_dist(args.center_frequency - halfBandwidth,
                                                  args.center_frequency + halfBandwidth);

  std::uniform_real_distribution<float> ampl_dist(args.amplitude - args.amplitude_width / 2.0f,
                                                  args.amplitude + args.amplitude_width / 2.0f);

  // Main transmission loop: update the preallocated buffer and transmit it.
  while (true) {
    // Fill the preallocated buffer with updated samples.
    for (size_t i = 0; i < args.num_samples; i++) {
      float current_freq = freq_dist(gen);
      float current_ampl = ampl_dist(gen);

      // Generate the complex sample using polar coordinates.
      tx_buffer[i] = std::polar(current_ampl, phase);

      // Update phase for the next sample.
      phase += 2 * PI * current_freq * delta_t;

      phase = fmod(phase, 2 * PI);  // Wrap phase to [0, 2π)
      
      if (phase < 0.0f) {
        phase += 2 * PI;
      }
    }

    // Transmit the updated buffer.
    tx_stream->send(tx_buffer.data(), tx_buffer.size(), metadata);
    std::cout << "Transmitting..." << std::endl;

    // After the first transmission, ensure start_of_burst is false.
    metadata.start_of_burst = false;
  }

  // Note: This loop runs indefinitely; you'll need to implement a stopping mechanism as appropriate.
}
