#include "noise.h"
#include <chrono>   // Required for time measurements
#include <iostream> // Required for std::cout
#include <thread>   // Potentially useful for timing, though chrono is primary

// Assuming uhd/usrp/multi_usrp.hpp, uhd/stream.hpp, uhd/types/tune_request.hpp, etc.
// are included via "noise.h" or are available in the build environment.

// Forward declaration of generateComplexSineWave if it's not in noise.h
// std::vector<std::complex<float>> generateComplexSineWave(const all_args_t args);

// Definition of generateComplexSineWave (as provided in the problem description)
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
       i++) { // generates args.num_samples samples

    float current_freq = freq_dist(gen);
    float current_ampl = ampl(gen);

    // Generate complex sample using polar coordinates
    samples.push_back(std::polar(current_ampl, phase));

    // Update phase for next sample (correct frequency ramp integration)
    phase += 2 * PI * current_freq * delta_t;

    // Keep phase wrapped to [0, 2π) to prevent precision loss
    phase = fmod(
        phase, 2 * PI); // remander angle after subtracting the multiples of 2pi
    if (phase < 0.0) {
      phase += 2 * PI;
    }
  }

  return samples;
}


// Modified transmission function
void transmission(uhd::usrp::multi_usrp::sptr usrp, const all_args_t args) {

  // Configure the USRP transmission stream
  uhd::stream_args_t stream_args("fc32",
                                 "sc16"); // Complex float to short conversion
  uhd::tx_streamer::sptr tx_stream = usrp->get_tx_stream(stream_args);

  uhd::tx_metadata_t metadata;
  metadata.start_of_burst =
      true; // First packet should have start_of_burst = true
  metadata.end_of_burst =
      false; // Will be set to true for the *last* packet
  metadata.has_time_spec = false;

  // Generate the samples ONCE before the loop
  std::vector<std::complex<float>> samples = generateComplexSineWave(args);
  if (samples.empty()) {
      std::cerr << "Warning: Generated sample buffer is empty. Nothing to transmit." << std::endl;
      return; // Exit if no samples were generated
  }

  // Define the transmission duration
  const auto transmission_duration = std::chrono::seconds(4);

  // Get the starting time point
  const auto start_time = std::chrono::steady_clock::now();

  std::cout << "Starting transmission for approximately 4 seconds..." << std::endl;

  // Loop while the elapsed time is less than the desired duration
  while (std::chrono::steady_clock::now() - start_time < transmission_duration) {

    // Transmit samples
    // Note: send() might block, the loop timing is approximate.
    size_t num_samps_sent = tx_stream->send(samples.data(), samples.size(), metadata);

    // Optional: Check if send was successful (num_samps_sent == samples.size())
    if (num_samps_sent != samples.size()) {
        std::cerr << "Warning: Failed to send all samples in one go." << std::endl;
        // Depending on requirements, you might want to handle this differently
        // (e.g., break the loop, retry, etc.)
    }

    // std::cout << "Transmitting chunk..." << std::endl; // Can be noisy

    // After the first packet, set `start_of_burst = false`
    // This only needs to be set once after the first successful send.
    if (metadata.start_of_burst) {
        metadata.start_of_burst = false;
    }
  }

  // --- Loop finished ---

  // Signal the end of the transmission burst
  // It's good practice to send a final packet (can be zero-length)
  // with end_of_burst set to true.
  std::cout << "Transmission time elapsed. Sending end-of-burst signal..." << std::endl;
  metadata.end_of_burst = true;
  // Send a zero-length packet with the EOB flag set
  tx_stream->send("", 0, metadata);

  std::cout << "Transmission stopped." << std::endl;

  // The function will now naturally return, stopping this part of the program.
}