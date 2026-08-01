#include "attack.h"

#include "barrage.h"
#include "constant.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <uhd/stream.hpp>

namespace {

size_t packets_for_duration_ms(float duration_ms, size_t num_samples,
                               float sampling_freq) {
  const float packet_duration_ms =
      (static_cast<float>(num_samples) / sampling_freq) * 1000.0f;
  if (packet_duration_ms <= 0.0f) {
    return 1;
  }

  const size_t packets =
      static_cast<size_t>(std::ceil(duration_ms / packet_duration_ms));
  return std::max(packets, size_t{1});
}

void transmission_continuous(uhd::usrp::multi_usrp::sptr usrp,
                           const all_args_t &args) {
  uhd::stream_args_t stream_args("fc32", "sc16");
  uhd::tx_streamer::sptr tx_stream = usrp->get_tx_stream(stream_args);

  uhd::tx_metadata_t metadata;
  metadata.start_of_burst = true;
  metadata.end_of_burst = false;
  metadata.has_time_spec = false;

  const std::vector<std::complex<float>> samples = generate_samples(args);

  while (true) {
    tx_stream->send(samples.data(), samples.size(), metadata);
    std::cout << "Transmitting (" << attackTypeToString(args.attack_type)
              << ")...." << std::endl;
    metadata.start_of_burst = false;
  }
}

void transmission_random(uhd::usrp::multi_usrp::sptr usrp,
                         const all_args_t &args) {
  uhd::stream_args_t stream_args("fc32", "sc16");
  uhd::tx_streamer::sptr tx_stream = usrp->get_tx_stream(stream_args);

  uhd::tx_metadata_t metadata;
  metadata.start_of_burst = true;
  metadata.end_of_burst = false;
  metadata.has_time_spec = false;

  const std::vector<std::complex<float>> burst_samples = generate_barrage(args);
  std::vector<std::complex<float>> silence_samples(args.num_samples, {0.0f, 0.0f});

  const size_t burst_packets = packets_for_duration_ms(
      args.burst_duration_ms, args.num_samples, args.sampling_freq);
  const size_t idle_packets = packets_for_duration_ms(
      args.idle_duration_ms, args.num_samples, args.sampling_freq);

  while (true) {
    for (size_t i = 0; i < burst_packets; ++i) {
      tx_stream->send(burst_samples.data(), burst_samples.size(), metadata);
      metadata.start_of_burst = false;
    }
    std::cout << "Transmitting (random) ON for " << args.burst_duration_ms
              << " ms...." << std::endl;

    for (size_t i = 0; i < idle_packets; ++i) {
      tx_stream->send(silence_samples.data(), silence_samples.size(), metadata);
    }
    std::cout << "Transmitting (random) IDLE for " << args.idle_duration_ms
              << " ms...." << std::endl;
  }
}

} // namespace

std::vector<std::complex<float>> generate_samples(const all_args_t &args) {
  switch (args.attack_type) {
  case attack_type_t::BARRAGE:
    return generate_barrage(args);
  case attack_type_t::CONSTANT:
    return generate_constant(args);
  case attack_type_t::RANDOM:
    return generate_barrage(args);
  }

  return generate_barrage(args);
}

void transmission(uhd::usrp::multi_usrp::sptr usrp, const all_args_t &args) {
  switch (args.attack_type) {
  case attack_type_t::BARRAGE:
  case attack_type_t::CONSTANT:
    transmission_continuous(usrp, args);
    return;
  case attack_type_t::RANDOM:
    transmission_random(usrp, args);
    return;
  }
}
