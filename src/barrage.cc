#include "barrage.h"

#include <cmath>
#include <random>

namespace {
constexpr float PI = std::acos(-1.0f);
} // namespace

std::vector<std::complex<float>> generate_barrage(const all_args_t &args) {
  std::vector<std::complex<float>> samples;
  samples.reserve(args.num_samples);

  const float delta_t = 1.0f / args.sampling_freq;
  const float half_bandwidth = args.bandwidth / 2.0f;
  float phase = args.initial_phase;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> freq_dist(
      args.center_frequency - half_bandwidth,
      args.center_frequency + half_bandwidth);
  std::uniform_real_distribution<float> ampl(
      args.amplitude - args.amplitude_width / 2.0f,
      args.amplitude + args.amplitude_width / 2.0f);

  for (size_t i = 0; i < args.num_samples; ++i) {
    const float current_freq = freq_dist(gen);
    const float current_ampl = ampl(gen);

    samples.push_back(std::polar(current_ampl, phase));

    phase += 2.0f * PI * current_freq * delta_t;
    phase = std::fmod(phase, 2.0f * PI);
    if (phase < 0.0f) {
      phase += 2.0f * PI;
    }
  }

  return samples;
}
