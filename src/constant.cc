#include "constant.h"

#include <cmath>
#include <complex>
#include <random>
#include <vector>

namespace {
constexpr float PI = std::acos(-1.0f);
} // namespace

// Constant jammer — v3 (narrowband Gaussian noise jamming).
//
// Method validated by published 5G NR jamming experiments on USRP B210 at
// 23.04 MS/s (e.g. "STORM: Stealth Jamming Attacks on 5G Synchronization",
// CEUR 2025; Antosik et al., "Narrowband jamming in 5G private networks",
// IEEE 2024). Key findings these papers establish:
//
//   * Effective narrowband 5G jamming uses **true complex Gaussian noise**,
//     not a sum of equal-amplitude tones. Equal-amplitude tones look like
//     discrete CW tones to the UE RX AGC, which suppresses each
//     individually; Gaussian noise raises the receiver's noise floor.
//   * Bandwidth around **1.92 MHz** (one SSB-worth, 128 subcarriers at
//     15 kHz SCS) concentrates power on a meaningful sub-band. Wider
//     bandwidth loses the ~25 dB "narrowband jamming gain" advantage
//     (Antosik 2024).
//   * The jammer must be **offset from DC**, because NR's DC subcarrier
//     is empty and the USRP B-series DC-offset cancellation suppresses
//     any tone near LO.
//
// Pipeline:
//   1. Generate complex white Gaussian noise samples at rate jam_bandwidth_hz
//      (using zero-order hold from a slow noise source to the USRP sample rate).
//      This naturally band-limits the energy to roughly ±jam_bw/2 around DC.
//   2. Frequency-shift by tone_offset_hz so the band lands where we want.
//   3. If the caller picks a near-zero offset, automatically shift to a safe
//      non-DC offset (default 1.92 MHz, matching the STORM recipe).
std::vector<std::complex<float>> generate_constant(const all_args_t &args) {
  std::vector<std::complex<float>> samples;
  samples.reserve(args.num_samples);

  const float dt = 1.0f / args.sampling_freq;
  const float jam_bw = std::max(args.jam_bandwidth_hz, 100.0e3f); // floor 100 kHz

  // Auto-shift off DC if the user requested ~0 offset. NR leaves DC empty
  // and the B-series DC-offset correction nulls anything close to LO.
  float center_offset = args.tone_offset_hz;
  if (std::fabs(center_offset) < 50.0e3f) {
    center_offset = 1.92e6f; // STORM-style: one SSB-width above LO
  }

  // Zero-order hold upsampling factor: how many output samples each random
  // sample is held. For 23.04 MS/s sampling and 1.92 MHz jam bandwidth this
  // is 12, producing a sinc-shaped band-limited noise of width ≈ jam_bw.
  const int hold =
      std::max(1, static_cast<int>(std::round(args.sampling_freq / jam_bw)));

  // Complex Gaussian noise with amplitude calibrated so the typical sample
  // magnitude is around args.amplitude. Sigma = amplitude * 0.5 gives an
  // RMS magnitude near amplitude/sqrt(2) and peak excursions around
  // 3 * sigma ~ 1.5 * amplitude, well within DAC headroom for amplitude ≤ 0.6.
  const float sigma = args.amplitude * 0.5f;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> normal(0.0f, sigma);

  float phase = args.initial_phase;
  const float dphase = 2.0f * PI * center_offset * dt;

  float re_held = 0.0f;
  float im_held = 0.0f;
  int hold_cnt = 0;

  for (size_t i = 0; i < args.num_samples; ++i) {
    // Draw a fresh complex Gaussian sample every `hold` output samples.
    if (hold_cnt == 0) {
      re_held = normal(gen);
      im_held = normal(gen);
    }
    if (++hold_cnt >= hold) {
      hold_cnt = 0;
    }

    // Mix the held noise sample up to the desired baseband offset:
    //   y[n] = noise[n] * exp(j * 2*pi * center_offset * n * dt)
    const float c = std::cos(phase);
    const float s = std::sin(phase);
    const float re_out = re_held * c - im_held * s;
    const float im_out = re_held * s + im_held * c;
    samples.emplace_back(re_out, im_out);

    phase += dphase;
    if (phase > 2.0f * PI) {
      phase -= 2.0f * PI;
    } else if (phase < 0.0f) {
      phase += 2.0f * PI;
    }
  }

  return samples;
}
