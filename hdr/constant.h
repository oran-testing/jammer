#ifndef CONSTANT_H
#define CONSTANT_H

#include "args.h"

#include <complex>
#include <vector>

// Constant (narrowband Gaussian noise) jamming: true complex Gaussian
// noise band-limited to jam_bandwidth_hz and centered at tone_offset_hz
// off LO. Method matches published 5G NR jamming experiments on USRP
// B-series at 23.04 MS/s (STORM CEUR 2025; Antosik IEEE 2024) which
// found Gaussian noise raises the UE noise floor instead of being
// rejected by AGC the way discrete CW tones are.
std::vector<std::complex<float>> generate_constant(const all_args_t &args);

#endif
