#ifndef BARRAGE_H
#define BARRAGE_H

#include "args.h"

#include <complex>
#include <vector>

// Wideband barrage jamming: random frequency and amplitude per sample
// across [center_frequency ± bandwidth/2].
std::vector<std::complex<float>> generate_barrage(const all_args_t &args);

#endif
