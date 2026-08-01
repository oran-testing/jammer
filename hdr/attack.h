#ifndef ATTACK_H
#define ATTACK_H

#include "args.h"

#include <complex>
#include <uhd/usrp/multi_usrp.hpp>
#include <vector>

std::vector<std::complex<float>> generate_samples(const all_args_t &args);

void transmission(uhd::usrp::multi_usrp::sptr usrp, const all_args_t &args);

#endif
