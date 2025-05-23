#pragma once

#include <uhd/usrp/multi_usrp.hpp>
#include "config.hpp" // For Config struct

// Forward declare to avoid circular dependency if Config needed them
// (not the case here, but good practice if it were)
// struct Config;

void rx_thread(uhd::usrp::multi_usrp::sptr usrp, const Config& cfg);
void tx_thread(uhd::usrp::multi_usrp::sptr usrp, const Config& cfg);