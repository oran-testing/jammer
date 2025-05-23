#pragma once

#include <string>
#include <uhd/usrp/multi_usrp.hpp>
#include <uhd/utils/log.hpp> // For severity_level
#include "config.hpp"       // For Config struct in perform_calibration

void perform_calibration(const Config& cfg, uhd::usrp::multi_usrp::sptr usrp);
uhd::log::severity_level string_to_severity(const std::string& level);