#include "usrp_utils.hpp"
#include <iostream> // For std::cout, std::cerr
#include <thread>   // For std::this_thread::sleep_for
#include <chrono>   // For 1s literal, std::chrono::duration
#include <boost/algorithm/string.hpp> // For boost::icontains, boost::iequals
#include <stdexcept> // For std::invalid_argument in string_to_severity

using namespace std::chrono_literals; // For 1s, 100ms, etc.

void perform_calibration(const Config& cfg, uhd::usrp::multi_usrp::sptr usrp) {
    std::cout << "\nPerforming device calibrations..." << std::endl;

    size_t channel = 0; // Assuming single channel for calibration for now
     if (cfg.enable_rx) {
         try {
             std::cout << "  Calibrating RX channel " << channel << "..." << std::endl;
             usrp->set_rx_agc(false, channel); // Disable AGC for manual cal

             std::cout << "    Performing RX DC offset calibration..." << std::endl;
             usrp->set_rx_dc_offset(true, channel); // Enable automatic DC offset cal
             std::this_thread::sleep_for(1s); // Give it time to settle

             std::cout << "    Performing RX IQ imbalance calibration..." << std::endl;
             usrp->set_rx_iq_balance(true, channel); // Enable automatic IQ imbalance cal
             std::this_thread::sleep_for(1s); // Give it time to settle

             std::cout << "  RX Calibration for channel " << channel << " complete." << std::endl;
         } catch (const uhd::exception& e) {
            std::cerr << "Warning: RX calibration failed for channel " << channel << ": " << e.what() << std::endl;
         }
     } else {
         std::cout << "  Skipping RX calibration (RX disabled)." << std::endl;
     }

     if (cfg.enable_tx) {
        try {
             std::cout << "  Calibrating TX channel " << channel << "..." << std::endl;
             auto sensor_names = usrp->get_tx_sensor_names(channel);
             bool has_tx_iq_cal = false;
             bool has_tx_dc_cal = false; // Some devices might have this

             if (!sensor_names.empty()) {
                 for (const auto& name : sensor_names) {
                     if (boost::icontains(name, "iq_balance")) {
                         has_tx_iq_cal = true;
                     }
                     if (boost::icontains(name, "dc_offset")) { // Check for DC offset sensor
                         has_tx_dc_cal = true;
                     }
                 }

                 if (has_tx_dc_cal) {
                    std::cout << "    Performing TX DC offset self-calibration..." << std::endl;
                    usrp->set_tx_dc_offset(true, channel); // Some USRPs might support this
                    std::this_thread::sleep_for(1s);
                 } else {
                    std::cout << "    TX DC offset self-calibration not detected/supported for this channel." << std::endl;
                 }


                 if (has_tx_iq_cal) {
                     std::cout << "    Performing TX IQ imbalance self-calibration..." << std::endl;
                     usrp->set_tx_iq_balance(true, channel);
                     std::this_thread::sleep_for(1s);
                 } else {
                     std::cout << "    TX IQ imbalance self-calibration not detected/supported for this channel." << std::endl;
                 }
              } else {
                 std::cout << "    Could not retrieve TX sensor names for channel " << channel << "." << std::endl;
              }
              std::cout << "  TX Calibration for channel " << channel << " complete (capabilities vary by device)." << std::endl;
        } catch (const uhd::key_error& e) {
             std::cerr << "Warning: TX calibration sensor not found for channel " << channel << ": " << e.what() << std::endl;
        } catch (const uhd::exception& e) {
             std::cerr << "Warning: TX calibration failed for channel " << channel << ": " << e.what() << std::endl;
         }
    } else {
         std::cout << "  Skipping TX calibration (TX disabled)." << std::endl;
     }
    std::cout << "Device calibrations finished.\n" << std::endl;
}

uhd::log::severity_level string_to_severity(const std::string& level) {
    if (boost::iequals(level, "trace"))
        return uhd::log::severity_level::trace;
    else if (boost::iequals(level, "debug"))
        return uhd::log::severity_level::debug;
    else if (boost::iequals(level, "info"))
        return uhd::log::severity_level::info;
    else if (boost::iequals(level, "warning"))
        return uhd::log::severity_level::warning;
    else if (boost::iequals(level, "error"))
        return uhd::log::severity_level::error;
    else if (boost::iequals(level, "fatal"))
        return uhd::log::severity_level::fatal;
    else
        throw std::invalid_argument("Invalid log level: " + level);
}