#pragma once

#include <string>
#include <vector>
#include <stdexcept> // For std::runtime_error
#include <iostream>  // For std::cerr in validate

// Forward declaration for YAML::Node to avoid full yaml-cpp include here if possible
// However, it's often cleaner to just include it.
#include <yaml-cpp/yaml.h>

// Constants
constexpr float TX_AMPLITUDE_DEFAULT = 0.5f;

struct Config {
    // USRP Settings
    std::string usrp_args = "type=b200";
    double sample_rate = 1e6;
    double rx_gain = 50.0;
    double tx_gain = 50.0;
    double tx_center_freq = 1.5e9;
    std::string subdev = "A:A";
    std::string tx_ant = "TX/RX";
    std::string rx_ant = "TX/RX";
    double clock_rate = 0.0;

    // Scanning Parameters
    double start_freq = 2e9;
    double end_freq = 3e9;
    double step_freq = 100e6;
    double settling_time = 0.05;

    // Processing Parameters
    std::string algorithm = "fft";
    size_t fft_size = 1024;
    size_t avg_num = 10;
    std::string fft_window_type = "hann";
    double peak_threshold_db = -60.0;
    double prominence_threshold_db = 5.0;
    size_t num_samples_block = 16384;

    // Transmission Parameters
    bool enable_tx = false;
    bool enable_rx = true;
    std::string tx_waveform_type = "tone";
    double tx_tone_freq_offset = 100e3;
    float tx_amplitude = TX_AMPLITUDE_DEFAULT;

    // ML Parameters
    std::string ml_model_path = "";

    // Performance & Misc
    std::string fft_wisdom_path = "";
    bool set_thread_priority = true;
    bool verbose = false;

    void load_from_yaml(const std::string& filename);
    void save_to_yaml(const std::string& filename) const;
    void validate() const;
};