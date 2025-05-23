#include "config.hpp"
#include <fstream>
#include <boost/algorithm/string.hpp> // For boost::iequals

void Config::load_from_yaml(const std::string& filename) {
    YAML::Node config_node; // Renamed from 'config' to avoid conflict with struct name
    try {
        config_node = YAML::LoadFile(filename);
    } catch (const YAML::Exception& e) {
        throw std::runtime_error("Error loading config file '" + filename + "': " + e.what());
    }

    #define LOAD_YAML_PARAM(param_name, type) \
        if (config_node[#param_name]) param_name = config_node[#param_name].as<type>()

    LOAD_YAML_PARAM(usrp_args, std::string);
    LOAD_YAML_PARAM(sample_rate, double);
    LOAD_YAML_PARAM(rx_gain, double);
    LOAD_YAML_PARAM(tx_gain, double);
    LOAD_YAML_PARAM(tx_center_freq, double);
    LOAD_YAML_PARAM(subdev, std::string);
    LOAD_YAML_PARAM(tx_ant, std::string);
    LOAD_YAML_PARAM(rx_ant, std::string);
    LOAD_YAML_PARAM(clock_rate, double);
    LOAD_YAML_PARAM(start_freq, double);
    LOAD_YAML_PARAM(end_freq, double);
    LOAD_YAML_PARAM(step_freq, double);
    LOAD_YAML_PARAM(settling_time, double);
    LOAD_YAML_PARAM(algorithm, std::string);
    LOAD_YAML_PARAM(fft_size, size_t);
    LOAD_YAML_PARAM(avg_num, size_t);
    LOAD_YAML_PARAM(fft_window_type, std::string);
    LOAD_YAML_PARAM(peak_threshold_db, double);
    LOAD_YAML_PARAM(prominence_threshold_db, double);
    LOAD_YAML_PARAM(num_samples_block, size_t);
    LOAD_YAML_PARAM(enable_tx, bool);
    LOAD_YAML_PARAM(enable_rx, bool);
    LOAD_YAML_PARAM(tx_waveform_type, std::string);
    LOAD_YAML_PARAM(tx_tone_freq_offset, double);
    LOAD_YAML_PARAM(tx_amplitude, float);
    LOAD_YAML_PARAM(ml_model_path, std::string);
    LOAD_YAML_PARAM(fft_wisdom_path, std::string);
    LOAD_YAML_PARAM(set_thread_priority, bool);
    LOAD_YAML_PARAM(verbose, bool);

    #undef LOAD_YAML_PARAM
}

void Config::save_to_yaml(const std::string& filename) const {
    YAML::Emitter emitter;
    emitter << YAML::BeginMap;

    #define SAVE_YAML_PARAM(param_name) \
        emitter << YAML::Key << #param_name << YAML::Value << param_name

    SAVE_YAML_PARAM(usrp_args);
    SAVE_YAML_PARAM(sample_rate);
    SAVE_YAML_PARAM(rx_gain);
    SAVE_YAML_PARAM(tx_gain);
    SAVE_YAML_PARAM(tx_center_freq);
    SAVE_YAML_PARAM(subdev);
    SAVE_YAML_PARAM(tx_ant);
    SAVE_YAML_PARAM(rx_ant);
    SAVE_YAML_PARAM(clock_rate);
    SAVE_YAML_PARAM(start_freq);
    SAVE_YAML_PARAM(end_freq);
    SAVE_YAML_PARAM(step_freq);
    SAVE_YAML_PARAM(settling_time);
    SAVE_YAML_PARAM(algorithm);
    SAVE_YAML_PARAM(fft_size);
    SAVE_YAML_PARAM(avg_num);
    SAVE_YAML_PARAM(fft_window_type);
    SAVE_YAML_PARAM(peak_threshold_db);
    SAVE_YAML_PARAM(prominence_threshold_db);
    SAVE_YAML_PARAM(num_samples_block);
    SAVE_YAML_PARAM(enable_tx);
    SAVE_YAML_PARAM(enable_rx);
    SAVE_YAML_PARAM(tx_waveform_type);
    SAVE_YAML_PARAM(tx_tone_freq_offset);
    SAVE_YAML_PARAM(tx_amplitude);
    SAVE_YAML_PARAM(ml_model_path);
    SAVE_YAML_PARAM(fft_wisdom_path);
    SAVE_YAML_PARAM(set_thread_priority);
    SAVE_YAML_PARAM(verbose);

    #undef SAVE_YAML_PARAM

    emitter << YAML::EndMap;

    std::ofstream fout(filename);
    if (!fout.is_open()) {
        throw std::runtime_error("Error opening file for saving config: " + filename);
    }
    fout << emitter.c_str();
}

void Config::validate() const {
    if (start_freq >= end_freq) throw std::runtime_error("Start frequency must be less than end frequency.");
    if (step_freq <= 0) throw std::runtime_error("Step frequency must be positive.");
    if (sample_rate <= 0) throw std::runtime_error("Sample rate must be positive.");
    if (fft_size <= 0) throw std::runtime_error("FFT size must be positive.");
    if (num_samples_block < fft_size && enable_rx) throw std::runtime_error("num_samples_block must be >= fft_size when RX is enabled.");
    if (avg_num == 0) throw std::runtime_error("Averaging number must be at least 1.");
    if (!boost::iequals(algorithm, "fft") && !boost::iequals(algorithm, "ml")) {
        throw std::runtime_error("Unsupported algorithm: " + algorithm + ". Choose 'fft' or 'ml'.");
    }
     if (boost::iequals(algorithm, "ml") && ml_model_path.empty()) {
        std::cerr << "Warning: ML algorithm selected but no model path provided." << std::endl;
    }
}