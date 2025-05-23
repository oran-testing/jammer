#include "config.hpp"
#include "globals.hpp"
#include "shared_data.hpp"
#include "usrp_scanner.hpp"
#include "usrp_utils.hpp"
#include "fft_processor.hpp" // Specifically for FFTProcessor::save_wisdom

#include <uhd/usrp/multi_usrp.hpp>
#include <uhd/utils/safe_main.hpp> // For UHD_SAFE_MAIN and signal handling
#include <uhd/utils/log.hpp>       // For uhd::log::set_log_level
#include <uhd/types/tune_request.hpp> // For uhd::tune_request_t

#include <boost/program_options.hpp>
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp> // For boost::iequals in main options

#include <thread>
#include <iostream>
#include <chrono>    // For literals like 100ms
#include <stdexcept> // For std::invalid_argument from string_to_severity
#include <csignal>   // For signal handling (though safe_main helps)

// Namespaces
namespace po = boost::program_options;
using namespace std::chrono_literals;


// UHD_SAFE_MAIN handles SIGINT (Ctrl+C) and sets stop_signal_called
int UHD_SAFE_MAIN(int argc, char* argv[]) {
    Config cfg; // Default config

    // --- Command Line Options ---
    po::options_description desc("Advanced USRP Spectrum Scanner Options");
    desc.add_options()
        ("help,h", "Show help message")
        ("config,c", po::value<std::string>(), "Load configuration from YAML file")
        ("save-config", po::value<std::string>(), "Save current configuration to YAML file and exit")
        ("verbose,v", po::bool_switch(&cfg.verbose)->default_value(cfg.verbose), "Enable verbose output")

        // USRP Settings
        ("usrp-args", po::value<std::string>(&cfg.usrp_args)->default_value(cfg.usrp_args), "UHD device arguments (e.g., 'type=b210')")
        ("rate", po::value<double>(&cfg.sample_rate)->default_value(cfg.sample_rate)->notifier([](double r){ if(r <= 0) throw po::validation_error(po::validation_error::invalid_option_value, "rate", std::to_string(r)); }), "Sample rate (Sps)")
        ("rx-gain", po::value<double>(&cfg.rx_gain)->default_value(cfg.rx_gain), "RX gain (dB)")
        ("tx-gain", po::value<double>(&cfg.tx_gain)->default_value(cfg.tx_gain), "TX gain (dB)")
        ("tx-freq", po::value<double>(&cfg.tx_center_freq)->default_value(cfg.tx_center_freq), "TX center frequency (Hz)")
        ("rx-ant", po::value<std::string>(&cfg.rx_ant)->default_value(cfg.rx_ant), "RX Antenna")
        ("tx-ant", po::value<std::string>(&cfg.tx_ant)->default_value(cfg.tx_ant), "TX Antenna")
        ("subdev", po::value<std::string>(&cfg.subdev)->default_value(cfg.subdev), "USRP Subdevice Spec")
        ("clock-rate", po::value<double>(&cfg.clock_rate)->default_value(cfg.clock_rate), "Optional Master clock rate (Hz, 0 for default)")

        // Scanning Parameters
        ("start-freq", po::value<double>(&cfg.start_freq)->default_value(cfg.start_freq), "Scan start frequency (Hz)")
        ("end-freq", po::value<double>(&cfg.end_freq)->default_value(cfg.end_freq), "Scan end frequency (Hz)")
        ("step-freq", po::value<double>(&cfg.step_freq)->default_value(cfg.step_freq)->notifier([](double r){ if(r <= 0) throw po::validation_error(po::validation_error::invalid_option_value, "step-freq", std::to_string(r)); }), "Scan frequency step (Hz)")
        ("settling", po::value<double>(&cfg.settling_time)->default_value(cfg.settling_time)->notifier([](double r){ if(r < 0) throw po::validation_error(po::validation_error::invalid_option_value, "settling", std::to_string(r)); }), "RX/TX tune settling time (s)")

        // Processing Parameters
        ("alg", po::value<std::string>(&cfg.algorithm)->default_value(cfg.algorithm), "Processing algorithm ('fft' or 'ml')")
        ("fft-size", po::value<size_t>(&cfg.fft_size)->default_value(cfg.fft_size)->notifier([](size_t r){ if(r == 0) throw po::validation_error(po::validation_error::invalid_option_value, "fft-size", std::to_string(r)); }), "FFT size (points)")
        ("avg", po::value<size_t>(&cfg.avg_num)->default_value(cfg.avg_num)->notifier([](size_t r){ if(r == 0) throw po::validation_error(po::validation_error::invalid_option_value, "avg", std::to_string(r)); }), "Number of PSDs to average")
        ("window", po::value<std::string>(&cfg.fft_window_type)->default_value(cfg.fft_window_type), "FFT window ('none', 'hann', 'hamming', 'blackmanharris')")
        ("threshold", po::value<double>(&cfg.peak_threshold_db)->default_value(cfg.peak_threshold_db), "Peak detection threshold (dB)")
        ("prominence", po::value<double>(&cfg.prominence_threshold_db)->default_value(cfg.prominence_threshold_db)->notifier([](double r){ if(r < 0) throw po::validation_error(po::validation_error::invalid_option_value, "prominence", std::to_string(r)); }), "Peak prominence threshold (dB)")
        ("block-size", po::value<size_t>(&cfg.num_samples_block)->default_value(cfg.num_samples_block)->notifier([](size_t r){ if(r == 0) throw po::validation_error(po::validation_error::invalid_option_value, "block-size", std::to_string(r)); }), "RX receive block size (samples)")
        ("wisdom-path", po::value<std::string>(&cfg.fft_wisdom_path)->default_value(cfg.fft_wisdom_path), "Path for FFTW wisdom file (load/save)")

        // Transmission Parameters
        ("enable-tx", po::value<bool>(&cfg.enable_tx)->default_value(cfg.enable_tx)->implicit_value(true), "Enable transmitter")
        ("enable-rx", po::value<bool>(&cfg.enable_rx)->default_value(cfg.enable_rx)->implicit_value(true), "Enable receiver")
        ("tx-wave", po::value<std::string>(&cfg.tx_waveform_type)->default_value(cfg.tx_waveform_type), "TX Waveform ('none', 'tone', 'noise')")
        ("tx-offset", po::value<double>(&cfg.tx_tone_freq_offset)->default_value(cfg.tx_tone_freq_offset), "TX tone frequency offset from TX center (Hz)")
        ("tx-amp", po::value<float>(&cfg.tx_amplitude)->default_value(cfg.tx_amplitude)->notifier([](float r){ if(r <= 0.0f || r > 1.0f) throw po::validation_error(po::validation_error::invalid_option_value, "tx-amp", std::to_string(r)); }), "TX baseband amplitude (0.0 to 1.0)")

        // ML Parameters
        ("ml-model", po::value<std::string>(&cfg.ml_model_path)->default_value(cfg.ml_model_path), "Path to ML model file")

        // Performance & Misc
        ("no-priority", po::bool_switch()->notifier([&](bool v){ if(v) cfg.set_thread_priority = false; else cfg.set_thread_priority = true; })->default_value(!cfg.set_thread_priority), "Disable setting real-time thread priority")
        ("log-level", po::value<std::string>()->default_value("info")->notifier([](const std::string& level_str) {
             try {
                uhd::log::set_log_level(string_to_severity(level_str));
             } catch (const std::invalid_argument& e) {
                 throw po::validation_error(po::validation_error::invalid_option_value, "log-level", level_str);
             }
        }), "Set UHD log level ('trace', 'debug', 'info', 'warning', 'error', 'fatal')")
    ;

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);

        if (vm.count("help")) {
            std::cout << desc << std::endl;
            return EXIT_SUCCESS;
        }

        if (vm.count("config")) {
            std::string config_path = vm["config"].as<std::string>();
            std::cout << "Loading configuration from: " << config_path << std::endl;
            cfg.load_from_yaml(config_path);
            // Re-parse command line AFTER loading config to allow command line to override YAML
            po::store(po::parse_command_line(argc, argv, desc), vm); // Overwrite cfg members again
        }

        po::notify(vm); // This will run notifiers and update cfg members

        if (vm.count("save-config")) {
            std::string save_path = vm["save-config"].as<std::string>();
            std::cout << "Saving current configuration to: " << save_path << std::endl;
            // cfg already reflects command-line args due to notify()
            cfg.save_to_yaml(save_path);
            return EXIT_SUCCESS;
        }
        if (!cfg.enable_rx && !cfg.enable_tx) {
            std::cerr << "Error: Both RX and TX are disabled. Nothing to do." << std::endl;
            return EXIT_FAILURE;
        }
        cfg.validate(); // Validate the final configuration

    } catch (const po::error &e) {
        std::cerr << "Command Line Error: " << e.what() << std::endl;
        std::cerr << desc << std::endl;
        return EXIT_FAILURE;
    } catch (const std::exception& e) {
        std::cerr << "Configuration Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    // --- Configuration Summary ---
    std::cout << "\n--- Configuration Summary ---" << std::endl;
    std::cout << boost::format("USRP Args:        '%s'\n") % cfg.usrp_args;
    std::cout << boost::format("Sample Rate:      %.2f Msps\n") % (cfg.sample_rate / 1e6);
    if(cfg.enable_rx) {
        std::cout << boost::format("RX Freq Range:    %.2f - %.2f MHz (Step: %.2f MHz)\n")
                     % (cfg.start_freq / 1e6) % (cfg.end_freq / 1e6) % (cfg.step_freq / 1e6);
        std::cout << boost::format("RX Gain:          %.1f dB | RX Antenna: %s\n") % cfg.rx_gain % cfg.rx_ant;
    }
    if (cfg.enable_tx) {
        std::cout << boost::format("TX Freq:          %.2f MHz | TX Gain: %.1f dB | TX Antenna: %s | Wave: %s (Amp: %.2f, Offset: %.1f kHz)\n")
                     % (cfg.tx_center_freq / 1e6) % cfg.tx_gain % cfg.tx_ant % cfg.tx_waveform_type % cfg.tx_amplitude % (cfg.tx_tone_freq_offset / 1e3);
    }
    std::cout << boost::format("Algorithm:        %s\n") % cfg.algorithm;
    if (boost::iequals(cfg.algorithm, "fft") && cfg.enable_rx) {
        std::cout << boost::format("  FFT Size: %u | Avg: %u | Window: %s | Threshold: %.1f dB | Prominence: %.1f dB | Block Size: %u\n")
                     % cfg.fft_size % cfg.avg_num % cfg.fft_window_type % cfg.peak_threshold_db % cfg.prominence_threshold_db % cfg.num_samples_block;
    }
    if (boost::iequals(cfg.algorithm, "ml") && !cfg.ml_model_path.empty()) {
        std::cout << boost::format("  ML Model Path:  %s\n") % cfg.ml_model_path;
    }
    std::cout << "Verbose:          " << (cfg.verbose ? "Enabled" : "Disabled") << std::endl;
    std::cout << "Thread Priority:  " << (cfg.set_thread_priority ? "Real-time" : "Normal") << std::endl;
    if (!cfg.fft_wisdom_path.empty() && cfg.enable_rx && boost::iequals(cfg.algorithm, "fft")) {
        std::cout << boost::format("FFTW Wisdom:      %s\n") % cfg.fft_wisdom_path;
    }
    std::cout << "---------------------------\n" << std::endl;


    // --- USRP Initialization ---
    uhd::usrp::multi_usrp::sptr usrp;
    try {
        std::cout << "Creating USRP device with args: " << cfg.usrp_args << "..." << std::endl;
        usrp = uhd::usrp::multi_usrp::make(cfg.usrp_args);

        if (cfg.clock_rate > 0.0) {
            std::cout << "Setting master clock rate: " << cfg.clock_rate / 1e6 << " MHz..." << std::endl;
            usrp->set_master_clock_rate(cfg.clock_rate);
        }
        std::cout << "Setting clock source to internal..." << std::endl;
        usrp->set_clock_source("internal"); // Default, but good to be explicit

        // Subdevice specification
        std::cout << "Setting subdevice spec: " << cfg.subdev << "..." << std::endl;
        usrp->set_rx_subdev_spec(uhd::usrp::subdev_spec_t(cfg.subdev), 0); // Channel 0
        usrp->set_tx_subdev_spec(uhd::usrp::subdev_spec_t(cfg.subdev), 0); // Channel 0


        std::cout << "Setting sample rate: " << cfg.sample_rate / 1e6 << " Msps..." << std::endl;
        if(cfg.enable_rx) usrp->set_rx_rate(cfg.sample_rate, 0);
        if(cfg.enable_tx) usrp->set_tx_rate(cfg.sample_rate, 0);

        // Verify actual rates
        if(cfg.enable_rx) {
            double actual_rx_rate = usrp->get_rx_rate(0);
            std::cout << "Actual RX Rate: " << actual_rx_rate / 1e6 << " Msps" << std::endl;
            if (std::abs(actual_rx_rate - cfg.sample_rate) > 1.0) { // Tolerance of 1 Hz
                std::cerr << "Warning: Actual RX sample rate (" << actual_rx_rate
                          << ") deviates significantly from requested rate (" << cfg.sample_rate << ")!" << std::endl;
            }
        }
        if (cfg.enable_tx) {
            double actual_tx_rate = usrp->get_tx_rate(0);
            std::cout << "Actual TX Rate: " << actual_tx_rate / 1e6 << " Msps" << std::endl;
            if (std::abs(actual_tx_rate - cfg.sample_rate) > 1.0) {
                std::cerr << "Warning: Actual TX sample rate (" << actual_tx_rate
                          << ") deviates significantly from requested rate (" << cfg.sample_rate << ")!" << std::endl;
            }
        }

        // Configure RX Path
        if (cfg.enable_rx) {
            std::cout << "Setting RX Gain: " << cfg.rx_gain << " dB..." << std::endl;
            usrp->set_rx_gain(cfg.rx_gain, 0);
            std::cout << "Actual RX Gain: " << usrp->get_rx_gain(0) << " dB" << std::endl;

            std::cout << "Setting RX Antenna: " << cfg.rx_ant << "..." << std::endl;
            usrp->set_rx_antenna(cfg.rx_ant, 0);

            std::cout << "Setting initial RX Freq: " << cfg.start_freq / 1e6 << " MHz..." << std::endl;
            usrp->set_rx_freq(uhd::tune_request_t(cfg.start_freq), 0);
            std::this_thread::sleep_for(std::chrono::duration<double>(cfg.settling_time)); // Wait for LO to settle
        }

        // Configure TX Path
        if (cfg.enable_tx) {
            std::cout << "Setting TX Gain: " << cfg.tx_gain << " dB..." << std::endl;
            usrp->set_tx_gain(cfg.tx_gain, 0);
            std::cout << "Actual TX Gain: " << usrp->get_tx_gain(0) << " dB" << std::endl;

            std::cout << "Setting TX Antenna: " << cfg.tx_ant << "..." << std::endl;
            usrp->set_tx_antenna(cfg.tx_ant, 0);

            std::cout << "Setting TX Freq: " << cfg.tx_center_freq / 1e6 << " MHz..." << std::endl;
            usrp->set_tx_freq(uhd::tune_request_t(cfg.tx_center_freq), 0);
            std::this_thread::sleep_for(std::chrono::duration<double>(cfg.settling_time)); // Wait for TX LO lock

            try { // Check TX LO lock
                if (!usrp->get_tx_sensor("lo_locked", 0).to_bool()) {
                    std::cerr << "Warning: TX LO failed to lock at " << cfg.tx_center_freq / 1e6 << " MHz!" << std::endl;
                } else if (cfg.verbose) {
                     std::cout << "TX LO locked at " << cfg.tx_center_freq / 1e6 << " MHz." << std::endl;
                }
            } catch (const uhd::key_error&) { /* Sensor not present, common for some devices */
                if (cfg.verbose) std::cout << "  (TX LO lock sensor not found)." << std::endl;
            } catch (const uhd::exception& e) { std::cerr << "Warning: Error checking TX LO lock: " << e.what() << std::endl; }
        }

        perform_calibration(cfg, usrp);

        std::cout << "USRP Initialization Complete." << std::endl;

    } catch (const uhd::exception& e) {
        std::cerr << "UHD Error during initialization: " << e.what() << std::endl;
        return EXIT_FAILURE;
    } catch (const std::exception& e) {
        std::cerr << "Initialization Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    // --- Launch Threads ---
    std::thread rx_t, tx_t;
    bool rx_thread_started = false;
    bool tx_thread_started = false;

    try {
        if (cfg.enable_rx) {
            std::cout << "Launching RX thread..." << std::endl;
            rx_t = std::thread(rx_thread, usrp, std::cref(cfg));
            rx_thread_started = true;
        }
        if (cfg.enable_tx) {
            std::cout << "Launching TX thread..." << std::endl;
            tx_t = std::thread(tx_thread, usrp, std::cref(cfg));
            tx_thread_started = true;
        }
    } catch (const std::system_error& e) {
        std::cerr << "Error launching threads: " << e.what() << " (code: " << e.code() << ")" << std::endl;
        stop_signal_called = true; // Signal any potentially started threads to stop
    } catch (const std::exception& e) {
        std::cerr << "Error during thread launch setup: " << e.what() << std::endl;
        stop_signal_called = true;
    }

    if (!rx_thread_started && !tx_thread_started && (cfg.enable_rx || cfg.enable_tx)) {
         std::cerr << "Failed to start any worker threads. Exiting." << std::endl;
         // Ensure USRP object is released if threads didn't take ownership or run
         // (unique_ptr/sptr handles this, but good to be mindful)
         return EXIT_FAILURE;
    }


    std::cout << "\nScanner running. Press Ctrl+C to stop." << std::endl;

    // Main loop: wait for stop_signal_called (set by UHD_SAFE_MAIN on Ctrl+C)
    while(not stop_signal_called.load()){
        std::this_thread::sleep_for(100ms);
        // Can add periodic status updates here if desired, independent of RX thread reports
    }

    std::cout << "\nCtrl+C detected or stop signal received. Shutting down..." << std::endl;

    // --- Join Threads ---
    if (rx_thread_started && rx_t.joinable()) {
        std::cout << "Waiting for RX thread to join..." << std::endl;
        rx_t.join();
        std::cout << "RX thread joined." << std::endl;
    }
    if (tx_thread_started && tx_t.joinable()) {
        std::cout << "Waiting for TX thread to join..." << std::endl;
        tx_t.join();
        std::cout << "TX thread joined." << std::endl;
    }

    std::cout << "\nAll threads joined." << std::endl;

    // --- Cleanup ---
    if (cfg.enable_rx && boost::iequals(cfg.algorithm, "fft") && !cfg.fft_wisdom_path.empty()) {
        FFTProcessor::save_wisdom(cfg.fft_wisdom_path);
    }

    // USRP object (sptr) will be automatically released.

    std::cout << "Exiting gracefully." << std::endl;
    return EXIT_SUCCESS;
}