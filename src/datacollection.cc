#include <uhd/usrp/multi_usrp.hpp>
#include <uhd/types/tune_request.hpp>
#include <uhd/types/tune_result.hpp>
#include <uhd/stream.hpp>
#include <uhd/utils/thread.hpp>
#include <uhd/utils/safe_main.hpp> // For simplified signal handling in main
//#include <uhd/utils/msg.hpp>      // For UHD logging control

#include <boost/program_options.hpp>
#include <boost/format.hpp>
#include <boost/math/constants/constants.hpp> // For pi
#include <boost/algorithm/string.hpp>       // For algorithm name comparison

#include <yaml-cpp/yaml.h>
#include <fftw3.h>
//#include <uhd/utils/logging.hpp>

#include <atomic>
#include <thread>
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <numeric>
#include <fstream>
#include <mutex>
#include <memory> // For smart pointers
#include <stdexcept>
#include <condition_variable> // Could be useful for more advanced synchronization
#include <deque>

// Namespaces
namespace po = boost::program_options;
using namespace std::chrono_literals; // For time literals like 100ms

// Constants
constexpr double PI = boost::math::constants::pi<double>();
constexpr float TX_AMPLITUDE_DEFAULT = 0.5f; // Default amplitude for TX signal components

// ----------------------------------------
// Configuration Structure
// ----------------------------------------
struct Config {
    // USRP Settings
    std::string usrp_args = "";         // Arguments to find USRP (e.g., "type=b210")
    double sample_rate = 4e6;           // Sample rate (Sps) - Increased default
    double rx_gain = 50.0;              // RX gain (dB) - Adjusted default
    double tx_gain = 50.0;              // TX gain (dB) - Adjusted default
    double tx_center_freq = 1.5e9;      // Center frequency for TX (Hz)
    std::string subdev = "A:A";         // Subdevice spec (A:A for TX/RX A on B210)
    std::string tx_ant = "TX/RX";       // TX Antenna
    std::string rx_ant = "TX/RX";       // RX Antenna
    double clock_rate = 0.0;            // Optional master clock rate

    // Scanning Parameters
    double start_freq = 1e9;          // Start frequency for RX scan (Hz)
    double end_freq = 2e9;            // End frequency for RX scan (Hz)
    double step_freq = 1e6;           // Frequency step for scan (Hz)
    double settling_time = 0.05;      // Time to wait after tuning (s) - Reduced default

    // Processing Parameters
    std::string algorithm = "fft";      // "fft" or "ml" (placeholder)
    size_t fft_size = 2048;             // FFT size (points) - Increased default
    size_t avg_num = 10;                // Number of PSDs to average - Increased default
    std::string fft_window_type = "hann"; // Window type ("none", "hann", "hamming", "blackmanharris")
    double peak_threshold_db = -60.0;   // Threshold for peak detection (dB rel. to full scale approx) - Adjusted default
    double prominence_threshold_db = 5.0; // Minimum prominence for a peak (dB)
    size_t num_samples_block = 16384;   // Number of samples to request in one recv call (larger than FFT)
    
    // Transmission Parameters
    bool enable_tx = false;             // Default to RX only for safety/focus
    bool enable_rx = true;              // Default to RX enabled
    std::string tx_waveform_type = "tone"; // "none", "tone", "noise"
    double tx_tone_freq_offset = 100e3;  // Frequency offset for TX tone from tx_center_freq (Hz)
    float tx_amplitude = TX_AMPLITUDE_DEFAULT; // TX signal amplitude

    // ML Parameters
    std::string ml_model_path = "";      // Path to ML model file

    // Performance & Misc
    std::string fft_wisdom_path = "";   // Path to save/load FFTW wisdom
    bool set_thread_priority = true;   // Attempt to set real-time thread priority
    bool verbose = false;               // Enable verbose output


    // Load from YAML
    void load_from_yaml(const std::string& filename) {
        YAML::Node config;
        try {
            config = YAML::LoadFile(filename);
        } catch (const YAML::Exception& e) {
            throw std::runtime_error("Error loading config file '" + filename + "': " + e.what());
        }

        #define LOAD_YAML_PARAM(param_name, type) \
            if (config[#param_name]) param_name = config[#param_name].as<type>()

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

    // Save to YAML
    void save_to_yaml(const std::string& filename) const {
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

     // Validate config parameters
    void validate() const {
        if (start_freq >= end_freq) throw std::runtime_error("Start frequency must be less than end frequency.");
        if (step_freq <= 0) throw std::runtime_error("Step frequency must be positive.");
        if (sample_rate <= 0) throw std::runtime_error("Sample rate must be positive.");
        if (fft_size <= 0) throw std::runtime_error("FFT size must be positive.");
        if (num_samples_block < fft_size) throw std::runtime_error("num_samples_block must be >= fft_size.");
        if (avg_num == 0) throw std::runtime_error("Averaging number must be at least 1.");
        if (!boost::iequals(algorithm, "fft") && !boost::iequals(algorithm, "ml")) {
            throw std::runtime_error("Unsupported algorithm: " + algorithm + ". Choose 'fft' or 'ml'.");
        }
         if (boost::iequals(algorithm, "ml") && ml_model_path.empty()) {
            // Warning or error, depending on desired behavior
            std::cerr << "Warning: ML algorithm selected but no model path provided." << std::endl;
        }
        // Add more validations as needed (e.g., gain ranges for specific USRPs)
    }
};

// ----------------------------------------
// Global Atomic Stop Signal
// ----------------------------------------
std::atomic<bool> stop_signal_called(false);

// ----------------------------------------
// Shared Data (Minimal for now - just reporting)
// ----------------------------------------
struct DetectedPeak {
    double frequency_hz;
    double power_db;
    double center_freq_hz; // The RX center frequency when detected
};

struct SharedData {
    std::mutex mtx;
    std::deque<DetectedPeak> current_sweep_peaks; // Store peaks from the latest completed sweep
    std::atomic<bool> sweep_complete{false};
    // Could add condition variables if TX needs to react immediately
} shared_data;

// ----------------------------------------
// RAII Wrapper for FFTW Memory
// ----------------------------------------
template <typename T> struct fftw_allocator {
    typedef T value_type;
    T* allocate(size_t n) {
        T* p = static_cast<T*>(fftwf_malloc(sizeof(T) * n));
        if (!p) throw std::bad_alloc();
        return p;
    }
    void deallocate(T* p, size_t) noexcept { fftwf_free(p); }
};

template <typename T>
using fftw_vector = std::vector<T, fftw_allocator<T>>;

struct fftwf_plan_deleter {
    void operator()(fftwf_plan p) const { fftwf_destroy_plan(p); }
};
using fftwf_plan_ptr = std::unique_ptr<std::remove_pointer<fftwf_plan>::type, fftwf_plan_deleter>;

// ----------------------------------------
// Signal Processing Abstraction
// ----------------------------------------
class Processor {
public:
    virtual ~Processor() = default;
    // Processes a block of data and returns detected peaks relative to the current center freq.
    virtual std::vector<std::pair<double, double>> process_block(const std::vector<std::complex<float>>& data) = 0;
    virtual void reset() = 0; // Reset internal state (like averaging buffers)
};

// --- FFT Processor Implementation ---
class FFTProcessor : public Processor {
private:
    const Config& cfg;
    fftw_vector<std::complex<float>> fft_in;
    fftw_vector<std::complex<float>> fft_out;
    fftw_vector<float> window;
    fftwf_plan_ptr fft_plan;
    std::vector<std::vector<double>> psd_buffers; // Buffers to store PSDs for averaging
    size_t current_avg_count = 0;

    void generate_window() {
        window.resize(cfg.fft_size);
        if (boost::iequals(cfg.fft_window_type, "hann")) {
            for (size_t i = 0; i < cfg.fft_size; ++i)
                window[i] = 0.5f * (1.0f - std::cos(2.0f * PI * i / (cfg.fft_size - 1)));
        } else if (boost::iequals(cfg.fft_window_type, "hamming")) {
            for (size_t i = 0; i < cfg.fft_size; ++i)
                window[i] = 0.54f - 0.46f * std::cos(2.0f * PI * i / (cfg.fft_size - 1));
        } else if (boost::iequals(cfg.fft_window_type, "blackmanharris")) {
             const float a0 = 0.35875f, a1 = 0.48829f, a2 = 0.14128f, a3 = 0.01168f;
            for (size_t i = 0; i < cfg.fft_size; ++i)
                window[i] = a0 - a1 * std::cos(2*PI*i/(cfg.fft_size-1)) + a2 * std::cos(4*PI*i/(cfg.fft_size-1)) - a3 * std::cos(6*PI*i/(cfg.fft_size-1));
        }
        // else "none" -> default to all 1s (or could optimize to skip multiplication)
        else { 
            std::fill(window.begin(), window.end(), 1.0f);
        }
        // Normalize window power (optional, depends on desired PSD units)
        double window_power = 0.0;
        for(float w : window) window_power += w*w;
        float norm_factor = std::sqrt(static_cast<float>(cfg.fft_size) / window_power);
         for (size_t i = 0; i < cfg.fft_size; ++i) window[i] *= norm_factor;

    }

    std::vector<std::pair<double, double>> find_peaks(const std::vector<double>& psd) {
        std::vector<std::pair<double, double>> peaks;
        std::vector<size_t> peak_indices;

        // Basic peak finding (local maxima above threshold)
        for (size_t i = 1; i < psd.size() - 1; ++i) {
             // Use shifted indices for finding peaks (assuming fftshift)
             // psd contains N points, peak freq calc assumes -Fs/2 to +Fs/2 centered at N/2
            double current_val = psd[i]; 
            if (current_val > cfg.peak_threshold_db && current_val > psd[i-1] && current_val > psd[i+1]) {
                 peak_indices.push_back(i);
             }
        }

         // Prominence calculation
        for (size_t idx : peak_indices) {
            double peak_val = psd[idx];
            double left_min = peak_val;
            for (long i = (long)idx - 1; i >= 0; --i) {
                left_min = std::min(left_min, psd[i]);
                if (psd[i] >= peak_val) break; // Found higher peak to the left
            }

            double right_min = peak_val;
            for (size_t i = idx + 1; i < psd.size(); ++i) {
                right_min = std::min(right_min, psd[i]);
                if (psd[i] >= peak_val) break; // Found higher peak to the right
            }

            double prominence = peak_val - std::max(left_min, right_min);

            if (prominence >= cfg.prominence_threshold_db) {
                 // Calculate frequency offset relative to center freq
                 // Bin 'i' corresponds to frequency: (i - N/2) * Fs / N
                 double freq_offset = (static_cast<double>(idx) - static_cast<double>(cfg.fft_size) / 2.0) 
                                    * cfg.sample_rate / static_cast<double>(cfg.fft_size);
                peaks.emplace_back(freq_offset, peak_val);
            }
        }

        return peaks;
    }


public:
    FFTProcessor(const Config& config) : cfg(config) {
        if (cfg.fft_size == 0) throw std::runtime_error("FFT size cannot be zero.");
        fft_in.resize(cfg.fft_size);
        fft_out.resize(cfg.fft_size);
        generate_window();
        psd_buffers.reserve(cfg.avg_num);

        unsigned flags = FFTW_ESTIMATE; // Default: faster planning
         // Check for wisdom file
         if (!cfg.fft_wisdom_path.empty()) {
             if (fftwf_import_wisdom_from_filename(cfg.fft_wisdom_path.c_str())) {
                 std::cout << "Successfully loaded FFTW wisdom from " << cfg.fft_wisdom_path << std::endl;
                 flags = FFTW_WISDOM_ONLY; // Use only wisdom if loaded
             } else {
                 std::cerr << "Warning: Failed to load FFTW wisdom from " << cfg.fft_wisdom_path << ". Planning may take longer." << std::endl;
                 flags = FFTW_MEASURE; // Measure if wisdom failed
            }
         } else {
             // Optional: measure if no wisdom path given, or stick with estimate
             // flags = FFTW_MEASURE; // Takes longer on first run
             std::cout << "No FFTW wisdom file specified. Using FFTW_ESTIMATE for planning." << std::endl;
         }


        fft_plan.reset(fftwf_plan_dft_1d(cfg.fft_size,
                                         reinterpret_cast<fftwf_complex*>(fft_in.data()),
                                         reinterpret_cast<fftwf_complex*>(fft_out.data()),
                                         FFTW_FORWARD,
                                         flags)); // Use flags determined above

        if (!fft_plan) {
            throw std::runtime_error("FFTW failed to create plan.");
        }
    }

    ~FFTProcessor() override = default; // Smart pointers handle cleanup

     void reset() override {
        current_avg_count = 0;
        psd_buffers.clear();
    }

    std::vector<std::pair<double, double>> process_block(const std::vector<std::complex<float>>& data) override {
        if (data.size() != cfg.fft_size) {
            throw std::runtime_error("Invalid data size passed to FFTProcessor::process_block");
        }

        // 1. Apply window
        for (size_t i = 0; i < cfg.fft_size; ++i) {
            fft_in[i] = data[i] * window[i];
        }

        // 2. Execute FFT
        fftwf_execute(fft_plan.get());

        // 3. Compute PSD (magnitude squared, normalize by N, convert to dBFS approximation)
        //    Also perform fftshift implicitly during calculation for dB plot.
        std::vector<double> current_psd(cfg.fft_size);
        double norm_factor = 1.0 / static_cast<double>(cfg.fft_size); // Normalization for power
        for (size_t i = 0; i < cfg.fft_size; ++i) {
            // Implicit fftshift: copy second half of FFT output to first half of PSD buffer, first half to second
             size_t shifted_idx = (i + cfg.fft_size / 2) % cfg.fft_size;
             double power = std::norm(fft_out[shifted_idx]) * norm_factor; // std::norm is real^2 + imag^2
            // Convert to dB. Adding 1e-12 (~-120 dB) to avoid log(0) and handle noise floor.
            // Max power of complex float (1.0 + 1.0j) is 2.0. Let's assume max display is ~0 dB.
             current_psd[i] = 10.0 * std::log10(power + 1e-12); 
         }
        
        // 4. Averaging
        psd_buffers.push_back(std::move(current_psd));
        current_avg_count++;

        if (current_avg_count < cfg.avg_num) {
            return {}; // Not enough data yet
        }

        // 5. Calculate Average PSD
        std::vector<double> averaged_psd(cfg.fft_size, 0.0);
        for(const auto& psd_buf : psd_buffers) {
            for(size_t i = 0; i < cfg.fft_size; ++i) {
                averaged_psd[i] += psd_buf[i];
            }
        }
         double avg_factor = 1.0 / static_cast<double>(cfg.avg_num);
        for(size_t i = 0; i < cfg.fft_size; ++i) {
            averaged_psd[i] *= avg_factor;
        }

        // 6. Find peaks in the averaged PSD
        auto peaks = find_peaks(averaged_psd);

        // 7. Reset for next average
        reset(); 

        return peaks;
    }

    // Static method to save wisdom
    static void save_wisdom(const std::string& path) {
        if (!path.empty()) {
            if (fftwf_export_wisdom_to_filename(path.c_str())) {
                std::cout << "Saved FFTW wisdom to " << path << std::endl;
            } else {
                std::cerr << "Error saving FFTW wisdom to " << path << std::endl;
            }
        }
    }
};

// --- ML Processor Placeholder ---
class MLProcessor : public Processor {
private:
    const Config& cfg;
public:
     MLProcessor(const Config& config) : cfg(config) {
         std::cout << "Initializing ML Processor (Placeholder)..." << std::endl;
         if (cfg.ml_model_path.empty()) {
             std::cerr << "Warning: ML Processor created, but ml_model_path is empty." << std::endl;
         } else {
             std::cout << " (Model path: " << cfg.ml_model_path << ")" << std::endl;
             // Placeholder: Add code here to load the model using an inference engine (ONNX Runtime, etc.)
         }
         // Potentially need FFT internally for feature extraction
     }
     ~MLProcessor() override = default;

      void reset() override {
         // Reset any ML-specific state if necessary
     }

     std::vector<std::pair<double, double>> process_block(const std::vector<std::complex<float>>& data) override {
        // --- ML PROCESSING LOGIC WOULD GO HERE ---
        // 1. Preprocess data (e.g., normalization, maybe FFT)
        // 2. Extract features relevant to the trained model
        // 3. Run inference using the loaded ML model
        // 4. Post-process model output to get frequency/power pairs
        // 5. Return results
         if (cfg.verbose) {
             std::cout << "ML Processor: Processing block of size " << data.size() << " (Not Implemented)" << std::endl;
         }
        // Return empty for now
         return {};
         // throw std::runtime_error("ML processing not yet implemented."); 
    }
};

// ----------------------------------------
// Calibration Functions
// ----------------------------------------
void perform_calibration(const Config& cfg, uhd::usrp::multi_usrp::sptr usrp) {
    std::cout << "\nPerforming device calibrations..." << std::endl;
    
    // Loop over channels - assume only channel 0 for now for simplicity
    size_t channel = 0; 
     if (cfg.enable_rx) {
         try {
             std::cout << "  Calibrating RX channel " << channel << "..." << std::endl;
             usrp->set_rx_agc(false, channel); // Disable AGC for calibration stability

             // DC Offset Calibration
             std::cout << "    Performing RX DC offset calibration..." << std::endl;
             usrp->set_rx_dc_offset(true, channel); // Enable automatic DC offset correction
             std::this_thread::sleep_for(1s); // Wait for it to stabilize (heuristic)
             //auto dc_offset = usrp->get_rx_dc_offset(channel);
             // std::cout << "    RX DC Offset calibrated value: " << dc_offset << std::endl;

             // IQ Imbalance Calibration
             std::cout << "    Performing RX IQ imbalance calibration..." << std::endl;
             usrp->set_rx_iq_balance(true, channel); // Enable automatic IQ balance correction
             std::this_thread::sleep_for(1s); // Wait for it to stabilize (heuristic)
             //auto iq_balance = usrp->get_rx_iq_balance(channel);
            // std::cout << "    RX IQ Balance calibrated value: " << iq_balance << std::endl;

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
             
             // TX IQ Imbalance Calibration (if available)
             // Note: TX DC offset often handled differently or less critical
             auto sensor_names = usrp->get_tx_sensor_names(channel);
             bool has_tx_iq_cal = false;

             if (!sensor_names.empty()) { // Check if any sensors exist
                 
                 for (const auto& name : sensor_names) {
                     if (boost::icontains(name, "iq_balance")) { // Look for relevant sensor name
                         has_tx_iq_cal = true;
                         break;
                     }
                 }

                 if (has_tx_iq_cal) {
                     std::cout << "    Performing TX IQ imbalance self-calibration..." << std::endl;
                     usrp->set_tx_iq_balance(true, channel); // Enable cal if possible
                     std::this_thread::sleep_for(1s);
                     //auto tx_iq_cal_value = usrp->get_tx_iq_balance(channel);
                    // std::cout << "    TX IQ Balance calibrated value: " << tx_iq_cal_value << std::endl;
                 } 
                 
                 else {
                     std::cout << "    TX IQ imbalance self-calibration not detected/supported for this channel." << std::endl;
                 }
              } 
              
              else {
                 std::cout << "    Could not retrieve TX sensor names for channel " << channel << "." << std::endl;
              }

              std::cout << "  TX Calibration for channel " << channel << " complete (limited)." << std::endl;

        } 
        
        catch (const uhd::exception& e) {
             std::cerr << "Warning: TX calibration failed for channel " << channel << ": " << e.what() << std::endl;
         }
    } 
    
    else {
         std::cout << "  Skipping TX calibration (TX disabled)." << std::endl;
     }

    std::cout << "Device calibrations finished.\n" << std::endl;
}

// ----------------------------------------
// RX Thread Function
// ----------------------------------------
void rx_thread(uhd::usrp::multi_usrp::sptr usrp, const Config& cfg) {
    if (cfg.set_thread_priority) {
        uhd::set_thread_priority_safe();
    }

    std::cout << "RX Thread: Starting." << std::endl;

    // Create processor based on config
    std::unique_ptr<Processor> processor;
    try {
        if (boost::iequals(cfg.algorithm, "fft")) {
            processor = std::make_unique<FFTProcessor>(cfg);
        } else if (boost::iequals(cfg.algorithm, "ml")) {
             processor = std::make_unique<MLProcessor>(cfg);
        } else {
             // Should have been caught by config validation, but good to double-check
             throw std::runtime_error("RX Thread: Invalid algorithm selected: " + cfg.algorithm);
        }
    } catch (const std::exception& e) {
        std::cerr << "RX Thread: Error initializing processor: " << e.what() << std::endl;
        return;
    }

    // Setup RX stream
    uhd::stream_args_t stream_args("fc32", "sc16"); // CPU=complex<float>, OTA=complex<short>
    stream_args.channels = {0}; // Use channel 0 for RX
    uhd::rx_streamer::sptr rx_stream = usrp->get_rx_stream(stream_args);

    // Setup stream command and receive buffer
    uhd::stream_cmd_t stream_cmd(uhd::stream_cmd_t::STREAM_MODE_START_CONTINUOUS);
    stream_cmd.stream_now = true; // Important: start immediately after tuning each step
    rx_stream->issue_stream_cmd(stream_cmd);

    std::vector<std::complex<float>> rx_buffer(cfg.num_samples_block);
    uhd::rx_metadata_t md;
    size_t samples_collected_for_fft = 0;
    std::vector<std::complex<float>> fft_input_buffer(cfg.fft_size);
    
    std::chrono::steady_clock::time_point last_report_time = std::chrono::steady_clock::now();
    long total_peaks_detected = 0;

    std::cout << "RX Thread: Starting frequency sweep..." << std::endl;

    while (!stop_signal_called) {
         // ---- FREQUENCY SWEEP LOOP ----
        shared_data.current_sweep_peaks.clear(); // Clear peaks from previous sweep

        for (double current_freq = cfg.start_freq; 
             current_freq <= cfg.end_freq && !stop_signal_called; 
             current_freq += cfg.step_freq) 
        {
            if(cfg.verbose) std::cout << "\nRX Tuning to: " << current_freq / 1e6 << " MHz" << std::endl;

            // 1. Tune RX
            uhd::tune_request_t tune_request(current_freq);
            // Optional: Set tuning policy (e.g., AUTO, MANUAL, UNKNOWN_PPS)
             // tune_request.args = uhd::device_addr_t("mode_n=integer"); // Example: integer-N tuning
            usrp->set_rx_freq(tune_request, 0); // Tune channel 0
             
             // Wait for settling AND check lock time
             try {
                 double actual_settle_time = std::max(cfg.settling_time, usrp->get_rx_sensor("lo_locked", 0).to_real());
                  if (cfg.verbose) std::cout << "RX waiting " << actual_settle_time * 1e3 << " ms for LO lock/settling..." << std::endl;
                 std::this_thread::sleep_for(std::chrono::duration<double>(actual_settle_time));
                 
                 if (!usrp->get_rx_sensor("lo_locked", 0).to_bool()){
                    std::cerr << "Warning: RX LO failed to lock at " << current_freq / 1e6 << " MHz." << std::endl;
                    // Decide whether to continue or skip this freq step
                 }

             } catch (const uhd::key_error&){
                // Sensor "lo_locked" doesn't exist, fall back to configured settling time
                 std::this_thread::sleep_for(std::chrono::duration<double>(cfg.settling_time));
             } catch (const uhd::exception &e) {
                std::cerr << "Warning: Error checking RX LO lock status: " << e.what() << std::endl;
                 std::this_thread::sleep_for(std::chrono::duration<double>(cfg.settling_time));
            }

            // 2. Reset processor state for this frequency step (e.g., averaging)
             processor->reset();
             samples_collected_for_fft = 0; // Reset collection count for this frequency

             // 3. Collect and Process Samples until enough averages are done
             bool averaging_complete = false;
             while (!averaging_complete && !stop_signal_called) {
                // Receive a block of samples
                 size_t num_rx_samps = rx_stream->recv(rx_buffer.data(), rx_buffer.size(), md, 0.1); // 100ms timeout

                 // Check for errors
                 if (md.error_code != uhd::rx_metadata_t::ERROR_CODE_NONE) {
                     std::cerr << "RX Error at " << current_freq / 1e6 << " MHz: " << md.strerror() << std::endl;
                     // Handle error (e.g., break inner loop, attempt recovery?)
                     break; // Break processing loop for this frequency step
                 }
                 if (num_rx_samps == 0) {
                    // Timeout occurred, might happen if sample rate is very low or system is stalled
                    std::cerr << "Warning: RX receive timeout at " << current_freq / 1e6 << " MHz." << std::endl;
                    continue; 
                 }

                // Process received samples in fft_size chunks
                size_t current_pos_in_block = 0;
                 while (current_pos_in_block < num_rx_samps && !averaging_complete) {
                    size_t remaining_in_block = num_rx_samps - current_pos_in_block;
                    size_t needed_for_fft = cfg.fft_size - samples_collected_for_fft;
                    size_t samples_to_copy = std::min(remaining_in_block, needed_for_fft);

                    // Copy data into the FFT input buffer
                     std::copy(rx_buffer.begin() + current_pos_in_block, 
                              rx_buffer.begin() + current_pos_in_block + samples_to_copy,
                              fft_input_buffer.begin() + samples_collected_for_fft);
                    
                    samples_collected_for_fft += samples_to_copy;
                    current_pos_in_block += samples_to_copy;

                     // If a full FFT buffer is ready, process it
                    if (samples_collected_for_fft == cfg.fft_size) {
                        auto peaks = processor->process_block(fft_input_buffer);
                        samples_collected_for_fft = 0; // Reset for next FFT buffer

                         // If process_block returned peaks, it means averaging is done for this frequency
                         if (!peaks.empty()) {
                             averaging_complete = true; 
                             
                            if (cfg.verbose) std::cout << "  Peaks found at " << current_freq / 1e6 << " MHz: " << peaks.size() << std::endl;

                             // Lock shared data and store the results
                            { // Lock scope
                                 std::lock_guard<std::mutex> lock(shared_data.mtx);
                                for (const auto& [offset, power] : peaks) {
                                     double absolute_freq = current_freq + offset;
                                    // Optional: Filter peaks to ensure they fall within expected range if needed
                                     if (absolute_freq >= cfg.start_freq && absolute_freq <= cfg.end_freq) { // Basic sanity check
                                         shared_data.current_sweep_peaks.push_back({absolute_freq, power, current_freq});
                                         total_peaks_detected++;
                                         if (cfg.verbose) printf("    -> Peak: %.3f MHz (Power: %.2f dB)\n", absolute_freq / 1e6, power);
                                    }
                                 }
                            } // Unlock scope
                             break; // Exit inner processing loop (current_pos_in_block)
                         }
                     }
                 } // End while (processing received block)
             } // End while (!averaging_complete)

             if (stop_signal_called) break; // Exit frequency loop if stopped during processing

         } // ---- END FREQUENCY SWEEP LOOP ----

        // Indicate sweep completion (could be used by other threads/UI)
        shared_data.sweep_complete = true; 
        
         // Reporting (example: report peaks once per sweep)
         auto now = std::chrono::steady_clock::now();
         auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_report_time);
        if (elapsed >= 1000ms || stop_signal_called) { // Report every second or on exit
            std::lock_guard<std::mutex> lock(shared_data.mtx);
            std::cout << "\n--- Sweep Report (" << elapsed.count() << "ms) ---" << std::endl;
            if (shared_data.current_sweep_peaks.empty()) {
                 std::cout << "  No significant peaks detected in this sweep." << std::endl;
             } else {
                 std::cout << "  Detected Peaks (" << shared_data.current_sweep_peaks.size() << "):" << std::endl;
                 // Sort peaks by frequency for better readability
                 std::sort(shared_data.current_sweep_peaks.begin(), shared_data.current_sweep_peaks.end(), 
                          [](const DetectedPeak& a, const DetectedPeak& b){ return a.frequency_hz < b.frequency_hz; });

                for (const auto& peak : shared_data.current_sweep_peaks) {
                     printf("    - Freq: %9.3f MHz | Power: %6.2f dB | (Center: %.1f MHz)\n", 
                            peak.frequency_hz / 1e6, peak.power_db, peak.center_freq_hz / 1e6);
                 }
             }
             std::cout << "--- End Report ---\n" << std::endl;
             last_report_time = now;
         }


        // Slow down the loop slightly if sweep is very fast
        if (!stop_signal_called) {
             std::this_thread::sleep_for(10ms); // Prevents busy-looping if range is tiny
        }

     } // End while (!stop_signal_called)

    // Cleanup
    stream_cmd.stream_mode = uhd::stream_cmd_t::STREAM_MODE_STOP_CONTINUOUS;
    rx_stream->issue_stream_cmd(stream_cmd);

    std::cout << "RX Thread: Stopped. Total peaks detected across all sweeps: " << total_peaks_detected << std::endl;
}


// ----------------------------------------
// TX Thread Function
// ----------------------------------------
void tx_thread(uhd::usrp::multi_usrp::sptr usrp, const Config& cfg) {
    if (cfg.set_thread_priority) {
        uhd::set_thread_priority_safe();
    }

    std::cout << "TX Thread: Starting." << std::endl;
    
    if (boost::iequals(cfg.tx_waveform_type, "none")) {
        std::cout << "TX Thread: Waveform type is 'none'. Exiting." << std::endl;
        return;
    }

    // Setup TX stream
    uhd::stream_args_t stream_args("fc32", "sc16");
    stream_args.channels = {0}; // Use channel 0 for TX
    uhd::tx_streamer::sptr tx_stream = usrp->get_tx_stream(stream_args);

    size_t spb = tx_stream->get_max_num_samps() * 4; // Request slightly larger buffer size
     std::vector<std::complex<float>> tx_buffer(spb);

     // --- Generate Waveform ---
     if (boost::iequals(cfg.tx_waveform_type, "tone")) {
        std::cout << "TX Thread: Generating single tone at offset " << cfg.tx_tone_freq_offset / 1e3 << " kHz." << std::endl;
         double phase = 0.0;
         double delta_phase = 2.0 * PI * cfg.tx_tone_freq_offset / cfg.sample_rate;
         for (size_t i = 0; i < spb; ++i) {
             tx_buffer[i] = std::polar(cfg.tx_amplitude, static_cast<float>(phase));
             phase += delta_phase;
             if (phase > PI) phase -= 2.0 * PI; // Keep phase wrapped
         }
     } else if (boost::iequals(cfg.tx_waveform_type, "noise")) {
         std::cout << "TX Thread: Generating complex white noise." << std::endl;
        // Basic noise generation - consider better PRNG and filtering if specific noise profile needed
        srand(time(NULL)); // Seed random number generator (simple approach)
        for(size_t i=0; i<spb; ++i){
             float real = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f * cfg.tx_amplitude;
             float imag = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f * cfg.tx_amplitude;
             tx_buffer[i] = {real, imag};
         }
     } 
     // Add "file" option later if needed: Load IQ data from a file
     else {
        std::cerr << "TX Thread: Unknown or unsupported tx_waveform_type: " << cfg.tx_waveform_type << ". Exiting." << std::endl;
         return;
     }

    uhd::tx_metadata_t md;
    md.start_of_burst = true;  // Set for the first packet
    md.end_of_burst = false;    // Streaming continuously
    md.has_time_spec = false; // Send immediately

    std::cout << "TX Thread: Starting continuous transmission at " << cfg.tx_center_freq / 1e6 << " MHz..." << std::endl;

    size_t total_samps_sent = 0;

    while (!stop_signal_called) {
        // In a real adaptive system, you might regenerate the buffer here based on shared_data
        
        size_t num_sent = tx_stream->send(tx_buffer.data(), tx_buffer.size(), md);
        if (num_sent < tx_buffer.size()) {
            std::cerr << "TX Warning: Underrun possible or send incomplete. Sent " << num_sent << "/" << tx_buffer.size() << std::endl;
            // Consider adding delay or reducing load if underruns ('U' printed by UHD) are persistent
        }
        total_samps_sent += num_sent;

        md.start_of_burst = false; // Only true for the first packet

        // Optional: Add a small sleep to prevent hyperactive busy-loop if needed,
        // but often letting send() block is sufficient for flow control.
        // std::this_thread::sleep_for(1ms); 
    }

    // Cleanup: Send a zero buffer to flush the stream (optional but good practice)
    md.end_of_burst = true;
    std::fill(tx_buffer.begin(), tx_buffer.end(), std::complex<float>(0.0f, 0.0f));
    tx_stream->send(tx_buffer.data(), tx_buffer.size(), md);
    std::this_thread::sleep_for(100ms); // Give time for zeros to transmit


    std::cout << "TX Thread: Stopped. Total samples sent: ~" << total_samps_sent << std::endl;
}


// ----------------------------------------
// Main Function (using safe_main)
// ----------------------------------------


namespace po = boost::program_options;

// Helper function: Convert string log level to uhd::log::severity_level enum.
// Note: Adjust the enum member names to match your UHD version.
uhd::log::severity_level string_to_severity(const std::string& level) {
    if (level == "debug")
        return uhd::log::severity_level::debug;
    else if (level == "info")
        return uhd::log::severity_level::info;
    else if (level == "warning")
        return uhd::log::severity_level::warning;
    else if (level == "error")
        return uhd::log::severity_level::error;
    else
        throw std::invalid_argument("Invalid log level: " + level);
}


int UHD_SAFE_MAIN(int argc, char* argv[]) {
    // === Configuration ===
    Config cfg;

    // Setup command line options
    po::options_description desc("Advanced USRP Spectrum Scanner Options");
    desc.add_options()
        ("help,h", "Show help message")
        ("config,c", po::value<std::string>(), "Load configuration from YAML file")
        ("save-config", po::value<std::string>(), "Save current configuration to YAML file and exit")
        ("verbose,v", po::bool_switch(&cfg.verbose)->default_value(false), "Enable verbose output")

        // USRP Params
        ("usrp-args", po::value<std::string>(&cfg.usrp_args)->default_value(cfg.usrp_args), "UHD device arguments (e.g., 'type=b210')")
        ("rate", po::value<double>(&cfg.sample_rate)->notifier([](double r){ if(r <= 0) throw po::validation_error(po::validation_error::invalid_option_value, "rate"); }), "Sample rate (Sps)")
        ("rx-gain", po::value<double>(&cfg.rx_gain), "RX gain (dB)")
        ("tx-gain", po::value<double>(&cfg.tx_gain), "TX gain (dB)")
        ("tx-freq", po::value<double>(&cfg.tx_center_freq), "TX center frequency (Hz)")
        ("rx-ant", po::value<std::string>(&cfg.rx_ant), "RX Antenna")
        ("tx-ant", po::value<std::string>(&cfg.tx_ant), "TX Antenna")
        ("subdev", po::value<std::string>(&cfg.subdev), "USRP Subdevice Spec")
        ("clock-rate", po::value<double>(&cfg.clock_rate), "Optional Master clock rate (Hz)")

        // Scan Params
        ("start-freq", po::value<double>(&cfg.start_freq), "Scan start frequency (Hz)")
        ("end-freq", po::value<double>(&cfg.end_freq), "Scan end frequency (Hz)")
        ("step-freq", po::value<double>(&cfg.step_freq)->notifier([](double r){ if(r <= 0) throw po::validation_error(po::validation_error::invalid_option_value, "step-freq"); }), "Scan frequency step (Hz)")
        ("settling", po::value<double>(&cfg.settling_time)->notifier([](double r){ if(r < 0) throw po::validation_error(po::validation_error::invalid_option_value, "settling"); }), "RX tune settling time (s)")

        // Processing Params
        ("alg", po::value<std::string>(&cfg.algorithm), "Processing algorithm ('fft' or 'ml')")
        ("fft-size", po::value<size_t>(&cfg.fft_size), "FFT size (points)")
        ("avg", po::value<size_t>(&cfg.avg_num)->default_value(cfg.avg_num), "Number of PSDs to average")
        ("window", po::value<std::string>(&cfg.fft_window_type), "FFT window ('none', 'hann', 'hamming', 'blackmanharris')")
        ("threshold", po::value<double>(&cfg.peak_threshold_db), "Peak detection threshold (dB)")
        ("prominence", po::value<double>(&cfg.prominence_threshold_db)->notifier([](double r){ if(r < 0) throw po::validation_error(po::validation_error::invalid_option_value, "prominence"); }), "Peak prominence threshold (dB)")
        ("block-size", po::value<size_t>(&cfg.num_samples_block)->notifier([](size_t r){ if(r == 0) throw po::validation_error(po::validation_error::invalid_option_value, "block-size"); }), "RX receive block size (samples)")
        ("wisdom-path", po::value<std::string>(&cfg.fft_wisdom_path), "Path for FFTW wisdom file")

         // TX Params
        ("enable-tx", po::value<bool>(&cfg.enable_tx)->default_value(cfg.enable_tx)->implicit_value(true), "Enable transmitter")
        ("enable-rx", po::value<bool>(&cfg.enable_rx)->default_value(cfg.enable_rx)->implicit_value(true), "Enable receiver")
        ("tx-wave", po::value<std::string>(&cfg.tx_waveform_type), "TX Waveform ('none', 'tone', 'noise')")
        ("tx-offset", po::value<double>(&cfg.tx_tone_freq_offset), "TX tone frequency offset (Hz)")
        ("tx-amp", po::value<float>(&cfg.tx_amplitude)->notifier([](float r){ if(r <= 0) throw po::validation_error(po::validation_error::invalid_option_value, "tx-amp"); }), "TX baseband amplitude")

         // ML Params
        ("ml-model", po::value<std::string>(&cfg.ml_model_path), "Path to ML model file")

         // Misc
        ("no-priority", po::bool_switch()->notifier([&](bool v){ if(v) cfg.set_thread_priority = false; }), "Disable setting real-time thread priority")
        ("log-level", po::value<std::string>()->notifier([](const std::string& level) {
            // Convert the string to the proper severity level enum using our helper function.
            uhd::log::set_log_level(string_to_severity(level));
        }), "Set UHD log level (e.g., 'info', 'warning', 'error')")
    ;

    // Parse command line
    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);

        if (vm.count("help")) {
            std::cout << desc << std::endl;
            return EXIT_SUCCESS;
        }

        // Handle config file loading first
        if (vm.count("config")) {
            std::string config_path = vm["config"].as<std::string>();
            std::cout << "Loading configuration from: " << config_path << std::endl;
            cfg.load_from_yaml(config_path);
            // Re-parse command line to allow overrides AFTER loading config file
            po::store(po::parse_command_line(argc, argv, desc), vm);
        }

        po::notify(vm); // Finalize options storage and run notifiers

        // Handle saving config (after potential overrides)
        if (vm.count("save-config")) {
            std::string save_path = vm["save-config"].as<std::string>();
            std::cout << "Saving current configuration to: " << save_path << std::endl;
            cfg.save_to_yaml(save_path);
            return EXIT_SUCCESS;
        }

        // Validate the final configuration
        cfg.validate();

    } catch (const po::error &e) {
        std::cerr << "Command Line Error: " << e.what() << std::endl;
        std::cerr << desc << std::endl;
        return EXIT_FAILURE;
    } catch (const std::exception& e) {
        std::cerr << "Configuration Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    // Print summary of key config params
    std::cout << "--- Configuration Summary ---" << std::endl;
    std::cout << boost::format("USRP Args:        '%s'\n") % cfg.usrp_args;
    std::cout << boost::format("Sample Rate:      %.2f Msps\n") % (cfg.sample_rate / 1e6);
    std::cout << boost::format("RX Freq Range:    %.2f - %.2f MHz (Step: %.2f MHz)\n")
                 % (cfg.start_freq / 1e6) % (cfg.end_freq / 1e6) % (cfg.step_freq / 1e6);
    std::cout << boost::format("RX Gain:          %.1f dB | RX Antenna: %s\n") % cfg.rx_gain % cfg.rx_ant;
    if (cfg.enable_tx)
        std::cout << boost::format("TX Freq:          %.2f MHz | TX Gain: %.1f dB | TX Antenna: %s | Wave: %s\n")
                     % (cfg.tx_center_freq / 1e6) % cfg.tx_gain % cfg.tx_ant % cfg.tx_waveform_type;
    std::cout << boost::format("Algorithm:        %s\n") % cfg.algorithm;
    if (boost::iequals(cfg.algorithm, "fft"))
        std::cout << boost::format("  FFT Size: %d | Avg: %d | Window: %s | Threshold: %.1f dB | Prominence: %.1f dB\n")
                     % cfg.fft_size % cfg.avg_num % cfg.fft_window_type % cfg.peak_threshold_db % cfg.prominence_threshold_db;
    std::cout << "---------------------------\n" << std::endl;

    // === USRP Initialization ===
    uhd::usrp::multi_usrp::sptr usrp;
    try {
        std::cout << "Creating USRP device with args: " << cfg.usrp_args << "..." << std::endl;
        usrp = uhd::usrp::multi_usrp::make(cfg.usrp_args);

        // Lock mboard clocks - essential for phase alignment IF daughterboards share clocks (like B210)
        if (cfg.clock_rate > 0.0)
            usrp->set_master_clock_rate(cfg.clock_rate);
        usrp->set_clock_source("internal"); // Use internal oscillator

        // Set common properties
        usrp->set_rx_subdev_spec(cfg.subdev); // Usually "A:A" or "A:B" for B210
        usrp->set_tx_subdev_spec(cfg.subdev);
        
        // Set Sample Rate
        std::cout << "Setting sample rate: " << cfg.sample_rate / 1e6 << " Msps..." << std::endl;
        usrp->set_rx_rate(cfg.sample_rate, 0); // Set for channel 0
        usrp->set_tx_rate(cfg.sample_rate, 0);

        // Check actual rates
        double actual_rx_rate = usrp->get_rx_rate(0);
        double actual_tx_rate = usrp->get_tx_rate(0);
        std::cout << "Actual RX Rate: " << actual_rx_rate / 1e6 << " Msps" << std::endl;
        if (cfg.enable_tx)
            std::cout << "Actual TX Rate: " << actual_tx_rate / 1e6 << " Msps" << std::endl;
        if (std::abs(actual_rx_rate - cfg.sample_rate) > 1.0 ||
            (cfg.enable_tx && std::abs(actual_tx_rate - cfg.sample_rate) > 1.0)) {
            std::cerr << "Warning: Actual sample rate deviates significantly from requested rate!" << std::endl;
            // Optionally update cfg.sample_rate if desired.
        }
        
        // Set Gains
        if (cfg.enable_rx) {
            std::cout << "Setting RX Gain: " << cfg.rx_gain << " dB..." << std::endl;
            usrp->set_rx_gain(cfg.rx_gain, 0);
            std::cout << "Actual RX Gain: " << usrp->get_rx_gain(0) << " dB" << std::endl;
            std::cout << "Setting RX Antenna: " << cfg.rx_ant << "..." << std::endl;
            usrp->set_rx_antenna(cfg.rx_ant, 0);
        }
        if (cfg.enable_tx) {
            std::cout << "Setting TX Gain: " << cfg.tx_gain << " dB..." << std::endl;
            usrp->set_tx_gain(cfg.tx_gain, 0);
            std::cout << "Actual TX Gain: " << usrp->get_tx_gain(0) << " dB" << std::endl;
            std::cout << "Setting TX Antenna: " << cfg.tx_ant << "..." << std::endl;
            usrp->set_tx_antenna(cfg.tx_ant, 0);
            
            std::cout << "Setting TX Freq: " << cfg.tx_center_freq / 1e6 << " MHz..." << std::endl;
            usrp->set_tx_freq(uhd::tune_request_t(cfg.tx_center_freq), 0);
            std::this_thread::sleep_for(std::chrono::duration<double>(cfg.settling_time)); // Wait for TX LO lock.
            try {
                if (!usrp->get_tx_sensor("lo_locked", 0).to_bool()) {
                    std::cerr << "Warning: TX LO failed to lock!" << std::endl;
                }
            } catch (...) { /* Ignore errors if sensor not present */ }
        }
        if (cfg.enable_rx) {
            std::cout << "Setting initial RX Freq: " << cfg.start_freq / 1e6 << " MHz..." << std::endl;
            usrp->set_rx_freq(uhd::tune_request_t(cfg.start_freq), 0);
            std::this_thread::sleep_for(std::chrono::duration<double>(cfg.settling_time));
        }

        // Perform Calibration
        perform_calibration(cfg, usrp);

        std::cout << "USRP Initialization Complete." << std::endl;

    } catch (const uhd::exception& e) {
        std::cerr << "UHD Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    } catch (const std::exception& e) {
        std::cerr << "Initialization Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    // === Setup Signal Handler and Start Threads ===
    std::thread rx_t, tx_t;
    try {
        if (cfg.enable_rx) {
            std::cout << "Launching RX thread..." << std::endl;
            rx_t = std::thread(rx_thread, usrp, std::cref(cfg));
        }
        if (cfg.enable_tx) {
            std::cout << "Launching TX thread..." << std::endl;
            tx_t = std::thread(tx_thread, usrp, std::cref(cfg));
        }
    } catch (const std::system_error& e) {
        std::cerr << "Error launching threads: " << e.what() << " (" << e.code() << ")" << std::endl;
        stop_signal_called = true; // Signal any running threads to stop.
    } catch (const std::exception& e) {
        std::cerr << "Error during thread launch setup: " << e.what() << std::endl;
        stop_signal_called = true;
    }

    std::cout << "\nScanner running. Press Ctrl+C to stop." << std::endl;
    if (rx_t.joinable()) rx_t.join();
    if (tx_t.joinable()) tx_t.join();

    std::cout << "\nThreads joined. Exiting." << std::endl;

    // === Cleanup ===
    if (boost::iequals(cfg.algorithm, "fft")) {
        FFTProcessor::save_wisdom(cfg.fft_wisdom_path);
    }
    // USRP object automatically released via shared_ptr.

    return EXIT_SUCCESS;
}

