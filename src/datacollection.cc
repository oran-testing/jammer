#include <uhd/usrp/multi_usrp.hpp>
#include <uhd/types/tune_request.hpp>
#include <uhd/stream.hpp>
#include <uhd/utils/thread_priority.hpp>
#include <boost/program_options.hpp>
#include <yaml-cpp/yaml.h>
#include <fftw3.h>
#include <atomic>
#include <thread>
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <numeric>
#include <csignal>
#include <fstream>
#include <mutex>

using namespace std;
namespace po = boost::program_options;

struct Config {
    // Basic Parameters
    double start_freq = 1e9, end_freq = 2e9, step_freq = 1e6;
    double sample_rate = 1e6, tx_gain = 30.0, rx_gain = 30.0;
    string algorithm = "fft";
    bool enable_tx = true, enable_rx = true;
    
    // Advanced Processing
    size_t fft_size = 1024;
    size_t avg_num = 4;
    double threshold = -30.0;
    double settling_time = 0.1;
    
    // ML Parameters
    string model_path;

    void load_from_yaml(const string& filename) {
        YAML::Node config = YAML::LoadFile(filename);
        start_freq = config["start_freq"].as<double>();
        end_freq = config["end_freq"].as<double>();
        step_freq = config["step_freq"].as<double>();
        sample_rate = config["sample_rate"].as<double>();
        tx_gain = config["tx_gain"].as<double>();
        rx_gain = config["rx_gain"].as<double>();
        algorithm = config["algorithm"].as<string>();
        enable_tx = config["enable_tx"].as<bool>();
        enable_rx = config["enable_rx"].as<bool>();
        fft_size = config["fft_size"].as<size_t>();
        avg_num = config["avg_num"].as<size_t>();
        threshold = config["threshold"].as<double>();
        settling_time = config["settling_time"].as<double>();
    }

    void save_to_yaml(const string& filename) const {
        YAML::Emitter emitter;
        emitter << YAML::BeginMap;
        emitter << YAML::Key << "start_freq" << YAML::Value << start_freq;
        emitter << YAML::Key << "end_freq" << YAML::Value << end_freq;
        emitter << YAML::Key << "step_freq" << YAML::Value << step_freq;
        emitter << YAML::Key << "sample_rate" << YAML::Value << sample_rate;
        emitter << YAML::Key << "tx_gain" << YAML::Value << tx_gain;
        emitter << YAML::Key << "rx_gain" << YAML::Value << rx_gain;
        emitter << YAML::Key << "algorithm" << YAML::Value << algorithm;
        emitter << YAML::Key << "enable_tx" << YAML::Value << enable_tx;
        emitter << YAML::Key << "enable_rx" << YAML::Value << enable_rx;
        emitter << YAML::Key << "fft_size" << YAML::Value << fft_size;
        emitter << YAML::Key << "avg_num" << YAML::Value << avg_num;
        emitter << YAML::Key << "threshold" << YAML::Value << threshold;
        emitter << YAML::Key << "settling_time" << YAML::Value << settling_time;
        emitter << YAML::EndMap;

        ofstream fout(filename);
        fout << emitter.c_str();
    }
};

struct SharedData {
    mutex mtx;
    vector<pair<double, double>> detected_peaks;
    atomic<bool> new_data{false};
} shared_data;

atomic<bool> stop_signal(false);
void sigint_handler(int) { stop_signal = true; }

// ------------------------
// Calibration Routines
// ------------------------
void perform_dc_offset_calibration(shared_ptr<uhd::usrp::multi_usrp> usrp) {
    cout << "Performing RX DC offset calibration..." << endl;
    usrp->set_rx_dc_offset(true);
}

void perform_iq_balance_calibration(shared_ptr<uhd::usrp::multi_usrp> usrp) {
    cout << "Performing RX IQ balance calibration..." << endl;
    usrp->set_rx_iq_balance(true);
}

// ------------------------
// Signal Processing Methods
// ------------------------
class Processor {
public:
    virtual ~Processor() = default;
    virtual vector<pair<double, double>> process(const vector<complex<float>>&) = 0;
};

class FFTProcessor : public Processor {
    size_t fft_size;
    size_t avg_num;
    double threshold;
    vector<vector<complex<float>>> avg_buffers;
    fftwf_plan fft_plan;
    fftwf_complex *fft_in, *fft_out;

public:
    FFTProcessor(size_t size, size_t avg, double thresh)
        : fft_size(size), avg_num(avg), threshold(thresh) {
        fft_in = fftwf_alloc_complex(fft_size);
        fft_out = fftwf_alloc_complex(fft_size);
        fft_plan = fftwf_plan_dft_1d(fft_size, fft_in, fft_out, FFTW_FORWARD, FFTW_MEASURE);
    }

    ~FFTProcessor() {
        fftwf_destroy_plan(fft_plan);
        fftwf_free(fft_in);
        fftwf_free(fft_out);
    }

    vector<pair<double, double>> process(const vector<complex<float>>& data) override {
        if (data.size() != fft_size) throw runtime_error("Invalid FFT size");
        
        avg_buffers.push_back(data);
        if (avg_buffers.size() < avg_num) return {};

        vector<double> psd(fft_size, 0.0);
        for (const auto& buf : avg_buffers) {
            vector<double> tmp_psd;
            compute_fft(buf, tmp_psd);
            transform(psd.begin(), psd.end(), tmp_psd.begin(), psd.begin(), plus<double>());
        }
        avg_buffers.clear();

        transform(psd.begin(), psd.end(), psd.begin(),
            [this](double val) { return val / avg_num; });

        return find_peaks(psd);
    }

private:
    void compute_fft(const vector<complex<float>>& data, vector<double>& psd) {
        for (size_t i = 0; i < fft_size; ++i) {
            fft_in[i][0] = data[i].real();
            fft_in[i][1] = data[i].imag();
        }
        
        fftwf_execute(fft_plan);
        psd.resize(fft_size);

        for (size_t i = 0; i < fft_size/2; ++i) {
            double real = fft_out[i][0];
            double imag = fft_out[i][1];
            psd[i] = 10 * log10((real*real + imag*imag) / (fft_size * fft_size) + 1e-20);
        }
    }

    vector<pair<double, double>> find_peaks(const vector<double>& psd) {
        vector<pair<double, double>> peaks;
        for (size_t i = 1; i < psd.size() - 1; ++i) {
            if (psd[i] > threshold && psd[i] > psd[i-1] && psd[i] > psd[i+1]) {
                double prominence = calculate_prominence(psd, i);
                if (prominence > 3.0) {
                    peaks.emplace_back(i, psd[i]);
                }
            }
        }
        return peaks;
    }

    double calculate_prominence(const vector<double>& psd, size_t idx) {
        double left = *max_element(psd.begin(), psd.begin() + idx);
        double right = *max_element(psd.begin() + idx, psd.end());
        return psd[idx] - max(left, right);
    }
};

// ------------------------
// Transmitter Thread
// ------------------------
void transmit_thread(shared_ptr<uhd::usrp::multi_usrp> usrp, const Config& cfg) {
    uhd::stream_args_t stream_args("fc32", "sc16");
    auto tx_stream = usrp->get_tx_stream(stream_args);
    size_t spb = tx_stream->get_max_num_samps();
    
    vector<complex<float>> buffer(spb);
    vector<double> current_freqs = {1e3, 3e3, 5e3, 7e3, 9e3};  // Default frequencies

    uhd::tx_metadata_t md;
    md.start_of_burst = true;
    md.end_of_burst = false;

    while (!stop_signal) {
        if (shared_data.new_data) {
            lock_guard<mutex> lock(shared_data.mtx);
            
            // Update transmission frequencies based on detected peaks
            vector<double> new_freqs;
            for (const auto& [bin, power] : shared_data.detected_peaks) {
                if (power > -20.0) {  // Threshold for responsive transmission
                    new_freqs.push_back(bin);
                }
            }
            
            if (!new_freqs.empty()) {
                current_freqs = new_freqs;
            }
            
            shared_data.new_data = false;
        }

        // Regenerate buffer with current frequencies
        for (size_t n = 0; n < spb; ++n) {
            buffer[n] = 0;
            for (double f : current_freqs) {
                double t = n / cfg.sample_rate;
                buffer[n] += complex<float>(
                    cos(2 * M_PI * f * t),
                    sin(2 * M_PI * f * t)
                ) * 0.2f;
            }
        }

        size_t sent = tx_stream->send(&buffer[0], buffer.size(), md);
        md.start_of_burst = false;
        
        if (sent < buffer.size()) {
            cerr << "TX Underrun!" << endl;
        }
    }
}

// ------------------------
// Receiver Thread
// ------------------------
void rx_thread(shared_ptr<uhd::usrp::multi_usrp> usrp, const Config& cfg) {
    unique_ptr<Processor> processor;
    if (cfg.algorithm == "fft") {
        processor = make_unique<FFTProcessor>(cfg.fft_size, cfg.avg_num, cfg.threshold);
    } else {
        throw runtime_error("Unsupported algorithm");
    }

    uhd::stream_args_t stream_args("fc32", "sc16");
    auto rx_stream = usrp->get_rx_stream(stream_args);
    vector<complex<float>> buffer(cfg.fft_size);
    uhd::rx_metadata_t md;

    if (cfg.fft_size > buffer.size()) {
        throw runtime_error("FFT size exceeds buffer capacity");
    }

    while (!stop_signal) {
        for (double freq = cfg.start_freq; freq <= cfg.end_freq && !stop_signal; freq += cfg.step_freq) {
            usrp->set_rx_freq(freq);
            this_thread::sleep_for(chrono::milliseconds(static_cast<int>(cfg.settling_time * 1000)));
            
            size_t total = 0;
            while (total < cfg.fft_size) {
                size_t recv = rx_stream->recv(&buffer[total], buffer.size() - total, md, 1.0);
                if (md.error_code != uhd::rx_metadata_t::ERROR_CODE_NONE) {
                    cerr << "RX Error: " << md.strerror() << endl;
                    break;
                }
                total += recv;
            }
            
            auto peaks = processor->process(buffer);
            for (const auto& [bin, power] : peaks) {
                double freq_actual = freq + (bin * cfg.sample_rate / cfg.fft_size);
                {
                    lock_guard<mutex> lock(shared_data.mtx);
                    shared_data.detected_peaks.emplace_back(freq_actual, power);
                    shared_data.new_data = true;
                }
                cout << "Peak at " << freq_actual / 1e6 << " MHz, Power: " << power << " dB" << endl;
            }
        }
    }
}

// ------------------------
// Main Function
// ------------------------
int main(int argc, char** argv) {
    signal(SIGINT, &sigint_handler);
    Config cfg;

    po::options_description desc("Spectrum Analyzer Options");
    desc.add_options()
        ("help", "Help message")
        ("config", po::value<string>(), "YAML config file")
        ("save-config", po::value<string>(), "Save current config to YAML file")
        ("start", po::value<double>(&cfg.start_freq), "Start freq (Hz)")
        ("end", po::value<double>(&cfg.end_freq), "End freq (Hz)")
        ("step", po::value<double>(&cfg.step_freq), "Step freq (Hz)")
        ("sample_rate", po::value<double>(&cfg.sample_rate), "Sample rate")
        ("tx_gain", po::value<double>(&cfg.tx_gain), "TX gain")
        ("rx_gain", po::value<double>(&cfg.rx_gain), "RX gain")
        ("algorithm", po::value<string>(&cfg.algorithm), "Algorithm (fft/ml)")
        ("enable_tx", po::value<bool>(&cfg.enable_tx), "Enable TX")
        ("enable_rx", po::value<bool>(&cfg.enable_rx), "Enable RX")
        ("fft_size", po::value<size_t>(&cfg.fft_size), "FFT size")
        ("avg_num", po::value<size_t>(&cfg.avg_num), "Averaging count")
        ("threshold", po::value<double>(&cfg.threshold), "Peak threshold (dB)")
        ("settling_time", po::value<double>(&cfg.settling_time), "Tune settling (s)")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        cout << desc << endl;
        return 1;
    }

    if (vm.count("config")) {
        cfg.load_from_yaml(vm["config"].as<string>());
    }

    if (vm.count("save-config")) {
        cfg.save_to_yaml(vm["save-config"].as<string>());
        return 0;
    }

    auto usrp = uhd::usrp::multi_usrp::make("");
    if (!usrp) {
        cerr << "Failed to create USRP device!" << endl;
        return 1;
    }

    perform_dc_offset_calibration(usrp);
    perform_iq_balance_calibration(usrp);

    usrp->set_tx_rate(cfg.sample_rate);
    usrp->set_rx_rate(cfg.sample_rate);
    usrp->set_tx_gain(cfg.tx_gain);
    usrp->set_rx_gain(cfg.rx_gain);

    thread tx_t, rx_t;
    if (cfg.enable_tx) tx_t = thread(transmit_thread, usrp, cfg);
    if (cfg.enable_rx) rx_t = thread(rx_thread, usrp, cfg);

    if (tx_t.joinable()) tx_t.join();
    if (rx_t.joinable()) rx_t.join();

    return 0;
}