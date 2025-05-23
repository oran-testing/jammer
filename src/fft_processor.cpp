#include "fft_processor.hpp"
#include <cmath>
#include <algorithm> // For std::min, std::max, std::fill, std::sort
#include <numeric>   // For std::accumulate (if needed, not currently)
#include <iostream>  // For std::cout, std::cerr
#include <stdexcept> // For std::runtime_error
#include <boost/math/constants/constants.hpp>
#include <boost/algorithm/string.hpp> // For boost::iequals

// Constants
constexpr double PI = boost::math::constants::pi<double>();

FFTProcessor::FFTProcessor(const Config& config) : cfg(config) {
    if (cfg.fft_size == 0) throw std::runtime_error("FFT size cannot be zero.");
    fft_in.resize(cfg.fft_size);
    fft_out.resize(cfg.fft_size);
    generate_window();
    psd_buffers.reserve(cfg.avg_num); // Pre-allocate for efficiency

    unsigned flags = FFTW_ESTIMATE;
     if (!cfg.fft_wisdom_path.empty()) {
         if (fftwf_import_wisdom_from_filename(cfg.fft_wisdom_path.c_str())) {
             std::cout << "Successfully loaded FFTW wisdom from " << cfg.fft_wisdom_path << std::endl;
             flags = FFTW_WISDOM_ONLY; // Try to use wisdom only, fall back if plan fails
         } else {
             std::cerr << "Warning: Failed to load FFTW wisdom from " << cfg.fft_wisdom_path << ". Planning may take longer." << std::endl;
             flags = FFTW_MEASURE; // If load fails, measure to create new wisdom
        }
     } else {
         std::cout << "No FFTW wisdom file specified. Using FFTW_ESTIMATE for planning." << std::endl;
     }

    fft_plan.reset(fftwf_plan_dft_1d(cfg.fft_size,
                                     reinterpret_cast<fftwf_complex*>(fft_in.data()),
                                     reinterpret_cast<fftwf_complex*>(fft_out.data()),
                                     FFTW_FORWARD,
                                     flags));
    // If FFTW_WISDOM_ONLY failed to find a plan, try again with FFTW_ESTIMATE or FFTW_MEASURE
    if (!fft_plan && (flags & FFTW_WISDOM_ONLY)) {
        std::cerr << "Warning: FFTW_WISDOM_ONLY failed to create a plan. Trying FFTW_ESTIMATE." << std::endl;
        flags = FFTW_ESTIMATE; // or FFTW_MEASURE for better subsequent performance but longer init
        fft_plan.reset(fftwf_plan_dft_1d(cfg.fft_size,
                                     reinterpret_cast<fftwf_complex*>(fft_in.data()),
                                     reinterpret_cast<fftwf_complex*>(fft_out.data()),
                                     FFTW_FORWARD,
                                     flags));
    }


    if (!fft_plan) {
        throw std::runtime_error("FFTW failed to create plan.");
    }
}

void FFTProcessor::generate_window() {
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
    else { // "none" or unrecognized
        std::fill(window.begin(), window.end(), 1.0f);
    }

    // Normalize window for power
    double window_power = 0.0;
    for(float w : window) window_power += w*w;
    float norm_factor = 1.0f;
    if (window_power > 1e-9) { // Avoid division by zero for all-zero window (though unlikely)
      norm_factor = std::sqrt(static_cast<float>(cfg.fft_size) / window_power);
    }
    for (size_t i = 0; i < cfg.fft_size; ++i) window[i] *= norm_factor;
}

std::vector<std::pair<double, double>> FFTProcessor::find_peaks(const std::vector<double>& psd) {
    std::vector<std::pair<double, double>> peaks;
    if (psd.size() < 3) return peaks; // Need at least 3 points to find a peak

    std::vector<size_t> peak_indices;

    for (size_t i = 1; i < psd.size() - 1; ++i) {
        double current_val = psd[i];
        if (current_val > cfg.peak_threshold_db && current_val > psd[i-1] && current_val > psd[i+1]) {
             peak_indices.push_back(i);
         }
    }

    for (size_t idx : peak_indices) {
        double peak_val = psd[idx];
        
        // Find left minimum for prominence calculation
        double left_min = peak_val;
        for (long i = (long)idx - 1; i >= 0; --i) { // Use long for i to correctly go to -1
            left_min = std::min(left_min, psd[i]);
            if (psd[i] >= peak_val) break; // Stop if we go up again before finding a true trough
        }

        // Find right minimum for prominence calculation
        double right_min = peak_val;
        for (size_t i = idx + 1; i < psd.size(); ++i) {
            right_min = std::min(right_min, psd[i]);
            if (psd[i] >= peak_val) break; // Stop if we go up again
        }

        double prominence = peak_val - std::max(left_min, right_min);

        if (prominence >= cfg.prominence_threshold_db) {
             double freq_offset = (static_cast<double>(idx) - static_cast<double>(cfg.fft_size) / 2.0)
                                * cfg.sample_rate / static_cast<double>(cfg.fft_size);
            peaks.emplace_back(freq_offset, peak_val);
        }
    }
    return peaks;
}

void FFTProcessor::reset() {
    current_avg_count = 0;
    psd_buffers.clear(); // Clears the data, capacity remains
}

std::vector<std::pair<double, double>> FFTProcessor::process_block(const std::vector<std::complex<float>>& data) {
    if (data.size() != cfg.fft_size) {
        // This should ideally not happen if rx_thread logic is correct
        std::cerr << "Error: FFTProcessor::process_block received data of size " << data.size()
                  << ", expected " << cfg.fft_size << std::endl;
        return {}; // Return empty vector on error
    }

    for (size_t i = 0; i < cfg.fft_size; ++i) {
        fft_in[i] = data[i] * window[i];
    }

    fftwf_execute(fft_plan.get());

    std::vector<double> current_psd(cfg.fft_size);
    double norm_factor = 1.0 / (static_cast<double>(cfg.fft_size) * static_cast<double>(cfg.fft_size)); // Correct normalization for power
    // The window normalization already accounts for some of this.
    // For power spectral density, it's |X(f)|^2 / (Fs * N) if window is rectangular and scaled by 1/N
    // Or more generally, 1 / (sum_sq_window * Fs)
    // Let's stick to a simpler dB relative power for now, a calibration step would be needed for absolute power.
    // The normalization factor of 1.0/cfg.fft_size for amplitude, so 1.0/(cfg.fft_size^2) for power before sqrt,
    // then 1.0/cfg.fft_size for power.
    // The window normalization to unit *energy* (sum_sq_win = N) is common.
    // If sum_sq_win = N, then the factor is 1/N.
    // The current window normalization factor is sqrt(N / sum_sq_w_orig). So w_new[i] = w_orig[i] * sqrt(N / sum_sq_w_orig).
    // sum_sq_w_new = sum (w_orig[i]^2 * N / sum_sq_w_orig) = (sum_sq_w_orig * N / sum_sq_w_orig) = N.
    // So effective normalization is 1.0 / N for power.

    norm_factor = 1.0 / static_cast<double>(cfg.fft_size); // For power after windowing (if window is energy normalized)

    for (size_t i = 0; i < cfg.fft_size; ++i) {
         size_t shifted_idx = (i + cfg.fft_size / 2) % cfg.fft_size; // FFT shift
         double power_val = std::norm(fft_out[shifted_idx]) * norm_factor;
         current_psd[i] = 10.0 * std::log10(std::max(power_val, 1e-20)); // Avoid log(0) or very small numbers
     }

    psd_buffers.push_back(std::move(current_psd));
    current_avg_count++;

    if (current_avg_count < cfg.avg_num) {
        return {}; // Not enough data for an average yet
    }

    std::vector<double> averaged_psd(cfg.fft_size, 0.0);
    if (cfg.avg_num > 0 && !psd_buffers.empty()) {
        for(const auto& psd_buf : psd_buffers) {
            for(size_t i = 0; i < cfg.fft_size; ++i) {
                averaged_psd[i] += psd_buf[i];
            }
        }
        double avg_factor = 1.0 / static_cast<double>(psd_buffers.size()); // Use actual number of buffers, robust if less than avg_num
        for(size_t i = 0; i < cfg.fft_size; ++i) {
            averaged_psd[i] *= avg_factor;
        }
    } else {
        // Should not happen if avg_num >= 1
        return {};
    }


    auto peaks = find_peaks(averaged_psd);
    reset(); // Reset for the next averaging cycle

    return peaks;
}

void FFTProcessor::save_wisdom(const std::string& path) {
    if (!path.empty()) {
        if (fftwf_export_wisdom_to_filename(path.c_str())) {
            std::cout << "Saved FFTW wisdom to " << path << std::endl;
        } else {
            std::cerr << "Error saving FFTW wisdom to " << path << std::endl;
        }
    }
}