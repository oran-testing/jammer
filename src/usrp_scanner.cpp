#include "usrp_scanner.hpp"
#include "globals.hpp"       // For stop_signal_called
#include "shared_data.hpp"   // For shared_data, DetectedPeak
#include "processor.hpp"     // For Processor base class
#include "fft_processor.hpp" // For FFTProcessor
#include "ml_processor.hpp"  // For MLProcessor

#include <uhd/types/tune_request.hpp>
#include <uhd/types/tune_result.hpp> // Not strictly used but good for context
#include <uhd/stream.hpp>
#include <uhd/utils/thread.hpp>     // For uhd::set_thread_priority_safe

#include <iostream>
#include <vector>
#include <complex>
#include <cmath>      // For std::polar, std::log10, std::norm
#include <algorithm>  // For std::sort, std::min, std::copy, std::fill
#include <chrono>
#include <numeric>    // Not strictly used but often useful
#include <memory>     // For std::unique_ptr
#include <stdexcept>  // For std::runtime_error
#include <cstdio>     // For printf
#include <cstdlib>    // For rand, srand
#include <ctime>      // For time (for srand)

#include <boost/math/constants/constants.hpp>
#include <boost/algorithm/string.hpp> // For boost::iequals

using namespace std::chrono_literals;
constexpr double PI = boost::math::constants::pi<double>();

// RX Thread Function
void rx_thread(uhd::usrp::multi_usrp::sptr usrp, const Config& cfg) {
    if (cfg.set_thread_priority) {
        uhd::set_thread_priority_safe();
    }

    std::cout << "RX Thread: Starting." << std::endl;

    std::unique_ptr<Processor> processor;
    try {
        if (boost::iequals(cfg.algorithm, "fft")) {
            processor = std::make_unique<FFTProcessor>(cfg);
        } else if (boost::iequals(cfg.algorithm, "ml")) {
             processor = std::make_unique<MLProcessor>(cfg);
        } else {
             throw std::runtime_error("RX Thread: Invalid algorithm selected: " + cfg.algorithm);
        }
    } catch (const std::exception& e) {
        std::cerr << "RX Thread: Error initializing processor: " << e.what() << std::endl;
        stop_signal_called = true; // Signal other threads to stop
        return;
    }

    uhd::stream_args_t stream_args("fc32", "sc16"); // CPU format fc32, over-the-wire sc16
    stream_args.channels = {0}; // Use channel 0
    uhd::rx_streamer::sptr rx_stream = usrp->get_rx_stream(stream_args);

    uhd::stream_cmd_t stream_cmd(uhd::stream_cmd_t::STREAM_MODE_START_CONTINUOUS);
    stream_cmd.stream_now = true; // Start streaming immediately
    rx_stream->issue_stream_cmd(stream_cmd);

    std::vector<std::complex<float>> rx_buffer(cfg.num_samples_block);
    uhd::rx_metadata_t md;
    size_t samples_collected_for_fft = 0;
    std::vector<std::complex<float>> fft_input_buffer(cfg.fft_size); // Buffer specifically for one FFT input

    std::chrono::steady_clock::time_point last_report_time = std::chrono::steady_clock::now();
    long total_peaks_detected_session = 0; // Renamed to avoid confusion with shared_data

    std::cout << "RX Thread: Starting frequency sweep..." << std::endl;

    while (!stop_signal_called) {
        { // Scope for lock
            std::lock_guard<std::mutex> lock(shared_data.mtx);
            shared_data.current_sweep_peaks.clear(); // Clear peaks from previous sweep
            shared_data.sweep_complete = false;
        }

        for (double current_freq = cfg.start_freq;
             current_freq <= cfg.end_freq && !stop_signal_called;
             current_freq += cfg.step_freq)
        {
            if(cfg.verbose) std::cout << "\nRX Tuning to: " << current_freq / 1e6 << " MHz" << std::endl;

            uhd::tune_request_t tune_request(current_freq);
            // tune_request.args = uhd::device_addr_t("mode_n=integer"); // Example for integer-N tuning if needed
            usrp->set_rx_freq(tune_request, 0); // Tune channel 0

            // Wait for settling and check lock status
            if (cfg.verbose) std::cout << "RX waiting " << cfg.settling_time * 1e3 << " ms for settling..." << std::endl;
            std::this_thread::sleep_for(std::chrono::duration<double>(cfg.settling_time));

            try {
                bool locked = usrp->get_rx_sensor("lo_locked", 0).to_bool();
                if (!locked) {
                   std::cerr << "Warning: RX LO failed to lock at " << current_freq / 1e6 << " MHz after settling time." << std::endl;
                } else if (cfg.verbose) {
                   std::cout << "RX LO locked at " << current_freq / 1e6 << " MHz." << std::endl;
                }
            } catch (const uhd::key_error&) { // Sensor not found
               if (cfg.verbose) std::cout << "  (LO lock sensor not found, proceeding after wait)." << std::endl;
            } catch (const uhd::exception& e) { // Other UHD errors
               std::cerr << "Warning: Error checking RX LO lock status: " << e.what() << std::endl;
            }

             processor->reset(); // Reset processor state (e.g., averaging buffers) for the new frequency
             samples_collected_for_fft = 0; // Reset collection count

             bool processing_at_this_freq_done = false; // Flag to indicate if one round of processing (e.g. averaging) is done
             while (!processing_at_this_freq_done && !stop_signal_called) {
                 size_t num_rx_samps = rx_stream->recv(rx_buffer.data(), rx_buffer.size(), md, 0.1); // 100ms timeout

                 if (md.error_code != uhd::rx_metadata_t::ERROR_CODE_NONE) {
                     std::cerr << "RX Error at " << current_freq / 1e6 << " MHz: " << md.strerror() << std::endl;
                     if (md.error_code == uhd::rx_metadata_t::ERROR_CODE_OVERFLOW) {
                         // Potentially try to recover or just break
                     }
                     break; // Break from inner loop for this frequency on error
                 }
                 if (num_rx_samps == 0) { // Timeout
                    if (!stop_signal_called) {
                        std::cerr << "Warning: RX receive timeout at " << current_freq / 1e6 << " MHz." << std::endl;
                    }
                    continue; // Try receiving again
                 }

                size_t current_pos_in_block = 0;
                 while (current_pos_in_block < num_rx_samps && !processing_at_this_freq_done) {
                    size_t remaining_in_block = num_rx_samps - current_pos_in_block;
                    size_t needed_for_fft = cfg.fft_size - samples_collected_for_fft;
                    size_t samples_to_copy = std::min(remaining_in_block, needed_for_fft);

                     std::copy(rx_buffer.begin() + current_pos_in_block,
                              rx_buffer.begin() + current_pos_in_block + samples_to_copy,
                              fft_input_buffer.begin() + samples_collected_for_fft);

                    samples_collected_for_fft += samples_to_copy;
                    current_pos_in_block += samples_to_copy;

                    if (samples_collected_for_fft == cfg.fft_size) { // We have a full buffer for the processor
                        auto peaks = processor->process_block(fft_input_buffer);
                        samples_collected_for_fft = 0; // Reset for the next FFT input

                         if (!peaks.empty()) { // Processor decided it's done averaging and found peaks
                             processing_at_this_freq_done = true;

                            if (cfg.verbose) std::cout << "  Peaks found at " << current_freq / 1e6 << " MHz: " << peaks.size() << std::endl;

                            { // Scope for lock
                                 std::lock_guard<std::mutex> lock(shared_data.mtx);
                                for (const auto& [offset, power] : peaks) {
                                     double absolute_freq = current_freq + offset;
                                     // Ensure peak is within the intended scan range (can sometimes be slightly outside due to FFT bin width)
                                     if (absolute_freq >= cfg.start_freq && absolute_freq <= cfg.end_freq) {
                                         shared_data.current_sweep_peaks.push_back({absolute_freq, power, current_freq});
                                         total_peaks_detected_session++;
                                         if (cfg.verbose) printf("    -> Peak: %.3f MHz (Power: %.2f dB)\n", absolute_freq / 1e6, power);
                                    }
                                 }
                            }
                             // break; // from the while current_pos_in_block < num_rx_samps loop
                         } else if (cfg.algorithm == "fft" && static_cast<FFTProcessor*>(processor.get())->current_avg_count == 0) {
                            // This condition means FFTProcessor decided it's done averaging but found NO peaks.
                            // Or if current_avg_count became 0 it means it finished an averaging cycle.
                            processing_at_this_freq_done = true;
                         }
                     }
                 } // end while current_pos_in_block
             } // end while !processing_at_this_freq_done
             if (stop_signal_called) break; // Break from frequency loop
         } // end for current_freq loop

        { // Scope for lock
            std::lock_guard<std::mutex> lock(shared_data.mtx);
            shared_data.sweep_complete = true;
        }


         auto now = std::chrono::steady_clock::now();
         auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_report_time);
        // Report if enough time has passed or if stopping, but not if already stopped to avoid double report
        if ((elapsed >= 1000ms || (stop_signal_called && !shared_data.current_sweep_peaks.empty())) && !stop_signal_called.load()) {
            std::lock_guard<std::mutex> lock(shared_data.mtx); // Lock before accessing shared_data
            std::cout << "\n--- Sweep Report (" << elapsed.count() << "ms) ---" << std::endl;
            if (shared_data.current_sweep_peaks.empty()) {
                 std::cout << "  No significant peaks detected in this sweep." << std::endl;
             } else {
                 std::cout << "  Detected Peaks (" << shared_data.current_sweep_peaks.size() << "):" << std::endl;
                 // Sort peaks by frequency for display
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

        if (!stop_signal_called) {
             // Small sleep to yield, and prevent busy-looping if sweep is very fast
             std::this_thread::sleep_for(10ms);
        }
     } // end while !stop_signal_called

    stream_cmd.stream_mode = uhd::stream_cmd_t::STREAM_MODE_STOP_CONTINUOUS;
    rx_stream->issue_stream_cmd(stream_cmd);

    std::cout << "RX Thread: Stopped. Total peaks detected across all sweeps this session: " << total_peaks_detected_session << std::endl;
}


// TX Thread Function
void tx_thread(uhd::usrp::multi_usrp::sptr usrp, const Config& cfg) {
    if (cfg.set_thread_priority) {
        uhd::set_thread_priority_safe();
    }

    std::cout << "TX Thread: Starting." << std::endl;

    if (boost::iequals(cfg.tx_waveform_type, "none")) {
        std::cout << "TX Thread: Waveform type is 'none'. TX thread will idle." << std::endl;
        while(!stop_signal_called) { // Keep thread alive but idle
            std::this_thread::sleep_for(100ms);
        }
        std::cout << "TX Thread: Exiting (waveform was 'none')." << std::endl;
        return;
    }

    uhd::stream_args_t stream_args("fc32", "sc16");
    stream_args.channels = {0}; // Use channel 0
    uhd::tx_streamer::sptr tx_stream = usrp->get_tx_stream(stream_args);

    size_t spb = tx_stream->get_max_num_samps(); // Samples Per Buffer
    if (spb > 16384 || spb == 0) spb = (spb == 0) ? 1024: 16384; // Cap or set default
    std::vector<std::complex<float>> tx_buffer(spb);

    double current_phase = 0.0; // Static for tone phase continuity across buffer refills

     if (boost::iequals(cfg.tx_waveform_type, "tone")) {
        std::cout << "TX Thread: Generating single tone at offset " << cfg.tx_tone_freq_offset / 1e3 << " kHz." << std::endl;
         // Initial buffer fill is done inside the loop
     } else if (boost::iequals(cfg.tx_waveform_type, "noise")) {
         std::cout << "TX Thread: Generating complex white noise." << std::endl;
         srand(static_cast<unsigned int>(time(NULL))); // Seed RNG
         // Buffer fill is done inside the loop if we want changing noise, or once if static noise is ok
     }
     else {
        std::cerr << "TX Thread: Unknown or unsupported tx_waveform_type: " << cfg.tx_waveform_type << ". Exiting." << std::endl;
         return;
     }

    uhd::tx_metadata_t md;
    md.start_of_burst = true; // First packet is start of burst
    md.end_of_burst = false;  // Continuous stream
    md.has_time_spec = false; // Send immediately

    std::cout << "TX Thread: Starting continuous transmission at " << cfg.tx_center_freq / 1e6 << " MHz..." << std::endl;

    size_t total_samps_sent = 0;

    while (!stop_signal_called) {
        // Fill buffer for each send operation
        if (boost::iequals(cfg.tx_waveform_type, "tone")) {
            double delta_phase = 2.0 * PI * cfg.tx_tone_freq_offset / cfg.sample_rate;
            for (size_t i = 0; i < spb; ++i) {
                tx_buffer[i] = std::polar(cfg.tx_amplitude, static_cast<float>(current_phase));
                current_phase += delta_phase;
                while (current_phase >= 2.0 * PI) current_phase -= 2.0 * PI; // Wrap phase
                while (current_phase < 0.0) current_phase += 2.0 * PI;
            }
        } else if (boost::iequals(cfg.tx_waveform_type, "noise")) {
            for(size_t i=0; i<spb; ++i){
                 // Normalized to [-1, 1) approx
                 float real_part = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) * 2.0f - 1.0f;
                 float imag_part = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) * 2.0f - 1.0f;
                 tx_buffer[i] = std::complex<float>(real_part, imag_part) * cfg.tx_amplitude;
             }
        }

        size_t num_sent = tx_stream->send(tx_buffer.data(), tx_buffer.size(), md, 0.1); // 100ms timeout
        
        if (num_sent < tx_buffer.size() && !stop_signal_called) {
            // UHD Manual: "If the number of samples returned is less than the number of samples
            // provided, this indicates an underrun has occurred or the call timed out."
            std::cerr << "TX Warning: Send incomplete or potential underrun. Sent " << num_sent << "/" << tx_buffer.size() << std::endl;
        } else if (num_sent == 0 && !stop_signal_called) {
            std::cerr << "TX Warning: Send timed out." << std::endl;
        }
        total_samps_sent += num_sent;
        md.start_of_burst = false; // Subsequent packets are not start of burst
    }

    // Send a packet with end_of_burst = true to stop the transmitter gracefully
    md.end_of_burst = true;
    std::fill(tx_buffer.begin(), tx_buffer.end(), std::complex<float>(0.0f, 0.0f)); // Send zeros
    tx_stream->send(tx_buffer.data(), tx_buffer.size(), md); // Send final packet
    std::this_thread::sleep_for(100ms); // Allow time for EOB to be processed

    std::cout << "TX Thread: Stopped. Total samples sent: ~" << total_samps_sent << std::endl;
}