#pragma once

#include <mutex>
#include <deque>
#include <atomic>
#include <string> // Not strictly needed by structs but good for consistency if future changes
#include <vector> // Not strictly needed by structs

struct DetectedPeak {
    double frequency_hz;
    double power_db;
    double center_freq_hz;
};

struct SharedData {
    std::mutex mtx;
    std::deque<DetectedPeak> current_sweep_peaks;
    std::atomic<bool> sweep_complete{false};
};

extern SharedData shared_data;