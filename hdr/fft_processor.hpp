#pragma once

#include "processor.hpp"
#include "config.hpp" // Needs Config for its parameters
#include <fftw3.h>
#include <vector>
#include <complex>
#include <memory>    // For std::unique_ptr
#include <string>    // For std::string
#include <stdexcept> // For std::bad_alloc

// RAII Wrapper for FFTW Memory
template <typename T> struct fftw_allocator {
    typedef T value_type;
    T* allocate(size_t n) {
        T* p = static_cast<T*>(fftwf_malloc(sizeof(T) * n));
        if (!p) throw std::bad_alloc();
        return p;
    }
    void deallocate(T* p, size_t) noexcept { fftwf_free(p); }

    // Required for std::vector allocator concept
    template <class U> struct rebind { typedef fftw_allocator<U> other; };
    fftw_allocator() = default;
    template <class U> fftw_allocator(const fftw_allocator<U>&) {}

    bool operator==(const fftw_allocator&) const { return true; }
    bool operator!=(const fftw_allocator&) const { return false; }
};

template <typename T>
using fftw_vector = std::vector<T, fftw_allocator<T>>;

struct fftwf_plan_deleter {
    void operator()(fftwf_plan p) const { fftwf_destroy_plan(p); }
};
using fftwf_plan_ptr = std::unique_ptr<std::remove_pointer<fftwf_plan>::type, fftwf_plan_deleter>;


class FFTProcessor : public Processor {
private:
    const Config& cfg;
    fftw_vector<std::complex<float>> fft_in;
    fftw_vector<std::complex<float>> fft_out;
    fftw_vector<float> window;
    fftwf_plan_ptr fft_plan;
    std::vector<std::vector<double>> psd_buffers; // For averaging
    size_t current_avg_count = 0;

    void generate_window();
    std::vector<std::pair<double, double>> find_peaks(const std::vector<double>& psd);

public:
    FFTProcessor(const Config& config);
    ~FFTProcessor() override = default;

    void reset() override;
    std::vector<std::pair<double, double>> process_block(const std::vector<std::complex<float>>& data) override;

    static void save_wisdom(const std::string& path);
};