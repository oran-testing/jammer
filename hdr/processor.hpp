#pragma once

#include <vector>
#include <complex>
#include <utility> // For std::pair

class Processor {
public:
    virtual ~Processor() = default;
    virtual std::vector<std::pair<double, double>> process_block(const std::vector<std::complex<float>>& data) = 0;
    virtual void reset() = 0;
};