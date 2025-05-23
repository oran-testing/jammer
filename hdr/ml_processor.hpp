#pragma once

#include "processor.hpp"
#include "config.hpp" // Needs Config for its parameters
#include <vector>
#include <complex>
#include <string>

class MLProcessor : public Processor {
private:
    const Config& cfg;
    // Placeholder for ML model, preprocessors, etc.
public:
     MLProcessor(const Config& config);
     ~MLProcessor() override = default;

      void reset() override;
      std::vector<std::pair<double, double>> process_block(const std::vector<std::complex<float>>& data) override;
};