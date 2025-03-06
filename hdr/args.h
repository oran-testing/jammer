#ifndef ARGS_H
#define ARGS_H

#include <cstring>
#include <iostream>
#include <yaml-cpp/yaml.h>

typedef struct rf_args_s {
  std::string device_args;
  float tx_gain;
} rf_args_t;

typedef struct all_args_s {
  float amplitude;
  float amplitude_width;
  float center_frequency;
  float bandwidth;
  float initial_phase;
  size_t num_samples;
  float sampling_freq;
  std::string output_iq_file;
  std::string output_csv_file;
  bool write_iq;
  bool write_csv;
  rf_args_t rf;
} all_args_t;

all_args_t parseConfig(const std::string &filename);

void overrideConfig(all_args_t &args, int argc, char *argv[]);

#endif // !ARGS_H
