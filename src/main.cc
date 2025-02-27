#include <cstdio>
#include <iostream>

#include "args.h"
#include "noise.h"
#include "rf.h"
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <uhd/error.h>
#include <uhd/stream.hpp>
#include <uhd/types/device_addr.hpp>
#include <vector>

void handle_uhd_error(uhd_error err) {
  if (err != UHD_ERROR_NONE) {
    fprintf(stderr, "UHD ERROR: %d", UHD_ERROR_NONE);
    exit(EXIT_FAILURE);
  }
}

void writeIQBinary(
    const std::string &filename,
    const std::vector<std::complex<double>>
        &samples) { // Remember to change the filename to a real name
  std::ofstream outfile(filename, std::ios::binary);
  if (!outfile) {
    std::cerr << "Error opening file for IQ binary output: " << filename
              << std::endl;
    return;
  }
  for (const auto &sample : samples) {
    float real = static_cast<float>(sample.real());
    float imag = static_cast<float>(sample.imag());
    outfile.write(reinterpret_cast<const char *>(&real), sizeof(float));
    outfile.write(reinterpret_cast<const char *>(&imag), sizeof(float));
  }
  outfile.close();
}

// Write CSV file: index, real, imag
void writeCSV(const std::string &filename,
              const std::vector<std::complex<double>> &samples) {
  std::ofstream outfile(filename);
  if (!outfile) {
    std::cerr << "Error opening file for CSV output: " << filename << std::endl;
    return;
  }

  outfile << "index,real,imag\n";
  for (size_t i = 0; i < samples.size(); ++i) {
    outfile << i << "," << samples[i].real() << "," << samples[i].imag()
            << "\n";
  }
  outfile.close();
}

int main(int argc, char *argv[]) {
  // Default config file name
  std::string config_file = "";

  // Check if --config option is provided first
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
      config_file = argv[++i];
      break;
    }
  }

  if (config_file == "") {
    fprintf(stderr, "Usage: jammer --config [config file]\n");
    return EXIT_FAILURE;
  }

  // Load config from YAML
  all_args_t args = parseConfig(config_file);

  // Override config with any command-line arguments provided
  overrideConfig(args, argc, argv);

  // Generate the complex sine wave
  auto samples = generateComplexSineWave(
      args.amplitude, args.initial_frequency, args.frequency_change_rate,
      args.initial_phase, args.num_samples, args.sample_rate);

  // Write IQ binary file if enabled
  if (args.write_iq) {
    writeIQBinary(args.output_iq_file, samples);
    std::cout << "IQ binary data written to " << args.output_iq_file
              << std::endl;
  }

  // Write CSV file if enabled
  if (args.write_csv) {
    writeCSV(args.output_csv_file, samples);
    std::cout << "CSV data written to " << args.output_csv_file << std::endl;
  }

  rf_handler rf_dev = rf_handler();
  uint32_t nof_channels = 1;
  const uhd::device_addr_t dev_addr = uhd::device_addr_t(args.rf.device_args);
  handle_uhd_error(rf_dev.usrp_make(dev_addr, nof_channels));

  size_t channel_no = 0;
  handle_uhd_error(rf_dev.set_tx_gain(channel_no, args.rf.tx_gain));
  handle_uhd_error(rf_dev.set_tx_rate(args.sample_rate));
  double actual_frequency = 0.0;
  handle_uhd_error(
      rf_dev.set_tx_freq(0, args.initial_frequency, actual_frequency));

  // uhd::stream_args_t stream_args;
  //  tx_stream = rf_dev.get_tx_stream(stream_args);

  return 0;
}

// parse args into all_args_t using yaml-cpp
// write sin wave complex float to file .fc32
// use pysdr in python to read the constellation
// write sin wave as csv file as well, I will handle cartesian plotting
// Make soem argmetns for write to iq and write to csv
