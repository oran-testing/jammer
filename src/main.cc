#include <cstdio>
#include <iostream>

#include "noise.h"
#include "rf.h"
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>
#include <yaml-cpp/yaml.h>

const double PI = std::acos(-1.0);

// Struct to hold all configuration parameters
struct all_args_t {
  double amplitude;
  double initial_frequency;
  double frequency_change_rate;
  double initial_phase;
  int num_samples;
  double sample_rate;
  std::string output_iq_file;
  std::string output_csv_file;
  bool write_iq;
  bool write_csv;
};

// Function to parse YAML config into struct
all_args_t parseConfig(
    const std::string &filename) { // change the filename to the real filename

  YAML::Node config =
      YAML::LoadFile(filename); // change the filename to its real filename

  all_args_t args; // an instance of struct
  args.amplitude = config["amplitude"].as<double>();
  args.initial_frequency = config["initial_frequency"].as<double>();
  args.frequency_change_rate = config["frequency_change_rate"].as<double>();
  args.initial_phase = config["initial_phase"].as<double>();
  args.num_samples = config["num_samples"].as<int>();
  args.sample_rate = config["sample_rate"].as<double>();
  args.output_iq_file = config["output_iq_file"].as<std::string>();
  args.output_csv_file = config["output_csv_file"].as<std::string>();
  args.write_iq = config["write_iq"].as<bool>();
  args.write_csv = config["write_csv"].as<bool>();
  return args;
}

// Function to update config from command-line arguments
// (simple parsing: expects "--key value")

void overrideConfig(all_args_t &args, int argc, char *argv[]) {
  for (int i = 1; i < argc; ++i) {

    if (std::strcmp(argv[i], "--amplitude") == 0 && i + 1 < argc) {
      args.amplitude = std::atof(argv[++i]);
    } else if (std::strcmp(argv[i], "--initial_frequency") == 0 &&
               i + 1 < argc) {
      args.initial_frequency = std::atof(argv[++i]);
    } else if (std::strcmp(argv[i], "--frequency_change_rate") == 0 &&
               i + 1 < argc) {
      args.frequency_change_rate = std::atof(argv[++i]);
    } else if (std::strcmp(argv[i], "--initial_phase") == 0 && i + 1 < argc) {
      args.initial_phase = std::atof(argv[++i]);
    } else if (std::strcmp(argv[i], "--num_samples") == 0 && i + 1 < argc) {
      args.num_samples = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--sample_rate") == 0 && i + 1 < argc) {
      args.sample_rate = std::atof(argv[++i]);
    } else if (std::strcmp(argv[i], "--output_iq_file") == 0 && i + 1 < argc) {
      args.output_iq_file = argv[++i];
    } else if (std::strcmp(argv[i], "--output_csv_file") == 0 && i + 1 < argc) {
      args.output_csv_file = argv[++i];
    } else if (std::strcmp(argv[i], "--write_iq") == 0 && i + 1 < argc) {
      args.write_iq = (std::string(argv[++i]) == "true");
    } else if (std::strcmp(argv[i], "--write_csv") == 0 && i + 1 < argc) {
      args.write_csv = (std::string(argv[++i]) == "true");
    } else if (std::strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
      // Already processed separately, so skip here.
      ++i;
    } else {
      std::cerr << "Unknown or incomplete option: " << argv[i] << std::endl;
    }
  }
}

// Write IQ data as binary file (.fc32) with 32-bit float interleaved (real,
// imag)
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

  return 0;
}

// parse args into all_args_t using yaml-cpp
// write sin wave complex float to file .fc32
// use pysdr in python to read the constellation
// write sin wave as csv file as well, I will handle cartesian plotting
// Make soem argmetns for write to iq and write to csv
