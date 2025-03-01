#include "args.h"
#include <string>

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
  args.rf.device_args = config["device_args"].as<std::string>();
  args.rf.tx_gain = config["tx_gain"].as<double>();
  return args;
}

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
    } else if (std::strcmp(argv[i], "--device_args") == 0 && i + 1 < argc) {
      args.rf.device_args = std::string(argv[++i]);
    } else if (std::strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
      // Already processed separately, so skip here.
      ++i;
    } else if (std::strcmp(argv[i], "--tx_gain") == 0 && i + 1 < argc) {
      args.rf.tx_gain = std::stof(argv[++i]);
    } else {
      std::cerr << "Unknown or incomplete option: " << argv[i] << std::endl;
    }
  }
}
