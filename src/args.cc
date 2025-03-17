#include "args.h"
#include <string>



all_args_t parseConfig(
    const std::string &filename) { // change the filename to the real filename

  YAML::Node config =
      YAML::LoadFile(filename); // change the filename to its real filename

  all_args_t args; // an instance of struct
  args.amplitude = config["amplitude"].as<float>();
  args.num_samples= config["num_samples"].as<size_t>();
  args.amplitude_width = config["amplitude_width"].as<float>();
  args.center_frequency = config["center_frequency"].as<float>();
  args.initial_phase = config["bandwidth"].as<float>();
  args.initial_phase = config["initial_phase"].as<float>();
  args.sampling_freq = config["sampling_freq"].as<float>();
  args.output_iq_file = config["output_iq_file"].as<std::string>();
  args.output_csv_file = config["output_csv_file"].as<std::string>();
  args.write_iq = config["write_iq"].as<bool>();
  args.write_csv = config["write_csv"].as<bool>();
  args.rf.device_args = config["device_args"].as<std::string>();
  args.rf.tx_gain = config["tx_gain"].as<float>();
  return args;
}




void overrideConfig(all_args_t &args, int argc, char *argv[]) {
  for (int i = 1; i < argc; ++i) {

    if (std::strcmp(argv[i], "--amplitude") == 0 && i + 1 < argc) {
      args.amplitude = std::atof(argv[++i]);
    } else if (std::strcmp(argv[i], "--amplitude_witdth") == 0 &&
               i + 1 < argc) {
      args.center_frequency = std::atof(argv[++i]);
    } else if (std::strcmp(argv[i], "--center_frequency") == 0 &&
               i + 1 < argc) {
      args.bandwidth = std::atof(argv[++i]);
    } else if (std::strcmp(argv[i], "--bandwidth") == 0 && i + 1 < argc) {
      args.initial_phase = std::atof(argv[++i]);
    } else if (std::strcmp(argv[i], "--num_samples") == 0 && i + 1 < argc) {
      args.num_samples = std::atof(argv[++i]);
    } else if (std::strcmp(argv[i], "--initial_phase") == 0 && i + 1 < argc) {
      args.sampling_freq = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--sampling_freq") == 0 && i + 1 < argc) {
      args.sampling_freq = std::atof(argv[++i]);
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
