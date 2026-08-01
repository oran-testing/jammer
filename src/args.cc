#include "args.h"
#include <stdexcept>
#include <string>

#define REQUIRE_FIELD(node, key)                                               \
  if (!node[#key]) {                                                           \
    throw std::runtime_error("Missing required field in config: '" #key "'");  \
  }

attack_type_t parseAttackType(const std::string &type) {
  if (type == "barrage") {
    return attack_type_t::BARRAGE;
  }
  if (type == "constant") {
    return attack_type_t::CONSTANT;
  }
  if (type == "random") {
    return attack_type_t::RANDOM;
  }
  throw std::runtime_error("Unknown attack_type: '" + type +
                           "' (expected barrage, constant, or random)");
}

const char *attackTypeToString(attack_type_t type) {
  switch (type) {
  case attack_type_t::BARRAGE:
    return "barrage";
  case attack_type_t::CONSTANT:
    return "constant";
  case attack_type_t::RANDOM:
    return "random";
  }
  return "unknown";
}

all_args_t parseConfig(
    const std::string &filename) { // change the filename to the real filename

  YAML::Node config =
      YAML::LoadFile(filename); // change the filename to its real filename

  REQUIRE_FIELD(config, amplitude);
  REQUIRE_FIELD(config, num_samples);
  REQUIRE_FIELD(config, amplitude_width);
 // REQUIRE_FIELD(config, center_frequency);
  REQUIRE_FIELD(config, bandwidth);
  REQUIRE_FIELD(config, initial_phase);
 // REQUIRE_FIELD(config, sampling_freq);
  REQUIRE_FIELD(config, output_iq_file);
  REQUIRE_FIELD(config, output_csv_file);
  REQUIRE_FIELD(config, write_iq);
  REQUIRE_FIELD(config, write_csv);
  REQUIRE_FIELD(config, device_args);
 // REQUIRE_FIELD(config, tx_gain);

  all_args_t args; // an instance of struct
  args.amplitude        = config["amplitude"].as<float>();
  args.num_samples      = config["num_samples"].as<size_t>();
  args.amplitude_width  = config["amplitude_width"].as<float>();
 // args.center_frequency = config["center_frequency"].as<float>();
  args.bandwidth        = config["bandwidth"].as<float>();
  args.initial_phase    = config["initial_phase"].as<float>();
 // args.sampling_freq    = config["sampling_freq"].as<float>();
  args.output_iq_file   = config["output_iq_file"].as<std::string>();
  args.output_csv_file  = config["output_csv_file"].as<std::string>();
  args.write_iq         = config["write_iq"].as<bool>();
  args.write_csv        = config["write_csv"].as<bool>();
  args.rf.device_args   = config["device_args"].as<std::string>();
 // args.rf.tx_gain       = config["tx_gain"].as<float>();


  // These three are optional — autoconfig will supply them if enable_autoconfigure is true
  args.center_frequency = config["center_frequency"] ? config["center_frequency"].as<float>() : 0.0f;
  args.sampling_freq    = config["sampling_freq"]    ? config["sampling_freq"].as<float>()    : 0.0f;
  args.rf.tx_gain       = config["tx_gain"]          ? config["tx_gain"].as<float>()          : 0.0f;

  args.attack_type = attack_type_t::BARRAGE;
  if (config["attack_type"]) {
    args.attack_type = parseAttackType(config["attack_type"].as<std::string>());
  }

  args.tone_offset_hz = 0.0f;
  if (config["tone_offset_hz"]) {
    args.tone_offset_hz = config["tone_offset_hz"].as<float>();
  }

  args.burst_duration_ms = 50.0f;
  if (config["burst_duration_ms"]) {
    args.burst_duration_ms = config["burst_duration_ms"].as<float>();
  }

  args.idle_duration_ms = 50.0f;
  if (config["idle_duration_ms"]) {
    args.idle_duration_ms = config["idle_duration_ms"].as<float>();
  }

  args.jam_bandwidth_hz = 5.0e6f;
  if (config["jam_bandwidth_hz"]) {
    args.jam_bandwidth_hz = config["jam_bandwidth_hz"].as<float>();
  }

  args.num_tones = 64;
  if (config["num_tones"]) {
    args.num_tones = config["num_tones"].as<int>();
  }

  // autoconfig field
  args.enable_autoconfigure = false;
  if (config["enable_autoconfigure"]) {
	  args.enable_autoconfigure = config["enable_autoconfigure"].as<bool>();
  }
  if (config["database"]){
	YAML::Node db = config["database"];
	if (db["host"]) args.db.host = db["host"].as<std::string>();
	
	if (db["port"]) args.db.port = db["port"].as<uint32_t>();
	
	if (db["org"]) args.db.org = db["org"].as<std::string>();

	if (db["token"]) args.db.token = db["token"].as<std::string>();

	if (db["bucket"]) args.db.bucket = db["bucket"].as<std::string>();

	if (db["data_id"]) args.db.data_id = db["data_id"].as<std::string>();
  } 
  return args;
}

void overrideConfig(all_args_t &args, int argc, char *argv[]) {
  for (int i = 1; i < argc; ++i) {

    if (std::strcmp(argv[i], "--amplitude") == 0 && i + 1 < argc) {
      args.amplitude = std::atof(argv[++i]);
    } else if (std::strcmp(argv[i], "--amplitude_width") == 0 &&
               i + 1 < argc) {
      args.amplitude_width = std::atof(argv[++i]);
    } else if (std::strcmp(argv[i], "--center_frequency") == 0 &&
               i + 1 < argc) {
      args.center_frequency = std::atof(argv[++i]);
    } else if (std::strcmp(argv[i], "--bandwidth") == 0 && i + 1 < argc) {
      args.bandwidth = std::atof(argv[++i]);
    } else if (std::strcmp(argv[i], "--num_samples") == 0 && i + 1 < argc) {
      args.num_samples = std::atof(argv[++i]);
    } else if (std::strcmp(argv[i], "--initial_phase") == 0 && i + 1 < argc) {
      args.initial_phase = std::atof(argv[++i]);
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
	
    } else if (std::strcmp(argv[i], "--enable_autoconfigure") == 0 && i + 1 < argc) {
      args.enable_autoconfigure = (std::string(argv[++i]) == "true");
    } else if (std::strcmp(argv[i], "--attack_type") == 0 && i + 1 < argc) {
      args.attack_type = parseAttackType(argv[++i]);
    } else if (std::strcmp(argv[i], "--tone_offset_hz") == 0 && i + 1 < argc) {
      args.tone_offset_hz = std::atof(argv[++i]);
    } else if (std::strcmp(argv[i], "--burst_duration_ms") == 0 && i + 1 < argc) {
      args.burst_duration_ms = std::atof(argv[++i]);
    } else if (std::strcmp(argv[i], "--idle_duration_ms") == 0 && i + 1 < argc) {
      args.idle_duration_ms = std::atof(argv[++i]);
    } else if (std::strcmp(argv[i], "--jam_bandwidth_hz") == 0 && i + 1 < argc) {
      args.jam_bandwidth_hz = std::atof(argv[++i]);
    } else if (std::strcmp(argv[i], "--num_tones") == 0 && i + 1 < argc) {
      args.num_tones = std::atoi(argv[++i]);
    } else {
      std::cerr << "Unknown or incomplete option: " << argv[i] << std::endl;
    }
  }
}

void validateAttackArgs(const all_args_t &args) {
  if (args.attack_type != attack_type_t::RANDOM) {
    return;
  }

  if (args.burst_duration_ms <= 0.0f) {
    throw std::runtime_error(
        "burst_duration_ms must be > 0 for random jamming");
  }
  if (args.idle_duration_ms <= 0.0f) {
    throw std::runtime_error("idle_duration_ms must be > 0 for random jamming");
  }
}
