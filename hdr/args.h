#ifndef ARGS_H
#define ARGS_H

#include <cstring>
#include <iostream>
#include <yaml-cpp/yaml.h>
#include <string>

typedef struct rf_args_s {
  std::string device_args;
  float tx_gain;
} rf_args_t;


//influx_db

typedef struct db_args_s {
  std::string host = "127.0.0.1";
  uint32_t    port = 8086;
  std::string org  = "";
  std::string token = "";
  std::string bucket = "";
  std::string data_id = "";
} db_args_t;

enum class attack_type_t { BARRAGE, CONSTANT, RANDOM };

attack_type_t parseAttackType(const std::string &type);
const char *attackTypeToString(attack_type_t type);

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

  attack_type_t attack_type = attack_type_t::BARRAGE;
  float tone_offset_hz = 0.0f;
  float burst_duration_ms = 50.0f;
  float idle_duration_ms = 50.0f;

  // Constant (partial-band noise) jamming controls
  float jam_bandwidth_hz = 5.0e6f; // width of narrowband noise around tone_offset_hz
  int   num_tones        = 64;    // tones used to synthesize the noise comb

  //autoconfig control
  bool enable_autoconfigure = false;
  db_args_t db;
} all_args_t;

all_args_t parseConfig(const std::string &filename);

void overrideConfig(all_args_t &args, int argc, char *argv[]);

void validateAttackArgs(const all_args_t &args);

#endif // !ARGS_H
