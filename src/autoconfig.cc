#include "autoconfig.h"
#include "rt-recon-sdk/autoconfig/influx_worker.h"
#include "rt-recon-sdk/autoconfig/logger.h"
#include <iostream>

bool applyInfluxAutoconfig(all_args_t& args)
{
  logger_init(1);
  logger_set_level(LOG_INFO);
  rtrs::DatabaseConfig db_cfg;
  db_cfg.host = args.db.host;
  db_cfg.port = args.db.port;
  db_cfg.org = args.db.org;
  db_cfg.token = args.db.token;
  db_cfg.bucket = args.db.bucket;
  db_cfg.data_id = args.db.data_id;

  rtrs::InfluxWorker influx(db_cfg);


  // tx_gain, tx_red, offsets
  rtrs::ChannelConfig ch;
  if (!influx.pull_msg(ch))
  {
    std::cerr << "[autoconfig] Failed to pull channelConfig from InfluxDB\n";
    return false;
  }


  // frq, sample_rate. scs
  rtrs::recon_band_report_t band;
  if(!influx.pull_msg(band))
  {
    std::cerr << "[autoconfig] Failed to pull band report from InfluxDB\n";
    return false;
  }


  // map to jammer args
  args.center_frequency = static_cast<float>(band.ssb_freq);
  args.sampling_freq = static_cast<float>(band.sample_rate);
  args.rf.tx_gain = static_cast<float>(ch.tx_gain);


  std::cout << "[autoconfig] center_frequency : " << args.center_frequency << "\n";
  
  std::cout << "[autoconfig] sampling_frequency : " << args.sampling_freq << "\n";

  std::cout << "[autoconfig] tx_gain : " << args.rf.tx_gain << "\n";
  return true;
}

