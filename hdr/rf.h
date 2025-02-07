#ifndef RF_H
#define RF_H

#include <uhd.h>
#include <uhd/types/sensors.h>
#include <uhd/usrp/multi_usrp.hpp>

class rf_handler {
public:
  uhd::usrp::multi_usrp::sptr usrp = nullptr;
  const uhd::fs_path TREE_DBOARD_RX_FRONTEND_NAME =
      "/mboards/0/dboards/A/rx_frontends/A/name";
  const std::chrono::milliseconds FE_RX_RESET_SLEEP_TIME_MS =
      std::chrono::milliseconds(2000UL);
  uhd::stream_args_t stream_args = {};
  double lo_freq_tx_hz = 0.0;
  // double lo_freq_rx_hz = 0.0;
  double lo_freq_offset_hz = 0.0;

  rf_handler();
  ~rf_handler();

private:
};

#endif
