#ifndef RF_H
#define RF_H

#include <uhd.h>
#include <uhd/error.h>
#include <uhd/types/sensors.h>
#include <uhd/usrp/multi_usrp.hpp>
#include <uhd/usrp/usrp.h>

class rf_handler {
public:
  uhd::usrp::multi_usrp::sptr usrp = nullptr;
  uhd::stream_args_t stream_args = {};
  double lo_freq_tx_hz = 0.0;
  double lo_freq_rx_hz = 0.0;
  double lo_freq_offset_hz = 0.0;

  uhd::rx_streamer::sptr rx_stream = nullptr;
  uhd::tx_streamer::sptr tx_stream = nullptr;

  rf_handler();
  ~rf_handler();

  uhd_error usrp_make(const uhd::device_addr_t &dev_addr_,
                      uint32_t nof_channels);

  uhd_error get_mboard_name(std::string &mboard_name) {
    mboard_name = usrp->get_mboard_name();
    return UHD_ERROR_NONE;
  }
  uhd_error get_mboard_sensor_names(std::vector<std::string> &sensors) {
    sensors = usrp->get_mboard_sensor_names();
    return UHD_ERROR_NONE;
  }
  uhd_error get_rx_sensor_names(std::vector<std::string> &sensors) {
    sensors = usrp->get_rx_sensor_names();
    return UHD_ERROR_NONE;
  }
  uhd_error get_sensor(const std::string &sensor_name, double &sensor_value) {
    sensor_value = usrp->get_mboard_sensor(sensor_name).to_real();
    return UHD_ERROR_NONE;
  }
  uhd_error get_sensor(const std::string &sensor_name, bool &sensor_value) {
    sensor_value = usrp->get_mboard_sensor(sensor_name).to_bool();
    return UHD_ERROR_NONE;
  }
  uhd_error get_rx_sensor(const std::string &sensor_name, bool &sensor_value) {
    sensor_value = usrp->get_rx_sensor(sensor_name).to_bool();
    return UHD_ERROR_NONE;
  }
  uhd_error set_time_unknown_pps(const uhd::time_spec_t &timespec) {
    std::cout << "Setting Time at next PPS..." << "\n";
    usrp->set_time_unknown_pps(timespec);
    return UHD_ERROR_NONE;
  }
  uhd_error get_time_now(uhd::time_spec_t &timespec) {
    timespec = usrp->get_time_now();
    return UHD_ERROR_NONE;
  }
  uhd_error set_sync_source(const std::string &sync_source,
                            const std::string &clock_source) {
    std::cout << "Setting PPS source to '" << sync_source
              << "' and clock source to '" << clock_source << "'\n";
#if UHD_VERSION < 3140099
    usrp->set_clock_source(clock_source; usrp->set_time_source(sync_source);
#else
    usrp->set_sync_source(clock_source, sync_source);
#endif
    return UHD_ERROR_NONE;
  }
  uhd_error get_gain_range(uhd::gain_range_t &tx_gain_range,
                           uhd::gain_range_t &rx_gain_range) {
    tx_gain_range = usrp->get_tx_gain_range();
    rx_gain_range = usrp->get_rx_gain_range();

    return UHD_ERROR_NONE;
  }
  uhd_error set_master_clock_rate(double rate) {
    std::cout << "Setting master clock rate to " << rate / 1e6 << " MHz"
              << "\n";
    usrp->set_master_clock_rate(rate);
    return UHD_ERROR_NONE;
  }
  uhd_error set_rx_rate(double rate) {
    std::cout << "Setting Rx Rate to " << rate / 1e6 << "MHz" << "\n";
    usrp->set_rx_rate(rate);
    return UHD_ERROR_NONE;
  }
  uhd_error set_tx_rate(double rate) {
    std::cout << "Setting Tx Rate to " << rate / 1e6 << "MHz" << "\n";
    usrp->set_tx_rate(rate);
    return UHD_ERROR_NONE;
  }
  uhd_error set_command_time(const uhd::time_spec_t &timespec) {
    usrp->set_command_time(timespec);
    return UHD_ERROR_NONE;
  }
  uhd_error get_rx_stream(size_t &max_num_samps) {
    std::cout << "Creating Rx stream" << "\n";
    rx_stream = nullptr;
    rx_stream = usrp->get_rx_stream(stream_args);
    max_num_samps = rx_stream->get_max_num_samps();
    if (max_num_samps == 0UL) {
      std::cerr << "The maximum number of receive samples is zero." << "\n";
      return UHD_ERROR_VALUE;
    }
    return UHD_ERROR_NONE;
  }
  uhd_error get_tx_stream(size_t &max_num_samps) {
    std::cout << "Creating Tx stream" << "\n";
    tx_stream = nullptr;
    tx_stream = usrp->get_tx_stream(stream_args);
    max_num_samps = tx_stream->get_max_num_samps();
    if (max_num_samps == 0UL) {
      std::cerr << "The maximum number of transmit samples is zero."
                << "\n";
      return UHD_ERROR_VALUE;
    }
    return UHD_ERROR_NONE;
  }
  uhd_error set_tx_gain(size_t ch, double gain) {
    std::cout << "Setting channel " << ch << " Tx gain to " << gain << " dB"
              << "\n";
    usrp->set_tx_gain(gain, ch);
    return UHD_ERROR_NONE;
  }
  uhd_error set_rx_gain(size_t ch, double gain) {
    std::cout << "Setting channel " << ch << " Rx gain to " << gain << " dB"
              << "\n";
    usrp->set_rx_gain(gain, ch);
    return UHD_ERROR_NONE;
  }
  uhd_error get_rx_gain(double &gain) {
    gain = usrp->get_rx_gain();
    return UHD_ERROR_NONE;
  }
  uhd_error get_tx_gain(double &gain) {
    gain = usrp->get_tx_gain();
    return UHD_ERROR_NONE;
  }
  uhd_error set_tx_freq(uint32_t ch, double target_freq, double &actual_freq) {
    std::cout << "Setting channel " << ch << " Tx frequency to "
              << target_freq / 1e6 << " MHz\n";

    // Create Tune request
    uhd::tune_request_t tune_request(target_freq);

    if (std::isnormal(lo_freq_offset_hz)) {
      lo_freq_tx_hz = target_freq + lo_freq_offset_hz;
    }

    // If the LO frequency is defined, force a LO frequency and use the
    if (std::isnormal(lo_freq_tx_hz)) {
      tune_request.rf_freq = lo_freq_tx_hz;
      tune_request.rf_freq_policy = uhd::tune_request_t::POLICY_MANUAL;
      tune_request.dsp_freq_policy = uhd::tune_request_t::POLICY_AUTO;
    }

    uhd::tune_result_t tune_result = usrp->set_tx_freq(tune_request, ch);
    actual_freq = tune_result.target_rf_freq;
    return UHD_ERROR_NONE;
  }
  uhd_error set_rx_freq(uint32_t ch, double target_freq, double &actual_freq) {
    std::cout << "Setting channel " << ch << " Rx frequency to "
              << target_freq / 1e6 << " MHz\n";

    // Create Tune request
    uhd::tune_request_t tune_request(target_freq);

    if (std::isnormal(lo_freq_offset_hz)) {
      lo_freq_rx_hz = target_freq + lo_freq_offset_hz;
    }

    // If the LO frequency is defined, force a LO frequency and use the
    if (std::isnormal(lo_freq_rx_hz)) {
      tune_request.rf_freq = lo_freq_rx_hz;
      tune_request.rf_freq_policy = uhd::tune_request_t::POLICY_MANUAL;
      tune_request.dsp_freq_policy = uhd::tune_request_t::POLICY_AUTO;
    }

    uhd::tune_result_t tune_result = usrp->set_rx_freq(tune_request, ch);
    actual_freq = tune_result.target_rf_freq;

    return UHD_ERROR_NONE;
  }

  uhd_error stop_rx_stream() {
    std::cout << "Stopping Rx stream\n";
    uhd::stream_cmd_t stream_cmd(
        uhd::stream_cmd_t::STREAM_MODE_STOP_CONTINUOUS);
    stream_cmd.stream_now = true;
    rx_stream->issue_stream_cmd(stream_cmd);
    return UHD_ERROR_NONE;
  }

  uhd_error receive(void **buffs, const size_t nsamps_per_buff,
                    uhd::rx_metadata_t &metadata, const double timeout,
                    const bool one_packet, size_t &nof_rxd_samples) {
    uhd::rx_streamer::buffs_type buffs_cpp(buffs,
                                           rx_stream->get_num_channels());
    nof_rxd_samples = rx_stream->recv(buffs_cpp, nsamps_per_buff, metadata,
                                      timeout, one_packet);
    return UHD_ERROR_NONE;
  }

  uhd_error start_rx_stream(double delay) {
    std::cout << "Starting Rx stream\n";
    uhd::time_spec_t time_spec;
    uhd_error err = get_time_now(time_spec);
    if (err != UHD_ERROR_NONE) {
      return err;
    }

    uhd::stream_cmd_t stream_cmd(
        uhd::stream_cmd_t::STREAM_MODE_START_CONTINUOUS);
    stream_cmd.time_spec = time_spec;
    stream_cmd.time_spec += delay;
    stream_cmd.stream_now = not std::isnormal(delay);

    rx_stream->issue_stream_cmd(stream_cmd);
    return UHD_ERROR_NONE;
  }

private:
  const uhd::fs_path TREE_DBOARD_RX_FRONTEND_NAME =
      "/mboards/0/dboards/A/rx_frontends/A/name";
  const std::chrono::milliseconds FE_RX_RESET_SLEEP_TIME_MS =
      std::chrono::milliseconds(2000UL);

  uhd_error set_tx_subdev(const std::string &string) {
    std::cout << "Setting tx_subdev_spec to '" << string << "'\n";
    usrp->set_tx_subdev_spec(string);
    return UHD_ERROR_NONE;
  }
  uhd_error set_rx_subdev(const std::string &string) {
    std::cout << "Setting rx_subdev_spec to '" << string << "'\n";
    usrp->set_rx_subdev_spec(string);
    return UHD_ERROR_NONE;
  }

  uhd_error test_ad936x_device(uint32_t nof_channels) {
    uhd_error err = set_rx_rate(1.92e6);
    if (err != UHD_ERROR_NONE) {
      return err;
    }

    size_t max_samp = 0;
    err = get_rx_stream(max_samp);
    if (err != UHD_ERROR_NONE) {
      return err;
    }

    // Allocate buffers
    std::vector<float> data(max_samp * 2);
    std::vector<void *> buf(nof_channels);
    for (auto &b : buf) {
      b = data.data();
    }

    uhd::rx_metadata_t md = {};
    size_t nof_rxd_samples = 0;

    // If no error getting RX stream, try to receive once
    err = start_rx_stream(0.1);
    if (err != UHD_ERROR_NONE) {
      return err;
    }

    // Flush Stream
    do {
      err = receive(buf.data(), max_samp, md, 0.0f, false, nof_rxd_samples);
      if (err != UHD_ERROR_NONE) {
        return err;
      }
    } while (md.error_code != uhd::rx_metadata_t::ERROR_CODE_TIMEOUT);

    // Receive
    err = receive(buf.data(), max_samp, md, 2.0f, false, nof_rxd_samples);

    if (err != UHD_ERROR_NONE) {
      return err;
    }

    if (md.error_code == uhd::rx_metadata_t::ERROR_CODE_TIMEOUT) {
      std::cerr << md.strerror();
      return UHD_ERROR_IO;
    }

    // Stop stream
    err = stop_rx_stream();
    if (err != UHD_ERROR_NONE) {
      return err;
    }

    // Flush Stream
    do {
      err = receive(buf.data(), max_samp, md, 0.0f, false, nof_rxd_samples);
      if (err != UHD_ERROR_NONE) {
        return err;
      }
    } while (md.error_code != uhd::rx_metadata_t::ERROR_CODE_TIMEOUT);

    return err;
  }
};

#endif
