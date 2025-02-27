#include "rf.h"
#include <uhd/error.h>

rf_handler::rf_handler() {}

rf_handler::~rf_handler() {}

uhd_error rf_handler::usrp_make(const uhd::device_addr_t &dev_addr_,
                                uint32_t nof_channels) {
  uhd_error err = UHD_ERROR_NONE;
  uhd::device_addr_t dev_addr = dev_addr_;

  // Set transmitter subdevice spec string
  std::string tx_subdev;
  if (dev_addr.has_key("tx_subdev_spec")) {
    tx_subdev = dev_addr.pop("tx_subdev_spec");
  }

  // Set receiver subdevice spec string
  std::string rx_subdev;
  if (dev_addr.has_key("rx_subdev_spec")) {
    rx_subdev = dev_addr.pop("rx_subdev_spec");
  }

  // Set over the wire format
  std::string otw_format = "sc16";
  if (dev_addr.has_key("otw_format")) {
    otw_format = dev_addr.pop("otw_format");
  }

  // Samples-Per-Packet option, 0 means automatic
  std::string spp;
  if (dev_addr.has_key("spp")) {
    spp = dev_addr.pop("spp");
  }

  // Tx LO frequency
  if (dev_addr.has_key("lo_freq_tx_hz")) {
    lo_freq_tx_hz = dev_addr.cast("lo_freq_tx_hz", lo_freq_tx_hz);
    dev_addr.pop("lo_freq_tx_hz");
  }

  // Rx LO frequency
  if (dev_addr.has_key("lo_freq_rx_hz")) {
    lo_freq_rx_hz = dev_addr.cast("lo_freq_rx_hz", lo_freq_rx_hz);
    dev_addr.pop("lo_freq_rx_hz");
  }

  // LO Frequency offset automatic
  if (dev_addr.has_key("lo_freq_offset_hz")) {
    lo_freq_offset_hz = dev_addr.cast("lo_freq_offset_hz", lo_freq_offset_hz);
    dev_addr.pop("lo_freq_offset_hz");

    if (std::isnormal(lo_freq_tx_hz)) {
      std::cout << "'lo_freq_offset_hz' overrides 'lo_freq_tx_hz' ("
                << lo_freq_tx_hz / 1e6 << " MHz)\n";
    }

    if (std::isnormal(lo_freq_rx_hz)) {
      std::cout << "'lo_freq_offset_hz' overrides 'lo_freq_rx_hz' ("
                << lo_freq_rx_hz / 1e6 << " MHz)\n";
    }
  }

  // Make USRP
  usrp = uhd::usrp::multi_usrp::make(dev_addr);
  if (err != UHD_ERROR_NONE) {
    return err;
  }

  // Set transmitter subdev spec if specified
  if (not tx_subdev.empty()) {
    err = set_tx_subdev(tx_subdev);
    if (err != UHD_ERROR_NONE) {
      return err;
    }
  }

  // Set receiver subdev spec if specified
  if (not rx_subdev.empty()) {
    err = set_rx_subdev(rx_subdev);
    if (err != UHD_ERROR_NONE) {
      return err;
    }
  }

  // Initialize TX/RX stream args
  stream_args.cpu_format = "fc32";
  stream_args.otw_format = otw_format;
  if (not spp.empty()) {
    if (spp == "0") {
      std::cout
          << "The parameter spp is 0, some UHD versions do not handle it as "
             "default and receive method will overflow.\n";
    }
    stream_args.args.set("spp", spp);
  }
  stream_args.channels.resize(nof_channels);
  for (size_t i = 0; i < (size_t)nof_channels; i++) {
    stream_args.channels[i] = i;
  }

  // Try to get dboard name from property tree
  uhd::property_tree::sptr tree = usrp->get_device()->get_tree();
  if (tree == nullptr || not tree->exists(TREE_DBOARD_RX_FRONTEND_NAME)) {
    // Couldn't find dboard name in property tree
    return err;
  }

  std::string dboard_name =
      usrp->get_device()
          ->get_tree()
          ->access<std::string>(TREE_DBOARD_RX_FRONTEND_NAME)
          .get();

  // Detect if it a AD9361 based device
  if (dboard_name.find("FE-RX") != std::string::npos and false) {
    std::cout << "The device is based on AD9361, get RX stream for checking "
                 "LIBUSB_TRANSFER_ERROR\n";
    uint32_t ntrials = 10;
    do {
      // If no error getting RX stream, return
      err = test_ad936x_device(nof_channels);
      if (err == UHD_ERROR_NONE) {
        return err;
      }

      // Otherwise, close USRP and open again
      usrp = nullptr;

      std::cout << "Failed to open Rx stream, trying to open device again. "
                << ntrials << " trials left. Waiting for "
                << FE_RX_RESET_SLEEP_TIME_MS.count() << " ms\n";

      // Sleep
      // std::this_thread::sleep_for(FE_RX_RESET_SLEEP_TIME_MS);

      // Try once more making the device
      usrp = uhd::usrp::multi_usrp::make(dev_addr);

    } while (err == UHD_ERROR_NONE and --ntrials != 0);
  }

  return err;
}
