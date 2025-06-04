# Jammer

A configurable signal jamming tool for testing and development purposes. This project provides a simple CLI-based interface to simulate jamming behavior, useful for controlled testing environments. 
The jammer generates random noise over the given central frequency and bandwidth in the config file. 

---

## Building the Jammer (Linux CLI)

Follow the steps below to build and run the jammer on a Linux-based command line interface.

### 1. Clone the Repository

```bash
git clone https://www.github.com/oran-testing/jammer.git
cd jammer
```

### 2. Build the Project

```bash
mkdir build
cd build
cmake ..
make -j
```

> **Note:** Ensure that `cmake` and `make` are installed on your system. You may also need a C++ compiler (e.g., `g++`).

---

## Running the Jammer

After successfully building the project, you can run the jammer using a configuration file.

```bash
./jammer --config ../configs/basic_jammer.yaml
```

### Configuration

- The jammer uses YAML configuration files to define parameters such as signal types, frequency ranges, timing, etc.
- You can modify the parameters in `configs/basic_jammer.yaml` to suit your specific use case.

---

## Directory Structure

```
jammer/
├── configs/
│   └── basic_jammer.yaml
├── src/
├── build/
└── README.md
```

---

## Requirements

- Linux-based OS
- CMake (version 3.10 or higher recommended)
- Make
- C++ compiler (GCC or Clang)

To install the necessary tools on Ubuntu/Debian:

```bash
sudo apt update
sudo apt install build-essential cmake
```

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contact

For questions or contributions, feel free to open an issue or submit a pull request.
