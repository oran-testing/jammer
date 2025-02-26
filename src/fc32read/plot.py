import matplotlib.pyplot as plt
import sys
import numpy as np
import pysdr 

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_iq.py <input_file.fc32>")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    # Read binary file as float32
    data = np.fromfile(filename, dtype=np.float32) #Change the Filename to the real name of the file
    
    # Each sample consists of a pair: [real, imag]
    
    if len(data) % 2 != 0:
        print("Error: Data length is not even. Check the file format.")
        sys.exit(1)
    data = data.reshape(-1, 2)
    
    # Form complex numbers
    iq_samples = data[:, 0] + 1j * data[:, 1]
    
    # Plot IQ constellation using pysdr (assuming pysdr has a suitable plotting function)
    # For example, if pysdr provides plot_constellation, you can use:
    plt.scatter(np.real(iq_samples), np.imag(iq_samples), s=1)
    plt.xlabel("Real")
    plt.ylabel("Imag")
    plt.title("IQ Constellation")
    plt.show()
    
if __name__ == '__main__':
    main()
