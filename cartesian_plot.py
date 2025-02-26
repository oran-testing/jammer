import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Plot real and imaginary components from a CSV file.')
parser.add_argument('csv_file', type=str, help='Path to the CSV file')
parser.add_argument('--num_lines', type=int, default=None, help='Number of lines to read from the CSV file')
args = parser.parse_args()

# Read data from CSV
data = pd.read_csv(args.csv_file, nrows=args.num_lines)
index = data.iloc[:, 0].values
real = data.iloc[:, 1].values
imag = data.iloc[:, 2].values

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(index, real, label='Real Part', marker='o', linestyle='-', color='b')
plt.plot(index, imag, label='Imaginary Part', marker='s', linestyle='--', color='r')

# Labels and Title
plt.xlabel('Index')
plt.ylabel('Amplitude')
plt.title('Real and Imaginary Components as Waves')
plt.legend()
plt.grid(True)

# Show plot
plt.show()

