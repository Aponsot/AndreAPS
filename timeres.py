import numpy as np
import matplotlib.pyplot as plt
import argparse
import h5py

with h5py.File("argparse.h5", "r") as f:
    int_val = f["intensity_stack"][:]
    tth = f["tth"][:]
    q = f["q_values"][:] 

    plt.figure(figsize=(10, 6))
    plt.imshow(int_val, aspect='auto', cmap='jet', extent=(tth.min(), tth.max(), 0, int_val.shape[0]))
    plt.colorbar(label='Intensity')
    plt.xlabel('q (1/Å)')
    plt.ylabel('Frame Index') 
    plt.show()


if __name__ == "__main__":
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Polar integration of diffraction experiments.")
    parser.add_argument("h5_file", type=str, help="Path to the h5 file.")
                        

    args = parser.parse_args() 