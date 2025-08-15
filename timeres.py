import argparse 
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt


def graph(h5):
    with h5py.File(h5, "r") as f:
        q = f["q"][:]                       
        int_val = f["int"][:]
        frames = int_val.shape[0]
        plt.figure(figsize=(10, 6))
        plt.imshow(int_val, aspect='auto', extent=[2, 7, 0, frames],
                   origin='lower', cmap='jet')
        plt.colorbar(label='Intensity')
        plt.xlabel('q (1/Å)', fontsize = 20)
        plt.ylabel('Frame Index', fontsize = 20)
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)
        plt.show()


if __name__ == "__main__":
    # Command-line argument parsing
    
    p = argparse.ArgumentParser(
        description="Time Resolved plot"
    )
    p.add_argument("h5", type=str)
    args = p.parse_args()
    h5 = args.h5
    graph(h5)
