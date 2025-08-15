import argparse 
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt


def graph(h5):
    with h5py.File(h5, "r") as f:
        q = f["q"][:]                       
        int_val = f["int"][:]
        
        plt.figure(figsize=(10, 6))
        plt.imshow(int_val, aspect='auto', extent=[q.min(), q.max(), 0, int_val[:,]],
                   origin='lower', cmap='jet')
        plt.colorbar(label='Intensity')
        plt.xlabel('q (1/Å)', fontsize = 20)
        plt.ylabel('Frame Index', fontsize = 20)
        plt.show()

def _parse_args():
    p = argparse.ArgumentParser(
        description="Time Resolved plot"
    )
    p.add_argument("h5", type=str)
    p.add_argument("frame_number", type=int)
if __name__ == "__main__":
    args = _parse_args()
    graph(args.h5)
