import h5py
import numpy as np
import argparse

import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser(description="Track peak movement for 6 datasets.")
    ap.add_argument("--h5", nargs=7, required=True, help="6 HDF5 files with 'q' (or 'tth') and 'int'")
    args = ap.parse_args()
    h5_files = args.h5

    plt.figure(figsize=(10, 6))

    # Store each dataset's normalized intensity separately
    all_norm_intensities = []

    for fname in h5_files:
        with h5py.File(fname, 'r') as f:
            data = f["int"][:]
            frame_intensity = np.sum(data, axis=tuple(range(1, data.ndim)))
            norm_intensity = frame_intensity
            all_norm_intensities.append(norm_intensity)
            plt.plot(norm_intensity, label=fname)

    plt.xlabel('Frame')
    plt.ylabel('Normalized Signal Intensity (%)')
    plt.title('Signal Intensity Over Experiment')
    plt.xlim(0,1000)
    plt.legend([f"Beam Index {i}" for i in range(len(h5_files))])
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
