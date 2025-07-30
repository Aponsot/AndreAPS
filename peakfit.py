import numpy as np
import h5py
import matplotlib.pyplot as plt
import hexrd
import hexrd.fitting.fitpeak as fitpeak
import scipy.signal as signal

def load_hdf5_data(h5_file,frame):
    """Load data from an HDF5 file."""
    with h5py.File(h5_file, "r") as f:
        int_val = f["int"][:]
        print(f'Number of frames in intensity stack: {int_val.shape[0]}')
        data = int_val[frame]
   
    prominence = 0.05
    peaks = signal.find_peaks(data, prominence=prominence)[0]
    print(f"Found {len(peaks)} peaks in frame {frame}.")
    print(f"Peak indices: {peaks}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(data, label='Intensity Data')
    plt.overlay_peaks = fitpeak.OverlayPeaks(peaks, data, prominence=prominence)
    plt.plot(peaks, data[peaks], "x", label="Detected Peaks")
    plt.title(f"Peak Detection in Frame {frame}")
    plt.xlabel('q (1/Å)')
    plt.ylabel('Intensity')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Load and process HDF5 data.")
    parser.add_argument("h5_file", type=str, help="Path to the HDF5 file")
    parser.add_argument("Frame", type=int, help="Frame number to process")
    args = parser.parse_args()
    
    h5_file = args.h5_file
    frame = args.Frame
    
load_hdf5_data(h5_file, frame)
   
