import numpy as np
import h5py
import matplotlib.pyplot as plt
import hexrd
import hexrd.fitting.fitpeak as fitpeak
import scipy.signal as signal
from scipy.ndimage import gaussian_filter1d
def load_hdf5_data(h5_file,frame):
    """Load data from an HDF5 file and apply Gaussian background subtraction."""
    with h5py.File(h5_file, "r") as f:
        int_val = f["int"][:]
        print(f'Number of frames in intensity stack: {int_val.shape[0]}')
        data = int_val[frame]

    # Estimate background using a Gaussian filter
    sigma = 10  # Adjust sigma as needed for your data
    background = gaussian_filter1d(data, sigma=sigma)

    # Subtract background
    data_bg_sub = data - background

    # Plotting function
    def plot_function_peak_positions(data, peaks, title):
        plt.figure(figsize=(10, 6))
        plt.plot(data, label='Intensity Data')
        for peak in peaks:
            plt.axvline(x=peak, color='r', linestyle='--', alpha=0.7)
        plt.plot(peaks, data[peaks], "x", label="Detected Peaks")
        plt.title(title)
        plt.xlabel('q (1/Å)')
        plt.ylabel('Intensity')
        plt.legend()
        plt.show()

    # Find peaks on background-subtracted data
    prominence = 0.1
    peaks = signal.find_peaks(data_bg_sub, prominence=prominence)[0]
    print(f"Found {len(peaks)} peaks in frame {frame} after background subtraction.")
    print(f"Peak indices: {peaks}")

    title = f'Scipy peak picking after Gaussian background subtraction (prominence={prominence:.2f})'
    plot_function_peak_positions(data_bg_sub, peaks, title)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Load and process HDF5 data.")
    parser.add_argument("h5_file", type=str, help="Path to the HDF5 file")
    parser.add_argument("Frame", type=int, help="Frame number to process")
    args = parser.parse_args()
    
    h5_file = args.h5_file
    frame = args.Frame

   
    
load_hdf5_data(h5_file, frame)
   
