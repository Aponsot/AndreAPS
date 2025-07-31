import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import simpson

def load_hdf5_data(h5_file, frame, sigma=10, prominence=2):
    """Load data from an HDF5 file and apply Gaussian background subtraction."""
    with h5py.File(h5_file, "r") as f:
        int_val = f["int"][:]
        print(f'Number of frames in intensity stack: {int_val.shape[0]}')
       
        data = int_val[frame]

    # Estimate background using a Gaussian filter
    background = gaussian_filter1d(data, sigma=sigma)

    # Subtract background
    data_bg_sub = data - background

    # Find peaks on background-subtracted data
    peaks, properties = signal.find_peaks(data_bg_sub, prominence=prominence)
    
    print(f"Found {len(peaks)} peaks in frame {frame} after background subtraction.")
    print(f"Peak indices: {peaks}")

    # Plotting function
    def plot_function_peak_positions(data, peaks, title, background=None):
        plt.figure(figsize=(10, 6))
        plt.plot(data, label='Intensity Data')
        if background is not None:
            plt.plot(background, label='Estimated Background', color='orange', linestyle='--')
        for peak in peaks:
            plt.axvline(x=peak, color='r', linestyle='--', alpha=0.7)
        plt.plot(peaks, data[peaks], "x", label="Detected Peaks")
        plt.title(title)
        plt.xlabel('q (1/Å)')
        plt.ylabel('Intensity')
        plt.legend()
        plt.show()

    title = f'Scipy peak picking after Gaussian background subtraction (prominence={prominence:.2f})'
    plot_function_peak_positions(data_bg_sub, peaks, title, background=background)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Load and process HDF5 data.")
    parser.add_argument("h5_file", type=str, help="Path to the HDF5 file")
    parser.add_argument("Frame", type=int, help="Frame number to process")
    parser.add_argument("--sigma", type=float, default=1000, help="Sigma for Gaussian filter")
    parser.add_argument("--prominence", type=float, default=2, help="Prominence for peak detection")
    args = parser.parse_args()

    h5_file = args.h5_file
    frame = args.Frame
    sigma = args.sigma
    prominence = args.prominence

    load_hdf5_data(h5_file, frame, sigma=sigma, prominence=prominence)
