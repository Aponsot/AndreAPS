import numpy as np
import h5py
import argparse
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d

def peak_fit(h5, frame_number, peak_pos, window=0.1):
    with h5py.File(h5, 'r') as f:
        int_val = f['int'][:]  # shape: (nframes, q)
        q = f['q'][:]          # shape: (q,)
        cake_intensity_stack = f['cake_int'][:]  # shape: (nframes, n_cakes, q)
       
    cake_slices = [0, 10, 19, 28]
    sigma = 1.0  # Smoothing parameter for background
    prominence = 0.1  # Minimum prominence for peak finding 
    full_prominence = 0.1  # Minimum prominence for full data peak finding
    # Prepare compound plot
    fig, axes = plt.subplots(len(cake_slices), 2, figsize=(10, 3 * len(cake_slices)))

    # Fit and plot for each cake slice
    for i, cs in enumerate(cake_slices):
        cake_data = cake_intensity_stack[frame_number, cs, :]
        q_min = peak_pos - window
        q_max = peak_pos + window
        mask = (q >= q_min) & (q <= q_max)
        q_limited = q[mask]
        cake_data_limited = cake_data[mask]
        
        background = gaussian_filter1d(cake_data_limited, sigma=sigma)

        data_bg_sub = cake_data_limited - background

        peaks_cake, properties = signal.find_peaks(data_bg_sub, prominence=prominence) 
        axes[i, 0].plot(q_limited, cake_data_limited, label='Cake Slice Data', linestyle='b-')
        axes[i, 0].plot(q_limited, background, label='Background', linestyle='r-')
        axes[i, 0].plot(q_limited[peaks_cake], data_bg_sub[peaks_cake], 'x', label='Peaks')
        axes[i, 0].set_title(f'Cake Slice {cs}')    
        axes[i, 0].set_xlabel('q')
        axes[i, 0].set_ylabel('Intensity')

        print(f"Cake slice {cs}: Found peaks at {q_limited[peaks_cake]}")

        int_full_limited = int_val[frame_number, mask]

        data_bg_sub_full = int_full_limited - background 
        peaks_full, properties_full = signal.find_peaks(data_bg_sub_full, prominence= full_prominence) 
        axes[i, 1].plot(q_limited, int_full_limited, label='Full Data')
        axes[i, 1].plot(q_limited, background, label='Background', linestyle = "r-") 
        axes[i, 1].plot(q_limited[peaks_full], data_bg_sub_full[peaks_full], 'x', label='Peaks')
        axes[i, 1].set_title(f'Full Data Cake Slice {cs}')
        axes[i, 1].set_xlabel('q')
        axes[i, 1].set_ylabel('Intensity')
    plt.tight_layout()
    plt.show()

     

if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Fit peak function")
        parser.add_argument("h5", type=str, help="Input h5 file containing processed data.") 
        parser.add_argument("frame_number", type=int, help="Frame number to process (0-indexed).")
        parser.add_argument("peak_pos", type=float, help="Position of the peak to fit (in q units). If not provided, peaks will be automatically detected.")
        parser.add_argument("--window", type=float, default=0.1, help="Window size around the peak position for peak finding.")
        args = parser.parse_args()
 
        h5 = args.h5
        frame_number = args.frame_number
        peak_pos = args.peak_pos
        window = args.window

peak_fit(h5, frame_number, peak_pos, window)
