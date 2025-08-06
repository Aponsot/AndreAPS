
import numpy as np
import h5py
import argparse
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d


import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy import signal

def peakrip(h5, peak_pos, window=0.1):
    with h5py.File(h5, 'r') as f:
        int_val = f['int'][:]  # shape (frames, q)
        q = f['q'][:]          # shape (q,)
        frames = int_val.shape[0]
    
    print(f'int_val shape: {int_val.shape}') 
    
    # Initialize lists to store frame numbers and peak positions
    frame_numbers = []
    all_peak_positions = []

    for cs in range(frames):
        prominence = 5
        sigma = 50
        
        # Extract the intensity data for the current frame
        int_val_frame = int_val[cs, :]
        print(f'int_val_frame shape: {int_val_frame.shape}') 
        
        # Apply the window to limit the q range
        q_min = peak_pos - window
        q_max = peak_pos + window
        mask = (q >= q_min) & (q <= q_max)
        q_limited = q[mask] 
        int_val_mask = int_val_frame[mask]   
        
        # Subtract the background
        background_full = gaussian_filter1d(int_val_mask, sigma=sigma)
        data_bg_sub_full = int_val_mask - background_full
        
        # Find peaks in the background-subtracted data
        peaks_full, properties_full = signal.find_peaks(data_bg_sub_full, prominence=prominence)
        
        # Append the frame number and the actual peak positions in q
        for peak in peaks_full:
            frame_numbers.append(cs)
            all_peak_positions.append(q_limited[peak])

    # Plot the peak positions versus frame numbers
    plt.figure(figsize=(8, 5))
    plt.scatter(frame_numbers, all_peak_positions, color='blue', label='Peak Positions')
    plt.xlabel("Frame Number")
    plt.ylabel("Peak Positions (q)")
    plt.title(f"Peak Positions vs Frame Number (q = {peak_pos} ± {window})")
    plt.legend()
    plt.grid(True)
    plt.show()
if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Fit peak function")
        parser.add_argument("h5", type=str, help="Input h5 file containing processed data.") 
        parser.add_argument("--peak_pos", type=float, default=None, help="Position of the peak to fit (in q units). If not provided, peaks will be automatically detected.")
        args = parser.parse_args()
 
        h5 = args.h5
        peak_pos = args.peak_pos

peakrip(h5, peak_pos)
