import numpy as np
import h5py
import argparse
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d


def peakrip(h5, peak_pos, window=0.1):
        with h5py.File(h5, 'r') as f:
            int_val = f['int'][:]  
            q = f['q'][:]  
        peak_positions = []
    
        for i in enumerate(range(frames)):
            prominence = 5
            sigma = 50
           
            int_val_frame = int_val_limited[i,:]
            q_min = peak_pos - window
            q_max = peak_pos + window
            mask = (q >= q_min) & (q <= q_max)
            q_limited = q[mask]
            int_val_limited = int_val_frame[mask]       
            frames = int_val.shape[0]
        # Fit a Gaussian model to the peak
            int_full_limited = int_val[frames, mask]
            background_full = gaussian_filter1d(int_full_limited, sigma=sigma)
            data_bg_sub_full = int_full_limited - background_full
            peaks_full, properties_full = signal.find_peaks(data_bg_sub_full, prominence= prominence) 
            peak_positions.append(peaks_full)  
        

            plt.figure(figsize=(8, 5))
            plt.scatter(peak_positions, frames, label="Intensity vs q")
            plt.xlabel("peak positions", q_limited)
            plt.ylabel("Frame Number")
            plt.title(f"Peak Positions for Frame {frames} in window q = {peak_pos} +/- {window}")
            plt.legend()
            plt.show()
    
if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Fit peak function")
        parser.add_argument("h5", type=str, help="Input h5 file containing processed data.") 
        parser.add_argument("--peak_pos", type=float, default=None, help="Position of the peak to fit (in q units). If not provided, peaks will be automatically detected.")
        args = parser.parse_args()
 
        h5 = args.h5
        peak_pos = args.peak_pos

peakrip(h5, peak_pos)
