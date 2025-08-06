
import numpy as np
import h5py
import argparse
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d


def peakrip(h5, peak_pos, window=0.1):
        with h5py.File(h5, 'r') as f:
            int_val = f['int'][:]  # shape (frames,q) 
            q = f['q'][:]  
            frames = int_val.shape[0]
        peak_positions = []
        print(f'int_val shape {int_val.shape}') 
        for i ,cs in enumerate(range(frames)):
            prominence = 5
            sigma = 50
           
            int_val_frame = int_val[cs,:]
            print(f'int_val_frame shapp {int_val_frame.shape}') 
            q_min = peak_pos - window
            q_max = peak_pos + window
            mask = (q >= q_min) & (q <= q_max)
            q_limited = q[mask] 
            int_val_mask = int_val_frame[mask]   
      
        # Fit a Gaussian model to the peak
          
            background_full = gaussian_filter1d(int_val_mask, sigma=sigma)
            data_bg_sub_full = int_val_mask - background_full
            peaks_full, properties_full = signal.find_peaks(data_bg_sub_full, prominence= prominence) 
            peak_positions.append(cs,peaks_full)  

            frame_number = peak_positions[-1][0]
            peak_indices = peak_positions[-1][1]
            plt.figure(figsize=(8, 5))
            plt.scatter(frame_number, peak_indices, label=f"Frame {frame_number}, Peak Positions", color='blue')
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
