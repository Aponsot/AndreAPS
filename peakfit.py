


import numpy as np
import h5py
import argparse
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from lmfit.models import GaussianModel, PolynomialModel

def peak_fit(h5, frame_number, peak_pos, window=0.1):
    with h5py.File(h5, 'r') as f:
        int_val = f['int'][:]  # shape: (nframes, q)
        q = f['q'][:]          # shape: (q,)
        cake_intensity_stack = f['cake_int'][:]  # shape: (nframes, n_cakes, q)
       
    cake_slices = [0, 10, 19, 28]
    sigma = 50 # Smoothing parameter for background
    prominence = 2 # Minimum prominence for peak finding 
    full_prominence = 2 # Minimum prominence for full data peak finding
    
    #mask 
    q_min = peak_pos - window
    q_max = peak_pos + window
    mask = (q >= q_min) & (q <= q_max)
    q_limited = q[mask]
    # Prepare compound plot
    fig, axes = plt.subplots(
    2, 3,  # 2 rows, 3 columns
    figsize=(15, 10),  # Adjust the figure size as needed
    gridspec_kw={"width_ratios": [1, 1, 2]},  # Left columns are equal, right column is wider
)

    fig.suptitle(f'Peak Fit for Frame {frame_number}', fontsize=16)
    
    #Full Az 
    int_full_limited = int_val[frame_number, mask]
    background_full = gaussian_filter1d(int_full_limited, sigma=sigma)
    data_bg_sub_full = int_full_limited - background_full
    peaks_full, properties_full = signal.find_peaks(data_bg_sub_full, prominence= full_prominence, width=1 ) 
    prominence_peak = properties_full['prominences']
    widths = properties_full['widths'] 
    print(f"Prominence of peaks: {prominence_peak}, Widths of peaks: {widths}")
    print(f"peaks found: {q_limited[peaks_full][1]}")
    print(f"peak shape: {len(q_limited[peaks_full])}")
    num_peaks = int(len(q_limited[peaks_full]))
    peak_fits = [] 
    composite_model = PolynomialModel(degree=1, prefix="bg_")  # Background model
    params = composite_model.make_params(c0=10, c1=0.5)  # Initial guesses for background

# Dynamically add Gaussian models for each peak
    for i, peak_index in enumerate(peaks_full):
        peak_value = q_limited[peak_index]  # Peak location
        gaussian_model = GaussianModel(prefix=f"g{i}_")  # Unique prefix for each Gaussian

    # Add the Gaussian model to the composite model
        composite_model += gaussian_model

    # Initialize parameters for this Gaussian
        params.update(gaussian_model.make_params(
        center=peak_value,  # Peak location
        sigma=2,           # Initial width guess
        amplitude=int_full_limited[peak_index]  # Initial amplitude guess
    ))
    params[f"g{i}_center"].set(value=peak_value, min=peak_value - 0.05, max=peak_value + 0.05)
# Fit the composite model to the data
    result = composite_model.fit(int_full_limited, params, x=q_limited)

# Store the fit results for each peak
    peak_fits = []  # List to store fit results for each peak
    for i in range(len(peaks_full)):
        peak_fit = {
            "peak_index": i,
            "fit_result": result.best_fit,  # The fitted curve
            "fit_params": {key: result.params[key] for key in result.params if key.startswith(f"g{i}_")}  # Parameters for this peak
        }
        peak_fits.append(peak_fit)

    # Plot the fitted curve for this peak
    axes[0, 2].plot(q_limited, result.best_fit, label=f'Fitted Peak {i}', color='orange')

# Optionally, plot the background separately
    background_fit = result.eval_components()["bg_"]
    axes[0, 2].plot(q_limited, background_fit, label="Background", color="green")

# Add a legend to the plot
    axes[0, 2].legend()


   
    axes[0, 2].plot(q_limited, int_full_limited, label='Full Data', linestyle = '--', color='green')
    axes[0, 2].plot(q_limited, background_full, label='Background', linestyle = "-",color='blue') 
    axes[0, 2].plot(q_limited[peaks_full], data_bg_sub_full[peaks_full], 'x', label='Peaks')
    axes[0, 2].set_title(f'Full Azumutal Integration')
    axes[0, 2].set_xlabel('q')
    axes[0, 2].set_ylabel('Intensity')
    
    
    
    # Fit and plot for each cake slice
    for i, cs in enumerate(cake_slices):
        row = i // 2 
        col = i % 2
        cake_data = cake_intensity_stack[frame_number, cs, :]
        cake_data_limited = cake_data[mask]
        
        background = gaussian_filter1d(cake_data_limited, sigma=sigma)

        data_bg_sub = cake_data_limited - background

        peaks_cake, properties = signal.find_peaks(data_bg_sub, prominence=prominence) 
        axes[row, col].plot(q_limited, cake_data_limited, label='Cake Slice Data', linestyle='--', color='green')
        axes[row, col].plot(q_limited, background, label='Background', linestyle='-', color='red')
        axes[row, col].plot(q_limited[peaks_cake], data_bg_sub[peaks_cake], 'x', label='Peaks')
        axes[row, col].set_title(f'Cake Slice integration {cs}')    
        axes[row, col].set_xlabel('q')
        axes[row, col].set_ylabel('Intensity')

        print(f"Cake slice {cs}: Found peaks at {q_limited[peaks_cake]}")


    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit the title
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
