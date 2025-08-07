


import numpy as np
import h5py
import argparse
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from lmfit.models import GaussianModel, PolynomialModel

def peak_fit(h5, frame_number, peak_pos, window=0.1):
    import h5py
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks
    from lmfit.models import GaussianModel, PolynomialModel

    with h5py.File(h5, 'r') as f:
        int_val = f['int'][:]  # shape: (nframes, q)
        q = f['q'][:]          # shape: (q,)
        cake_intensity_stack = f['cake_int'][:]  # shape: (nframes, n_cakes, q)

    cake_slices = [0, 10, 19, 28]
    sigma = 50  # Smoothing parameter for background
    prominence = 2  # Minimum prominence for peak finding
    full_prominence = 2  # Minimum prominence for full data peak finding

    # Mask
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

    # Full Azimuthal Integration
    int_full_limited = int_val[frame_number, mask]
    background_full = gaussian_filter1d(int_full_limited, sigma=sigma)
    data_bg_sub_full = int_full_limited - background_full
    peaks_full, properties_full = find_peaks(data_bg_sub_full, prominence=full_prominence, width=1)
    prominence_peak = properties_full['prominences']
    widths = properties_full['widths']
    print(f"Prominence of peaks: {prominence_peak}, Widths of peaks: {widths}")
    print(f"peaks found: {q_limited[peaks_full]}")
    print(f"peak shape: {len(q_limited[peaks_full])}")

    # Initialize storage for peak fits
    peak_fits = []

    # Fit each peak individually within a narrow range
    window_size = 0.05  # Narrow window around each peak (adjust as needed)
    for i, peak_index in enumerate(peaks_full):
        # Define a narrow range around the current peak
        peak_value = q_limited[peak_index]
        q_min_peak = peak_value - window_size
        q_max_peak = peak_value + window_size
        mask_peak = (q_limited >= q_min_peak) & (q_limited <= q_max_peak)
        q_fit = q_limited[mask_peak]
        int_fit = int_full_limited[mask_peak]

        # Create a Gaussian model for this peak
        gaussian_model = GaussianModel()
        width_guess = widths[i] / 2.355  # Convert FWHM to sigma
        amplitude_guess = int_full_limited[peak_index]

        # Initialize parameters for the Gaussian
        params = gaussian_model.make_params(
            center=peak_value,
            sigma=width_guess,
            amplitude=amplitude_guess
        )

        # Fit the Gaussian model to the data in the narrow range
        result = gaussian_model.fit(int_fit, params, x=q_fit)

        # Store the fit results for this peak
        peak_fit = {
            "peak_index": i,
            "fit_result": result.best_fit,  # The fitted curve
            "fit_params": result.params     # The optimized parameters
        }
        peak_fits.append(peak_fit)

        # Plot the fitted curve for this peak
        axes[0, 2].plot(q_fit, result.best_fit, label=f'Fitted Peak {i}', color='orange')

        # Print the fit parameters for debugging
        print(f"Peak {i} Fit Parameters:")
        for param_name, param in result.params.items():
            print(f"  {param_name}: {param.value:.4f} ± {param.stderr:.4f}")

    # Plot the original data and background
    axes[0, 2].plot(q_limited, int_full_limited, label='Full Data', linestyle='--', color='green')
    axes[0, 2].plot(q_limited, background_full, label='Background', linestyle="-", color='blue')
    axes[0, 2].plot(q_limited[peaks_full], data_bg_sub_full[peaks_full], 'x', label='Peaks')
    axes[0, 2].set_title(f'Full Azimuthal Integration')
    axes[0, 2].set_xlabel('q')
    axes[0, 2].set_ylabel('Intensity')

    # Add a legend to the plot
    axes[0, 2].legend()

    plt.show()
    
    
    
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
