import numpy as np
import h5py
import argparse
import matplotlib.pyplot as plt
from lmfit.models import GaussianModel, PolynomialModel
from scipy.signal import find_peaks

def peak_fit(h5, frame_number, peak_pos=None, window=0.1):
    with h5py.File(h5, 'r') as f:
        int_val = f['int'][:]  # shape: (nframes, q)
        q = f['q'][:]          # shape: (q,)
        cake_intensity_stack = f['cake_int'][:]  # shape: (nframes, n_cakes, q)
       
    cake_slices = [0, 10, 19, 28]

    # Prepare compound plot
    fig, axes = plt.subplots(len(cake_slices), 2, figsize=(10, 3 * len(cake_slices)))

    # Fit and plot for each cake slice
    for i, cs in enumerate(cake_slices):
        cake_data = cake_intensity_stack[frame_number, cs, :]
        intensity_data = int_val[frame_number, :]

        # Limit the region for peak finding based on the peak_pos and window
        if peak_pos is not None:
            q_min = peak_pos - window
            q_max = peak_pos + window
            mask = (q >= q_min) & (q <= q_max)
            q_limited = q[mask]
            cake_data_limited = cake_data[mask]
        else:
            q_limited = q
            cake_data_limited = cake_data

        # Automatically find peaks in the limited region
        peaks, _ = find_peaks(cake_data_limited, height=np.max(cake_data_limited) * 0.5, distance=10)
        if len(peaks) > 0:
            detected_peak_pos = q_limited[peaks[0]]  # Use the first detected peak as the initial guess
        else:
            print(f"No peaks found in the specified region for cake slice {cs}. Skipping.")
            continue

        # Fit cake slice
        if cake_data is not None:
            poly_model = PolynomialModel(degree=1)
            gauss_model = GaussianModel()
            model = gauss_model + poly_model
            params = model.make_params(
                center=detected_peak_pos,  # Use the detected peak position
                sigma=1,  # Start with a smaller sigma for narrow peaks
                amplitude=np.max(cake_data) - np.min(cake_data),  # Amplitude based on data range
                c0=np.mean(cake_data),  # Background intercept as the mean intensity
                c1=0  # Assume a flat background initially
            )
            result_cake = model.fit(cake_data, params, x=q)

            # Plot the cake slice data and fit
            axes[i, 0].plot(q, cake_data, 'b.', label='Cake Slice Data')
            axes[i, 0].plot(q, result_cake.best_fit, 'r-', label='Fit')
            axes[i, 0].set_title(f'Cake Slice {cs}')
            axes[i, 0].legend()
        else:
            axes[i, 0].set_title(f'Cake Slice {cs} (No Data)')
            axes[i, 0].axis('off')

        # Fit full azimuthal
        poly_model = PolynomialModel(degree=1)
        gauss_model = GaussianModel()
        model = gauss_model + poly_model
        params = model.make_params(
            center=detected_peak_pos,  # Use the detected peak position
            sigma=1,  # Start with a smaller sigma for narrow peaks
            amplitude=np.max(intensity_data) - np.min(intensity_data),  # Amplitude based on data range
            c0=np.mean(int_val),  # Background intercept as the mean intensity
            c1=0  # Assume a flat background initially
        )
        result_full = model.fit(intensity_data, params, x=q)

        # Plot the full azimuthal data and fit
        axes[i, 1].plot(q, intensity_data, 'g.', label='Full Azimuthal Data')
        axes[i, 1].plot(q, result_full.best_fit, 'r-', label='Fit')
        axes[i, 1].set_title('Full Azimuthal')
        axes[i, 1].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit peak function")
    parser.add_argument("h5", type=str, help="Input h5 file containing processed data.") 
    parser.add_argument("frame_number", type=int, help="Frame number to process (0-indexed).")
    parser.add_argument("--peak_pos", type=float, default=None, help="Position of the peak to fit (in q units). If not provided, peaks will be automatically detected.")
    parser.add_argument("--window", type=float, default=0.1, help="Window size around the peak position for peak finding.")
    args = parser.parse_args()
 
    h5 = args.h5
    frame_number = args.frame_number
    peak_pos = args.peak_pos
    window = args.window

    peak_fit(h5, frame_number, peak_pos, window)
