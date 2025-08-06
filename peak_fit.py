import numpy as np
import h5py
import argparse
import matplotlib.pyplot as plt
from lmfit.models import GaussianModel, PolynomialModel

def peak_fit(h5, frame_number,peak_pos):
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
       
        # Full azimuthal 
        intensity_data = int_val[frame_number, :]
        # Fit cake slice
        if cake_data is not None:
            poly_model = PolynomialModel(degree=1)
            gauss_model = GaussianModel()
            model = gauss_model + poly_model
            params = model.make_params(
                center=peak_pos,  # Adjust this to a more accurate guess
                sigma=1,  # Start with a smaller sigma for narrow peaks
                amplitude=np.max(cake_data) - np.min(cake_data),  # Amplitude based on data range
                c0=np.mean(cake_data),  # Background intercept as the mean intensity
                c1=0  # Assume a flat background initially
            )
            result_cake = model.fit(cake_data, params, x=q)
            window = .2
            q_min = peak_pos - window
            q_max = peak_pos + window

            # Mask data within the window
            mask = (q >= q_min) & (q <= q_max)
            q_plot = q[mask]
            cake_plot = int_val[mask]
            fit_plot = result_cake.best_fit[mask]
            axes[i, 0].plot(q_plot, cake_plot, 'b.', label='Cake Slice Data')
            axes[i, 0].plot(q_plot, fit_plot, 'r-', label='Fit')
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
            center=peak_pos,  # Adjust this to a more accurate guess
            sigma=1,  # Start with a smaller sigma for narrow peaks
            amplitude=np.max(intensity_data) - np.min(intensity_data),  # Amplitude based on data range
            c0=np.mean(int_val),  # Background intercept as the mean intensity
            c1=0  # Assume a flat background initially
        )
        result_full = model.fit(intensity_data, params, x=q)
        axes[1, 1].plot(q, intensity_data, 'g.', label='Full Azimuthal Data')
        axes[1, 1].plot(q, result_full.best_fit, 'r-', label='Fit')
        axes[1, 1].set_title('Full Azimuthal')
        axes[1, 1].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="fit peak fuction")
    parser.add_argument("h5", type=str, help="Input h5 file containing processed data.") 
    parser.add_argument("frame_number", type=int ,  help="Frame number to process (0-indexed).")
    parser.add_argument("peak_pos", type=float, help="Position of the peak to fit (in q units).")
    args = parser.parse_args()
 
    h5 = args.h5
    frame_number = args.frame_number
    peak_pos = args.peak_pos

peak_fit(h5, frame_number, peak_pos)
