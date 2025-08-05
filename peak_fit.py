import numpy as np
import h5py
import argparse
import matplotlib.pyplot as plt
from lmfit.models import GaussianModel, PolynomialModel

def peak_fit(h5, frame_number,peak_pos, cake_slice=None):
    with h5py.File(h5, 'r') as f:
        int_val = f['int'][:]  # shape: (nframes, tth)
        q = f['q'][:]          # shape: (tth,)
        nframes = f.attrs['nframes']
        if 'cake_int' in f:
            cake_intensity_stack = f['cake_int'][:]  # shape: (nframes, n_cakes, tth)
        else:
            cake_intensity_stack = None

    # Default cake slices for 0, 90, 180, 270 degrees
    default_cake_slices = [0, 10, 19, 28]

    # If no cake_slice is provided, use default slices
    if cake_slice is None:
        cake_slice_indices = default_cake_slices
    else:
        cake_slice_indices = [cake_slice]

    # Prepare compound plot
    fig, axes = plt.subplots(len(cake_slice_indices), 2, figsize=(10, 3 * len(cake_slice_indices)))

    # Fit and plot for each cake slice
    for i, cs in enumerate(cake_slice_indices):
        # Cake slice fit
        if cake_intensity_stack is not None:
            cake_data = cake_intensity_stack[frame_number, cs, :]
        else:
            cake_data = None

        # Full azimuthal fit
        intensity_data = int_val[frame_number, :]

        # Use peak_pos as center guess, or max position if not provided
     

        # Fit cake slice
        if cake_data is not None:
            poly_model = PolynomialModel(degree=1)
            gauss_model = GaussianModel()
            combined_model = gauss_model + poly_model
            params = combined_model.make_params(amplitude=np.max(cake_data), center=peak_pos, sigma=2)
            result_cake = combined_model.fit(cake_data, params, x=q)
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
        combined_model = gauss_model + poly_model
        params = combined_model.make_params(amplitude=np.max(intensity_data), center=peak_pos, sigma=2)
        result_full = combined_model.fit(intensity_data, params, x=q)
        axes[i, 1].plot(q, intensity_data, 'g.', label='Full Azimuthal Data')
        axes[i, 1].plot(q, result_full.best_fit, 'r-', label='Fit')
        axes[i, 1].set_title('Full Azimuthal')
        axes[i, 1].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="fit peak fuction")
    parser.add_argument("h5", type=str, help="Input h5 file containing processed data.") 
    parser.add_argument("frame_number", type=int ,  help="Frame number to process (0-indexed).")
    parser.add_argument("cake_slice", type=int, default = 36, help="cake slice to investigate (0-indexed). leave black for 0,90,180,270 degree slices.")
    parser.add_argument("peak_pos", type=float, help="Position of the peak to fit (in q units).")
    args = parser.parse_args()
 
    h5 = args.h5
    frame_number = args.frame_number
    cake_slice = args.cake_slice
    peak_pos = args.peak_pos
peak_fit(h5, frame_number, peak_pos, cake_slice)