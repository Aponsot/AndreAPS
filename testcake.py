import h5py
import argparse 
import matplotlib.pyplot as plt
import numpy as np


def test_cake(input_file, frame_number, cake_slice):
    with h5py.File(input_file, 'r') as f:
        intensity_stack = f['int'][:]
        q_values = f['q'][:]
        nframes = f.attrs['nframes']

        if 'cake_int' in f and 'cake_eta_ranges' in f:
            cake_intensity_stack = f['cake_int'][:]  # shape: (nframes, nslices, nq)
            cake_slices = f['cake_eta_ranges'][:]   # shape: (nslices, 2)
        else:
            print("Cake data not available in file.")
            return

        # Validate indices
        if frame_number < 0 or frame_number >= nframes or cake_slice < 0 or cake_slice >= cake_intensity_stack.shape[1]:
            print("Invalid frame number or cake slice index.")
            return

        # Extract 1D diffraction data for the selected frame and cake slice
        cake_int_1d = cake_intensity_stack[frame_number, cake_slice, :]
        plt.figure(figsize=(8, 5))
        plt.plot(q_values, cake_int_1d, label=f"Cake slice {cake_slice}")
        plt.title(f"1D Diffraction Plot - Frame {frame_number}, Cake Slice {cake_slice}")
        plt.xlabel("q values")
        plt.ylabel("Intensity")
        plt.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import h5 file and display its contents")
    parser.add_argument("input_file", type=str, help="Input file containing 2D diffraction data.") 
    parser.add_argument("frame_number", type=int ,  help="Frame number to process (0-indexed).")
    parser.add_argument("cake_slice", type=int, help="cake slice to investigate.")
    args = parser.parse_args()

    input_file = args.input_file
    frame_number = args.frame_number
    cake_slice = args.cake_slice
    

    # Load the 2D diffraction data (this part is assumed to be implemented)
    # intensity_stack, q_values, nframes, cake_intensity_stack, cake_slices = load_data(input_file)
    
test_cake(input_file, frame_number, cake_slice) 
