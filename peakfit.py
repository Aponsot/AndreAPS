import numpy as np
import h5py
import matplotlib.pyplot as ax
import hexrd
import hexrd.fitting.fitpeak as fitpeak

def load_hdf5_data(h5_file,frame):
    """Load data from an HDF5 file."""
    with h5py.File(h5_file, "r") as f:
        int_val = f["intensity_stack"][:]
        print(f'Number of frames in intensity stack: {int_val.shape[0]}')
        data = int_val[frame]
    ax.plot(data)
    ax.xlabel('q (1/A)')
    ax.ylabel('Intensity')
    ax.title('1D Intensity vs q')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Load and process HDF5 data.")
    parser.add_argument("h5_file", type=str, help="Path to the HDF5 file")
    parser.add_argument("Frame", type=int, help="Frame number to process")
    args = parser.parse_args()
    
    h5_file = args.h5_file
    frame = args.Frame
    
   