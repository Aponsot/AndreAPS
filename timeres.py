import argparse 
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
energy = 61.335  # Energy in keV
h = 4.135667696e-18  # Planck's constant in keV·s
c = 2.99792458e10  # Speed of light in cm/s
wavelength = h * c / energy  # Wavelength in cm
wavelength *= 1e8  # Convert to angstroms

def graph(h5):
    with h5py.File(h5, "r") as f:
                             
        int_val = f["int"][:]
        frames = int_val.shape[0]
        q = (4*np.pi/ wavelength) * np.sin(np.deg2rad(int_val[1]/2.0))
        plt.figure(figsize=(10, 6))
        plt.imshow(int_val, aspect='auto', extent=[2, 7, 0, frames],
                   origin='lower', cmap='jet')
        plt.colorbar(label='Intensity')
        plt.xlabel('q (1/Å)', fontsize = 20)
        plt.ylabel('Frame Index', fontsize = 20)
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)
        plt.show()


if __name__ == "__main__":
    # Command-line argument parsing
    
    p = argparse.ArgumentParser(
        description="Time Resolved plot"
    )
    p.add_argument("h5", type=str)
    args = p.parse_args()
    h5 = args.h5
    graph(h5)
