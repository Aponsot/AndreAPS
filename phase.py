import numpy as np
import h5py
import matplotlib.pyplot as plt
import argparse

def phase_overlay(input_file, frame_indices):
    with h5py.File(input_file, 'r') as f:
        intensity_stack = f['int'][:]
        q_values = f['q'][:]

    peak_data = [
        (np.array([2.48,2.693,2.82,3.66,4.295]), 'α-Ti', 'red'),
        (np.array([2.687,3.8,4.656,5.375,6.01]), 'β-Ti', 'blue'),
        (np.array([3.02,3.49,4.94,5.789,6.05]), "γ'", 'green'),
        (np.array([.97,1.118,1.58,1.854,1.935]), "Ti2Ni", 'cyan'),
        (np.array([1.435,1.527,1.627,2.096,2.49]), "TiNi3", 'magenta'),
        (np.array([3.341,4.85,5.94,6.862,7.67]), "TiNi", 'orange'),
        (np.array([1.496,1.603,2.192,2.59,1.697]), "Fe2Ti", 'purple'),
    ]

    print(f"\nLoaded {intensity_stack.shape[0]} frames.")
    frame_indices = sorted(frame_indices)

    plt.figure(figsize=(10, 6))
    for idx in frame_indices:
        if 0 <= idx < intensity_stack.shape[0]:
            plt.plot(q_values, intensity_stack[idx], label=f'Frame {idx}')
        else:
            print(f"Frame {idx} is out of range.")

    # Overlay phase lines
    for peaks, label, color in peak_data:
        for q in peaks:
            plt.axvline(q, color=color, linestyle='--', alpha=0.7)
        plt.text(peaks[0], plt.ylim()[1]*0.95, label, color=color, fontsize=12, rotation=90, va='top')
    plt.rcParams.update({
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.size": 14,
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        })
    plt.xlabel('q')
    plt.ylabel('Intensity')
    plt.legend()
    plt.title('Intensity vs q with Phase Overlay')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overlay phase lines on diffraction data.")
    parser.add_argument("h5", type=str, help="Path to the HDF5 file.")
    parser.add_argument(
        "--frames", "-f", type=str, required=True,
        help="Comma-separated list of frame indices to plot (e.g., 0,10,100)"
    )
    args = parser.parse_args()
    frame_indices = [int(num.strip()) for num in args.frames.split(',')]
    phase_overlay(args.h5, frame_indices)
