import h5py
import numpy as np
from scipy.optimize import curve_fit

# Gaussian peak function
def gaussian(x, amp, cen, wid, offset):
    return amp * np.exp(-(x - cen)**2 / (2 * wid**2)) + offset

# List of input files
beam_files = [
    '/Data/h5/oct25/',
]

peak_centers_all = []

center_guess = 4.94  # Known peak location
frame = 0           # Use frame 0, change as needed

num_frames_to_process = 200  # Cap on number of frames to process per file
num_frames_for_avg = 20      # Number of initial frames to use for average peak location

peak_deviations = []
final_avg_centers = []

# Collect peak centers for all datasets
for h5_path in beam_files:
    with h5py.File(h5_path, "r") as f:
        x = f["q"][:] if "q" in f else f["tth"][:]
        y_all = f["int"][:num_frames_to_process, :]  # shape: (frames, x)

        peak_centers = []
        for frame_idx in range(num_frames_to_process):
            y = y_all[frame_idx, :]
            mask = (x > center_guess - 0.1) & (x < center_guess + 0.02)
            x_fit = x[mask]
            y_fit = y[mask]
            p0 = [y_fit.max(), center_guess, 0.1, y_fit.min()]
            try:
                popt, _ = curve_fit(gaussian, x_fit, y_fit, p0=p0)
                peak_centers.append(popt[1])
            except Exception:
                peak_centers.append(np.nan)

        peak_centers_all.append(np.array(peak_centers))

# Compute global average for first 20 frames across all datasets
initial_centers = np.array([centers[:num_frames_for_avg] for centers in peak_centers_all])
global_avg_center = np.nanmean(initial_centers)

# Compute deviations and final averages
for centers in peak_centers_all:
    deviation = centers - global_avg_center
    peak_deviations.append(deviation)
    final_avg_centers.append(np.nanmean(centers[num_frames_for_avg:]))

# Plot deviations for each dataset, with final average as last point
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
for i, deviation in enumerate(peak_deviations):
    frames = np.arange(len(deviation))
    plt.plot(frames, deviation, label=f'beam_{i}')
    # Add final average as last point
    plt.scatter(len(deviation), final_avg_centers[i] - global_avg_center, color=plt.gca().lines[-1].get_color(), marker='x')

plt.axhline(0, color='gray', linestyle='--', linewidth=1)
plt.xlabel('Frame')
plt.ylabel('Peak Center Deviation')
plt.title('Peak Center Deviation from Global Initial Average at 50um')
plt.legend()
plt.tight_layout()
plt.show()
