import numpy as np
from tkinter import filedialog, Tk
import matplotlib.pyplot as plt

# --- Select .npz file ---
Tk().withdraw()
input_file = filedialog.askopenfilename(title="Select .npz file", filetypes=[("NumPy zipped", "*.npz")])
if not input_file:
    raise ValueError("No file selected.")

# --- Load arrays from .npz ---
with np.load(input_file) as npz:
    tth_array = npz["tth"]
    intensities = npz["intensities"]  # shape: (frames, tth)

n_frames = intensities.shape[0]

# --- Time axis in ms ---
ms_per_frame = 4
time_axis = np.arange(n_frames) * ms_per_frame / 1000  # in seconds

# --- Logarithmic scaling ---
log_intensity = np.log1p(intensities)  # log(1 + I)
log_max = np.max(log_intensity)
scaled_log = 100 * log_intensity / log_max  # normalize to 0-100

# --- Plot heatmap ---
fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(scaled_log, aspect='auto', origin='lower',
               extent=[tth_array.min(), tth_array.max(), time_axis[0], time_axis[-1]],
               cmap='plasma', vmin=0, vmax=100)

ax.set_xlabel("2θ (degrees)", fontsize=20)
ax.set_ylabel("Time (s)", fontsize=20)
ax.set_title("Time-Resolved Diffraction (Log-Scaled Intensity)", fontsize=16)
ax.tick_params(axis='both', labelsize=16) 

# --- Colorbar ---
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Log Intensity", fontsize=20)

plt.tight_layout()
plt.show()