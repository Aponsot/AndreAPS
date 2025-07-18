


import os
import numpy as np
from tkinter import filedialog, Tk
import matplotlib.pyplot as plt

# --- Select folder containing .txt files ---
Tk().withdraw()
input_folder = filedialog.askdirectory(title="Select folder containing integrated .txt files")
if not input_folder:
    raise ValueError("No folder selected.")

# --- Get all .txt files ---
txt_files = sorted(
    [f for f in os.listdir(input_folder) if f.lower().endswith(".txt")],
    key=lambda x: int(x.split('_')[-1].split('.')[0])  # assumes filename ends with _frame_XXXX.txt
)
if not txt_files:
    raise RuntimeError("No .txt files found in selected folder.")

# --- Load all files into arrays ---
intensity_list = []
tth_array = None

for i, fname in enumerate(txt_files):
    data = np.loadtxt(os.path.join(input_folder, fname))
    tth, intensity = data[:, 0], data[:, 1]

    if tth_array is None:
        tth_array = tth
    elif not np.allclose(tth_array, tth):
        raise ValueError(f"2θ mismatch in file: {fname}")

    intensity_list.append(intensity)

intensities = np.array(intensity_list)  # shape: (frames, tth)
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


# Save the log-scaled version
log_script_path = "/mnt/data/Time_Resolved_LogScaled.py"
with open(log_script_path, "w") as f:
    f.write(log_scaled_script)

log_script_path
