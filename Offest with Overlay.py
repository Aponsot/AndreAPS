import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog, Tk
import matplotlib.cm as cm

# --- Select NPZ File ---
Tk().withdraw()
npz_path = filedialog.askopenfilename(title="Select .npz time-resolved file", filetypes=[("NumPy Zip files", "*.npz")])
if not npz_path:
    raise ValueError("No .npz file selected.")

# --- Load NPZ Data ---
data = np.load(npz_path)
tth = data["tth"]
intensities = data["intensities"]

# Crop data range
tth_min, tth_max = 1, 12
mask = (tth >= tth_min) & (tth <= tth_max)
cropped_tth = tth[mask]
cropped_intensities = intensities[:, mask]

# --- Grouped peak positions ---
peak_data = [
    (np.array([5.58, 6.44, 9.11, 10.69, 11.17, 12.90]), 'γ', 'red'),
    (np.array([5.54, 6.40, 9.06, 10.62, 11.10, 12.82,13.98]), 'Cu', 'purple'),
    (np.array([5.54, 6.40, 9.06, 10.62, 11.10, 12.82]), "γ'", 'blue'),
    (np.array([2.88, 4.70, 5.51, 5.76, 6.65]), 'Cr₂Nb', 'green'),
    (np.array([6.44,5.53,9.11,10.56]), "γ''", 'cyan')
]

# --- Frame input ---
n_frames = intensities.shape[0]
print(f"\nLoaded {n_frames} frames.")
frame_input = input("Enter frame indices to plot (e.g., 0,10,100): ").strip()
frame_indices = sorted([int(num.strip()) for num in frame_input.split(',')])

# --- Set up color map and vertical offset ---
max_intensity = np.max(intensities)
offset_step = 0.2 * max_intensity
cmap = plt.colormaps['viridis'].resampled(len(frame_indices))

# --- Plot selected frames ---
fig, ax = plt.subplots(figsize=(14, 6))
for i, idx in enumerate(frame_indices):
    if idx < 0 or idx >= n_frames:
        print(f"Frame {idx} is out of range. Skipping.")
        continue
    offset = i * offset_step
    ax.plot(tth, intensities[idx] + offset, label=f"Frame {idx}", color=cmap(i))

# --- Overlay grouped peak positions with separate legend handles ---
overlay_lines = []
overlay_labels = []

for positions, label, color in peak_data:
    added = False
    for pos in positions:
        if tth_min <= pos <= tth_max:
            line = ax.axvline(x=pos, color=color, linestyle='--', alpha=0.4)
            if not added:
                overlay_lines.append(line)
                overlay_labels.append(label)
                added = True

# --- Plot settings ---
ax.set_xlabel("2θ (degrees)", fontsize=16)
ax.set_ylabel("Intensity", fontsize=16)
sample = "Sample 1 Phase Overlay"
ax.set_title(sample, fontsize=20)
legend1 = ax.legend(title="Frame", loc="upper right", fontsize=14)
legend2 = ax.legend(overlay_lines, overlay_labels, title="Phase Overlay", loc="upper left", fontsize=14)
ax.add_artist(legend1)

ax.grid(True)
fig.tight_layout()

# --- Save figure ---
save_dir = '/'.join(npz_path.replace("\\", "/").split('/')[:-1])
save_name = f"{save_dir}/{sample.replace(' ', '_')}.png"
plt.savefig(save_name, dpi=300)
print(f"\n Saved plot to: {save_name}")

plt.show()
