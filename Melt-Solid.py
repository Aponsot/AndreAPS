import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog, Tk
import matplotlib.cm as cm

# --- Load NPZ File ---
Tk().withdraw()
npz_path = filedialog.askopenfilename(title="Select .npz time-resolved file", filetypes=[("NumPy Zip files", "*.npz")])
if not npz_path:
    raise ValueError("No .npz file selected.")

data = np.load(npz_path)
tth = data["tth"]
intensities = data["intensities"]

tth_min, tth_max = 1, 18
mask = (tth >= tth_min) & (tth <= tth_max)
tth = tth[mask]
intensities = intensities[:, mask]


n_frames = intensities.shape[0]
ms_per_frame = 4
offset = 2.0

print(f"\nLoaded {n_frames} frames.")
melting_frame = int(input(""))


before_frames = int(80/ ms_per_frame)  # 10 frames
after_frames = int(80/ ms_per_frame)   # 20 frames
frame_range = range(max(0, melting_frame - before_frames),
                    min(n_frames, melting_frame + after_frames + 1))


fig, ax = plt.subplots(figsize=(8, 10))
for i, frame in enumerate(frame_range):
    y_offset = i * offset
    intensity = intensities[frame]
    time_ms = (frame - melting_frame) * ms_per_frame  
    color = cm.viridis(i / len(frame_range))
    
    ax.plot(tth, intensity + y_offset, color=color, linewidth=1.0)
    ax.text(tth[0] + 3, y_offset + 0.2, f"{time_ms:+} ms", fontsize=14)

ax.set_xlim(tth_min, tth_max)
ax.set_ylim(-offset, y_offset + offset * 1.5)
ax.set_xlabel(r"$2\theta$ (degrees)", fontsize=20)
ax.set_ylabel("Intensity", fontsize=20)
ax.set_title("Melting–Solidification Window Sample 1" \
"", fontsize=15)
ax.tick_params(left=False, labelleft=False)
ax.grid(False)
plt.tight_layout()
plt.show()
