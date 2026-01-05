#!/usr/bin/env python3
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt


# ---------- Plot style (publication-ish, easy to tweak) ----------
plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})


def read_lineout(h5_path: str, frame: int):
    """Read q and intensity lineout at a given frame from an HDF5 file."""
    with h5py.File(h5_path, "r") as f:
        q = np.asarray(f["q"][:]).squeeze()
        I = np.asarray(f["int"][frame, :]).squeeze()
    return q, I


def safe_label(path: str) -> str:
    return path.split("/")[-1]


def normalize(y: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return y
    y = y.astype(float)
    if mode == "max":
        m = np.nanmax(y)
        return y / m if m != 0 else y
    if mode == "area":
        a = np.trapz(y, dx=1.0)
        return y / a if a != 0 else y
    raise ValueError(f"Unknown normalization mode: {mode}")


def compare_frames_across_datasets(h5_files, frames, norm="none", offset=0.0, title=None):
    frames = sorted(set(frames))

    fig, ax = plt.subplots(figsize=(10.5, 6.0))

    # For consistent offset scaling, peek max intensity if offset requested
    offset_step = 0.0
    if offset > 0:
        max_all = 0.0
        for fp in h5_files:
            with h5py.File(fp, "r") as f:
                max_all = max(max_all, float(np.nanmax(f["int"][:])))
        offset_step = offset * max_all

    # Plot: for each frame, overlay all datasets (grouped by frame)
    # This makes it easy to compare dataset-to-dataset at a fixed frame.
    for fi, frame in enumerate(frames):
        base_off = fi * offset_step
        for di, fp in enumerate(h5_files):
            try:
                q, I = read_lineout(fp, frame)
            except Exception as e:
                print(f"[WARN] {safe_label(fp)}: could not read frame {frame}: {e}")
                continue

            I = normalize(I, norm)
            ax.plot(q, I + base_off, label=f"{safe_label(fp)} | frame {frame}")

    ax.set_xlabel("q")
    ax.set_ylabel("Intensity" + (" (offset)" if offset_step != 0 else ""))
    ax.set_title(title if title else f"Lineout comparison across datasets ({len(h5_files)} files)")

    ax.legend(fontsize=10, ncols=1, frameon=True)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    plt.show()


def parse_frames_arg(frames_str: str):
    # supports "10,20,30"
    return [int(x.strip()) for x in frames_str.split(",") if x.strip()]


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Compare diffraction lineouts at the same frame(s) across multiple HDF5 datasets."
    )
    p.add_argument("h5", nargs="+", help="One or more HDF5 files (must contain datasets: /q and /int).")

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--frame", type=int, help="Single frame index to compare across all datasets.")
    g.add_argument("--frames", type=str, help="Comma-separated frames, e.g. 0,50,100")

    p.add_argument("--norm", choices=["none", "max", "area"], default="none",
                   help="Optional normalization per curve: none, max, or area.")
    p.add_argument("--offset", type=float, default=0.0,
                   help="Vertical offset between different frames (fraction of global max intensity). "
                        "Example: 0.15 gives 15%% of global max per frame. Default 0 (no offset).")
    p.add_argument("--title", type=str, default=None, help="Custom plot title.")

    args = p.parse_args()

    frames = [args.frame] if args.frame is not None else parse_frames_arg(args.frames)
    compare_frames_across_datasets(args.h5, frames, norm=args.norm, offset=args.offset, title=args.title)
