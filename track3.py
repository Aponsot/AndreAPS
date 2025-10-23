#!/usr/bin/env python3
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
from lmfit.models import GaussianModel, LinearModel

WINDOW = 0.30  # fit window width in x-units (q or 2θ)

def robust_sigma(y):
    y = np.asarray(y, float)
    med = np.median(y)
    return 15 * np.median(np.abs(y - med)) + 1e-12

def fwhm_to_sigma(fwhm):
    return fwhm / 2.354820045

def fit_single_peak(h5_path, frame, center):
    with h5py.File(h5_path, "r") as f:
        x = f["q"][:] if "q" in f else f["tth"][:]
        I = f["int"][:]
    yfull = np.asarray(I[frame], float)
    x = np.asarray(x, float)

    half = WINDOW / 2.0
    m = (x >= center - half) & (x <= center + half)
    xw, yw = x[m], yfull[m]
    mfin = np.isfinite(xw) & np.isfinite(yw)
    xw, yw = xw[mfin], yw[mfin]
    if xw.size < 5:
        raise ValueError("Too few points in window.")

    baseline = np.median(yw)
    noise = robust_sigma(yw)
    dx = float(np.mean(np.diff(xw))) if len(xw) > 1 else WINDOW
    min_sigma = max(dx / 50.0, 1e-6)
    max_sigma = 5.0 * WINDOW  

    peak_idx = np.abs(xw - center).argmin()
    height0 = max(yw[peak_idx] - baseline, noise)
    sigma0 = fwhm_to_sigma(WINDOW / 2)
    amp0 = height0 * sigma0 * np.sqrt(2 * np.pi)

    bkg = LinearModel(prefix="bkg_")
    gauss = GaussianModel(prefix="g_")
    model = bkg + gauss
    params = model.make_params(
        bkg_slope=0.0,
        bkg_intercept=baseline,
        g_center=float(xw[peak_idx]),
        g_sigma=sigma0,
        g_amplitude=amp0,
    )
    params["g_center"].set(min=xw[0], max=xw[-1])
    params["g_sigma"].set(min=min_sigma, max=max_sigma)
    params["g_amplitude"].set(min=0.0)

    result = model.fit(yw, params, x=xw)

    center_fit = result.params["g_center"].value
    return center_fit

def process_dataset(h5_path, initial_guess):
    with h5py.File(h5_path, "r") as f:
        I = f["int"][:]
        nframes = I.shape[0]

    centers = []
    failed_frames = []

    # Get average center from first 10 frames using initial_guess only
    first10_centers = []
    for frame in range(min(20, nframes)):
        try:
            c = fit_single_peak(h5_path, frame, initial_guess)
            first10_centers.append(c)
        except Exception:
            pass
    if not first10_centers:
        avg_center = initial_guess
    else:
        avg_center = np.mean(first10_centers)

    # Track differential movement using a single guess (avg_center) for all frames
    for frame in range(nframes):
        try:
            c = fit_single_peak(h5_path, frame, avg_center)
            centers.append(c)
        except Exception:
            centers.append(np.nan)
            failed_frames.append(frame)

    centers = np.array(centers)
    diff_centers = centers - avg_center


def main():
    ap = argparse.ArgumentParser(description="Track peak movement for 7 datasets.")
    ap.add_argument("--h5", nargs=7, required=True, help="7 HDF5 files with 'q' (or 'tth') and 'int'")
    ap.add_argument("--center", nargs=7, type=float, required=True, help="Initial guess for peak center for each dataset")
    args = ap.parse_args()

     # --- same publication style as your sequential plot ---
    plt.rcParams.update({
         "figure.figsize": (6.5, 4.8),   # similar aspect
    "figure.dpi": 160,
    "savefig.dpi": 300,             # publication export
    "font.size": 12,                # base font
    "axes.titlesize": 12,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.linewidth": 1.0,          # frame thickness
    "axes.grid": False,             # no grid in the sample
    "legend.frameon": False,        # legend without box
    "legend.fontsize": 12,
    })
    max_moves = []
    labels = []

    for i in range(7):
        diff_centers, failed_frames, nframes = process_dataset(args.h5[i], args.center[i])

        # NEW: record max abs movement for this dataset
        max_move = float(np.nanmax(np.abs(diff_centers)))
        max_moves.append(max_move)
        labels.append(f"Beam Index {i}")

        frames = np.arange(nframes)
        plt.scatter(frames, diff_centers, label=f"Beam Index {i}")
        if failed_frames:
            print(f"Warning: Peak fitting failed for frames in dataset {i}: {failed_frames}")

    plt.xlabel("Frame")
    plt.xlim(50, 200)
    plt.ylabel("Peak Center Differential (q or 2θ)")
    plt.title("Differential Peak Center Movement Over Frames (7 Datasets)")
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

    # NEW: print and plot comparison of maxima
    print("\nMax absolute peak movement per dataset:")
    for lab, mm in zip(labels, max_moves):
        print(f"  {lab}: {mm:.6g}")

    plt.figure(figsize=(8, 4.5))
    x = np.arange(len(max_moves))
    plt.bar(x, max_moves)
    plt.xticks(x, labels)
    plt.ylabel("Max |Δcenter| (q or 2θ)")
    plt.title("Max Peak Movement per Dataset")
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    main()
