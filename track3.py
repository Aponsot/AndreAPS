#!/usr/bin/env python3
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
from lmfit.models import GaussianModel, LinearModel

WINDOW = 0.50  # fit window width in x-units (q or 2θ)

def robust_sigma(y):
    y = np.asarray(y, float)
    med = np.median(y)
    return 15 * np.median(np.abs(y - med)) + 1e-12

def fwhm_to_sigma(fwhm):
    return fwhm / 2.354820045

def fit_single_peak(h5_path, frame, center, plot=False):
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
    sigma_fit = result.params["g_sigma"].value
    amp_fit = result.params["g_amplitude"].value
    height_fit = amp_fit / (sigma_fit * np.sqrt(2 * np.pi)) if sigma_fit > 0 else 0.0
    fwhm_fit = 2 * np.sqrt(2 * np.log(2)) * sigma_fit

    if plot:
        plt.figure(figsize=(8, 4))
        plt.plot(xw, yw, 'b.', label="Data")
        plt.plot(xw, result.best_fit, 'r-', label="Fit")
        plt.axvline(center_fit, color='k', ls='--', alpha=0.5, label="Peak Center")
        plt.xlabel("q (1/Å)")
        plt.ylabel("Intensity")
        plt.title(f"Frame {frame} | Center={center_fit:.5f} | FWHM={fwhm_fit:.5f}")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {
        "center": center_fit,
        "height": height_fit,
        "fwhm": fwhm_fit,
        "amplitude": amp_fit,
        "background": {
            "slope": result.params["bkg_slope"].value,
            "intercept": result.params["bkg_intercept"].value,
        },
        "noise": noise,
        "result": result,
        "x": xw,
        "y": yw,
        "yfit": result.best_fit,
    }

def main():
    ap = argparse.ArgumentParser(description="Track peak movement over all frames.")
    ap.add_argument("h5", help="HDF5 with 'q' (or 'tth') and 'int'")
    ap.add_argument("center", type=float, help="Initial guess for peak center")
    args = ap.parse_args()

    with h5py.File(args.h5, "r") as f:
        I = f["int"][:]
        nframes = I.shape[0]

    centers = []
    heights = []
    fwhms = []
    failed_frames = []

    # Fit first 20 frames to determine the dataset's own zero value
    initial_centers = []
    for frame in range(min(20, nframes)):
        try:
            res = fit_single_peak(args.h5, frame, args.center, plot=False)
            initial_centers.append(res["center"])
        except Exception:
            continue
    if initial_centers:
        zero_value = np.mean(initial_centers)
    else:
        zero_value = args.center  # fallback if all fits fail

    for frame in range(nframes):
        try:
            res = fit_single_peak(args.h5, frame, zero_value, plot=False)
            centers.append(res["center"])
            heights.append(res["height"])
            fwhms.append(res["fwhm"])
        except Exception:
            centers.append(np.nan)
            heights.append(np.nan)
            fwhms.append(np.nan)
            failed_frames.append(frame)

    frames = np.arange(nframes)

    plt.figure(figsize=(10, 5))
    plt.plot(frames, centers, 'o-', label="Peak Center")
    plt.xlabel("Frame")
    plt.ylabel("Peak Center (q or 2θ)")
    plt.title("Peak Center Movement Over Frames")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Optionally plot FWHM and height
    plt.figure(figsize=(10, 5))
    plt.plot(frames, fwhms, 'o-', label="FWHM")
    plt.xlabel("Frame")
    plt.ylabel("FWHM")
    plt.title("Peak FWHM Over Frames")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(frames, heights, 'o-', label="Peak Height")
    plt.xlabel("Frame")
    plt.ylabel("Peak Height")
    plt.title("Peak Height Over Frames")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    if failed_frames:
        print(f"Warning: Peak fitting failed for frames: {failed_frames}")

if __name__ == "__main__":
    main()
