#!/usr/bin/env python3
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from lmfit.models import GaussianModel, LinearModel


# ---------- helpers (minimal) ----------
def robust_sigma(y):
    y = np.asarray(y, float)
    med = np.median(y)
    return 1.4826 * np.median(np.abs(y - med)) + 1e-12

def fwhm_to_sigma(fwhm):
    return fwhm / 2.354820045  # 2*sqrt(2*ln2)

def compute_r2(y, yfit):
    y = np.asarray(y, float)
    yfit = np.asarray(yfit, float)
    ss_res = np.sum((y - yfit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2) + 1e-16
    return 1.0 - ss_res / ss_tot


# ---------- core ----------
def fit_peaks_in_window(h5_path, frame, center, window=0.1, plot=True):
    """
    Fits all peaks found by scipy.signal.find_peaks within ±window/2 around 'center'
    using a sum of GaussianModel(s) + LinearModel background.

    Expects:
      f['q'][:] -> x-axis (preferred; falls back to 'tth' if missing)
      f['int'][:] -> intensity stack, shape (nframes, nx)
    """
    with h5py.File(h5_path, "r") as f:
        if "q" in f:
            x = f["q"][:]
        I = f["int"][:]
    yfull = np.asarray(I[frame], float)
    x = np.asarray(x, float)

    half = window / 2.0
    mask = (x >= center - half) & (x <= center + half)
    xw = x[mask]
    yw = yfull[mask]

    # Remove non-finite points in window
    mfin = np.isfinite(xw) & np.isfinite(yw)
    xw, yw = xw[mfin], yw[mfin]
    if xw.size < 5:
        raise ValueError("Too few finite points in the window after masking.")

    dx = float(np.mean(np.diff(xw))) if len(xw) > 1 else window

    # ---- peak finding (adaptive, shoulder/faint-friendly) ----
    noise = robust_sigma(yw)
    baseline = np.median(yw)
    peaks, props = find_peaks(
        yw,
        prominence=0.8 * noise,
        height=baseline + 0.5 * noise,
        width=(1, None),  # allow very narrow up to very wide
    )

    # Fallback to a single-peak guess at the window center if none found
    if len(peaks) == 0:
        peaks = np.array([np.argmin(np.abs(xw - center))])
        fwhm_pts = np.array([max(3, int(0.1 * len(xw)))])
    else:
        wcalc = peak_widths(yw, peaks, rel_height=0.5)
        fwhm_pts = wcalc[0]  # in sample points

    # ---- build model: linear bkg + sum of gaussians (one per detected peak) ----
    bkg = LinearModel(prefix="bkg_")
    model = bkg
    params = bkg.make_params(slope=0.0, intercept=baseline)

    min_sigma = max(dx / 3.0, 1e-6)
    max_sigma = max(window / 3.0, min_sigma * 2.0)

    for i, pidx in enumerate(peaks):
        gi = GaussianModel(prefix=f"g{i}_")
        model += gi

        x0 = float(xw[pidx])
        fwhm0 = (fwhm_pts[i] * dx) if i < len(fwhm_pts) else (window / 10.0)
        sigma0 = max(fwhm_to_sigma(max(fwhm0, dx)), min_sigma)

        height0 = max(float(yw[pidx] - baseline), noise)
        amp0 = height0 * sigma0 * np.sqrt(2.0 * np.pi)

        params.update(gi.make_params(center=x0, sigma=sigma0, amplitude=amp0))
        params[f"g{i}_center"].set(min=center - half, max=center + half, value=x0)
        params[f"g{i}_sigma"].set(min=min_sigma, max=max_sigma, value=sigma0)
        params[f"g{i}_amplitude"].set(min=0.0, value=amp0)

    result = model.fit(yw, params, x=xw)
    yfit = result.best_fit
    r2 = compute_r2(yw, yfit)

    # ---- collect per-peak stats ----
    rows = []
    for i in range(len(peaks)):
        amp = result.params[f"g{i}_amplitude"].value
        ctr = result.params[f"g{i}_center"].value
        sig = result.params[f"g{i}_sigma"].value
        hgt = amp / (sig * np.sqrt(2.0 * np.pi))
        fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sig
        rows.append([i, ctr, hgt, fwhm, amp])

    bkg_slope = result.params["bkg_slope"].value
    bkg_intercept = result.params["bkg_intercept"].value

    # ---- optional plot with stats table below ----
    if plot:
        plt.rcParams.update({
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.size": 14,
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        })

        fig, (ax, ax_tbl) = plt.subplots(
            2, 1, figsize=(9, 6),
            gridspec_kw={"height_ratios": [3, 1]}
        )

        ax.plot(xw, yw, lw=1.8, label="Data")
        ax.plot(xw, yfit, lw=2.2, label="Fit")
        comps = result.eval_components(x=xw)
        if "bkg_" in comps:
            ax.plot(xw, comps["bkg_"], ls="--", label="Background")
        for i in range(len(peaks)):
            key = f"g{i}_"
            if key in comps:
                ax.plot(xw, comps[key], ls=":", label=f"Peak {i}")
                ax.axvline(rows[i][1], color="k", alpha=0.2)

        ax.set_xlabel("q (1/A)")
        ax.set_ylabel("Intensity")
        ax.set_title(
            f"Frame {frame}"  
        )
        ax.legend(loc="best")
        ax.grid(alpha=0.3)

        ax_tbl.axis("off")
        col_labels = ["Peak #", "Center", "Height", "FWHM", "Amplitude"]
        cell_text = [[
            f"{r[0]}",
            f"{r[1]:.6g}",
            f"{r[2]:.6g}",
            f"{r[3]:.6g}",
            f"{r[4]:.6g}",
        ] for r in rows] if rows else [["—"] * len(col_labels)]

        table = ax_tbl.table(
            cellText=cell_text,
            colLabels=col_labels,
            loc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.3)

        plt.tight_layout()
        plt.show()

    return {
        "frame": frame,
        "window": (center - half, center + half),
        "background": {"slope": bkg_slope, "intercept": bkg_intercept},
        "r2": r2,
        "rows": rows,           # [ [idx, center, height, fwhm, amplitude], ... ]
        "result": result,       # full lmfit result if you want the report
        "x": xw,
        "y": yw,
        "yfit": yfit,
    }


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Fit all peaks in a 0.1-wide window with Gaussian(s)+linear background.")
    ap.add_argument("h5", help="Path to HDF5 file with 'q' (or 'tth') and 'int'")
    ap.add_argument("frame", type=int, help="Frame index to fit")
    ap.add_argument("center", type=float, help="Center of the window")
    ap.add_argument("--window", type=float, default=0.1, help="Window width (default 0.1)")
    
    args = ap.parse_args()

    out = fit_peaks_in_window(
        h5_path=args.h5,
        frame=args.frame,
        center=args.center,
        window=args.window,
    )

    print(f"\nFrame {out['frame']}  window: {out['window'][0]:.6f}..{out['window'][1]:.6f}")
    print(f"Background: slope={out['background']['slope']:.6g}, intercept={out['background']['intercept']:.6g}")
    print(f"R^2: {out['r2']:.6g}")
    if out["rows"]:
        for r in out["rows"]:
            print(f"Peak {r[0]}: center={r[1]:.6g}, height={r[2]:.6g}, FWHM={r[3]:.6g}, amplitude={r[4]:.6g}")
    else:
        print("No peaks found (fallback single-peak guess may have been used).")

if __name__ == "__main__":
    main()
