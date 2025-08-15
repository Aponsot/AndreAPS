#!/usr/bin/env python3
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
from lmfit.models import GaussianModel, LinearModel


# ------------------------ helpers (minimal) ------------------------ #
def robust_sigma(y):
    y = np.asarray(y, float)
    med = np.median(y)
    return 1.4826 * np.median(np.abs(y - med)) + 1e-12

def fwhm_to_sigma(fwhm):
    return fwhm / 2.354820045  # 2*sqrt(2*ln2)

def compute_r2(y, yfit):
    y = np.asarray(y, float); yfit = np.asarray(yfit, float)
    ss_res = np.sum((y - yfit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2) + 1e-16
    return 1.0 - ss_res / ss_tot

def smooth_ma(y, win=7):
    """Simple moving average; win coerced to odd and >=1."""
    n = max(1, int(win))
    if n % 2 == 0: n += 1
    if n <= 1: return y.copy()
    k = np.ones(n, float) / n
    ypad = np.pad(y, (n//2, n//2), mode="edge")
    return np.convolve(ypad, k, mode="valid")

def derivative_peaks(x, y, smooth_win=7, min_sep_pts=3):
    """
    Return indices of local maxima using derivative sign change (+ to -).
    Also require negative second derivative at the peak.
    Apply simple non-maximum suppression with min separation (in samples).
    """
    ys = smooth_ma(y, smooth_win)
    dx = float(np.mean(np.diff(x))) if len(x) > 1 else 1.0
    dy = np.gradient(ys, dx)
    d2y = np.gradient(dy, dx)

    sgn = np.sign(dy)
    zc = (sgn[:-1] > 0) & (sgn[1:] <= 0)               # + to -
    cand = np.where(zc)[0] + 1
    cand = cand[d2y[cand] < 0]                         # concave down

    if cand.size == 0:
        return cand

    # Non-maximum suppression by height with a min spacing in points
    order = np.argsort(ys[cand])[::-1]
    kept = []
    taken = np.zeros(len(ys), dtype=bool)
    for idx in cand[order]:
        if not taken[max(0, idx - min_sep_pts): idx + min_sep_pts + 1].any():
            kept.append(idx)
            taken[max(0, idx - min_sep_pts): idx + min_sep_pts + 1] = True
    kept.sort()
    return np.array(kept, dtype=int)

def initial_width_pts(y, peak_idx, baseline, frac=0.5):
    """Estimate FWHM (in points) by half-height crossings; fallback if not found."""
    n = len(y)
    ypk = y[peak_idx]
    h = max(ypk - baseline, 1e-12)
    yhalf = baseline + frac * h

    # search left
    li = peak_idx
    while li > 0 and y[li] > yhalf:
        li -= 1
    # search right
    ri = peak_idx
    while ri < n - 1 and y[ri] > yhalf:
        ri += 1

    width = (ri - li) if (ri > li) else max(3, int(0.1 * n))
    return max(width, 3)


# ---------------------------- core ---------------------------- #
def fit_peaks_derivative(
    h5_path, frame, center, window=0.1, plot=True,
    smooth_win=5, min_sep_pts=3, min_height_sigma=1.5
):
    """
    Derivative-based peak finding (no scipy.find_peaks):
      1) smooth -> dy sign-change for maxima -> optional spacing filter
      2) build sum of Gaussians + linear background
      3) fit with lmfit
      4) keep peaks with fitted height >= min_height_sigma * noise

    Expects HDF5 with:
      - x-axis: 'q' (preferred) or 'tth'
      - intensities: 'int' shaped (nframes, nx)
    """
    with h5py.File(h5_path, "r") as f:
        if "q" in f: x = f["q"][:]
        elif "tth" in f: x = f["tth"][:]
        else: raise KeyError("No x-axis found (expected 'q' or 'tth').")
        I = f["int"][:]

    if not (0 <= frame < I.shape[0]):
        raise IndexError(f"Frame {frame} out of bounds (0..{I.shape[0]-1}).")

    yfull = np.asarray(I[frame], float)
    x = np.asarray(x, float)

    # window
    half = window / 2.0
    m = (x >= center - half) & (x <= center + half)
    if not np.any(m):
        raise ValueError("No points in requested window; check center/window.")
    xw, yw = x[m], yfull[m]

    # finite
    mfin = np.isfinite(xw) & np.isfinite(yw)
    xw, yw = xw[mfin], yw[mfin]
    if xw.size < 5:
        raise ValueError("Too few finite points in window.")

    dx = float(np.mean(np.diff(xw))) if len(xw) > 1 else window
    baseline = np.median(yw)
    noise = robust_sigma(yw)

    # -------- derivative maxima --------
    pk_idx = derivative_peaks(xw, yw, smooth_win=smooth_win, min_sep_pts=min_sep_pts)
    if pk_idx.size == 0:
        # Fallback: force a single seed at the nearest point to center
        pk_idx = np.array([np.abs(xw - center).argmin()])

    # -------- build model: linear bkg + sum of Gaussians --------
    bkg = LinearModel(prefix="bkg_")
    model = bkg
    params = bkg.make_params(slope=0.0, intercept=baseline)

    min_sigma = max(dx / 3.0, 1e-6)
    max_sigma = max(window / 3.0, min_sigma * 2.0)

    for i, p in enumerate(pk_idx):
        gi = GaussianModel(prefix=f"g{i}_")
        model += gi

        x0 = float(xw[p])
        # initial width guess from half-height crossings
        fwhm_pts = initial_width_pts(yw, p, baseline, frac=0.5)
        fwhm0 = max(fwhm_pts * dx, dx)
        sigma0 = max(fwhm_to_sigma(fwhm0), min_sigma)

        height0 = max(float(yw[p] - baseline), noise)
        amp0 = height0 * sigma0 * np.sqrt(2.0 * np.pi)

        params.update(gi.make_params(center=x0, sigma=sigma0, amplitude=amp0))
        params[f"g{i}_center"].set(min=center - half, max=center + half, value=x0)
        params[f"g{i}_sigma"].set(min=min_sigma, max=max_sigma, value=sigma0)
        params[f"g{i}_amplitude"].set(min=0.0, value=amp0)

    # fit
    result = model.fit(yw, params, x=xw)
    yfit = result.best_fit
    r2 = compute_r2(yw, yfit)

    # collect per-peak stats, then prominence-like filter
    kept_rows = []
    comps = result.eval_components(x=xw)
    for i in range(len(pk_idx)):
        amp = result.params[f"g{i}_amplitude"].value
        ctr = result.params[f"g{i}_center"].value
        sig = result.params[f"g{i}_sigma"].value
        hgt = amp / (sig * np.sqrt(2.0 * np.pi))
        fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sig

        # apply prominence-like threshold vs noise
        if hgt >= (min_height_sigma * noise):
            kept_rows.append([i, ctr, hgt, fwhm, amp])

    bkg_slope = result.params["bkg_slope"].value
    bkg_intercept = result.params["bkg_intercept"].value

    # optional plot + table
    if plot:
        plt.rcParams.update({
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.size": 16,      # user's preference for readable exports
            "axes.labelsize": 18,
            "axes.titlesize": 20,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
        })

        fig, (ax, ax_tbl) = plt.subplots(
            2, 1, figsize=(10, 6.8),
            gridspec_kw={"height_ratios": [3, 1]}
        )

        ax.plot(xw, yw, lw=1.8, label="Data")
        ax.plot(xw, yfit, lw=2.2, label="Fit")
        if "bkg_" in comps:
            ax.plot(xw, comps["bkg_"], ls="--", label="Background")
        # draw each component and center line; grey out those that failed threshold
        for i in range(len(pk_idx)):
            key = f"g{i}_"
            show = key in comps
            is_kept = any((r[0] == i) for r in kept_rows)
            if show:
                ax.plot(xw, comps[key], ls=":", alpha=1.0 if is_kept else 0.35,
                        label=f"Peak {i}" + ("" if is_kept else " (filtered)"))
            # mark center from fit (not seed)
            cx = result.params[f"g{i}_center"].value
            ax.axvline(cx, alpha=0.25)

        ax.set_xlabel("q (1/A)")
        ax.set_ylabel("Intensity")
        ax.set_title(
            f"Frame {frame} |"
            f"R²={r2:.4f} "
        )
        ax.legend(loc="best")
        ax.grid(alpha=0.3)

        ax_tbl.axis("off")
        cols = ["Peak #", "Center", "Height", "FWHM", "Amplitude"]
        cell_text = [[
            f"{r[0]}", f"{r[1]:.6g}", f"{r[2]:.6g}", f"{r[3]:.6g}", f"{r[4]:.6g}"
        ] for r in kept_rows] if kept_rows else [["—"] * len(cols)]
        table = ax_tbl.table(cellText=cell_text, colLabels=cols, loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.25)

        plt.tight_layout()
        plt.show()

    return {
        "frame": frame,
        "window": (center - half, center + half),
        "background": {"slope": bkg_slope, "intercept": bkg_intercept},
        "r2": r2,
        "rows": kept_rows,      # kept peaks after threshold
        "result": result,       # full lmfit result (includes filtered peaks)
        "x": xw, "y": yw, "yfit": yfit,
        "noise": noise,
    }


# ---------------------------- CLI ---------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Derivative-based peak fitting with Gaussian(s) + linear background.")
    ap.add_argument("h5", help="Path to HDF5 with 'q' (or 'tth') and 'int'")
    ap.add_argument("frame", type=int, help="Frame index")
    ap.add_argument("center", type=float, help="Center of the 0.1-wide window")
    ap.add_argument("--window", type=float, default=0.1, help="Window width (default 0.1)")
    args = ap.parse_args()

    out = fit_peaks_derivative(
        h5_path=args.h5, frame=args.frame, center=args.center, window=args.window,    
    )

    print(f"\nFrame {out['frame']} window: {out['window'][0]:.6f}..{out['window'][1]:.6f}")
    print(f"Background: slope={out['background']['slope']:.6g}, intercept={out['background']['intercept']:.6g}")
    print(f"Noise (MAD): {out['noise']:.6g}   R^2: {out['r2']:.6g}")
    if out["rows"]:
        for r in out["rows"]:
            print(f"Peak {r[0]}: center={r[1]:.6g}, height={r[2]:.6g}, FWHM={r[3]:.6g}, amplitude={r[4]:.6g}")
    else:
        print("No peaks passed the prominence threshold.")

if __name__ == "__main__":
    main()
