#!/usr/bin/env python3
# Barebones multi-peak tracker: Sum of Gaussians + linear background
# CLI: --h5 DATASET.h5 --centers 2.975,3.124 --frame 87
# If --frame is omitted, tracks all frames and writes a CSV next to the HDF5.

import argparse
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt

# You’ll need lmfit installed: pip install lmfit
from lmfit.models import GaussianModel, LinearModel

# ------------------------------
# Tunables (simple, minimal)
# ------------------------------
WINDOW = 0.50       # half-width to each side around each seed (q-units)
MIN_POINTS = 8      # minimum points in the combined window to attempt a fit

# Minimum "peak requirement" for reporting/plotting:
# now interpreted as the **minimum max measured intensity** near the fitted center.
PEAK_HEIGHT_MIN = 5.0

# ------------------------------
# Helpers
# ------------------------------
def parse_centers(s: str):
    vals = [float(v) for v in s.split(",") if v.strip() != ""]
    if not vals:
        raise ValueError("No centers parsed from --centers.")
    return np.array(vals, float)

def sigma_to_fwhm(sigma):
    return 2.354820045 * sigma

def load_q_and_I(h5_path):
    """Load q (or tth) and 1D intensity vs q for each frame.
    - Expects 'int' with shape (nframes, Nq) or (nframes, ...).
    - Uses 'q' if present, else 'tth' (will be treated as x-axis units generically).
    - If 'int' has extra dims, averages over non-frame axes to 1D.
    """
    with h5py.File(h5_path, "r") as f:
        if "q" in f:
            x = np.asarray(f["q"][:], float)
        elif "tth" in f:
            x = np.asarray(f["tth"][:], float)
        else:
            raise ValueError("HDF5 must contain 'q' (preferred) or 'tth' dataset.")

        I_full = np.asarray(f["int"][:], float)  # expect frames first
    if I_full.ndim == 1:
        # single frame already 1D; make it (1, Nq)
        I_full = I_full[None, :]
    elif I_full.ndim > 2:
        # average over non-frame axes
        axes = tuple(range(1, I_full.ndim))
        I_full = I_full.mean(axis=axes)

    if I_full.shape[1] != x.shape[0]:
        raise ValueError(f"Shape mismatch: int.shape={I_full.shape}, x.shape={x.shape}")

    return x, I_full  # x: (Nq,), I_full: (nframes, Nq)

def combined_window_mask(x, centers, halfwidth):
    lo = np.min(centers) - halfwidth
    hi = np.max(centers) + halfwidth
    m = (x >= lo) & (x <= hi)
    return m

def initial_params_for_frame(xw, yw, centers, halfwidth):
    """Build an lmfit model (linear background + N Gaussians) & params."""
    # background initial guess
    try:
        bkg_slope, bkg_intercept = np.polyfit(xw, yw, 1)
    except Exception:
        bkg_slope, bkg_intercept = 0.0, float(np.median(yw))

    model = LinearModel(prefix="bkg_")
    params = model.make_params(bkg_slope=bkg_slope, bkg_intercept=bkg_intercept)

    span = max(xw[-1] - xw[0], 1e-9)
    sigma0 = max(span / (7.0 * len(centers)), 1e-6)  # narrower as #peaks grows

    # Add one Gaussian per peak
    for i, c0 in enumerate(centers):
        g = GaussianModel(prefix=f"g{i}_")
        model = model + g

        # pick a point nearest the seed to estimate amplitude
        idx = np.abs(xw - c0).argmin()
        y_at_seed = yw[idx]
        y_bkg_at_seed = bkg_slope * xw[idx] + bkg_intercept

        # initial height guess (above bkg) with simple floor to stabilize seeding
        height0 = max(y_at_seed - y_bkg_at_seed, np.std(yw) * 0.5)

        # crude amplitude guess ~ height * sigma * sqrt(2pi)
        amp0 = max(height0 * sigma0 * np.sqrt(2.0 * np.pi), 0.0)

        params.update(g.make_params(
            center=c0,
            sigma=sigma0,
            amplitude=amp0
        ))

        # mild constraints to keep it stable but flexible
        params[f"g{i}_sigma"].set(min=1e-6, max=max(span, 1.0))
        params[f"g{i}_amplitude"].set(min=0.0)  # non-negative
        params[f"g{i}_center"].set(min=c0 - 0.6*halfwidth, max=c0 + 0.6*halfwidth)

    return model, params

def _max_intensity_near_center(xw, yw, center, fwhm):
    """
    Return the maximum measured intensity near 'center'.
    Uses a symmetric window of half-width = max(fwhm/2, small_eps) in x.
    Falls back to nearest point if window is empty/NaN.
    """
    # small fallback window if FWHM invalid
    if not np.isfinite(fwhm) or fwhm <= 0:
        # choose ~1% of x-span as a tiny window
        span = max(xw[-1] - xw[0], 1e-12)
        half = 0.01 * span
    else:
        half = 0.5 * fwhm

    m = (xw >= center - half) & (xw <= center + half)
    if not np.any(m):
        # fallback: just take nearest sample
        idx = np.abs(xw - center).argmin()
        return float(yw[idx])
    return float(np.max(yw[m]))

def fit_frame(x, y, centers, halfwidth):
    """Fit one frame; returns dict with centers, fwhm, max_int, success, and bestfit y."""
    m = combined_window_mask(x, centers, halfwidth)
    if not np.any(m):
        return {"success": False}

    xw, yw = x[m], y[m]
    if xw.size < MIN_POINTS:
        return {"success": False}

    model, params = initial_params_for_frame(xw, yw, centers, halfwidth)
    try:
        result = model.fit(yw, params, x=xw, nan_policy="omit")
        out_centers = []
        out_fwhm = []
        out_maxI = []
        for i in range(len(centers)):
            sigma_i = result.params[f"g{i}_sigma"].value
            center_i = result.params[f"g{i}_center"].value
            fwhm_i = sigma_to_fwhm(sigma_i) if np.isfinite(sigma_i) else np.nan
            out_centers.append(center_i)
            out_fwhm.append(fwhm_i)
            out_maxI.append(_max_intensity_near_center(xw, yw, center_i, fwhm_i))

        return {
            "success": True,
            "centers": np.array(out_centers, float),
            "fwhm": np.array(out_fwhm, float),
            "max_int": np.array(out_maxI, float),   # <-- max measured intensity
            "xw": xw,
            "yw": yw,
            "yfit": result.best_fit,
            "result": result
        }
    except Exception:
        return {"success": False}

# ------------------------------
# Plot style helper (Nature-like)
# ------------------------------
def apply_nature_style():
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "legend.frameon": False
    })

# ------------------------------
# Main
# ------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Barebones Gaussian peak tracking (linear background)."
    )
    ap.add_argument("--h5", required=True, help="HDF5 file with 'q' (or 'tth') and 'int'")
    ap.add_argument("--centers", required=True,
                    help="Comma-separated initial peak centers (in same units as x, e.g., q). Example: 2.975,3.124")
    ap.add_argument("--frame", type=int, default=None,
                    help="Fit a single frame index. Omit to track all frames.")
    args = ap.parse_args()

    centers0 = parse_centers(args.centers)
    x, I_full = load_q_and_I(args.h5)
    nframes = I_full.shape[0]

    if args.frame is not None:
        if args.frame < 0 or args.frame >= nframes:
            raise ValueError(f"--frame {args.frame} is out of range [0, {nframes-1}]")
        y = I_full[args.frame]
        res = fit_frame(x, y, centers0, WINDOW/2.0)  # WINDOW is full-width; use half-width around each seed
        if not res["success"]:
            print("Fit failed for the requested frame.")
            return

        # Apply omission by PEAK_HEIGHT_MIN (interpreted as min max intensity)
        valid = res["max_int"] >= PEAK_HEIGHT_MIN
        centers_v = res["centers"][valid]
        fwhm_v = res["fwhm"][valid]
        maxI_v = res["max_int"][valid]

        # Print results (only valid peaks)
        print(f"# Frame {args.frame}")
        for i_vis, (c, w, h) in enumerate(zip(centers_v, fwhm_v, maxI_v)):
            print(f"peak{i_vis}_center={c:.6f}, peak{i_vis}_FWHM={w:.6f}, peak{i_vis}_maxI={h:.6f}")

        # Nicer plot layout + table of Center/FWHM/MaxI
        apply_nature_style()
        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=(6.2, 4.6))
        gs = GridSpec(2, 1, height_ratios=[3.0, 1.4], hspace=0.15)
        ax = fig.add_subplot(gs[0])

        ax.plot(res["xw"], res["yw"], lw=1.0, label="data")
        ax.plot(res["xw"], res["yfit"], lw=1.2, label="fit")
        for c in centers_v:
            ax.axvline(c, linestyle="--", alpha=0.6, lw=0.9)
        ax.set_xlabel("q (1/A")
        ax.set_ylabel("Intensity (a.u.)")
        ax.set_title(f"Frame {args.frame} multi-peak fit")
        ax.minorticks_on()
        ax.legend(fontsize=8, ncol=2)

        # Table axis (clean, no frame)
        ax_tbl = fig.add_subplot(gs[1])
        ax_tbl.axis("off")
        table_data = [[f"peak{i}", f"{c:.6f}", f"{w:.6f}", f"{h:.6f}"]
                      for i, (c, w, h) in enumerate(zip(centers_v, fwhm_v, maxI_v))]
        col_labels = ["Peak", "Center", "FWHM", "Max Intensity"]

        tbl = ax_tbl.table(cellText=table_data,
                           colLabels=col_labels,
                           loc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        for key, cell in tbl.get_celld().items():
            cell.set_edgecolor("0.8")
            cell.set_linewidth(0.6)
            cell.set_height(0.18)
            cell.set_alpha(0.0 if key[0] == 0 else 0.15)

        fig.tight_layout()
        plt.show()
        return

    # Otherwise: track all frames (no cap)
    nuse = nframes
    npeaks = len(centers0)
    centers_trk = np.full((nuse, npeaks), np.nan, float)
    fwhm_trk = np.full((nuse, npeaks), np.nan, float)

    # Seed that follows the peak(s)
    seeds = centers0.copy()
    for f in range(nuse):
        y = I_full[f]
        res = fit_frame(x, y, seeds, WINDOW/2.0)
        if not res["success"]:
            # one retry with a 2× wider window
            res = fit_frame(x, y, seeds, WINDOW)
        if res["success"]:
            # enforce omission rule by max intensity threshold
            valid = res["max_int"] >= PEAK_HEIGHT_MIN
            # store only valid peaks; invalid -> NaN
            tmp_centers = res["centers"].copy()
            tmp_fwhm = res["fwhm"].copy()
            tmp_centers[~valid] = np.nan
            tmp_fwhm[~valid] = np.nan
            centers_trk[f, :] = tmp_centers
            fwhm_trk[f, :] = tmp_fwhm
            # update seeds only for peaks that remained valid; keep prior seeds otherwise
            seeds = np.where(valid, res["centers"], seeds)
        # else leave NaNs and keep previous seeds

    # Write CSV next to the HDF5 (centers/FWHM only; invalid peaks remain NaN)
    base = os.path.splitext(os.path.basename(args.h5))[0]
    csv_path = f"{base}_multi_peak_tracking.csv"
    header_cols = ["frame"] + [f"center_{i}" for i in range(npeaks)] + [f"FWHM_{i}" for i in range(npeaks)]
    arr = np.column_stack([
        np.arange(nuse),
        centers_trk,
        fwhm_trk
    ])
    np.savetxt(csv_path, arr, delimiter=",", header=",".join(header_cols), comments="", fmt="%.10g")
    print(f"Wrote: {csv_path}")

    # Console preview of first few lines
    preview_rows = min(5, nuse)
    print("# preview:")
    for r in range(preview_rows):
        parts = [str(r)] + [f"{v:.6f}" if np.isfinite(v) else "nan" for v in centers_trk[r]] + \
                [f"{v:.6f}" if np.isfinite(v) else "nan" for v in fwhm_trk[r]]
        print(",".join(parts))

if __name__ == "__main__":
    main()
