#!/usr/bin/env python3
# Barebones multi-peak tracker: Sum of Gaussians + linear background
# CLI: --h5 DATASET.h5 --centers 2.975,3.124 --frame 87
# If --frame is omitted, tracks all frames and writes a CSV next to the HDF5.

import argparse
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from lmfit.models import GaussianModel, LinearModel

# ------------------------------
# Tunables (simple, minimal)
# ------------------------------
WINDOW = 0.50       # half-width to each side around each seed (q-units)
MIN_POINTS = 8      # minimum points in the combined window to attempt a fit
PEAK_HEIGHT_MIN = 5.0  # minimum peak max intensity for reporting/plotting

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
    with h5py.File(h5_path, "r") as f:
        if "q" in f:
            x = np.asarray(f["q"][:], float)
        elif "tth" in f:
            x = np.asarray(f["tth"][:], float)
        else:
            raise ValueError("HDF5 must contain 'q' (preferred) or 'tth' dataset.")
        I_full = np.asarray(f["int"][:], float)
    if I_full.ndim == 1:
        I_full = I_full[None, :]
    elif I_full.ndim > 2:
        axes = tuple(range(1, I_full.ndim))
        I_full = I_full.mean(axis=axes)
    if I_full.shape[1] != x.shape[0]:
        raise ValueError(f"Shape mismatch: int.shape={I_full.shape}, x.shape={x.shape}")
    return x, I_full

def combined_window_mask(x, centers, halfwidth):
    lo = np.min(centers) - halfwidth
    hi = np.max(centers) + halfwidth
    return (x >= lo) & (x <= hi)

def initial_params_for_frame(xw, yw, centers, halfwidth):
    try:
        bkg_slope, bkg_intercept = np.polyfit(xw, yw, 1)
    except Exception:
        bkg_slope, bkg_intercept = 0.0, float(np.median(yw))
    model = LinearModel(prefix="bkg_")
    params = model.make_params(bkg_slope=bkg_slope, bkg_intercept=bkg_intercept)
    span = max(xw[-1] - xw[0], 1e-9)
    sigma0 = max(span / (7.0 * len(centers)), 1e-6)
    for i, c0 in enumerate(centers):
        g = GaussianModel(prefix=f"g{i}_")
        model = model + g
        idx = np.abs(xw - c0).argmin()
        y_at_seed = yw[idx]
        y_bkg_at_seed = bkg_slope * xw[idx] + bkg_intercept
        height0 = max(y_at_seed - y_bkg_at_seed, np.std(yw) * 0.5)
        amp0 = max(height0 * sigma0 * np.sqrt(2.0 * np.pi), 0.0)
        params.update(g.make_params(center=c0, sigma=sigma0, amplitude=amp0))
        params[f"g{i}_sigma"].set(min=1e-6, max=max(span, 1.0))
        params[f"g{i}_amplitude"].set(min=0.0)
        params[f"g{i}_center"].set(min=c0 - 0.6*halfwidth, max=c0 + 0.6*halfwidth)
    return model, params

def _max_intensity_near_center(xw, yw, center, fwhm):
    if not np.isfinite(fwhm) or fwhm <= 0:
        span = max(xw[-1] - xw[0], 1e-12)
        half = 0.01 * span
    else:
        half = 0.5 * fwhm
    m = (xw >= center - half) & (xw <= center + half)
    if not np.any(m):
        idx = np.abs(xw - center).argmin()
        return float(yw[idx])
    return float(np.max(yw[m]))

def fit_frame(x, y, centers, halfwidth):
    m = combined_window_mask(x, centers, halfwidth)
    if not np.any(m):
        return {"success": False}
    xw, yw = x[m], y[m]
    if xw.size < MIN_POINTS:
        return {"success": False}
    model, params = initial_params_for_frame(xw, yw, centers, halfwidth)
    try:
        result = model.fit(yw, params, x=xw, nan_policy="omit")
        out_centers, out_fwhm, out_maxI = [], [], []
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
            "max_int": np.array(out_maxI, float),
            "xw": xw,
            "yw": yw,
            "yfit": result.best_fit,
            "result": result
        }
    except Exception:
        return {"success": False}

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
        "legend.frameon": False
    })

# ------------------------------
# Main
# ------------------------------
def main():
    ap = argparse.ArgumentParser(description="Barebones Gaussian peak tracking (linear background).")
    ap.add_argument("--h5", required=True, help="HDF5 file with 'q' (or 'tth') and 'int'")
    ap.add_argument("--centers", required=True,
                    help="Comma-separated initial peak centers (e.g., 2.975,3.124)")
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
        res = fit_frame(x, y, centers0, WINDOW/2.0)
        if not res["success"]:
            print("Fit failed for the requested frame.")
            return

        valid = res["max_int"] >= PEAK_HEIGHT_MIN
        centers_v = res["centers"][valid]
        fwhm_v = res["fwhm"][valid]
        maxI_v = res["max_int"][valid]

        print(f"# Frame {args.frame}")
        for i_vis, (c, w, h) in enumerate(zip(centers_v, fwhm_v, maxI_v)):
            print(f"peak{i_vis}_center={c:.6f}, peak{i_vis}_FWHM={w:.6f}, peak{i_vis}_maxI={h:.6f}")

        apply_nature_style()
        from matplotlib.gridspec import GridSpec
        fig = plt.figure(figsize=(6.2, 4.6))
        gs = GridSpec(2, 1, height_ratios=[3.0, 1.4], hspace=0.15)
        ax = fig.add_subplot(gs[0])

        ax.plot(res["xw"], res["yw"], lw=1.0, label="data")
        ax.plot(res["xw"], res["yfit"], lw=1.2, label="fit")

        # Plot each individual Gaussian as a faint dotted line
        for i in range(len(centers0)):
            prefix = f"g{i}_"
            if prefix + "amplitude" in res["result"].params:
                amp = res["result"].params[prefix + "amplitude"].value
                cen = res["result"].params[prefix + "center"].value
                sig = res["result"].params[prefix + "sigma"].value
                if np.isfinite(amp) and np.isfinite(sig) and np.isfinite(cen):
                    g_curve = amp * np.exp(-(res["xw"] - cen)**2 / (2 * sig**2))
                    ax.plot(res["xw"], g_curve, ":", lw=0.8, alpha=0.5, label=None)

        for c in centers_v:
            ax.axvline(c, linestyle="--", alpha=0.6, lw=0.9)

        ax.set_xlabel("q (1/Å)")
        ax.set_ylabel("Intensity (a.u.)")
        ax.set_title(f"Frame {args.frame} multi-peak fit")
        ax.minorticks_on()
        ax.legend(fontsize=8, ncol=2)

        # Table axis
        ax_tbl = fig.add_subplot(gs[1])
        ax_tbl.axis("off")
        table_data = [[f"peak{i}", f"{c:.6f}", f"{w:.6f}", f"{h:.6f}"]
                      for i, (c, w, h) in enumerate(zip(centers_v, fwhm_v, maxI_v))]
        col_labels = ["Peak", "Center", "FWHM", "Max Intensity"]

        tbl = ax_tbl.table(cellText=table_data, colLabels=col_labels, loc="center")
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

    # otherwise track all frames
    nuse = nframes
    npeaks = len(centers0)
    centers_trk = np.full((nuse, npeaks), np.nan)
    fwhm_trk = np.full((nuse, npeaks), np.nan)
    seeds = centers0.copy()
    for f in range(nuse):
        y = I_full[f]
        res = fit_frame(x, y, seeds, WINDOW/2.0)
        if not res["success"]:
            res = fit_frame(x, y, seeds, WINDOW)
        if res["success"]:
            valid = res["max_int"] >= PEAK_HEIGHT_MIN
            tmp_c = res["centers"].copy()
            tmp_w = res["fwhm"].copy()
            tmp_c[~valid] = np.nan
            tmp_w[~valid] = np.nan
            centers_trk[f, :] = tmp_c
            fwhm_trk[f, :] = tmp_w
            seeds = np.where(valid, res["centers"], seeds)

    base = os.path.splitext(os.path.basename(args.h5))[0]
    csv_path = f"{base}_multi_peak_tracking.csv"
    header_cols = ["frame"] + [f"center_{i}" for i in range(npeaks)] + [f"FWHM_{i}" for i in range(npeaks)]
    arr = np.column_stack([np.arange(nuse), centers_trk, fwhm_trk])
    np.savetxt(csv_path, arr, delimiter=",", header=",".join(header_cols),
               comments="", fmt="%.10g")
    print(f"Wrote: {csv_path}")

    preview_rows = min(5, nuse)
    print("# preview:")
    for r in range(preview_rows):
        parts = [str(r)] + [f"{v:.6f}" if np.isfinite(v) else "nan" for v in centers_trk[r]] + \
                [f"{v:.6f}" if np.isfinite(v) else "nan" for v in fwhm_trk[r]]
        print(",".join(parts))

if __name__ == "__main__":
    main()
