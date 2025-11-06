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

# Minimum "peak requirement" — interpreted as the **fitted height above background**
# (NOT raw data). Peaks with fitted height < PEAK_HEIGHT_MIN are omitted.
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
        height0 = max(y_at_seed - y_bkg_at_seed, np.std(yw) * 0.5)

        # amplitude ~ height * sigma * sqrt(2π)
        amp0 = max(height0 * sigma0 * np.sqrt(2.0 * np.pi), 0.0)

        params.update(g.make_params(center=c0, sigma=sigma0, amplitude=amp0))
        params[f"g{i}_sigma"].set(min=1e-6, max=max(span, 1.0))
        params[f"g{i}_amplitude"].set(min=0.0)
        params[f"g{i}_center"].set(min=c0 - 0.6*halfwidth, max=c0 + 0.6*halfwidth)

    return model, params

def _gaussian_y(x, amp, cen, sig):
    # pure Gaussian (area-normalized amplitude)
    return amp * np.exp(-(x - cen)**2 / (2.0 * sig**2))

def fit_frame(x, y, centers, halfwidth):
    """
    Fit one frame; returns:
      centers, fwhm, height_fit (above background), peak_fit (background+height at center),
      component_curves (list of y arrays for bkg + each gaussian), yfit, xw/yw, success, result
    """
    m = combined_window_mask(x, centers, halfwidth)
    if not np.any(m):
        return {"success": False}

    xw, yw = x[m], y[m]
    if xw.size < MIN_POINTS:
        return {"success": False}

    model, params = initial_params_for_frame(xw, yw, centers, halfwidth)
    try:
        result = model.fit(yw, params, x=xw, nan_policy="omit")

        # background line over xw
        bkg_slope = result.params.get("bkg_slope").value if "bkg_slope" in result.params else 0.0
        bkg_intercept = result.params.get("bkg_intercept").value if "bkg_intercept" in result.params else 0.0
        bkg_line = bkg_slope * xw + bkg_intercept

        out_centers, out_fwhm, out_hfit, out_peakfit = [], [], [], []
        component_curves = []  # each = bkg_line + that gaussian

        for i in range(len(centers)):
            amp_i = result.params[f"g{i}_amplitude"].value
            cen_i = result.params[f"g{i}_center"].value
            sig_i = result.params[f"g{i}_sigma"].value

            # FWHM
            fwhm_i = sigma_to_fwhm(sig_i) if np.isfinite(sig_i) else np.nan

            # fitted height above bkg at the center: amp = height * sigma * sqrt(2π)
            if np.isfinite(sig_i) and sig_i > 0:
                height_fit_i = amp_i / (sig_i * np.sqrt(2.0 * np.pi))
            else:
                height_fit_i = np.nan

            # peak value at the center (background + height)
            peak_fit_i = (bkg_slope * cen_i + bkg_intercept) + (height_fit_i if np.isfinite(height_fit_i) else 0.0)

            out_centers.append(cen_i)
            out_fwhm.append(fwhm_i)
            out_hfit.append(height_fit_i)
            out_peakfit.append(peak_fit_i)

            # component curve (bkg + gaussian) to plot as dotted line
            if np.all(np.isfinite([amp_i, cen_i, sig_i])):
                comp = bkg_line + _gaussian_y(xw, amp_i, cen_i, sig_i)
            else:
                comp = np.full_like(xw, np.nan, dtype=float)
            component_curves.append(comp)

        return {
            "success": True,
            "centers": np.array(out_centers, float),
            "fwhm": np.array(out_fwhm, float),
            "height_fit": np.array(out_hfit, float),   # fitted height above background
            "peak_fit": np.array(out_peakfit, float),  # fitted peak value at the center (bkg+height)
            "components": component_curves,            # list of arrays: bkg + that gaussian
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

        # Omit peaks using **fitted height above background**
        valid = res["height_fit"] >= PEAK_HEIGHT_MIN
        centers_v = res["centers"][valid]
        fwhm_v = res["fwhm"][valid]
        hfit_v = res["height_fit"][valid]
        pfit_v = res["peak_fit"][valid]
        comps_v = [res["components"][i] for i, ok in enumerate(valid) if ok]

        print(f"# Frame {args.frame}")
        for i_vis, (c, w, h, p) in enumerate(zip(centers_v, fwhm_v, hfit_v, pfit_v)):
            print(f"peak{i_vis}_center={c:.6f}, peak{i_vis}_FWHM={w:.6f}, "
                  f"peak{i_vis}_height_fit={h:.6f}, peak{i_vis}_peak_fit={p:.6f}")

        # Plot
        apply_nature_style()
        from matplotlib.gridspec import GridSpec
        fig = plt.figure(figsize=(6.2, 4.6))
        gs = GridSpec(2, 1, height_ratios=[3.0, 1.4], hspace=0.15)
        ax = fig.add_subplot(gs[0])

        ax.plot(res["xw"], res["yw"], lw=1.0, label="data")
        ax.plot(res["xw"], res["yfit"], lw=1.2, label="fit")

        # Plot each valid component as faint dotted (bkg + that Gaussian)
        for comp in comps_v:
            ax.plot(res["xw"], comp, ":", lw=0.9, alpha=0.6, label=None)

        for c in centers_v:
            ax.axvline(c, linestyle="--", alpha=0.6, lw=0.9)

        ax.set_xlabel("q (1/Å)")
        ax.set_ylabel("Intensity (a.u.)")
        ax.set_title(f"Frame {args.frame} multi-peak fit")
        ax.minorticks_on()
        ax.legend(fontsize=8, ncol=2)

        # Table: fitted values only
        ax_tbl = fig.add_subplot(gs[1])
        ax_tbl.axis("off")
        table_data = [[f"peak{i}", f"{c:.6f}", f"{w:.6f}", f"{h:.6f}", f"{p:.6f}"]
                      for i, (c, w, h, p) in enumerate(zip(centers_v, fwhm_v, hfit_v, pfit_v))]
        col_labels = ["Peak", "Center", "FWHM", "Height (fit)", "Peak@Center (fit)"]

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

    # otherwise, track all frames
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
            # enforce omission by **fitted height**
            valid = res["height_fit"] >= PEAK_HEIGHT_MIN
            tmp_c = res["centers"].copy()
            tmp_w = res["fwhm"].copy()
            tmp_c[~valid] = np.nan
            tmp_w[~valid] = np.nan
            centers_trk[f, :] = tmp_c
            fwhm_trk[f, :] = tmp_w
            # update seeds only for peaks that remained valid; keep prior seeds otherwise
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
