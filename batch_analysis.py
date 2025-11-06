#!/usr/bin/env python3
# Barebones multi-peak tracker with optional residual-added shoulder
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
PEAK_HEIGHT_MIN = 5.0  # min fitted height above background for reporting/plotting

# Shoulder (residual-add) controls
ENABLE_RESIDUAL_SHOULDER = True   # set False to disable auto add
RESIDUAL_SNR = 3.0                # residual peak must exceed this SNR
MIN_SEP = 0.010                   # min separation from existing centers (x-units)
AIC_IMPROVE = 6.0                 # require ΔAIC <= -AIC_IMPROVE to accept added peak
SIGMA_MIN = 0.001                 # absolute sigma bounds for added peak
SIGMA_MAX = 0.080

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

def robust_sigma(y):
    y = np.asarray(y, float)
    med = np.median(y)
    return 1.4826 * np.median(np.abs(y - med)) + 1e-12

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

        # seed amplitude from local height above background
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

def _gaussian_y(x, amp, cen, sig):
    return amp * np.exp(-(x - cen)**2 / (2.0 * sig**2))

def _compute_fitted_metrics(result, xw):
    """Extract background line, per-peak centers/sigmas, heights, fwhm, peak@center."""
    bkg_slope = result.params.get("bkg_slope").value if "bkg_slope" in result.params else 0.0
    bkg_intercept = result.params.get("bkg_intercept").value if "bkg_intercept" in result.params else 0.0
    bkg_line = bkg_slope * xw + bkg_intercept

    # gather peaks
    i = 0
    centers = []
    sigmas = []
    amps = []
    while f"g{i}_center" in result.params:
        centers.append(result.params[f"g{i}_center"].value)
        sigmas.append(result.params[f"g{i}_sigma"].value)
        amps.append(result.params[f"g{i}_amplitude"].value)
        i += 1
    centers = np.asarray(centers, float)
    sigmas = np.asarray(sigmas, float)
    amps   = np.asarray(amps, float)

    # compute height (above bkg) and peak@center
    heights = np.full_like(centers, np.nan, dtype=float)
    peakfit = np.full_like(centers, np.nan, dtype=float)
    fwhm    = np.full_like(centers, np.nan, dtype=float)
    for j in range(centers.size):
        if np.isfinite(sigmas[j]) and sigmas[j] > 0:
            heights[j] = amps[j] / (sigmas[j] * np.sqrt(2.0 * np.pi))
            fwhm[j] = sigma_to_fwhm(sigmas[j])
        peakfit[j] = (bkg_slope * centers[j] + bkg_intercept) + (heights[j] if np.isfinite(heights[j]) else 0.0)
    return bkg_line, centers, sigmas, amps, heights, fwhm, peakfit

def _try_add_shoulder(xw, yw, model, result, halfwidth):
    """Propose one residual-based extra Gaussian and accept if it passes guardrails."""
    # Residuals and noise
    resid = yw - result.best_fit
    noise = robust_sigma(resid)
    if noise <= 0:
        return model, result, False

    # Candidate at max positive residual
    idx = int(np.argmax(resid))
    peak_resid = resid[idx]
    if peak_resid < RESIDUAL_SNR * noise:
        return model, result, False  # not strong enough

    x0 = xw[idx]

    # Separation from existing centers
    _, centers, sigmas, amps, heights, fwhm, peakfit = _compute_fitted_metrics(result, xw)
    if centers.size:
        if np.min(np.abs(centers - x0)) < MIN_SEP:
            return model, result, False

    # Seed amplitude/sigma for the new component
    span = max(xw[-1] - xw[0], 1e-9)
    mean_sigma = np.nanmean(sigmas) if sigmas.size else span / 10.0
    sigma_seed = np.clip(min(mean_sigma, span / 12.0), SIGMA_MIN, SIGMA_MAX)
    # Convert residual height to amplitude for initial guess
    amp_seed = max(peak_resid * sigma_seed * np.sqrt(2.0 * np.pi), 1e-9)

    # Build a new model with one extra Gaussian and refit
    new_idx = 0
    while f"g{new_idx}_center" in result.params:
        new_idx += 1
    prefix = f"g{new_idx}_"
    g = GaussianModel(prefix=prefix)
    new_model = model + g
    new_params = result.params.copy()

    # initialize new params
    new_params.update(g.make_params(center=float(x0), sigma=sigma_seed, amplitude=amp_seed))
    new_params[prefix + "sigma"].set(min=SIGMA_MIN, max=SIGMA_MAX)
    new_params[prefix + "amplitude"].set(min=0.0)
    # keep it within the existing combined window
    lo = xw[0]
    hi = xw[-1]
    new_params[prefix + "center"].set(min=lo, max=hi)

    # Refit with the extra component
    new_result = new_model.fit(yw, new_params, x=xw, nan_policy="omit")

    # Model selection: require AIC improvement and height >= threshold
    dAIC = new_result.aic - result.aic  # negative is better
    _, c_new, s_new, a_new, h_new, f_new, p_new = _compute_fitted_metrics(new_result, xw)
    # new peak is last one (index new_idx)
    if c_new.size <= new_idx:
        return model, result, False
    new_height = h_new[new_idx]

    if (dAIC <= -AIC_IMPROVE) and np.isfinite(new_height) and (new_height >= PEAK_HEIGHT_MIN):
        return new_model, new_result, True  # accept
    else:
        return model, result, False        # reject

def fit_frame(x, y, centers, halfwidth):
    """
    Fit one frame; returns:
      centers, fwhm, height_fit (above background), peak_fit (background+height at center),
      components (bkg+gaussian curves for plotted peaks that pass PEAK_HEIGHT_MIN),
      yfit, xw/yw, success, result
    """
    m = combined_window_mask(x, centers, halfwidth)
    if not np.any(m):
        return {"success": False}

    xw, yw = x[m], y[m]
    if xw.size < MIN_POINTS:
        return {"success": False}

    # 1) initial fit with user-specified peaks
    base_model, params = initial_params_for_frame(xw, yw, centers, halfwidth)
    try:
        result = base_model.fit(yw, params, x=xw, nan_policy="omit")
    except Exception:
        return {"success": False}

    model_used = base_model

    # 2) optional one-pass shoulder add via residual peak
    if ENABLE_RESIDUAL_SHOULDER:
        model_used, result, accepted = _try_add_shoulder(xw, yw, model_used, result, halfwidth)

    # Extract metrics & component curves
    bkg_line, c_all, s_all, a_all, h_all, w_all, p_all = _compute_fitted_metrics(result, xw)

    # Keep only peaks that pass height threshold for reporting/plotting
    valid = (h_all >= PEAK_HEIGHT_MIN) & np.isfinite(h_all)
    centers_out = c_all.copy()
    fwhm_out = w_all.copy()
    height_out = h_all.copy()
    peakfit_out = p_all.copy()
    centers_out[~valid] = np.nan
    fwhm_out[~valid] = np.nan
    height_out[~valid] = np.nan
    peakfit_out[~valid] = np.nan

    # Build component curves (bkg + that Gaussian) for valid peaks
    components = []
    for i in range(c_all.size):
        if not valid[i]:
            continue
        comp = bkg_line + _gaussian_y(xw, a_all[i], c_all[i], s_all[i])
        components.append(comp)

    return {
        "success": True,
        "centers": centers_out,
        "fwhm": fwhm_out,
        "height_fit": height_out,
        "peak_fit": peakfit_out,
        "components": components,
        "xw": xw,
        "yw": yw,
        "yfit": result.best_fit,
        "result": result
    }

def apply_nature_style():
    plt.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 300,
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 10,
        "axes.linewidth": 0.8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "xtick.direction": "in", "ytick.direction": "in",
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

        # Prepare visible results (drop NaNs)
        mask = np.isfinite(res["centers"])
        centers_v = res["centers"][mask]
        fwhm_v = res["fwhm"][mask]
        hfit_v = res["height_fit"][mask]
        pfit_v = res["peak_fit"][mask]

        print(f"# Frame {args.frame}")
        for i_vis, (c, w, h, p) in enumerate(zip(centers_v, fwhm_v, hfit_v, pfit_v)):
            print(f"peak{i_vis}_center={c:.6f}, peak{i_vis}_FWHM={w:.6f}, "
                  f"peak{i_vis}_height_fit={h:.6f}, peak{i_vis}_peak_fit={p:.6f}")

        apply_nature_style()
        from matplotlib.gridspec import GridSpec
        fig = plt.figure(figsize=(6.2, 4.6))
        gs = GridSpec(2, 1, height_ratios=[3.0, 1.4], hspace=0.15)
        ax = fig.add_subplot(gs[0])

        ax.plot(res["xw"], res["yw"], lw=1.0, label="data")
        ax.plot(res["xw"], res["yfit"], lw=1.2, label="fit")

        # Dotted component curves (bkg + each Gaussian that passed threshold)
        for comp in res["components"]:
            ax.plot(res["xw"], comp, ":", lw=0.9, alpha=0.6, label=None)

        for c in centers_v:
            ax.axvline(c, linestyle="--", alpha=0.6, lw=0.9)

        ax.set_xlabel("q (1/Å)")
        ax.set_ylabel("Intensity (a.u.)")
        ax.set_title(f"Frame {args.frame} multi-peak fit")
        ax.minorticks_on()
        ax.legend(fontsize=8, ncol=2)

        # Table with fitted metrics
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

    # Track all frames
    nuse = nframes
    npeaks_seeded = len(centers0)
    centers_trk = np.full((nuse, npeaks_seeded + 1), np.nan)  # +1 in case a shoulder is accepted
    fwhm_trk = np.full((nuse, npeaks_seeded + 1), np.nan)

    seeds = centers0.copy()
    for f in range(nuse):
        y = I_full[f]
        res = fit_frame(x, y, seeds, WINDOW/2.0)
        if not res["success"]:
            res = fit_frame(x, y, seeds, WINDOW)
        if res["success"]:
            # store up to (seeded+1) peaks that passed threshold
            vis = np.isfinite(res["centers"])
            cvis = res["centers"][vis]
            wvis = res["fwhm"][vis]
            k = min(cvis.size, centers_trk.shape[1])
            centers_trk[f, :k] = cvis[:k]
            fwhm_trk[f, :k] = wvis[:k]
            # update seeds from visible peaks if available, else keep old seeds
            if cvis.size >= seeds.size:
                seeds = cvis[:seeds.size]
        # else keep NaNs & previous seeds

    # CSV (fixed number of columns: seeded + one optional)
    base = os.path.splitext(os.path.basename(args.h5))[0]
    csv_path = f"{base}_multi_peak_tracking.csv"
    header_cols = ["frame"] + [f"center_{i}" for i in range(centers_trk.shape[1])] + \
                  [f"FWHM_{i}" for i in range(centers_trk.shape[1])]
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
