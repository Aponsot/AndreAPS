#!/usr/bin/env python3
# Pixel-integrated Gaussian multi-peak fit with ONE mechanism:
# If residual is high, FORCE-SPLIT the tallest peak into two and refit.
# Linear background. Height floor governs admission & reporting.
#
# CLI unchanged: --h5, --centers, --frame

import argparse, os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from lmfit.models import LinearModel
from lmfit import Model, Parameters
from numpy import sqrt
from scipy.special import erf

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

# ------------------------------
# Tunables (minimal + force-split)
# ------------------------------
HALF_WINDOW = 0.13
MIN_POINTS  = 8

# Peak reporting/admission floor (both children must pass after split)
PEAK_HEIGHT_MIN = 10.0

# Sigma bounds (per component)
SIGMA_MIN_FIT = 0.005
SIGMA_MAX_FIT = 0.15    # slight headroom helps in broad frames

# Per-seed drift limits (asymmetric, relative to seed)
DRIFT_NEG = 0.15
DRIFT_POS = 0.010

# Minimum separation between components
MIN_SEP = 0.00040

# Model selection requirement for accepting a split (ΔAIC improvement)
AIC_IMPROVE = 0.3

# Plot title time scaling (0 or None -> use frame index)
SEC_PER_FRAME = 0.004

# --- FORCE-SPLIT controls ---
# Fire split if max|residual| >= FORCE_SPLIT_NOISE_MULT * noise
FORCE_SPLIT_NOISE_MULT   = 6.0
# Children centers offset ~ frac * parent_sigma (clamped by MIN_SEP and residual geometry)
FORCE_SPLIT_DELTA_SIGMA_FRAC = 0.7
# Accept split if (ΔAIC >= AIC_IMPROVE) OR (resid_rms drops by >= this fraction)
FORCE_SPLIT_RESID_DROP_FRAC  = 0.20
# Children sigma bounds relative to parent
FORCE_SPLIT_SIGMA_FRAC_MIN = 0.6
FORCE_SPLIT_SIGMA_FRAC_MAX = 2.2

DEBUG = False  # flip to True for per-frame diagnostics

# ------------------------------
# Pixel-integrated Gaussian model
# ------------------------------
def bin_edges_from_centers(x):
    x = np.asarray(x, float)
    if x.size < 2:
        dx = 1.0
        return np.array([x[0]-0.5*dx, x[0]+0.5*dx], float)
    mids = 0.5 * (x[:-1] + x[1:])
    edges = np.empty(x.size + 1, float)
    edges[1:-1] = mids
    edges[0]  = x[0] - (mids[0] - x[0])
    edges[-1] = x[-1] + (x[-1] - mids[-1])
    return edges

# amplitude == total area
def pixint_gauss(x, amplitude, center, sigma):
    sigma = max(float(sigma), 1e-12)
    edges = bin_edges_from_centers(x)
    t1 = (edges[1:] - center) / (sigma * sqrt(2.0))
    t0 = (edges[:-1] - center) / (sigma * sqrt(2.0))
    return amplitude * 0.5 * (erf(t1) - erf(t0))

# ------------------------------
# Helpers
# ------------------------------
def parse_centers(s: str):
    vals = [float(v) for v in s.split(",") if v.strip()]
    if not vals:
        raise ValueError("No centers parsed from --centers.")
    return np.array(vals, float)

def sigma_to_fwhm(sigma):
    return 2.354820045 * sigma

def robust_sigma(y):
    y = np.asarray(y, float)
    med = np.median(y)
    return 1.4826 * np.median(np.abs(y - med)) + 1e-12

def r2_score(y_true, y_fit):
    y_true = np.asarray(y_true, float); y_fit = np.asarray(y_fit, float)
    ss_res = np.sum((y_true - y_fit)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2) + 1e-12
    return 1.0 - ss_res/ss_tot

def load_q_and_I(h5_path):
    with h5py.File(h5_path, "r") as f:
        if "q" in f:
            x = np.asarray(f["q"][:], float)
        elif "tth" in f:
            x = np.asarray(f["tth"][:], float)
        else:
            raise ValueError("HDF5 must contain 'q' or 'tth'.")
        I_full = np.asarray(f["int"][:], float)

    if I_full.ndim == 1:
        I_full = I_full[None, :]
    elif I_full.ndim > 2:
        I_full = I_full.mean(axis=tuple(range(1, I_full.ndim)))

    if I_full.shape[1] != x.shape[0]:
        raise ValueError(f"Shape mismatch: int.shape={I_full.shape}, x.shape={x.shape}")
    return x, I_full

def window_mask(x, centers, halfwidth):
    lo = float(np.min(centers) - halfwidth)
    hi = float(np.max(centers) + halfwidth)
    return (x >= lo) & (x <= hi)

def _build_seed_model(xw, yw, seeds):
    # linear background init
    try:
        bkg_slope, bkg_intercept = np.polyfit(xw, yw, 1)
    except Exception:
        bkg_slope, bkg_intercept = 0.0, float(np.median(yw))

    model = LinearModel(prefix="bkg_")
    params = model.make_params(bkg_slope=bkg_slope, bkg_intercept=bkg_intercept)

    span = max(xw[-1] - xw[0], 1e-9)
    sigma0_base = max(span / (7.0 * len(seeds)), 1e-6)
    sigma0 = float(np.clip(sigma0_base, SIGMA_MIN_FIT, SIGMA_MAX_FIT))

    for i, c_seed in enumerate(seeds):
        g = Model(pixint_gauss, prefix=f"g{i}_")
        model = model + g
        idx = np.abs(xw - c_seed).argmin()
        y_at_seed = yw[idx]
        y_bkg = bkg_slope * xw[idx] + bkg_intercept
        height0 = max(y_at_seed - y_bkg, np.std(yw) * 0.5)
        amp0 = max(height0 * sigma0 * np.sqrt(2.0 * np.pi), 0.0)

        params.update(g.make_params(amplitude=amp0, center=c_seed, sigma=sigma0))
        params[f"g{i}_sigma"].set(min=SIGMA_MIN_FIT, max=SIGMA_MAX_FIT)
        params[f"g{i}_amplitude"].set(min=0.0)
        params[f"g{i}_center"].set(min=c_seed - DRIFT_NEG, max=c_seed + DRIFT_POS)
    return model, params

def _extract_metrics(result, xw):
    bkg_slope = result.params.get("bkg_slope").value if "bkg_slope" in result.params else 0.0
    bkg_intercept = result.params.get("bkg_intercept").value if "bkg_intercept" in result.params else 0.0
    bkg_line = bkg_slope * xw + bkg_intercept

    centers, sigmas, amps = [], [], []
    i = 0
    while f"g{i}_center" in result.params:
        centers.append(result.params[f"g{i}_center"].value)
        sigmas.append (result.params[f"g{i}_sigma"].value)
        amps.append   (result.params[f"g{i}_amplitude"].value)  # AREA
        i += 1
    centers = np.asarray(centers, float)
    sigmas  = np.asarray(sigmas,  float)
    amps    = np.asarray(amps,    float)

    heights = np.full_like(centers, np.nan, float)
    fwhm    = np.full_like(centers, np.nan, float)
    peak_at_center = np.full_like(centers, np.nan, float)
    for j in range(centers.size):
        if np.isfinite(sigmas[j]) and sigmas[j] > 0:
            heights[j] = amps[j] / (sigmas[j] * np.sqrt(2.0 * np.pi))   # AREA→HEIGHT
            fwhm[j] = sigma_to_fwhm(sigmas[j])
        peak_at_center[j] = (bkg_slope * centers[j] + bkg_intercept) + (heights[j] if np.isfinite(heights[j]) else 0.0)

    comps = []
    for a, c, s in zip(amps, centers, sigmas):
        if np.all(np.isfinite([a, c, s])) and s > 0:
            # individual component (no bkg)
            edges = _gaussian_y(xw, a, c, s)
            comps.append(edges)
        else:
            comps.append(np.full_like(xw, np.nan, float))

    return bkg_line, centers, sigmas, amps, heights, fwhm, peak_at_center, comps

def _gaussian_y(x, amp_area, cen, sig):
    return pixint_gauss(x, amp_area, cen, sig)

def _rebuild_from_kept(xw, yw, result, keep_mask):
    params_new = Parameters()
    model_new = LinearModel(prefix="bkg_")
    for nm in ["bkg_slope", "bkg_intercept"]:
        if nm in result.params:
            p = result.params[nm]
            params_new.add(nm, value=p.value, min=p.min, max=p.max, vary=p.vary)
        else:
            params_new.add(nm, value=0.0)

    next_idx = 0
    j = 0
    while f"g{j}_center" in result.params:
        if keep_mask[j]:
            gk = Model(pixint_gauss, prefix=f"g{next_idx}_")
            model_new = model_new + gk
            ccur = result.params[f"g{j}_center"].value
            scur = result.params[f"g{j}_sigma"].value
            acur = result.params[f"g{j}_amplitude"].value
            params_new.update(gk.make_params(center=ccur, sigma=scur, amplitude=acur))
            params_new[f"g{next_idx}_sigma"].set(min=SIGMA_MIN_FIT, max=SIGMA_MAX_FIT)
            params_new[f"g{next_idx}_amplitude"].set(min=0.0)
            params_new[f"g{next_idx}_center"].set(min=ccur - DRIFT_NEG, max=ccur + DRIFT_POS)
            next_idx += 1
        j += 1

    refit = model_new.fit(yw, params_new, x=xw, nan_policy="omit")
    return model_new, refit

def _build_params_from_result(res, drop_idx=None):
    params_new = Parameters()
    model_expr = LinearModel(prefix="bkg_")
    for nm in ["bkg_slope","bkg_intercept"]:
        if nm in res.params:
            p = res.params[nm]
            params_new.add(nm, value=p.value, min=p.min, max=p.max, vary=p.vary)
        else:
            params_new.add(nm, value=0.0)
    next_idx = 0
    j = 0
    while f"g{j}_center" in res.params:
        if drop_idx is not None and j == drop_idx:
            j += 1; continue
        g = Model(pixint_gauss, prefix=f"g{next_idx}_")
        model_expr = model_expr + g
        params_new.update(g.make_params(
            center=res.params[f"g{j}_center"].value,
            sigma =res.params[f"g{j}_sigma"].value,
            amplitude=res.params[f"g{j}_amplitude"].value))
        params_new[f"g{next_idx}_sigma"].set(min=SIGMA_MIN_FIT, max=SIGMA_MAX_FIT)
        params_new[f"g{next_idx}_amplitude"].set(min=0.0)
        ccur = res.params[f"g{j}_center"].value
        params_new[f"g{next_idx}_center"].set(min=ccur-DRIFT_NEG, max=ccur+DRIFT_POS)
        next_idx += 1; j += 1
    return model_expr, params_new, next_idx

# ------------------------------
# FORCE-SPLIT of tallest peak
# ------------------------------
def _force_split_if_needed(xw, yw, result, seed_cap):
    """
    If residual is high, split tallest component into two children and refit.
    Respect seed_cap by optionally dropping the weakest other component.
    Accept if (ΔAIC >= AIC_IMPROVE) OR (RMS residual drops by >= FORCE_SPLIT_RESID_DROP_FRAC),
    AND both children heights >= PEAK_HEIGHT_MIN and separation >= MIN_SEP.
    """
    # current comps
    n_now = 0
    while f"g{n_now}_center" in result.params:
        n_now += 1
    if n_now == 0:
        return result, False

    # Extract current metrics
    bkg_line, centers, sigmas, amps, heights, _, _, _ = _extract_metrics(result, xw)
    resid_vec = yw - (bkg_line + sum_component_stack(xw, result))
    noise = robust_sigma(resid_vec)
    resid_max = float(np.max(np.abs(resid_vec))) if resid_vec.size else 0.0
    resid_rms0 = float(np.sqrt(np.mean(resid_vec**2))) if resid_vec.size else 0.0

    if noise <= 0:
        return result, False
    if resid_max < FORCE_SPLIT_NOISE_MULT * noise:
        return result, False

    # Identify tallest component (by height)
    j_main = int(np.nanargmax(heights))
    main_c = float(centers[j_main])
    main_s = max(float(sigmas[j_main]), 1e-12)
    main_a = max(float(amps[j_main]),   1e-12)

    # Residual centroid to choose split direction
    r = resid_vec
    rpos = np.maximum(r, 0.0)
    left_mask  = xw <  main_c
    right_mask = xw >  main_c

    def centroid(mask):
        if not np.any(mask):
            return None, 0.0
        w = rpos[mask]
        if np.sum(w) <= 0:
            return None, 0.0
        x_side = xw[mask]
        xc = float(np.sum(x_side * w) / np.sum(w))
        area = float(np.sum(w))
        return xc, area

    xL, areaL = centroid(left_mask)
    xR, areaR = centroid(right_mask)
    if (xL is None) and (xR is None):
        return result, False

    # Choose side by larger positive-residual area; set direction sign
    if (xR is not None and areaR >= areaL) or (xL is None):
        x_centroid = xR
        sign_dir = +1.0
    else:
        x_centroid = xL
        sign_dir = -1.0

    # Compute child centers: symmetric around main_c but nudged toward residual centroid
    # base delta from parent sigma
    delta0 = max(FORCE_SPLIT_DELTA_SIGMA_FRAC * main_s, MIN_SEP)
    # also consider distance to residual centroid (helps aim the split)
    if x_centroid is not None:
        delta0 = max(delta0, min(abs(x_centroid - main_c), 2.0*main_s))
    delta = float(np.clip(delta0, MIN_SEP, 2.5*main_s))

    c1 = main_c - 0.5 * delta
    c2 = main_c + 0.5 * delta

    # Enforce minimum separation
    if abs(c2 - c1) < MIN_SEP:
        c1 = main_c - 0.5 * MIN_SEP
        c2 = main_c + 0.5 * MIN_SEP

    # Children sigmas tied to parent within range
    s_child = float(np.clip(main_s,
                            FORCE_SPLIT_SIGMA_FRAC_MIN*main_s,
                            FORCE_SPLIT_SIGMA_FRAC_MAX*main_s))
    s1 = s2 = float(np.clip(s_child, SIGMA_MIN_FIT, SIGMA_MAX_FIT))

    # Split area: start 50/50
    a1 = a2 = 0.5 * main_a

    # Build model: drop main, add two children; if at cap, drop weakest other
    over_cap = (n_now >= seed_cap)  # split would increase count by +1
    drop_idx = None
    if over_cap:
        # choose weakest other (exclude j_main)
        idx_others = [i for i in range(n_now) if i != j_main]
        if idx_others:
            h_others = heights[idx_others]
            drop_idx = idx_others[int(np.nanargmin(h_others))]
        else:
            drop_idx = j_main  # degenerate; should not happen

    model_expr, params_new, next_idx = _build_params_from_result(result, drop_idx=j_main if drop_idx is None else drop_idx)

    # If we dropped someone else to honor cap, we need to keep main if not dropped.
    # Ensure main is removed (we're replacing it with two)
    if drop_idx is not None and drop_idx != j_main:
        # we removed a different weakest peak; now remove main too by rebuilding again
        model_expr, params_new, next_idx = _build_params_from_result(result, drop_idx=j_main)

    # Add the two children
    g1 = Model(pixint_gauss, prefix=f"g{next_idx}_")
    model_expr = model_expr + g1
    params_new.update(g1.make_params(center=c1, sigma=s1, amplitude=a1))
    params_new[f"g{next_idx}_sigma"].set(min=SIGMA_MIN_FIT, max=SIGMA_MAX_FIT)
    params_new[f"g{next_idx}_amplitude"].set(min=0.0)
    params_new[f"g{next_idx}_center"].set(min=xw[0], max=xw[-1])
    next_idx += 1

    g2 = Model(pixint_gauss, prefix=f"g{next_idx}_")
    model_expr = model_expr + g2
    params_new.update(g2.make_params(center=c2, sigma=s2, amplitude=a2))
    params_new[f"g{next_idx}_sigma"].set(min=SIGMA_MIN_FIT, max=SIGMA_MAX_FIT)
    params_new[f"g{next_idx}_amplitude"].set(min=0.0)
    params_new[f"g{next_idx}_center"].set(min=xw[0], max=xw[-1])

    trial = model_expr.fit(yw, params_new, x=xw, nan_policy="omit")

    # Evaluate acceptance
    bkg_line_t, _, _, _, h_all_t, _, _, comps_t = _extract_metrics(trial, xw)
    child_heights_ok = np.sum(np.isfinite(h_all_t) & (h_all_t >= PEAK_HEIGHT_MIN)) >= 2
    daic_ok = (trial.aic <= result.aic - AIC_IMPROVE)

    comp_sum_t = bkg_line_t + (np.sum(np.vstack(comps_t), axis=0) if len(comps_t) else 0.0)
    resid_vec_t = yw - comp_sum_t
    resid_rms1 = float(np.sqrt(np.mean(resid_vec_t**2))) if resid_vec_t.size else resid_rms0
    resid_drop_ok = (resid_rms0 > 0.0) and ((resid_rms0 - resid_rms1) / resid_rms0 >= FORCE_SPLIT_RESID_DROP_FRAC)

    # Ensure the two closest peaks are at least MIN_SEP apart
    # (extract updated centers and check min spacing)
    _, centers_t, _, _, _, _, _, _ = _extract_metrics(trial, xw)
    minsep_ok = True
    if centers_t.size >= 2:
        diff = np.sort(np.abs(np.subtract.outer(centers_t, centers_t) + np.eye(centers_t.size)*1e9).ravel())[:centers_t.size]
        minsep_ok = (np.min(diff) >= MIN_SEP)

    accept = child_heights_ok and minsep_ok and (daic_ok or resid_drop_ok)

    if DEBUG:
        print(f"[force-split] resid_max/noise={resid_max/noise:.2f}, ΔAIC={result.aic - trial.aic:.3g}, "
              f"RMS drop={(resid_rms0 - resid_rms1)/max(resid_rms0,1e-9):.2%}, "
              f"heights_ok={child_heights_ok}, minsep_ok={minsep_ok}, accept={accept}")

    if accept:
        return trial, True
    return result, False

def sum_component_stack(xw, result):
    """Sum of all Gaussian components only (no background), as a vector."""
    comps = []
    i = 0
    while f"g{i}_center" in result.params:
        c = result.params[f"g{i}_center"].value
        s = result.params[f"g{i}_sigma"].value
        a = result.params[f"g{i}_amplitude"].value
        if np.all(np.isfinite([a, c, s])) and s > 0:
            comps.append(_gaussian_y(xw, a, c, s))
        i += 1
    if not comps:
        return np.zeros_like(xw, float)
    return np.sum(np.vstack(comps), axis=0)

# ------------------------------
# Fit a single frame
# ------------------------------
def fit_frame(x, y, seeds, halfwidth):
    m = window_mask(x, seeds, halfwidth)
    if not np.any(m):
        return {"success": False}
    xw, yw = x[m], y[m]
    if xw.size < MIN_POINTS:
        return {"success": False}

    base_model, params = _build_seed_model(xw, yw, seeds)
    try:
        result = base_model.fit(yw, params, x=xw, nan_policy="omit")
    except Exception:
        return {"success": False}

    # Initial prune by height and refit
    bkg_line, c_all, s_all, a_all, h_all, w_all, p_all, comps = _extract_metrics(result, xw)
    keep = np.isfinite(h_all) & (h_all >= PEAK_HEIGHT_MIN)
    if keep.size and not np.all(keep):
        _, result = _rebuild_from_kept(xw, yw, result, keep)
        bkg_line, c_all, s_all, a_all, h_all, w_all, p_all, comps = _extract_metrics(result, xw)

    # FORCE-SPLIT if residual is high, respecting seed cap
    result, _ = _force_split_if_needed(xw, yw, result, seed_cap=len(seeds))
    bkg_line, c_all, s_all, a_all, h_all, w_all, p_all, comps = _extract_metrics(result, xw)

    # Final hard gate for outputs
    valid = np.isfinite(h_all) & (h_all >= PEAK_HEIGHT_MIN)
    centers_out = c_all.copy(); centers_out[~valid] = np.nan
    fwhm_out    = w_all.copy(); fwhm_out[~valid]    = np.nan
    height_out  = h_all.copy(); height_out[~valid]  = np.nan
    peakfit_out = p_all.copy(); peakfit_out[~valid] = np.nan

    comp_sum = bkg_line + (np.sum(np.vstack(comps), axis=0) if len(comps) else 0.0)
    r2 = r2_score(yw, result.best_fit)
    resid_vec = yw - comp_sum
    resid_max_abs = float(np.max(np.abs(resid_vec))) if resid_vec.size else 0.0

    return {
        "success": True,
        "xw": xw, "yw": yw, "yfit": result.best_fit, "bkg": bkg_line,
        "centers": centers_out, "fwhm": fwhm_out, "height_fit": height_out,
        "peak_fit": peakfit_out, "components": comps, "comp_sum": comp_sum,
        "r2": r2, "result": result,
        "resid_max_abs": resid_max_abs
    }

# ------------------------------
# Visual style
# ------------------------------
def apply_pub_style():
    plt.rcParams.update({
        "figure.figsize": (9.6, 4.8),   # wide
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "font.size": 12,
        "axes.labelsize": 14,
        "legend.fontsize": 11,
        "legend.frameon": False,
        "axes.linewidth": 1.15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "axes.grid": False,
    })

def style_axes(ax, light_grid=True):
    for side in ("top","right","bottom","left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.15)
    ax.minorticks_on()
    if light_grid:
        ax.grid(True, which="major", alpha=0.12, linestyle="-", linewidth=0.6)

COMP_COLORS = [
    "tab:purple","tab:red","tab:brown","tab:pink","tab:olive","tab:cyan",
    "#7f7f7f","#9467bd","#8c564b"
]

# ------------------------------
# Main (CLI unchanged)
# ------------------------------
def main():
    ap = argparse.ArgumentParser(description="Pixel-integrated Gaussian tracker (linear bkg, height-prune, FORCE-SPLIT on high residual).")
    ap.add_argument("--h5", required=True, help="HDF5 with 'q' (or 'tth') and 'int'")
    ap.add_argument("--centers", required=True, help="Comma-separated initial peak centers (e.g., 2.975,3.124)")
    ap.add_argument("--frame", type=int, default=None, help="Fit a single frame index. Omit to track all frames.")
    args = ap.parse_args()

    seeds0 = parse_centers(args.centers)
    x, I_full = load_q_and_I(args.h5)
    nframes = I_full.shape[0]

    # -------- Single frame --------
    if args.frame is not None:
        if not (0 <= args.frame < nframes):
            raise ValueError(f"--frame {args.frame} is out of range [0, {nframes-1}]")
        y = I_full[args.frame]
        res = fit_frame(x, y, seeds0, HALF_WINDOW)
        if not res["success"]:
            print("Fit failed for the requested frame.")
            return

        # Terminal table
        vis = np.isfinite(res["centers"])
        centers_v = res["centers"][vis]
        fwhm_v    = res["fwhm"][vis]
        hfit_v    = res["height_fit"][vis]
        print("\nPEAKS (kept >= height floor):")
        print("Index\tCenter\t\tFWHM\t\tHeight")
        for i, (c, w, h) in enumerate(zip(centers_v, fwhm_v, hfit_v), start=1):
            print(f"{i}\t{c:.6f}\t{w:.6f}\t{h:.6f}")
        print(f"R2 = {res['r2']:.4f} | max|residual| = {res['resid_max_abs']:.3f}\n")

        # Plot
        apply_pub_style()
        from matplotlib.gridspec import GridSpec
        plt.rcParams.update({"figure.figsize": (10.5, 5.2)})

        fig = plt.figure()
        gs = GridSpec(2, 1, height_ratios=[3.0, 1.2], hspace=0.50)

        ax = fig.add_subplot(gs[0]); style_axes(ax, light_grid=True)
        ax.plot(res["xw"], res["yw"],  lw=1.2, color="tab:blue",   label="Data")
        ax.plot(res["xw"], res["yfit"], lw=1.8, color="tab:orange", label="Total fit")
        ax.plot(res["xw"], res["bkg"],  "--", lw=1.2, color="tab:green", alpha=0.7, label="Linear bkg")

        for idx, comp in enumerate(res["components"]):
            ax.plot(res["xw"], comp, lw=1.0, linestyle="--",
                    color=COMP_COLORS[idx % len(COMP_COLORS)], alpha=0.7)
        ax.plot(res["xw"], res["comp_sum"], lw=1.0, color="k", alpha=0.35)

        for c in centers_v:
            ax.axvline(c, linestyle="--", alpha=0.35, lw=0.9, color="0.4")

        lo = float(np.min(seeds0) - HALF_WINDOW)
        hi = float(np.max(seeds0) + HALF_WINDOW)
        ax.set_xlim(lo, hi)

        title_prefix = (f"{args.frame*float(SEC_PER_FRAME):.1f} s | "
                        if (SEC_PER_FRAME is not None and SEC_PER_FRAME > 0)
                        else f"Frame {args.frame} | ")
        ax.set_title(title_prefix + f"R²={res['r2']:.4f}", pad=6)
        ax.set_xlabel("q (1/Å)")
        ax.set_ylabel("Intensity (a.u.)")
        ax.legend(loc="upper right", ncol=1, fontsize=10)

        axr = fig.add_subplot(gs[1]); style_axes(axr, light_grid=False)
        resid = res["yw"] - res["yfit"]
        axr.plot(res["xw"], resid, lw=1.0, color="0.2")
        axr.axhline(0.0, color="0.25", lw=0.8)
        axr.set_xlabel("q (1/Å)")
        axr.set_ylabel("Residual")

        fig.tight_layout()
        plt.show()
        return

    # -------- Mapping (all frames) --------
    nuse = nframes
    npeaks = len(seeds0)
    centers_trk = np.full((nuse, npeaks), np.nan)
    fwhm_trk    = np.full((nuse, npeaks), np.nan)
    height_trk  = np.full((nuse, npeaks), np.nan)

    iterator = range(nuse)
    if tqdm is not None:
        iterator = tqdm(iterator, desc="Fitting frames", ncols=80)

    for f in iterator:
        y = I_full[f]
        res = fit_frame(x, y, seeds0, HALF_WINDOW)
        if not res["success"]:
            continue

        valid = np.isfinite(res["centers"]) & np.isfinite(res["height_fit"])
        if np.any(valid):
            c = res["centers"][valid]
            w = res["fwhm"][valid]
            h = res["height_fit"][valid]
            hi_mask = h >= PEAK_HEIGHT_MIN
            c, w, h = c[hi_mask], w[hi_mask], h[hi_mask]
        else:
            c = w = h = np.array([])

        if c.size:
            order = np.argsort(c)
            c, w, h = c[order], w[order], h[order]

        k = min(c.size, npeaks)
        centers_trk[f, :k] = c[:k]
        fwhm_trk[f, :k]    = w[:k]
        height_trk[f, :k]  = h[:k]

    apply_pub_style()
    plt.rcParams.update({"figure.figsize": (11.5, 4.6)})

    fig, ax = plt.subplots(); style_axes(ax, light_grid=True)

    frames = np.arange(nuse)
    xvals = (frames * float(SEC_PER_FRAME)) if (SEC_PER_FRAME is not None and SEC_PER_FRAME > 0) else frames
    xlabel = "Time (s)" if (SEC_PER_FRAME is not None and SEC_PER_FRAME > 0) else "Frame"

    any_plotted = False
    for j in range(npeaks):
        mask = (
            np.isfinite(centers_trk[:, j]) &
            np.isfinite(height_trk[:, j]) &
            (height_trk[:, j] >= PEAK_HEIGHT_MIN)
        )
        if not np.any(mask):
            continue
        sc = ax.scatter(
            xvals[mask],
            centers_trk[mask, j],
            c=height_trk[mask, j],
            cmap="plasma",
            s=18,
            linewidths=0.0,
            edgecolors="none",
        )
        any_plotted = True

    lo = float(np.min(seeds0) - HALF_WINDOW)
    hi = float(np.max(seeds0) + HALF_WINDOW)
    ax.set_ylim(lo, hi)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Center (q or 2θ)")
    ax.set_title("Peak centers over frames (color = fitted height)")

    if any_plotted:
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label("Height (fit)")

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()





