#!/usr/bin/env python3
# Pixel-integrated Gaussian multi-peak fit with ONE add-path:
# area-based residual add (centroid + sigma tied to main sigma).
# Linear background. Height floor governs both admission and reporting.
#
# CLI: --h5, --centers, --frame   (unchanged)

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
# Tunables (minimal set)
# ------------------------------
HALF_WINDOW = 0.13
MIN_POINTS  = 8

# Peak reporting/admission floor (applied when adding AND at the END)
PEAK_HEIGHT_MIN = 5000.0

# Sigma bounds
SIGMA_MIN_FIT = 0.001
SIGMA_MAX_FIT = 0.10

# Per-seed drift limits (asymmetric, relative to seed)
DRIFT_NEG = 0.15
DRIFT_POS = 0.010

# Separation guard between components
MIN_SEP = 0.00050

# Model selection requirement for adding a component
AIC_IMPROVE = 1.0

# Time scaling for titles (0 or None -> use frame index)
SEC_PER_FRAME = 0.004

# Area-based add/rescue behavior
SIGMA_ADDED_MIN_FRAC = 0.7     # new comp sigma >= 0.7 * main_sigma
SIGMA_ADDED_MAX_FRAC = 1.5     # and <= 1.5 * main_sigma
NOISE_TRIGGER_MULT   = 10.0     # rescue fire level = NOISE_TRIGGER_MULT * noise

DEBUG = False  # set True to print why adds are accepted/rejected

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

# amplitude = total area under the continuous Gaussian
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

def _gaussian_y(x, amp_area, cen, sig):
    return pixint_gauss(x, amp_area, cen, sig)

def _build_seed_model(xw, yw, seeds):
    # background init via linear fit
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
            comps.append(_gaussian_y(xw, a, c, s))
        else:
            comps.append(np.full_like(xw, np.nan, float))
    return bkg_line, centers, sigmas, amps, heights, fwhm, peak_at_center, comps

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

# ---- ONE add path: area-based residual add ----
def _try_area_add(xw, yw, result, max_n,
                  height_floor=PEAK_HEIGHT_MIN,
                  aic_improve=AIC_IMPROVE):
    # count components
    n_now = 0
    while f"g{n_now}_center" in result.params:
        n_now += 1
    if n_now == 0:
        return result, False

    # extract centers/sigmas/heights
    centers, sigmas, heights = [], [], []
    for k in range(n_now):
        c = result.params[f"g{k}_center"].value
        s = max(result.params[f"g{k}_sigma"].value, 1e-12)
        a = result.params[f"g{k}_amplitude"].value
        h = a / (s * np.sqrt(2.0*np.pi))   # area→height
        centers.append(c); sigmas.append(s); heights.append(h)
    centers = np.asarray(centers, float)
    sigmas  = np.asarray(sigmas,  float)
    heights = np.asarray(heights, float)
    j_main  = int(np.nanargmax(heights))
    main_c  = float(centers[j_main])
    main_s  = float(sigmas[j_main])

    # residuals & noise
    resid = yw - result.best_fit
    noise = robust_sigma(resid)
    if noise <= 0:
        return result, False
    rpos = np.maximum(resid, 0.0)

    def side_stats(mask):
        if not np.any(mask):
            return None
        w = rpos[mask]
        if np.sum(w) <= 0:
            return None
        x_side = xw[mask]
        area   = float(np.sum(w))
        xc     = float(np.sum(x_side * w) / np.sum(w))
        var    = max(np.sum(w * (x_side - xc)**2) / np.sum(w), 1e-12)
        s_est  = float(np.clip(np.sqrt(var), SIGMA_MIN_FIT, SIGMA_MAX_FIT))
        h_est  = area / (np.sqrt(2.0*np.pi) * max(s_est, 1e-12))
        return (area, xc, s_est, h_est)

    left  = side_stats(xw <  main_c)
    right = side_stats(xw >  main_c)

    cand = None
    if left and right:
        cand = left if left[3] >= right[3] else right
    elif left:
        cand = left
    elif right:
        cand = right
    else:
        return result, False

    area_side, x0_centroid, sigma_est, h_est = cand

    # spacing guard
    if centers.size and np.min(np.abs(centers - x0_centroid)) < MIN_SEP:
        if DEBUG:
            print("[add] blocked by MIN_SEP")
        return result, False

    # tie sigma to main sigma to avoid needle splits
    sigma_seed = float(np.clip(main_s,
                               SIGMA_ADDED_MIN_FRAC*main_s,
                               SIGMA_ADDED_MAX_FRAC*main_s))
    sigma_seed = float(np.clip(sigma_seed, SIGMA_MIN_FIT, SIGMA_MAX_FIT))

    # amp seed equals residual area (pixel-integrated amp == area)
    amp_seed = max(float(area_side), 1e-9)

    # admission: height estimate and ΔAIC
    if not (np.isfinite(h_est) and (h_est >= height_floor)):
        if DEBUG:
            print(f"[add] blocked by height floor: h_est={h_est:.3g} < {height_floor}")
        return result, False

    # add or replace weakest, respecting max_n
    if n_now < max_n:
        model_expr, params_new, next_idx = _build_params_from_result(result)
        gnew = Model(pixint_gauss, prefix=f"g{next_idx}_")
        model_expr = model_expr + gnew
        params_new.update(gnew.make_params(center=x0_centroid,
                                           sigma=sigma_seed,
                                           amplitude=amp_seed))
        params_new[f"g{next_idx}_sigma"].set(min=SIGMA_MIN_FIT, max=SIGMA_MAX_FIT)
        params_new[f"g{next_idx}_amplitude"].set(min=0.0)
        params_new[f"g{next_idx}_center"].set(min=xw[0], max=xw[-1])
        trial = model_expr.fit(yw, params_new, x=xw, nan_policy="omit")
        _, _, _, _, h_all, _, _, _ = _extract_metrics(trial, xw)
        new_h = h_all[-1] if h_all.size else -np.inf
        daic_ok = (trial.aic <= result.aic - aic_improve)
        if DEBUG:
            print(f"[add] ΔAIC={result.aic - trial.aic:.3g}, new_h={new_h:.3g}")
        if daic_ok and np.isfinite(new_h) and (new_h >= height_floor):
            return trial, True
        return result, False

    # replace weakest by height if at cap
    _, _, _, _, h0, _, _, _ = _extract_metrics(result, xw)
    if h0.size == 0:
        return result, False
    weakest = int(np.nanargmin(h0))
    model_expr, params_new, next_idx = _build_params_from_result(result, drop_idx=weakest)
    gnew = Model(pixint_gauss, prefix=f"g{next_idx}_")
    model_expr = model_expr + gnew
    params_new.update(gnew.make_params(center=x0_centroid,
                                       sigma=sigma_seed,
                                       amplitude=amp_seed))
    params_new[f"g{next_idx}_sigma"].set(min=SIGMA_MIN_FIT, max=SIGMA_MAX_FIT)
    params_new[f"g{next_idx}_amplitude"].set(min=0.0)
    params_new[f"g{next_idx}_center"].set(min=xw[0], max=xw[-1])
    trial = model_expr.fit(yw, params_new, x=xw, nan_policy="omit")
    _, _, _, _, h_all, _, _, _ = _extract_metrics(trial, xw)
    new_h = h_all[-1] if h_all.size else -np.inf
    daic_ok = (trial.aic <= result.aic - aic_improve)
    if DEBUG:
        print(f"[add-replace] ΔAIC={result.aic - trial.aic:.3g}, new_h={new_h:.3g}")
    if daic_ok and np.isfinite(new_h) and (new_h >= height_floor):
        return trial, True
    return result, False

def _rescue_double_peak(xw, yw, result, seed_cap):
    """
    Use the SAME area-based add once more if:
     - only one kept peak after height floor, and
     - residual is large relative to noise
    """
    resid = yw - result.best_fit
    noise = robust_sigma(resid)
    trigger = NOISE_TRIGGER_MULT * noise

    _, _, _, _, h_all, _, _, _ = _extract_metrics(result, xw)
    kept = np.isfinite(h_all) & (h_all >= PEAK_HEIGHT_MIN)
    n_kept = int(np.sum(kept))

    if (n_kept == 1) and (np.max(np.abs(resid)) >= trigger) and (seed_cap >= 2):
        trial, ok = _try_area_add(xw, yw, result, max_n=seed_cap,
                                  height_floor=PEAK_HEIGHT_MIN,
                                  aic_improve=AIC_IMPROVE)
        if ok:
            return trial, True
    return result, False

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

    # initial prune by height (and refit)
    bkg_line, c_all, s_all, a_all, h_all, w_all, p_all, comps = _extract_metrics(result, xw)
    keep = np.isfinite(h_all) & (h_all >= PEAK_HEIGHT_MIN)
    if keep.size and not np.all(keep):
        _, result = _rebuild_from_kept(xw, yw, result, keep)
        bkg_line, c_all, s_all, a_all, h_all, w_all, p_all, comps = _extract_metrics(result, xw)

    # ONE add attempt (area-based), bounded by number of seeds
    max_allowed = len(seeds)
    if max_allowed > 0:
        result, _ = _try_area_add(xw, yw, result, max_n=max_allowed,
                                  height_floor=PEAK_HEIGHT_MIN,
                                  aic_improve=AIC_IMPROVE)
        bkg_line, c_all, s_all, a_all, h_all, w_all, p_all, comps = _extract_metrics(result, xw)

    # rescue if still looks like a wide single
    result, _ = _rescue_double_peak(xw, yw, result, seed_cap=len(seeds))
    bkg_line, c_all, s_all, a_all, h_all, w_all, p_all, comps = _extract_metrics(result, xw)

    # final hard gate for outputs
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

# Distinct colors for Gaussian components
COMP_COLORS = [
    "tab:purple","tab:red","tab:brown","tab:pink","tab:olive","tab:cyan",
    "#7f7f7f","#9467bd","#8c564b"
]

# ------------------------------
# Main (CLI unchanged)
# ------------------------------
def main():
    ap = argparse.ArgumentParser(description="Pixel-integrated Gaussian tracker (linear bkg, height-prune, single area-based add + rescue).")
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

        # ---- Plot (single frame, wide) ----
        apply_pub_style()
        from matplotlib.gridspec import GridSpec
        plt.rcParams.update({"figure.figsize": (10.5, 5.2)})

        fig = plt.figure()
        gs = GridSpec(2, 1, height_ratios=[3.0, 1.2], hspace=0.50)

        ax = fig.add_subplot(gs[0]); style_axes(ax, light_grid=True)

        # Data: BLUE solid
        ax.plot(res["xw"], res["yw"],  lw=1.2, color="tab:blue", label="Data")

        # Total fit: ORANGE solid
        ax.plot(res["xw"], res["yfit"], lw=1.8, color="tab:orange", label="Total fit")

        # Linear background: GREEN dashed, alpha 0.7
        ax.plot(res["xw"], res["bkg"],  "--", lw=1.2, color="tab:green", alpha=0.7, label="Linear bkg")

        # Each Gaussian: distinct color dashed, alpha 0.7
        for idx, comp in enumerate(res["components"]):
            ax.plot(res["xw"], comp, lw=1.0, linestyle="--",
                    color=COMP_COLORS[idx % len(COMP_COLORS)], alpha=0.7)

        # Composite sum (light)
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

        # Residual panel
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

    # ---- Map plot (wide, legend removed) ----
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

