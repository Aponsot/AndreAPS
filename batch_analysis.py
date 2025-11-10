#!/usr/bin/env python3
# Core v1.4 — Gaussian multi-peak fitting with linear background
# - Single-frame and full-experiment tracking
# - Fixed number of peaks from --centers (hard cap, constant across frames)
# - Prune low-height components (< PEAK_HEIGHT_MIN) BEFORE shoulder detection
# - Edge/flank shoulder detector + classic residual peak add (both respect cap)
# - Identity-preserving map assignment (nearest-neighbor to previous seeds)
# - Border-aware asymmetric auto-expansion for edge-add only (then refit back)
# - Polished plotting (publishable) + plasma colormap for map points
# - No CSV writing

import argparse, os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from lmfit.models import GaussianModel, LinearModel

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

# ------------------------------
# Tunables
# ------------------------------
HALF_WINDOW = 0.10      # symmetric window half-width around [min(seeds), max(seeds)]
MIN_POINTS  = 8         # min points in window to attempt a fit

# Peak reporting/pruning floor (for keeping/plotting components)
PEAK_HEIGHT_MIN = 5.2

# Global sigma bounds
SIGMA_MIN_FIT = 0.001
SIGMA_MAX_FIT = 0.080

# Per-seed drift limits per frame (asymmetric)
DRIFT_NEG = 0.10
DRIFT_POS = 0.010

# Residual-shoulder logic (point-peak style; cap respected)
ENABLE_RESIDUAL = True
RESIDUAL_SNR    = 0.35       # residual peak must be ≥ SNR * robust_noise
MIN_SEP         = 0.00010    # min separation from existing centers
AIC_IMPROVE     = 4.0        # require ΔAIC ≤ -AIC_IMPROVE to accept addition

# Admission looseners for new components (avoid height floor blocking)
ADD_SNR_MIN         = 0.40   # admit new comp if height >= ADD_SNR_MIN*noise (even if < PEAK_HEIGHT_MIN)
EDGE_SNR_AREA_MIN   = 0.6    # area-SNR threshold for edge/flank addition (raised from 0.1)

# Border-aware expansion (only for edge-add; then project back to tight window)
AUTO_EDGE_EXPAND = True
BORDER_SNR       = 0.9       # residual at border must exceed this * noise
EXPAND_FACTOR    = 1.6
MAX_HALF_WINDOW  = 0.50

SEC_PER_FRAME    = 0.004     # for titles; set None to show "Frame N" instead

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
    y_true = np.asarray(y_true, float)
    y_fit  = np.asarray(y_fit,  float)
    ss_res = np.sum((y_true - y_fit)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2) + 1e-12
    return 1.0 - ss_res/ss_tot

def load_q_and_I(h5_path):
    with h5py.File(h5_path, "r") as f:
        if "q" in f:     x = np.asarray(f["q"][:], float)
        elif "tth" in f: x = np.asarray(f["tth"][:], float)
        else: raise ValueError("HDF5 must contain 'q' or 'tth'.")
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

def _gaussian_y(x, amp, cen, sig):
    return amp * np.exp(-(x - cen)**2 / (2.0 * sig**2))

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
        g = GaussianModel(prefix=f"g{i}_")
        model = model + g
        idx = np.abs(xw - c_seed).argmin()
        y_at_seed = yw[idx]
        y_bkg = bkg_slope * xw[idx] + bkg_intercept
        height0 = max(y_at_seed - y_bkg, np.std(yw) * 0.5)
        amp0 = max(height0 * sigma0 * np.sqrt(2.0 * np.pi), 0.0)

        params.update(g.make_params(center=c_seed, sigma=sigma0, amplitude=amp0))
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
        amps.append   (result.params[f"g{i}_amplitude"].value)
        i += 1
    centers = np.asarray(centers, float)
    sigmas  = np.asarray(sigmas,  float)
    amps    = np.asarray(amps,    float)

    heights = np.full_like(centers, np.nan, float)
    fwhm    = np.full_like(centers, np.nan, float)
    peak_at_center = np.full_like(centers, np.nan, float)
    for j in range(centers.size):
        if np.isfinite(sigmas[j]) and sigmas[j] > 0:
            heights[j] = amps[j] / (sigmas[j] * np.sqrt(2.0 * np.pi))
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
    """Rebuild a model keeping only components where keep_mask is True; refit."""
    from lmfit import Parameters
    params_new = Parameters()
    model_new = LinearModel(prefix="bkg_")
    # background
    for nm in ["bkg_slope", "bkg_intercept"]:
        if nm in result.params:
            p = result.params[nm]
            params_new.add(nm, value=p.value, min=p.min, max=p.max, vary=p.vary)
        else:
            params_new.add(nm, value=0.0)

    next_idx = 0
    # keep selected components; reindex tightly
    j = 0
    while f"g{j}_center" in result.params:
        if keep_mask[j]:
            gk = GaussianModel(prefix=f"g{next_idx}_")
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

def _accept_new_component(trial, base_aic, xw, new_height, noise):
    """Admission rule for a new/edge component."""
    daic_ok = (trial.aic <= base_aic - AIC_IMPROVE)
    height_ok = (np.isfinite(new_height) and
                 (new_height >= PEAK_HEIGHT_MIN or new_height >= ADD_SNR_MIN * max(noise, 1e-12)))
    return daic_ok and height_ok

def _build_params_from_result(res, drop_idx=None):
    from lmfit import Parameters
    params_new = Parameters()
    model_expr = LinearModel(prefix="bkg_")
    for nm in ["bkg_slope","bkg_intercept"]:
        if nm in res.params:
            p = res.params[nm]
            params_new.add(nm, value=p.value, min=p.min, max=p.max, vary=p.vary)
        else:
            params_new.add(nm, value=0.0)
    # copy existing (optionally drop one)
    next_idx = 0
    j = 0
    while f"g{j}_center" in res.params:
        if drop_idx is not None and j == drop_idx:
            j += 1; continue
        g = GaussianModel(prefix=f"g{next_idx}_")
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

def _try_residual_add(xw, yw, result, max_n):
    """Point-peak shoulder add (never exceed max_n; replace weakest at cap if better)."""
    # current component count
    n_now = 0
    while f"g{n_now}_center" in result.params:
        n_now += 1

    resid = yw - result.best_fit
    noise = robust_sigma(resid)
    if noise <= 0:
        return result, False

    idx = int(np.argmax(resid))
    if resid[idx] < RESIDUAL_SNR * noise:
        return result, False

    x0 = float(xw[idx])

    # abort if too close to existing center
    centers_old = []
    j = 0
    while f"g{j}_center" in result.params:
        centers_old.append(result.params[f"g{j}_center"].value)
        j += 1
    centers_old = np.asarray(centers_old, float)
    if centers_old.size and np.min(np.abs(centers_old - x0)) < MIN_SEP:
        return result, False

    # seed width: modest
    span = max(xw[-1] - xw[0], 1e-9)
    mean_sigma = np.nanmean([result.params[f"g{k}_sigma"].value for k in range(n_now)]) if n_now else span/10.0
    sigma_seed = float(np.clip(min(max(mean_sigma, span/12.0), span/8.0), SIGMA_MIN_FIT, SIGMA_MAX_FIT))
    amp_seed = max(resid[idx] * sigma_seed * np.sqrt(2.0 * np.pi), 1e-9)

    if n_now < max_n:
        model_expr, params_new, next_idx = _build_params_from_result(result)
        gnew = GaussianModel(prefix=f"g{next_idx}_")
        model_expr = model_expr + gnew
        params_new.update(gnew.make_params(center=x0, sigma=sigma_seed, amplitude=amp_seed))
        params_new[f"g{next_idx}_sigma"].set(min=SIGMA_MIN_FIT, max=SIGMA_MAX_FIT)
        params_new[f"g{next_idx}_amplitude"].set(min=0.0)
        params_new[f"g{next_idx}_center"].set(min=xw[0], max=xw[-1])

        trial = model_expr.fit(yw, params_new, x=xw, nan_policy="omit")
        _, c, s, a, h, _, _, _ = _extract_metrics(trial, xw)
        new_h = h[-1] if h.size else -np.inf
        if _accept_new_component(trial, result.aic, xw, new_h, noise):
            return trial, True
        return result, False

    # At cap: try replacing the weakest height
    _, _, _, _, h0, _, _, _ = _extract_metrics(result, xw)
    if h0.size == 0:
        return result, False
    weakest = int(np.nanargmin(h0))
    model_expr, params_new, next_idx = _build_params_from_result(result, drop_idx=weakest)
    gnew = GaussianModel(prefix=f"g{next_idx}_")
    model_expr = model_expr + gnew
    params_new.update(gnew.make_params(center=x0, sigma=sigma_seed, amplitude=amp_seed))
    params_new[f"g{next_idx}_sigma"].set(min=SIGMA_MIN_FIT, max=SIGMA_MAX_FIT)
    params_new[f"g{next_idx}_amplitude"].set(min=0.0)
    params_new[f"g{next_idx}_center"].set(min=xw[0], max=xw[-1])
    trial = model_expr.fit(yw, params_new, x=xw, nan_policy="omit")
    _, c, s, a, h, _, _, _ = _extract_metrics(trial, xw)
    new_h = h[-1] if h.size else -np.inf
    if _accept_new_component(trial, result.aic, xw, new_h, noise):
        return trial, True
    return result, False

def _try_edge_add(xw, yw, result, max_n):
    """
    Edge/flank-based shoulder add:
    - choose dominant peak
    - compute positive residual area on each side
    - if area-SNR is high, seed a broad Gaussian at residual-weighted centroid with width from residual spread
    - add or replace weakest (if at cap) with standard AIC & SNR admission
    """
    _, centers, sigmas, amps, heights, fwhm, _, _ = _extract_metrics(result, xw)
    if centers.size == 0:
        return result, False

    resid = yw - result.best_fit
    noise = robust_sigma(resid)
    if noise <= 0:
        return result, False

    # pick dominant component by height
    j_main = int(np.nanargmax(heights))
    x_main = centers[j_main]

    left_mask  = xw <  x_main
    right_mask = xw >  x_main
    rpos = np.maximum(resid, 0.0)

    def side_stats(msk):
        if not np.any(msk):
            return 0.0, np.nan, np.nan
        area = np.sum(rpos[msk])
        snr_area = area / (max(noise,1e-12) * np.sqrt(int(np.sum(msk))))
        if area <= 0:
            return 0.0, np.nan, np.nan
        x_side = xw[msk]; w = rpos[msk]
        xc = np.sum(x_side * w) / np.sum(w)
        var = max(np.sum(w * (x_side - xc)**2) / np.sum(w), 1e-12)
        sig = float(np.clip(np.sqrt(var), SIGMA_MIN_FIT, SIGMA_MAX_FIT))
        return snr_area, xc, sig

    snrL, xL, sL = side_stats(left_mask)
    snrR, xR, sR = side_stats(right_mask)

    if snrL < EDGE_SNR_AREA_MIN and snrR < EDGE_SNR_AREA_MIN:
        return result, False
    if snrR > snrL:
        x0, sigma_seed = xR, sR
        side_mask = right_mask
    else:
        x0, sigma_seed = xL, sL
        side_mask = left_mask

    # min separation from existing centers
    if centers.size and np.min(np.abs(centers - x0)) < MIN_SEP:
        return result, False

    # amplitude from area on the chosen side: area ≈ amp * sqrt(2π) * sigma
    area_side = np.sum(rpos[side_mask])
    amp_seed  = max(area_side / (np.sqrt(2.0*np.pi) * max(sigma_seed, 1e-12)), 1e-9)

    # Build trial similar to residual add (with replacement if at cap)
    n_now = 0
    while f"g{n_now}_center" in result.params:
        n_now += 1
    n_now = int(n_now)

    if n_now < max_n:
        model_expr, params_new, next_idx = _build_params_from_result(result)
        gnew = GaussianModel(prefix=f"g{next_idx}_")
        model_expr = model_expr + gnew
        params_new.update(gnew.make_params(center=x0, sigma=sigma_seed, amplitude=amp_seed))
        params_new[f"g{next_idx}_sigma"].set(min=SIGMA_MIN_FIT, max=SIGMA_MAX_FIT)
        params_new[f"g{next_idx}_amplitude"].set(min=0.0)
        params_new[f"g{next_idx}_center"].set(min=xw[0], max=xw[-1])
        trial = model_expr.fit(yw, params_new, x=xw, nan_policy="omit")
        _, c, s, a, h, _, _, _ = _extract_metrics(trial, xw)
        new_h = h[-1] if h.size else -np.inf
        if _accept_new_component(trial, result.aic, xw, new_h, noise):
            return trial, True
        return result, False

    # at cap: replace weakest height
    _, _, _, _, h0, _, _, _ = _extract_metrics(result, xw)
    if h0.size == 0:
        return result, False
    weakest = int(np.nanargmin(h0))
    model_expr, params_new, next_idx = _build_params_from_result(result, drop_idx=weakest)
    gnew = GaussianModel(prefix=f"g{next_idx}_")
    model_expr = model_expr + gnew
    params_new.update(gnew.make_params(center=x0, sigma=sigma_seed, amplitude=amp_seed))
    params_new[f"g{next_idx}_sigma"].set(min=SIGMA_MIN_FIT, max=SIGMA_MAX_FIT)
    params_new[f"g{next_idx}_amplitude"].set(min=0.0)
    params_new[f"g{next_idx}_center"].set(min=xw[0], max=xw[-1])
    trial = model_expr.fit(yw, params_new, x=xw, nan_policy="omit")
    _, c, s, a, h, _, _, _ = _extract_metrics(trial, xw)
    new_h = h[-1] if h.size else -np.inf
    if _accept_new_component(trial, result.aic, xw, new_h, noise):
        return trial, True
    return result, False

def _edge_border_snr(resid, k=5):
    """Return (snr_left, snr_right) using average positive residual over k-edge points."""
    noise = robust_sigma(resid)
    if noise <= 0: 
        return 0.0, 0.0
    k = max(1, min(k, resid.size//10))
    rpos = np.maximum(resid, 0.0)
    sl = np.mean(rpos[:k])  / noise
    sr = np.mean(rpos[-k:]) / noise
    return sl, sr

def _assign_by_nearest(prev, cur_c, cur_w, cur_h):
    """
    Greedy nearest-neighbor assignment of current fits (cur_*) onto previous seeds (prev).
    Returns arrays (C, W, H) of length len(prev), with NaN where unmatched.
    """
    prev = np.asarray(prev, float)
    cur_c = np.asarray(cur_c, float); cur_w = np.asarray(cur_w, float); cur_h = np.asarray(cur_h, float)
    used = [False]*cur_c.size
    out_c = np.full(prev.size, np.nan); out_w = np.full(prev.size, np.nan); out_h = np.full(prev.size, np.nan)

    for i, p in enumerate(prev):
        best = -1; dmin = 1e18
        for j in range(cur_c.size):
            if used[j] or not np.isfinite(cur_c[j]): 
                continue
            d = abs(cur_c[j] - p)
            if d < dmin:
                dmin = d; best = j
        if best >= 0:
            used[best] = True
            out_c[i], out_w[i], out_h[i] = cur_c[best], cur_w[best], cur_h[best]
    return out_c, out_w, out_h

# ------------------------------
# Fit one frame
# ------------------------------
def fit_frame(x, y, seeds, halfwidth, max_allowed):
    # tight window
    m = window_mask(x, seeds, halfwidth)
    if not np.any(m):
        return {"success": False}
    xw, yw = x[m], y[m]
    if xw.size < MIN_POINTS:
        return {"success": False}

    # 1) Seeded fit
    base_model, params = _build_seed_model(xw, yw, seeds)
    try:
        result = base_model.fit(yw, params, x=xw, nan_policy="omit")
    except Exception:
        return {"success": False}

    # 2) Height-based PRUNE before shoulder detection
    bkg_line, c_all, s_all, a_all, h_all, w_all, p_all, comps = _extract_metrics(result, xw)
    keep = np.isfinite(h_all) & (h_all >= PEAK_HEIGHT_MIN)
    if keep.size and not np.all(keep):
        _, result = _rebuild_from_kept(xw, yw, result, keep)
        bkg_line, c_all, s_all, a_all, h_all, w_all, p_all, comps = _extract_metrics(result, xw)

    # 3) EDGE/FLANK addition — with optional border-aware expansion
    if AUTO_EDGE_EXPAND:
        resid0 = yw - result.best_fit
        sl, sr = _edge_border_snr(resid0, k=5)
        if max(sl, sr) >= BORDER_SNR:
            # expand asymmetrically toward pressured edge
            hwL = min(halfwidth * (EXPAND_FACTOR if sl >= sr else 1.0), MAX_HALF_WINDOW)
            hwR = min(halfwidth * (EXPAND_FACTOR if sr >  sl else 1.0), MAX_HALF_WINDOW)
            m2 = (x >= np.min(seeds) - hwL) & (x <= np.max(seeds) + hwR)
            xw2, yw2 = x[m2], y[m2]
            # refit on expanded window using current params as init
            model2, params2 = _build_seed_model(xw2, yw2, seeds)
            i = 0
            while f"g{i}_center" in params2 and f"g{i}_center" in result.params:
                for nm in ("center","sigma","amplitude"):
                    params2[f"g{i}_{nm}"].value = result.params[f"g{i}_{nm}"].value
                i += 1
            result2 = model2.fit(yw2, params2, x=xw2, nan_policy="omit")
            result2, changed = _try_edge_add(xw2, yw2, result2, max_allowed)
            if changed:
                # project back to tight window by refitting with the new params
                model_back, params_back = _build_seed_model(xw, yw, seeds)
                i = 0
                while f"g{i}_center" in params_back and f"g{i}_center" in result2.params:
                    for nm in ("center","sigma","amplitude"):
                        params_back[f"g{i}_{nm}"].value = result2.params[f"g{i}_{nm}"].value
                    i += 1
                result = model_back.fit(yw, params_back, x=xw, nan_policy="omit")

    # regular edge add on tight window (still useful if expansion didn’t add)
    result, _ = _try_edge_add(xw, yw, result, max_allowed)
    bkg_line, c_all, s_all, a_all, h_all, w_all, p_all, comps = _extract_metrics(result, xw)

    # 3b) Then classic point-residual addition (still capped)
    if ENABLE_RESIDUAL and max_allowed > 0:
        result, _ = _try_residual_add(xw, yw, result, max_allowed)
        bkg_line, c_all, s_all, a_all, h_all, w_all, p_all, comps = _extract_metrics(result, xw)

    # 4) Apply reporting mask (based on PEAK_HEIGHT_MIN)
    valid = np.isfinite(h_all) & (h_all >= PEAK_HEIGHT_MIN)
    centers_out = c_all.copy(); centers_out[~valid] = np.nan
    fwhm_out    = w_all.copy(); fwhm_out[~valid]    = np.nan
    height_out  = h_all.copy(); height_out[~valid]  = np.nan
    peakfit_out = p_all.copy(); peakfit_out[~valid] = np.nan

    comp_sum = bkg_line + (np.sum(np.vstack(comps), axis=0) if len(comps) else 0.0)
    r2 = r2_score(yw, result.best_fit)
    return {
        "success": True,
        "xw": xw, "yw": yw, "yfit": result.best_fit, "bkg": bkg_line,
        "centers": centers_out, "fwhm": fwhm_out, "height_fit": height_out,
        "peak_fit": peakfit_out, "components": comps, "comp_sum": comp_sum,
        "r2": r2, "result": result
    }

# ------------------------------
# Visual style
# ------------------------------
def apply_pub_style():
    plt.rcParams.update({
        "figure.figsize": (6.5, 4.8),
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "font.size": 12,
        "axes.labelsize": 14,
        "legend.fontsize": 12,
        "legend.frameon": False,   # per preference
        "axes.linewidth": 1.15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "axes.grid": False,        # clean; we’ll add light grid selectively
    })

def style_axes(ax, light_grid=True):
    for side in ("top","right","bottom","left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.15)
    ax.minorticks_on()
    if light_grid:
        ax.grid(True, which="major", alpha=0.12, linestyle="-", linewidth=0.6)

# ------------------------------
# Main
# ------------------------------
def main():
    ap = argparse.ArgumentParser(description="Gaussian multi-peak tracker (linear background, fixed cap, identity map, border-aware edge-add).")
    ap.add_argument("--h5", required=True, help="HDF5 with 'q' (or 'tth') and 'int'")
    ap.add_argument("--centers", required=True, help="Comma-separated initial peak centers (e.g., 2.975,3.124)")
    ap.add_argument("--frame", type=int, default=None, help="Fit a single frame index. Omit to track all frames.")
    args = ap.parse_args()

    seeds0 = parse_centers(args.centers)
    x, I_full = load_q_and_I(args.h5)
    nframes = I_full.shape[0]
    max_allowed = len(seeds0)  # CONSTANT cap across the run

    # -------- Single frame --------
    if args.frame is not None:
