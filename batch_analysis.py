#!/usr/bin/env python3
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
# Tunables
# ------------------------------
HALF_WINDOW = 0.20
MIN_POINTS  = 8

# Height floor (admission + reporting)
PEAK_HEIGHT_MIN = 1000

# Sigma bounds (per component)
SIGMA_MIN_FIT = 0.0000001
SIGMA_MAX_FIT = 0.15

# Per-seed drift limits (asymmetric, relative to seed)
DRIFT_NEG = 0.15
DRIFT_POS = 0.090

# Minimum separation
MIN_SEP = 0.000040

# Acceptance requirement (ΔAIC improvement)
AIC_IMPROVE = 1.0

# Plot title time scaling (0/None -> use frame index)
SEC_PER_FRAME = 0.004

# --- FORCE-SPLIT triggers & guards ---
# ABSOLUTE trigger (in same intensity units as your data). When set (not None),
# it OVERRIDES the ratio trigger for clean testing, but the relative trigger can still fire.
FORCE_SPLIT_ABS = 15.0

# Ratio trigger (fallback if FORCE_SPLIT_ABS is None)
FORCE_SPLIT_NOISE_MULT = 20.0   # split if max|resid| >= K * noise (noise = 1.4826*MAD)

# Relative trigger for weak/late frames (both must hold)
FORCE_SPLIT_REL_MAIN  = 0.30    # resid_max ≥ 30% of tallest peak height
FORCE_SPLIT_NOISE_MIN = 4.0     # and resid_max ≥ 4×noise

# Candidate geometry for children
FORCE_SPLIT_DELTA_SIGMA_FRAC = 0.5   # child offset ~ frac * parent_sigma (clamped below)
FORCE_SPLIT_SIGMA_FRAC_MIN   = 0.7
FORCE_SPLIT_SIGMA_FRAC_MAX   = 1.6
DELTA_MIN_SIGMA_FRAC         = 0.40
DELTA_MAX_SIGMA_FRAC         = 1.80

# Acceptance gates
FORCE_SPLIT_RESID_DROP_FRAC = 0.50  # require ≥50% RMS residual drop (AND ΔAIC pass)
CHILD_HEIGHT_FRAC           = 0.25  # each child ≥ 25% of parent height
AREA_CONSERVE_MIN_FRAC      = 0.70  # total child area within [70%,130%] of parent
AREA_CONSERVE_MAX_FRAC      = 1.30
SIDE_AREA_SNR_MULT          = 6.0   # side residual area ≥ K * noise * sqrt(N_side); set 0 to disable

# Debug toggles
DEBUG = True
DEBUG_PROVENANCE_TEXT = True

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

def pixint_gauss(x, amplitude, center, sigma):
    # amplitude = integrated area under continuous Gaussian
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

# ----- center bound utilities -----
def nearest_seed_bounds(center_value, seeds):
    seed = float(seeds[np.argmin(np.abs(seeds - center_value))])
    return seed - DRIFT_NEG, seed + DRIFT_POS

def clamp_center_param(params, name, center_value, seeds):
    lo, hi = nearest_seed_bounds(center_value, seeds)
    params[name].set(min=lo, max=hi)

# ------------------------------
# Provenance utilities
# ------------------------------
def prov_new():
    return {}

def prov_reindex(prov_in, old_to_new):
    prov_out = {}
    for old_i, new_i in old_to_new.items():
        old_key = f"g{old_i}_"; new_key = f"g{new_i}_"
        prov_out[new_key] = prov_in.get(old_key, "seed?")
    return prov_out

def prov_add(prov_map, idx, label):
    prov_map[f"g{idx}_"] = label

def prov_list_from_params(result, prov_map):
    out = []
    i = 0
    while f"g{i}_center" in result.params:
        out.append(prov_map.get(f"g{i}_", "unk"))
        i += 1
    return out

# ------------------------------
# Model builders / extractors
# ------------------------------
def _build_seed_model(xw, yw, seeds):
    try:
        bkg_slope, bkg_intercept = np.polyfit(xw, yw, 1)
    except Exception:
        bkg_slope, bkg_intercept = 0.0, float(np.median(yw))

    model = LinearModel(prefix="bkg_")
    params = model.make_params(bkg_slope=bkg_slope, bkg_intercept=bkg_intercept)
    prov = prov_new()

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
        prov_add(prov, i, f"seed:{i}")
    return model, params, prov

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
            heights[j] = amps[j] / (sigmas[j] * np.sqrt(2.0 * np.pi))   # area -> height
            fwhm[j] = sigma_to_fwhm(sigmas[j])
        peak_at_center[j] = bkg_slope * centers[j] + bkg_intercept + (heights[j] if np.isfinite(heights[j]) else 0.0)

    comps = []
    for a, c, s in zip(amps, centers, sigmas):
        if np.all(np.isfinite([a, c, s])) and s > 0:
            comps.append(_gaussian_y(xw, a, c, s))
        else:
            comps.append(np.full_like(xw, np.nan, float))
    return bkg_line, centers, sigmas, amps, heights, fwhm, peak_at_center, comps

def _rebuild_from_kept(xw, yw, result, keep_mask, seeds, prov_in):
    params_new = Parameters()
    model_new = LinearModel(prefix="bkg_")
    for nm in ["bkg_slope", "bkg_intercept"]:
        if nm in result.params:
            p = result.params[nm]
            params_new.add(nm, value=p.value, min=p.min, max=p.max, vary=p.vary)
        else:
            params_new.add(nm, value=0.0)

    old_to_new = {}
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
            clamp_center_param(params_new, f"g{next_idx}_center", ccur, seeds)
            old_to_new[j] = next_idx
            next_idx += 1
        j += 1

    prov_new_map = prov_reindex(prov_in, old_to_new)
    refit = model_new.fit(yw, params_new, x=xw, nan_policy="omit")
    return model_new, refit, prov_new_map

def _build_params_from_result(res, seeds, prov_in, drop_idx=None):
    params_new = Parameters()
    model_expr = LinearModel(prefix="bkg_")
    for nm in ["bkg_slope","bkg_intercept"]:
        if nm in res.params:
            p = res.params[nm]
            params_new.add(nm, value=p.value, min=p.min, max=p.max, vary=p.vary)
        else:
            params_new.add(nm, value=0.0)
    old_to_new = {}
    next_idx = 0
    j = 0
    while f"g{j}_center" in res.params:
        if drop_idx is not None and j == drop_idx:
            j += 1; continue
        g = Model(pixint_gauss, prefix=f"g{next_idx}_")
        model_expr = model_expr + g
        ccur = res.params[f"g{j}_center"].value
        scur = res.params[f"g{j}_sigma"].value
        acur = res.params[f"g{j}_amplitude"].value
        params_new.update(g.make_params(center=ccur, sigma=scur, amplitude=acur))
        params_new[f"g{next_idx}_sigma"].set(min=SIGMA_MIN_FIT, max=SIGMA_MAX_FIT)
        params_new[f"g{next_idx}_amplitude"].set(min=0.0)
        clamp_center_param(params_new, f"g{next_idx}_center", ccur, seeds)
        old_to_new[j] = next_idx
        next_idx += 1; j += 1
    prov_new_map = prov_reindex(prov_in, old_to_new)
    return model_expr, params_new, next_idx, prov_new_map

def sum_component_stack(xw, result):
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
# FORCE-SPLIT of tallest peak (cap respected, seed-bounded centers, provenance)
# ------------------------------
def _force_split_if_needed(xw, yw, result, seed_cap, seeds, prov_in, debug=False):
    # NEVER exceed seed_cap
    n_now = 0
    while f"g{n_now}_center" in result.params:
        n_now += 1
    if n_now == 0 or n_now >= seed_cap:
        return result, False, prov_in

    # Extract current metrics
    bkg_line, centers, sigmas, amps, heights, _, _, _ = _extract_metrics(result, xw)
    resid_vec = yw - (bkg_line + sum_component_stack(xw, result))
    noise = robust_sigma(resid_vec)
    resid_max = float(np.max(np.abs(resid_vec))) if resid_vec.size else 0.0
    resid_rms0 = float(np.sqrt(np.mean(resid_vec**2))) if resid_vec.size else 0.0

    # Tallest component (needed for relative trigger)
    j_main = int(np.nanargmax(heights))
    main_c = float(centers[j_main])
    main_s = max(float(sigmas[j_main]), 1e-12)
    main_a = max(float(amps[j_main]),   1e-12)
    main_h = float(np.max(heights)) if heights.size else 0.0

    # ---------- TRIGGERS ----------
    abs_trigger   = (FORCE_SPLIT_ABS is not None) and (resid_max >= float(FORCE_SPLIT_ABS))
    ratio_trigger = (FORCE_SPLIT_ABS is None) and (noise > 0) and (resid_max >= FORCE_SPLIT_NOISE_MULT * noise)
    rel_trigger   = (main_h > 0) and (noise > 0) and \
                    (resid_max >= FORCE_SPLIT_REL_MAIN * main_h) and \
                    (resid_max >= FORCE_SPLIT_NOISE_MIN * noise)
    if not (abs_trigger or ratio_trigger or rel_trigger):
        return result, False, prov_in

    # Side area SNR gate (optional; slightly relax when peak is weak)
    rpos = np.maximum(resid_vec, 0.0)
    left_mask  = xw <  main_c
    right_mask = xw >  main_c
    def side_area_snr(mask):
        if not np.any(mask): return 0.0, 0, 0.0
        w = rpos[mask]; n = int(np.sum(mask))
        area = float(np.sum(w))
        snr  = area / (max(noise,1e-12) * np.sqrt(max(n,1)))
        return area, n, snr
    areaL, nL, snrL = side_area_snr(left_mask)
    areaR, nR, snrR = side_area_snr(right_mask)
    side_gate = SIDE_AREA_SNR_MULT
    if (main_h > 0) and (noise > 0) and (main_h < 3.0 * noise):
        side_gate = max(0.0, SIDE_AREA_SNR_MULT - 2.0)
    if side_gate > 0 and max(snrL, snrR) < side_gate:
        return result, False, prov_in

    # Residual centroid & delta
    def centroid(mask):
        if not np.any(mask): return None, 0.0
        w = rpos[mask]
        if np.sum(w) <= 0: return None, 0.0
        xs = xw[mask]
        return float(np.sum(xs*w)/np.sum(w)), float(np.sum(w))
    xL, _ = centroid(left_mask)
    xR, _ = centroid(right_mask)
    x_centroid = xR if (xR is not None and areaR >= areaL) or (xL is None) else xL

    delta0 = max(FORCE_SPLIT_DELTA_SIGMA_FRAC * main_s, MIN_SEP)
    if x_centroid is not None:
        delta0 = max(delta0, min(abs(x_centroid - main_c), 2.0*main_s))
    delta = float(np.clip(
        delta0,
        max(DELTA_MIN_SIGMA_FRAC*main_s, MIN_SEP),
        DELTA_MAX_SIGMA_FRAC*main_s
    ))
    c1 = main_c - 0.5 * delta
    c2 = main_c + 0.5 * delta
    if abs(c2 - c1) < MIN_SEP:
        c1 = main_c - 0.5 * MIN_SEP
        c2 = main_c + 0.5 * MIN_SEP

    # Children σ relative to parent
    s_child = float(np.clip(main_s,
                            FORCE_SPLIT_SIGMA_FRAC_MIN*main_s,
                            FORCE_SPLIT_SIGMA_FRAC_MAX*main_s))
    s1 = s2 = float(np.clip(s_child, SIGMA_MIN_FIT, SIGMA_MAX_FIT))

    # Split area 50/50; later we check area conservation
    a1 = a2 = 0.5 * main_a

    # Build model: drop main, add two children (count increases by +1; we already know n_now < seed_cap)
    model_expr, params_new, next_idx, prov_after = _build_params_from_result(result, seeds, prov_in, drop_idx=j_main)

    g1 = Model(pixint_gauss, prefix=f"g{next_idx}_")
    model_expr = model_expr + g1
    params_new.update(g1.make_params(center=c1, sigma=s1, amplitude=a1))
    params_new[f"g{next_idx}_sigma"].set(min=SIGMA_MIN_FIT, max=SIGMA_MAX_FIT)
    params_new[f"g{next_idx}_amplitude"].set(min=0.0)
    clamp_center_param(params_new, f"g{next_idx}_center", c1, seeds)
    prov_add(prov_after, next_idx, f"split:{j_main}->A")
    next_idx += 1

    g2 = Model(pixint_gauss, prefix=f"g{next_idx}_")
    model_expr = model_expr + g2
    params_new.update(g2.make_params(center=c2, sigma=s2, amplitude=a2))
    params_new[f"g{next_idx}_sigma"].set(min=SIGMA_MIN_FIT, max=SIGMA_MAX_FIT)
    params_new[f"g{next_idx}_amplitude"].set(min=0.0)
    clamp_center_param(params_new, f"g{next_idx}_center", c2, seeds)
    prov_add(prov_after, next_idx, f"split:{j_main}->B")

    trial = model_expr.fit(yw, params_new, x=xw, nan_policy="omit")

    # Evaluate acceptance
    bkg_line_t, _, _, amps_t, h_all_t, _, _, comps_t = _extract_metrics(trial, xw)
    child_heights_ok = np.sum(np.isfinite(h_all_t) & (h_all_t >= PEAK_HEIGHT_MIN)) >= 2
    child_height_rel_ok = np.sum(np.isfinite(h_all_t) & (h_all_t >= max(PEAK_HEIGHT_MIN, CHILD_HEIGHT_FRAC*main_h))) >= 2
    daic_ok = (trial.aic <= result.aic - AIC_IMPROVE)

    comp_sum_t = bkg_line_t + (np.sum(np.vstack(comps_t), axis=0) if len(comps_t) else 0.0)
    resid_vec_t = yw - comp_sum_t
    resid_rms1 = float(np.sqrt(np.mean(resid_vec_t**2))) if resid_vec_t.size else 0.0
    resid_drop_ok = (resid_rms0 > 0.0) and ((resid_rms0 - resid_rms1) / max(resid_rms0,1e-12) >= FORCE_SPLIT_RESID_DROP_FRAC)

    # area conservation wrt parent
    a_sum = float(np.nansum(amps_t))
    area_ok = (a_sum >= AREA_CONSERVE_MIN_FRAC*main_a) and (a_sum <= AREA_CONSERVE_MAX_FRAC*main_a)

    # Ensure min separation post-refit
    _, centers_t, _, _, _, _, _, _ = _extract_metrics(trial, xw)
    minsep_ok = True
    if centers_t.size >= 2:
        diffs = np.abs(np.subtract.outer(centers_t, centers_t))
        diffs += np.eye(centers_t.size) * 1e9
        minsep_ok = (np.min(diffs) >= MIN_SEP)

    accept = (daic_ok and resid_drop_ok and child_heights_ok and child_height_rel_ok and minsep_ok and area_ok)

    if DEBUG and debug:
        print(f"[trigger] resid_max={resid_max:.3f}, noise={noise:.3f}, "
              f"abs_thr={FORCE_SPLIT_ABS}, ratio_thr={FORCE_SPLIT_NOISE_MULT}, "
              f"rel_main={FORCE_SPLIT_REL_MAIN}, rel_noise_min={FORCE_SPLIT_NOISE_MIN}, "
              f"abs_ok={abs_trigger}, ratio_ok={ratio_trigger}, rel_ok={rel_trigger}")
        print(f"[force-split] comps_before={n_now}, cap={seed_cap}, ΔAIC={result.aic - trial.aic:.3g}, "
              f"RMS drop={(resid_rms0 - resid_rms1)/max(resid_rms0,1e-9):.2%}, area_ok={area_ok}, accept={accept}")

    if accept:
        # sanity: still at or under cap
        n_after = 0
        while f"g{n_after}_center" in trial.params:
            n_after += 1
        if n_after <= seed_cap:
            return trial, True, prov_after
    return result, False, prov_in

# ------------------------------
# Fit a single frame
# ------------------------------
def fit_frame(x, y, seeds, halfwidth, debug=False):
    m = window_mask(x, seeds, halfwidth)
    if not np.any(m):
        return {"success": False}
    xw, yw = x[m], y[m]
    if xw.size < MIN_POINTS:
        return {"success": False}

    base_model, params, prov = _build_seed_model(xw, yw, seeds)
    try:
        result = base_model.fit(yw, params, x=xw, nan_policy="omit")
    except Exception:
        return {"success": False}

    # Initial prune by height and refit (with seed-bounded centers)
    bkg_line, c_all, s_all, a_all, h_all, w_all, p_all, comps = _extract_metrics(result, xw)
    keep = np.isfinite(h_all) & (h_all >= PEAK_HEIGHT_MIN)
    if keep.size and not np.all(keep):
        _, result, prov = _rebuild_from_kept(xw, yw, result, keep, seeds, prov)
        bkg_line, c_all, s_all, a_all, h_all, w_all, p_all, comps = _extract_metrics(result, xw)

    # FORCE-SPLIT if residual is high, honoring seed cap
    result, _, prov = _force_split_if_needed(xw, yw, result, seed_cap=len(seeds), seeds=seeds, prov_in=prov, debug=debug)
    bkg_line, c_all, s_all, a_all, h_all, w_all, p_all, comps = _extract_metrics(result, xw)

    # Final hard gate for outputs (reporting only)
    valid = np.isfinite(h_all) & (h_all >= PEAK_HEIGHT_MIN)

    centers_out = c_all.copy(); centers_out[~valid] = np.nan
    fwhm_out    = w_all.copy(); fwhm_out[~valid]    = np.nan
    height_out  = h_all.copy(); height_out[~valid]  = np.nan
    area_out    = a_all.copy(); area_out[~valid]    = np.nan   # NEW: expose area for mapping color

    peakfit_out = p_all.copy(); peakfit_out[~valid] = np.nan

    comp_sum = bkg_line + (np.sum(np.vstack(comps), axis=0) if len(comps) else 0.0)
    r2 = r2_score(yw, result.best_fit)
    resid_vec = yw - comp_sum
    resid_max_abs = float(np.max(np.abs(resid_vec))) if resid_vec.size else 0.0

    # Provenance list aligned to component index order
    labels = prov_list_from_params(result, prov)

    return {
        "success": True,
        "xw": xw, "yw": yw, "yfit": result.best_fit, "bkg": bkg_line,
        "centers": centers_out, "fwhm": fwhm_out, "height_fit": height_out,
        "area_fit": area_out,  # NEW
        "peak_fit": peakfit_out, "components": comps, "comp_sum": comp_sum,
        "labels": labels, "r2": r2, "result": result,
        "resid_max_abs": resid_max_abs
    }

# ------------------------------
# Visual style
# ------------------------------
def apply_pub_style():
    plt.rcParams.update({
        "figure.figsize": (7, 5),
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
    ap = argparse.ArgumentParser(description="Pixel-integrated Gaussian tracker (linear bkg, height-prune, FORCE-SPLIT with ABS/REL/ratio triggers; cap & seed-bounded centers; provenance).")
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
        res = fit_frame(x, y, seeds0, HALF_WINDOW, debug=True)
        if not res["success"]:
            print("Fit failed for the requested frame.")
            return

        # Terminal table (+ residual debug)
        print(f"\n[debug] ABS={FORCE_SPLIT_ABS}, NOISE_MULT={FORCE_SPLIT_NOISE_MULT}, "
              f"REL_MAIN={FORCE_SPLIT_REL_MAIN}, NOISE_MIN={FORCE_SPLIT_NOISE_MIN}")
        print(f"[debug] resid_max={res['resid_max_abs']:.3f}\n")

        vis = np.isfinite(res["centers"])
        centers_v = res["centers"][vis]
        fwhm_v    = res["fwhm"][vis]
        hfit_v    = res["height_fit"][vis]
        labels_all = res["labels"]
        labels_v  = np.array(labels_all)[np.where(vis)[0]]

        print("PEAKS (kept >= height floor):")
        print("Idx\tCenter\t\tFWHM\t\tHeight\t\tSource")
        for i, (c, w, h, lab) in enumerate(zip(centers_v, fwhm_v, hfit_v, labels_v), start=1):
            print(f"{i}\t{c:.6f}\t{w:.6f}\t{h:.6f}\t{lab}")
        print(f"R2 = {res['r2']:.4f}\n")

        # Plot
        apply_pub_style()
        from matplotlib.gridspec import GridSpec
        plt.rcParams.update({"figure.figsize": (7, 5.2)})

        fig = plt.figure()
        gs = GridSpec(2, 1, height_ratios=[3.0, 1.2], hspace=0.50)

        ax = fig.add_subplot(gs[0]); style_axes(ax, light_grid=True)
        ax.plot(res["xw"], res["yw"],  lw=1.2, color="tab:blue",   label="Data")
        ax.plot(res["xw"], res["yfit"], lw=1.8, color="tab:orange", label="Total fit")
        ax.plot(res["xw"], res["bkg"],  "--", lw=1.2, color="tab:green", alpha=0.7, label="Linear bkg")

        for idx, comp in enumerate(res["components"]):
            ax.plot(res["xw"], comp, lw=1.0, linestyle="--",
                    color=COMP_COLORS[idx % len(COMP_COLORS)], alpha=0.7)
            if DEBUG and DEBUG_PROVENANCE_TEXT and idx < len(labels_v):
                ycomp = comp
                imax = int(np.nanargmax(ycomp))
                xpk = res["xw"][imax]; ypk = ycomp[imax]
                

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
    height_trk  = np.full((nuse, npeaks), np.nan)  # height (not used for color)
    area_trk    = np.full((nuse, npeaks), np.nan)  # NEW: integrated area for color

    iterator = range(nuse)
    if tqdm is not None:
        iterator = tqdm(iterator, desc="Fitting frames", ncols=80)

    for f in iterator:
        y = I_full[f]
        res = fit_frame(x, y, seeds0, HALF_WINDOW, debug=False)  # map: silent
        if not res["success"]:
            continue
        valid = np.isfinite(res["centers"]) & np.isfinite(res["height_fit"])
        if np.any(valid):
            c = res["centers"][valid]
            w = res["fwhm"][valid]
            h = res["height_fit"][valid]
            a = res.get("area_fit", np.full_like(h, np.nan))[valid]  # NEW
            hi_mask = h >= PEAK_HEIGHT_MIN
            c, w, h, a = c[hi_mask], w[hi_mask], h[hi_mask], a[hi_mask]
        else:
            c = w = h = a = np.array([])
        if c.size:
            order = np.argsort(c)
            c, w, h, a = c[order], w[order], h[order], a[order]
        k = min(c.size, npeaks)
        centers_trk[f, :k] = c[:k]
        fwhm_trk[f, :k]    = w[:k]
        height_trk[f, :k]  = h[:k]
        area_trk[f, :k]    = a[:k]  # NEW

    apply_pub_style()
    plt.rcParams.update({"figure.figsize": (11.5, 4.6)})

    fig, ax = plt.subplots(); style_axes(ax, light_grid=True)

    frames = np.arange(nuse)
    xvals = (frames * float(SEC_PER_FRAME)) if (SEC_PER_FRAME is not None and SEC_PER_FRAME > 0) else frames
    xlabel = "Time (s)" if (SEC_PER_FRAME is not None and SEC_PER_FRAME > 0) else "Frame"

    # --- normalize fitted AREAS to 0–100 for color mapping (global) ---
    area_norm = np.zeros_like(area_trk, float)
    if np.any(np.isfinite(area_trk)):
        global_max = np.nanmax(area_trk)
        if global_max > 0:
            area_norm = 100.0 * area_trk / global_max

    any_plotted = False
    for j in range(npeaks):
        mask = (
            np.isfinite(centers_trk[:, j]) &
            np.isfinite(area_norm[:, j]) &
            (height_trk[:, j] >= PEAK_HEIGHT_MIN)
        )
        if not np.any(mask):
            continue
        sc = ax.scatter(
            xvals[mask],
            centers_trk[mask, j],
            c=area_norm[mask, j],
            cmap="plasma",
            s=18,
            linewidths=0.0,
            edgecolors="none",
            vmin=0, vmax=100,
        )
        any_plotted = True

    lo = float(np.min(seeds0) - HALF_WINDOW)
    hi = float(np.max(seeds0) + HALF_WINDOW)
    ax.set_ylim(lo, hi)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Center (q or 2θ)")
    ax.set_title("Peak centers over frames (color = normalized peak area)")

    if any_plotted:
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label("Normalized Peak Area (0–100)")

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()




















