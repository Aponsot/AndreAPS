#!/usr/bin/env python3
import argparse
import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
from lmfit.models import GaussianModel, LinearModel

# ==============================
# Tunable constants
# ==============================

# Windowing / view
WINDOW = 0.35                 # width of fitting+plot window (q-units)
GRAPH_CENTER = 3.0            # if None uses mean(peaks). Set float to lock for all frames.

# Peak width caps
MIN_SIGMA_ABS = 0.001
MAX_SIGMA_ABS = 0.040
MAX_SIGMA_FRAC = 0.25         # also cap sigma to this fraction of WINDOW

# Anchoring
ANCHOR_TOL = 0.005            # tol for peak 0 if anchored
ANCHOR_PEAK0 = True

# Asymmetric center drift (neg = left, pos = right) [global/baseline]
CENTER_TOL_NEG = 0.040
CENTER_TOL_POS = 0.015

# Detection / pruning
HEIGHT_MIN = 2.0
HEIGHT_MIN_SIGMA = 3.0
PRUNE_SMALL = True

# Background
BASELINE_QUANTILE = 0.20
BKG_EXCLUDE_RADIUS = 0.010
BKG_TRIM_FRACTION = 0.30
BKG_SLOPE_MAX_ABS = 2.0

# Robust loss
USE_ROBUST_LOSS = True

# Optional per-peak sigma limits (lists) - leave None to use global
PEAK_SIGMA_MIN = None
PEAK_SIGMA_MAX = None

# Rescue (fallback) for rough frames
RESCUE_ENABLED = True
RESCUE_R2_MIN = 0.85
RESCUE_MIN_KEPT = 1
RESCUE_EXPAND_WINDOW = 1.6
RESCUE_CENTER_TOL = 0.050
RESCUE_MAX_SIGMA_FRAC = 0.30
RESEED_SPAN = 0.060
RESCUE_SHIFT_FRAC = 0.25  # ignored if GRAPH_CENTER is set

# Residual placement bias (favor negative drift)
RESIDUAL_NEG_WEIGHT = 1.25
RESIDUAL_POS_WEIGHT = 1.00

# Sequential fit internals
_AIC_IMPROVE = 6.0
_MAX_PEAKS = 16
_RESIDUAL_PICK_SPAN = 0.030
_USE_GUESSES_FIRST = True
_VERBOSE = False

# Complexity control & guards
_USE_BIC = True               # prefer BIC over AIC to discourage overfitting
_IC_IMPROVE = 8.0             # min improvement to accept a new component (BIC units if _USE_BIC)
_EDGE_PENALTY_EPS = 0.010     # penalize peaks hugging the window edge
_EDGE_PENALTY = 6.0           # subtract from improvement if near edge
MIN_CENTER_SEP = 0.006        # don't place two peaks too close

# ----- Per-frame tracking (NEW) -----
DRIFT_TOL_NEG = 0.010     # how far a peak can move left per frame (q-units)
DRIFT_TOL_POS = 0.010     # how far a peak can move right per frame (q-units)
DRIFT_TOL_GROW = 1.6      # multiplier to expand drift after a miss
MISS_TOL_MAX   = 3.5      # cap for drift growth after repeated misses
SIGMA_CARRY_CLIP = 0.35   # cap sigma carry: fraction of WINDOW
AMP_SEED_FRACTION = 0.75  # (reserved) if carrying heights to amp seeds
ASSIGN_MAX_DIST = 0.025   # NN assign cutoff (q)

# ==============================
# Core utilities
# ==============================

def robust_sigma(y):
    y = np.asarray(y, float)
    med = np.median(y)
    mad = 1.4826 * np.median(np.abs(y - med)) + 1e-12
    std = np.std(y) + 1e-12
    return float(np.clip(mad, 0.5*std, 2.0*std))

def compute_r2(y, yfit):
    ss_res = np.sum((y - yfit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2) + 1e-16
    return 1.0 - ss_res / ss_tot

def _window_data(x, yfull, peak_positions, window=None, graph_center=None):
    window = WINDOW if window is None else window
    if graph_center is not None and np.isfinite(graph_center):
        center = float(graph_center)
    else:
        center = float(np.mean(peak_positions))
    half = window / 2.0
    m = (x >= center - half) & (x <= center + half)
    xw, yw = x[m], yfull[m]
    mfin = np.isfinite(xw) & np.isfinite(yw)
    xw, yw = xw[mfin], yw[mfin]
    if xw.size < 5:
        raise ValueError("Too few points in window.")
    return center, half, xw, yw

def _local_height_sigma_seeds(xw, yw, baseline, cx, w=0.010):
    m = np.abs(xw - cx) <= w
    if not np.any(m):
        sigma0 = max(np.mean(np.diff(xw)), MIN_SIGMA_ABS)
        height0 = max(np.max(yw) - baseline, robust_sigma(yw))
        return sigma0, height0
    xloc = xw[m]; yloc = yw[m]
    ypk = np.quantile(yloc, 0.9)
    height0 = max(ypk - baseline, robust_sigma(yw))
    half = baseline + 0.5 * (ypk - baseline)
    above = yloc >= half
    if np.any(above):
        xl = np.min(xloc[above]); xr = np.max(xloc[above])
        fwhm = max(xr - xl, np.mean(np.diff(xw)))
    else:
        fwhm = max(np.mean(np.diff(xw)), MIN_SIGMA_ABS)
    sigma0 = max(fwhm / 2.354820045, MIN_SIGMA_ABS)
    return sigma0, height0

def _robust_line_fit(x, y, max_iter=4, trim_frac=BKG_TRIM_FRACTION):
    m, b = np.polyfit(x, y, 1)
    for _ in range(max_iter):
        resid = y - (m * x + b)
        cutoff = np.quantile(np.abs(resid), 1.0 - trim_frac)
        keep = np.abs(resid) <= cutoff
        if keep.sum() < max(3, int(0.2 * len(x))):
            break
        m, b = np.polyfit(x[keep], y[keep], 1)
    return float(m), float(b)

def _background_init(xw, yw, centers, exclude_radius, sigma_seeds=None):
    if sigma_seeds is not None:
        radii = [max(exclude_radius, 2.5 * max(s, MIN_SIGMA_ABS)) for s in sigma_seeds]
    else:
        radii = [exclude_radius] * len(centers) if len(centers) else [exclude_radius]
    mask = np.ones_like(xw, dtype=bool)
    for cx, rad in zip(centers, radii):
        mask &= (np.abs(xw - cx) > rad)
    if mask.sum() >= max(5, int(0.2 * len(xw))):
        m, b = _robust_line_fit(xw[mask], yw[mask], trim_frac=BKG_TRIM_FRACTION)
    else:
        m = 0.0
        b = np.quantile(yw, BASELINE_QUANTILE)
    m = float(np.clip(m, -BKG_SLOPE_MAX_ABS, BKG_SLOPE_MAX_ABS))
    return m, float(b)

def _make_center_bounds(xmin, xmax, centers, anchor_peak0, anchor_tol,
                        center_tol_neg, center_tol_pos):
    bnds = []
    for i, cx in enumerate(centers):
        if anchor_peak0 and i == 0:
            cmin = max(xmin, cx - anchor_tol)
            cmax = min(xmax, cx + anchor_tol)
        else:
            cmin = max(xmin, cx - center_tol_neg)
            cmax = min(xmax, cx + center_tol_pos)
        if cmin > cmax:
            cmin, cmax = min(cmin, cmax), max(cmin, cmax)
        bnds.append((cmin, cmax))
    return bnds

def build_model(xw, yw, centers, baseline, center_bounds,
                height0_override=None, sigma0_override=None):
    dx = np.mean(np.diff(xw)) if len(xw) > 1 else WINDOW
    min_sigma_global = max(0.75 * dx, MIN_SIGMA_ABS)
    max_sigma_global = min(MAX_SIGMA_ABS, MAX_SIGMA_FRAC * WINDOW)

    sigma0_list, height0_list = [], []
    for cx in centers:
        sigma0_est, height0 = _local_height_sigma_seeds(xw, yw, baseline, cx, w=0.010)
        sigma0_clipped = np.clip(sigma0_est, min_sigma_global, max_sigma_global)
        sigma0_list.append(float(sigma0_clipped))
        height0_list.append(float(height0))

    init_slope, init_intercept = _background_init(xw, yw, centers, BKG_EXCLUDE_RADIUS, sigma_seeds=sigma0_list)
    bkg = LinearModel(prefix="bkg_")
    model = bkg
    params = bkg.make_params(slope=init_slope, intercept=init_intercept)
    params["bkg_slope"].set(min=-BKG_SLOPE_MAX_ABS, max=BKG_SLOPE_MAX_ABS, value=init_slope, vary=True)
    params["bkg_intercept"].set(value=init_intercept, vary=True)

    for i, (cx, sigma0, height0) in enumerate(zip(centers, sigma0_list, height0_list)):
        gi = GaussianModel(prefix=f"g{i}_")
        model += gi

        min_sig_i = max(0.0, min_sigma_global) if PEAK_SIGMA_MIN is None else max(min_sigma_global, PEAK_SIGMA_MIN[i])
        max_sig_i = min(max_sigma_global, MAX_SIGMA_FRAC * WINDOW) if PEAK_SIGMA_MAX is None else min(max_sigma_global, PEAK_SIGMA_MAX[i])

        # allow overrides from tracking if provided
        if sigma0_override is not None and i < len(sigma0_override) and np.isfinite(sigma0_override[i]):
            sigma0 = float(np.clip(sigma0_override[i], min_sig_i, max_sig_i))
        if height0_override is not None and i < len(height0_override) and np.isfinite(height0_override[i]):
            height0 = max(height0_override[i], height0)

        amp0 = max(height0, 0.0) * sigma0 * np.sqrt(2 * np.pi)
        cmin, cmax = center_bounds[i]

        params.update(gi.make_params(center=np.clip(cx, cmin, cmax),
                                     sigma=sigma0,
                                     amplitude=max(amp0, 0.0)))
        params[f"g{i}_center"].set(min=cmin, max=cmax)
        params[f"g{i}_sigma"].set(min=min_sig_i, max=max_sig_i)
        params[f"g{i}_amplitude"].set(min=0.0)

    return model, params

def extract_peaks(result):
    peaks = []
    i = 0
    while f"g{i}_center" in result.params:
        ctr = result.params[f"g{i}_center"].value
        sig = result.params[f"g{i}_sigma"].value
        amp = result.params[f"g{i}_amplitude"].value
        sig_abs = abs(sig) if np.isfinite(sig) else np.nan
        hgt = amp / (sig_abs * np.sqrt(2 * np.pi)) if (sig_abs > 0 and np.isfinite(sig_abs)) else 0.0
        fwhm = 2.354820045 * sig_abs if np.isfinite(sig_abs) else np.nan
        peaks.append({"index": i, "center": float(ctr), "height": float(hgt),
                      "fwhm": float(fwhm), "amplitude": float(amp), "sigma": float(sig_abs)})
        i += 1
    return peaks

def _weighted_local_argmax(xw, yw, cx, span, wneg=1.0, wpos=1.0):
    m = (xw >= cx - span) & (xw <= cx + span)
    if not np.any(m):
        return float(cx)
    xx = xw[m]; yy = yw[m]
    left_mask  = xx <= cx
    right_mask = xx >= cx
    left_score = wneg * np.max(yy[left_mask]) if np.any(left_mask) else -np.inf
    right_score = wpos * np.max(yy[right_mask]) if np.any(right_mask) else -np.inf
    if left_score >= right_score and np.any(left_mask):
        j = np.argmax(yy[left_mask]); return float(xx[left_mask][j])
    j = np.argmax(yy[right_mask]);   return float(xx[right_mask][j])

# ==============================
# Single-peak helpers
# ==============================

def _build_window_abs(x, y, center, width):
    half = width / 2.0
    m = (x >= center - half) & (x <= center + half)
    xw, yw = x[m], y[m]
    mfin = np.isfinite(xw) & np.isfinite(yw)
    return xw[mfin], yw[mfin]

def _fit_peak_single(xw, yw, seed_center):
    if len(xw) < 5:
        return None
    try:
        bkg_slope, bkg_intercept = np.polyfit(xw, yw, 1)
    except Exception:
        bkg_slope, bkg_intercept = 0.0, np.median(yw)

    y_detr = yw - (bkg_slope * xw + bkg_intercept)
    noise = robust_sigma(y_detr)

    peak_idx = int(np.abs(xw - seed_center).argmin())
    height0 = max(yw[peak_idx] - (bkg_slope * xw[peak_idx] + bkg_intercept), 0.5 * noise)
    span = max(xw[-1] - xw[0], 1e-9)
    sigma0 = max(span / 7.0, 1e-6)
    amp0 = max(height0 * sigma0 * np.sqrt(2*np.pi), noise * sigma0 * np.sqrt(2*np.pi))

    gm = GaussianModel(prefix="g_"); lm = LinearModel(prefix="bkg_")
    model = lm + gm
    pars = model.make_params(
        bkg_slope=bkg_slope, bkg_intercept=bkg_intercept,
        g_center=xw[peak_idx], g_sigma=sigma0, g_amplitude=amp0
    )
    pars["g_sigma"].set(min=1e-6, max=max(span, 1.0))
    pars["g_amplitude"].set(min=0.0)

    try:
        res = model.fit(yw, pars, x=xw, nan_policy="omit")
        p = res.params
        c = float(p["g_center"].value)
        s = float(p["g_sigma"].value)
        return {"center": c, "sigma": s, "result": res}
    except Exception:
        return None

# ==============================
# Sequential residual-add (single-frame solver) — now accepts bounds override
# ==============================

def _sequential_fit_single_frame(xw, yw, peak_positions, anchor_peak0, baseline, noise,
                                 center_bounds_override=None):
    dx = np.mean(np.diff(xw)) if len(xw) > 1 else WINDOW
    min_sigma_global = max(0.75 * dx, MIN_SIGMA_ABS)
    max_sigma_global = min(MAX_SIGMA_ABS, MAX_SIGMA_FRAC * WINDOW)

    init_slope, init_intercept = _background_init(xw, yw, [], BKG_EXCLUDE_RADIUS, sigma_seeds=None)
    model = LinearModel(prefix="bkg_")
    params = model.make_params(slope=init_slope, intercept=init_intercept)
    params["bkg_slope"].set(min=-BKG_SLOPE_MAX_ABS, max=BKG_SLOPE_MAX_ABS, value=init_slope, vary=True)
    params["bkg_intercept"].set(value=init_intercept, vary=True)

    loss_kwargs = {"loss": "soft_l1", "f_scale": noise} if USE_ROBUST_LOSS else {}
    best_res = model.fit(yw, params, x=xw, calc_covar=False, method="least_squares", max_nfev=800, **loss_kwargs)
    best_ic = (best_res.bic if _USE_BIC else best_res.aic)

    xmin, xmax = float(np.min(xw)), float(np.max(xw))
    peak_positions = list(sorted(peak_positions))
    if center_bounds_override is not None:
        center_bounds_guess = center_bounds_override
    else:
        center_bounds_guess = _make_center_bounds(xmin, xmax, peak_positions, anchor_peak0, ANCHOR_TOL, CENTER_TOL_NEG, CENTER_TOL_POS)

    seeds = []
    for cx in peak_positions:
        sig0, h0 = _local_height_sigma_seeds(xw, yw, baseline, cx, w=0.010)
        seeds.append((cx, float(h0), float(np.clip(sig0, min_sigma_global, max_sigma_global))))
    seeds.sort(key=lambda t: t[1], reverse=True)

    n_added = 0
    used_positions = []
    height_thresh = max(HEIGHT_MIN, HEIGHT_MIN_SIGMA * noise)

    def _place_next_center(resid):
        span = max(_RESIDUAL_PICK_SPAN, 2*np.mean(np.diff(xw)))
        if _USE_GUESSES_FIRST:
            for cx, h0, s0 in seeds:
                if cx in used_positions:
                    continue
                cx_new = _weighted_local_argmax(xw, resid + baseline, cx, span, wneg=RESIDUAL_NEG_WEIGHT, wpos=RESIDUAL_POS_WEIGHT)
                return cx, cx_new, h0, s0
        j = int(np.argmax(resid))
        cx_g = float(xw[j])
        sig0, h0 = _local_height_sigma_seeds(xw, yw, baseline, cx_g, w=0.010)
        s0 = float(np.clip(sig0, min_sigma_global, max_sigma_global))
        cx_new = _weighted_local_argmax(xw, resid + baseline, cx_g, span, wneg=RESIDUAL_NEG_WEIGHT, wpos=RESIDUAL_POS_WEIGHT)
        return cx_g, cx_new, h0, s0

    while n_added < min(_MAX_PEAKS, max(1, len(peak_positions))):
        resid = yw - best_res.best_fit
        guess_cx, place_cx, h0, s0 = _place_next_center(resid)

        # Too-close to existing peaks?
        ok_sep = True
        for k in range(n_added):
            if f"g{k}_center" in best_res.params:
                prev_c = best_res.params[f"g{k}_center"].value
                if abs(prev_c - place_cx) < MIN_CENTER_SEP:
                    ok_sep = False; break
        if not ok_sep:
            break

        if guess_cx in peak_positions:
            i_guess = peak_positions.index(guess_cx)
            cmin, cmax = center_bounds_guess[i_guess]
        else:
            cmin = max(xmin, place_cx - CENTER_TOL_NEG)
            cmax = min(xmax, place_cx + CENTER_TOL_POS)

        amp0 = max(h0, 0.0) * s0 * np.sqrt(2*np.pi)
        gi = GaussianModel(prefix=f"g{n_added}_")
        new_model = best_res.model + gi
        new_params = best_res.params.copy()
        new_params.update(gi.make_params(center=np.clip(place_cx, cmin, cmax), sigma=s0, amplitude=max(amp0, 0.0)))
        new_params[f"g{n_added}_center"].set(min=cmin, max=cmax)
        if anchor_peak0 and n_added == 0 and (guess_cx in peak_positions) and (peak_positions.index(guess_cx) == 0):
            new_params[f"g{n_added}_center"].set(min=max(xmin, place_cx - ANCHOR_TOL), max=min(xmax, place_cx + ANCHOR_TOL))
        new_params[f"g{n_added}_sigma"].set(min=min_sigma_global, max=max_sigma_global)
        new_params[f"g{n_added}_amplitude"].set(min=0.0)

        trial_res = new_model.fit(yw, new_params, x=xw, calc_covar=False, method="least_squares", max_nfev=800, **loss_kwargs)
        trial_ic = (trial_res.bic if _USE_BIC else trial_res.aic)
        dIC = best_ic - trial_ic

        # Edge penalty for the new component
        this_peak = extract_peaks(trial_res)
        this_peak = next((p for p in this_peak if p["index"] == n_added), None)
        edge_pen = 0.0
        if this_peak is not None:
            if (this_peak["center"] - xmin) < _EDGE_PENALTY_EPS or (xmax - this_peak["center"]) < _EDGE_PENALTY_EPS:
                edge_pen = _EDGE_PENALTY
        dIC_eff = dIC - edge_pen

        too_small = (this_peak is None) or (this_peak["height"] < height_thresh)
        if (dIC_eff < _IC_IMPROVE) or too_small:
            break

        best_res = trial_res
        best_ic = trial_ic
        used_positions.append(guess_cx)
        n_added += 1

    # residual repair (single-peak micro add)
    resid = yw - best_res.best_fit
    jmax = int(np.argmax(np.abs(resid)))
    cx_try = float(xw[jmax])
    w_local = max(0.5 * _RESIDUAL_PICK_SPAN, 2*np.mean(np.diff(xw)))
    xloc, yloc = _build_window_abs(xw, resid + best_res.best_fit, cx_try, 2*w_local)
    sp = _fit_peak_single(xloc, yloc, cx_try)
    if sp is not None:
        gi = GaussianModel(prefix=f"g{n_added}_")
        trial_model = best_res.model + gi
        new_params = best_res.params.copy()
        amp0 = max(robust_sigma(yloc) * sp["sigma"] * np.sqrt(2*np.pi), 0.0)
        cmin = max(xmin, sp["center"] - CENTER_TOL_NEG)
        cmax = min(xmax, sp["center"] + CENTER_TOL_POS)
        new_params.update(gi.make_params(center=np.clip(sp["center"], cmin, cmax),
                                         sigma=np.clip(sp["sigma"], max(0.75*np.mean(np.diff(xw)), MIN_SIGMA_ABS),
                                                       min(MAX_SIGMA_ABS, MAX_SIGMA_FRAC*WINDOW)),
                                         amplitude=amp0))
        new_params[f"g{n_added}_center"].set(min=cmin, max=cmax)
        new_params[f"g{n_added}_amplitude"].set(min=0.0)
        trial_res2 = trial_model.fit(yw, new_params, x=xw, calc_covar=False, method="least_squares",
                                     max_nfev=800, **loss_kwargs)
        trial_ic2 = (trial_res2.bic if _USE_BIC else trial_res2.aic)
        near_edge = (sp["center"] - xmin < _EDGE_PENALTY_EPS) or (xmax - sp["center"] < _EDGE_PENALTY_EPS)
        if (best_ic - trial_ic2) > (_IC_IMPROVE + 2.0) and not near_edge:
            best_res = trial_res2
            best_ic = trial_ic2

    # optional micro-refine
    refined = best_res
    try:
        comps = extract_peaks(refined)
        new_params = refined.params.copy()
        for p in comps:
            ci = int(p["index"]); c0 = p["center"]
            xloc, yloc = _build_window_abs(xw, yw, c0, max(0.5*_RESIDUAL_PICK_SPAN, 6*np.mean(np.diff(xw))))
            sp = _fit_peak_single(xloc, yloc, c0)
            if sp is not None and np.isfinite(sp["center"]):
                if abs(sp["center"] - c0) <= max(CENTER_TOL_NEG, CENTER_TOL_POS) and (MIN_SIGMA_ABS <= sp["sigma"] <= MAX_SIGMA_FRAC*WINDOW):
                    new_params[f"g{ci}_center"].set(value=sp["center"])
                    new_params[f"g{ci}_sigma"].set(value=np.clip(sp["sigma"], MIN_SIGMA_ABS, MAX_SIGMA_FRAC*WINDOW))
        refined = refined.model.fit(yw, new_params, x=xw, calc_covar=False, method="least_squares", max_nfev=600, **loss_kwargs)
        ic_ref = (refined.bic if _USE_BIC else refined.aic)
        if ic_ref < best_ic:
            best_res = refined
            best_ic = ic_ref
    except Exception:
        pass

    # prune tiny peaks
    final_res = best_res
    if PRUNE_SMALL:
        peaks_now = extract_peaks(final_res)
        thresh = max(HEIGHT_MIN, HEIGHT_MIN_SIGMA * noise)
        pruned = [p["index"] for p in peaks_now if p["height"] < thresh]
        if len(pruned) > 0:
            refit_params = final_res.params.copy()
            for i in pruned:
                refit_params[f"g{i}_amplitude"].set(value=0.0, vary=False)
                refit_params[f"g{i}_center"].set(vary=False)
                refit_params[f"g{i}_sigma"].set(vary=False)
            final_res = final_res.model.fit(yw, refit_params, x=xw, calc_covar=False, method="least_squares",
                                            max_nfev=800, **loss_kwargs)

    return final_res

# ==============================
# Fit a single frame (and plot)
# ==============================

def fit_single_frame(h5_path, frame, peak_positions, plot=True, anchor_peak0=ANCHOR_PEAK0,
                     center_bounds_override=None, sigma_override=None, height_override=None):
    with h5py.File(h5_path, "r") as f:
        x = f["q"][:] if "q" in f else f["tth"][:]
        yfull = f["int"][frame, :]

    x = np.asarray(x, float)
    yfull = np.asarray(yfull, float)
    if x[0] > x[-1]:
        x = x[::-1]; yfull = yfull[::-1]

    center, half, xw, yw = _window_data(x, yfull, peak_positions, window=WINDOW, graph_center=GRAPH_CENTER)
    xmin, xmax = float(np.min(xw)), float(np.max(xw))
    baseline = np.quantile(yw, BASELINE_QUANTILE)
    noise = robust_sigma(yw)

    # Solve with optional bounds override (for per-frame tracking)
    result = _sequential_fit_single_frame(
        xw, yw, peak_positions, anchor_peak0, baseline, noise,
        center_bounds_override=center_bounds_override
    )
    r2 = compute_r2(yw, result.best_fit)
    peaks = extract_peaks(result)

    did_rescue = False
    if RESCUE_ENABLED:
        kept_now = [p for p in peaks if p["height"] >= max(HEIGHT_MIN, HEIGHT_MIN_SIGMA * noise)]
        need_rescue = (r2 < RESCUE_R2_MIN) or (len(kept_now) < RESCUE_MIN_KEPT)
        if not need_rescue:
            # too many hits on bounds -> try rescue
            center_bounds = _make_center_bounds(xmin, xmax, peak_positions, anchor_peak0, ANCHOR_TOL, CENTER_TOL_NEG, CENTER_TOL_POS) \
                            if center_bounds_override is None else center_bounds_override
            hit_bounds = 0
            # If any peak center equals bound, count it
            i = 0
            for (cmin, cmax) in center_bounds:
                cval = result.params.get(f"g{i}_center", None)
                if cval is not None:
                    v = float(cval.value)
                    if abs(v - cmin) < 1e-9 or abs(v - cmax) < 1e-9:
                        hit_bounds += 1
                i += 1
            need_rescue = hit_bounds >= max(1, len(peak_positions)//2)

        if need_rescue:
            did_rescue = True
            expanded_window = WINDOW * RESCUE_EXPAND_WINDOW
            rescue = _refit_with_rescue(x, yfull, peak_positions=peak_positions, frame=frame,
                                        anchor_peak0=anchor_peak0, window=expanded_window,
                                        center_tol=RESCUE_CENTER_TOL, max_sigma_frac=RESCUE_MAX_SIGMA_FRAC)
            if rescue is not None:
                xw = rescue["xw"]; yw = rescue["yw"]
                result = rescue["result"]; r2 = rescue["r2"]
                peaks = rescue["peaks"]
                baseline = rescue["baseline"]; noise = rescue["noise"]
                center = rescue["center"]; half = rescue["half"]

    bkg_slope = result.params["bkg_slope"].value
    bkg_intercept = result.params["bkg_intercept"].value

    thresh = max(HEIGHT_MIN, HEIGHT_MIN_SIGMA * noise)
    kept = [p for p in peaks if p["height"] >= thresh]
    rows = [[p["index"], p["center"], p["height"], p["fwhm"], p["amplitude"]] for p in kept]

    if plot:
        plt.rcParams.update({"figure.dpi": 160, "savefig.dpi": 300,
                             "font.size": 16, "axes.labelsize": 18, "axes.titlesize": 20,
                             "xtick.labelsize": 14, "ytick.labelsize": 14})
        fig, (ax, ax_tbl) = plt.subplots(2, 1, figsize=(10, 6.8), gridspec_kw={"height_ratios": [3, 1]})

        ax.plot(xw, yw, lw=1.8, label="Data")
        ax.plot(xw, result.best_fit, lw=2.2, label="Fit")
        comps = result.eval_components(x=xw)
        if "bkg_" in comps: ax.plot(xw, comps["bkg_"], ls="--", label="Background")
        for p in kept:
            i = p["index"]; key = f"g{i}_"
            if key in comps: ax.plot(xw, comps[key], ls=":", alpha=0.8, label=f"Peak {i+1}")
            ax.axvline(result.params[f"g{i}_center"].value, alpha=0.25, ls="--")
        rescue_tag = " | rescue" if did_rescue else ""
        ax.set_xlabel("q (1/Å)"); ax.set_ylabel("Intensity")
        ax.set_title(f"Frame {frame} | {len(kept)} kept peaks | R²={r2:.4f}{rescue_tag} | "
                     f"height_min=max({HEIGHT_MIN}, {HEIGHT_MIN_SIGMA}·σ)")
        ax.legend(loc="best"); ax.grid(alpha=0.3)
        ax.set_xlim(center - half, center + half)
        ax2 = ax.twinx(); ax2.plot(xw, yw - result.best_fit, lw=1.0, alpha=0.35); ax2.set_ylabel("Residual"); ax2.grid(False)

        ax_tbl.axis("off")
        cols = ["Peak #", "Center", "Height", "FWHM", "Amplitude"]
        table = ax_tbl.table(cellText=[[f"{r[0]}", f"{r[1]:.6g}", f"{r[2]:.6g}", f"{r[3]:.6g}", f"{r[4]:.6g}"] for r in rows],
                             colLabels=cols, loc="center")
        table.auto_set_font_size(False); table.set_fontsize(12); table.scale(1, 1.25)
        plt.tight_layout(); plt.show()

    return {"frame": frame,
            "background": {"slope": bkg_slope, "intercept": bkg_intercept},
            "r2": r2, "rows": rows, "result": result, "peaks": peaks,
            "pruned_indices": [p["index"] for p in peaks if p["height"] < thresh]}

# ==============================
# Rescue helper (respects fixed center)
# ==============================

def _refit_with_rescue(x, yfull, peak_positions, frame, anchor_peak0,
                       window, center_tol, max_sigma_frac):
    if GRAPH_CENTER is not None and np.isfinite(GRAPH_CENTER):
        center = float(GRAPH_CENTER); half = (window / 2.0)
    else:
        center = float(np.mean(peak_positions)); half = (window / 2.0)
        if RESCUE_SHIFT_FRAC and RESCUE_SHIFT_FRAC != 0.0:
            center = center - (RESCUE_SHIFT_FRAC * window)

    m = (x >= center - half) & (x <= center + half)
    xw, yw = x[m], yfull[m]
    mfin = np.isfinite(xw) & np.isfinite(yw)
    xw, yw = xw[mfin], yw[mfin]
    if xw.size < 5:
        return None

    xmin, xmax = float(np.min(xw)), float(np.max(xw))
    baseline = np.quantile(yw, BASELINE_QUANTILE)
    noise = robust_sigma(yw)

    reseeded = []
    for cx in peak_positions:
        xloc, yloc = _build_window_abs(xw, yw, cx, RESEED_SPAN*2.0)
        sp = _fit_peak_single(xloc, yloc, cx)
        reseeded.append(sp["center"] if sp is not None else cx)
    reseeded = sorted(reseeded)

    center_bounds = _make_center_bounds(xmin, xmax, reseeded, anchor_peak0,
                                        max(ANCHOR_TOL, min(RESEED_SPAN, center_tol)),
                                        center_tol, center_tol)

    loss_kwargs = {"loss": "soft_l1", "f_scale": noise} if USE_ROBUST_LOSS else {}
    global MAX_SIGMA_FRAC
    old_max_sigma_frac = MAX_SIGMA_FRAC
    MAX_SIGMA_FRAC = max_sigma_frac
    try:
        model, params = build_model(xw, yw, reseeded, baseline, center_bounds)
        result = model.fit(yw, params, x=xw, calc_covar=False, method="least_squares", max_nfev=800, **loss_kwargs)
        r2 = compute_r2(yw, result.best_fit)
    finally:
        MAX_SIGMA_FRAC = old_max_sigma_frac

    peaks = extract_peaks(result)
    kept = [p for p in peaks if p["height"] >= max(HEIGHT_MIN, HEIGHT_MIN_SIGMA * noise)]
    return {"xw": xw, "yw": yw, "result": result, "r2": r2, "peaks": peaks,
            "kept": kept, "baseline": baseline, "noise": noise, "center": center, "half": half}

# ==============================
# Tracking helpers (NEW)
# ==============================

def _progress_bar(i, total, *, width=28, prefix="Mapping"):
    i = min(i, total)
    frac = 0 if total == 0 else i / total
    filled = int(width * frac)
    bar = "█" * filled + "·" * (width - filled)
    msg = f"\r{prefix} [{bar}] {i}/{total}"
    sys.stdout.write(msg); sys.stdout.flush()
    if i == total: sys.stdout.write("\n")

def _greedy_match(prev_centers, curr_centers, max_dist):
    i, j = 0, 0
    m = [-1]*len(prev_centers)
    while i < len(prev_centers) and j < len(curr_centers):
        d = abs(prev_centers[i] - curr_centers[j])
        if d <= max_dist:
            m[i] = j; i += 1; j += 1
        elif curr_centers[j] < prev_centers[i]:
            j += 1
        else:
            i += 1
    return m

def _carry_sigma(prev_sigma, dx, window):
    if not np.isfinite(prev_sigma) or prev_sigma <= 0:
        return max(0.75*dx, MIN_SIGMA_ABS)
    return float(np.clip(prev_sigma, MIN_SIGMA_ABS, min(MAX_SIGMA_ABS, SIGMA_CARRY_CLIP*window)))

# ==============================
# Shared tracker (used by map and by single-frame debugging)
# ==============================

def _run_tracking_up_to(h5_path, peak_positions, target_frame_inclusive, anchor_peak0=True,
                        show_anchor_progress=False, show_map_progress=False):
    """
    Runs the same per-frame tracking from frame 0 up to 'target_frame_inclusive',
    returning the drift-bounded center bounds to use on that target frame,
    plus the current track centers and sigmas. This guarantees --frame N uses
    identical seeding/bounds as the mapping pass would.
    """
    with h5py.File(h5_path, "r") as f:
        nframes = int(f["int"].shape[0])
        x_all = f["q"][:] if "q" in f else f["tth"][:]

    tf = int(np.clip(target_frame_inclusive, 0, nframes-1))

    # Build anchor series (same as map)
    anchor_centers = np.full(tf+1, np.nan)
    seed_anchor = float(np.mean(peak_positions))
    if show_anchor_progress: _progress_bar(0, tf+1, prefix="Anchoring")
    for fr in range(tf+1):
        with h5py.File(h5_path, "r") as f:
            y = f["int"][fr, :]
        _, _, xw, yw = _window_data(x_all, y, [seed_anchor], window=WINDOW, graph_center=GRAPH_CENTER)
        sp = _fit_peak_single(xw, yw, seed_anchor)
        if sp is not None and np.isfinite(sp["center"]):
            anchor_centers[fr] = sp["center"]; seed_anchor = sp["center"]
        if show_anchor_progress: _progress_bar(fr+1, tf+1, prefix="Anchoring")

    # Tracking state
    K = len(peak_positions)
    track_centers = np.array(sorted(peak_positions), float)
    track_sigmas  = np.full(K, np.nan)
    track_misses  = np.zeros(K, dtype=int)

    if show_map_progress: _progress_bar(0, tf+1, prefix="Tracking")

    for fr in range(tf):  # run up to frame tf-1, so we can build bounds for tf
        with h5py.File(h5_path, "r") as f:
            y = f["int"][fr, :]
        center, half, xw, yw = _window_data(x_all, y, track_centers, window=WINDOW, graph_center=GRAPH_CENTER)
        xmin, xmax = float(np.min(xw)), float(np.max(xw))
        dx = np.mean(np.diff(xw)) if len(xw) > 1 else WINDOW
        baseline = np.quantile(yw, BASELINE_QUANTILE)
        noise = robust_sigma(yw)

        merged_seeds = sorted(set(list(track_centers) + ([float(anchor_centers[fr])] if np.isfinite(anchor_centers[fr]) else [])))
        # Build drift bounds around current track centers
        center_bounds = []
        for k, cx_prev in enumerate(track_centers):
            grow = min(MISS_TOL_MAX, (DRIFT_TOL_GROW ** track_misses[k]))
            cmin = max(xmin, cx_prev - DRIFT_TOL_NEG * grow)
            cmax = min(xmax, cx_prev + DRIFT_TOL_POS * grow)
            if cmin > cmax: cmin, cmax = cmax, cmin
            center_bounds.append((cmin, cmax))

        # Fit this frame (no plotting)
        res = fit_single_frame(h5_path, fr, merged_seeds, plot=False, anchor_peak0=anchor_peak0,
                               center_bounds_override=center_bounds)
        kept = res["rows"]
        curr_centers = np.array([row[1] for row in kept], float)
        curr_sigmas  = np.array([row[3]/2.354820045 if np.isfinite(row[3]) else np.nan for row in kept], float)

        if curr_centers.size > 0:
            order = np.argsort(curr_centers)
            curr_centers = curr_centers[order]
            curr_sigmas  = curr_sigmas[order]
            mapping = _greedy_match(track_centers, curr_centers, ASSIGN_MAX_DIST)
        else:
            mapping = [-1]*K

        # Update track state
        for k in range(K):
            j = mapping[k]
            if j >= 0:
                c_new = float(curr_centers[j])
                s_prev = track_sigmas[k]
                s_new  = _carry_sigma(s_prev if np.isfinite(s_prev) else curr_sigmas[j], dx, WINDOW)
                track_centers[k] = c_new
                track_sigmas[k]  = s_new
                track_misses[k]  = 0
            else:
                track_misses[k] = min(track_misses[k] + 1, 10)

        if show_map_progress: _progress_bar(fr+1, tf, prefix="Tracking")

    # For target frame tf, build and return the drift bounds derived from the up-to-date track state
    # Also return a merged seed list including anchor(tf) if available.
    # We don't update with frame tf data here; we just prepare its seeds/bounds.
    with h5py.File(h5_path, "r") as f:
        ytf = f["int"][tf, :]
    _, _, xw_tf, _ = _window_data(x_all, ytf, track_centers, window=WINDOW, graph_center=GRAPH_CENTER)
    xmin_tf, xmax_tf = float(np.min(xw_tf)), float(np.max(xw_tf))

    center_bounds_tf = []
    for k, cx_prev in enumerate(track_centers):
        grow = min(MISS_TOL_MAX, (DRIFT_TOL_GROW ** track_misses[k]))
        cmin = max(xmin_tf, cx_prev - DRIFT_TOL_NEG * grow)
        cmax = min(xmax_tf, cx_prev + DRIFT_TOL_POS * grow)
        if cmin > cmax: cmin, cmax = cmax, cmin
        center_bounds_tf.append((cmin, cmax))

    anchor_tf = [float(anchor_centers[tf])] if (tf < len(anchor_centers) and np.isfinite(anchor_centers[tf])) else []
    merged_seeds_tf = sorted(set(list(track_centers) + anchor_tf))

    return {
        "track_centers": track_centers.copy(),
        "track_sigmas": track_sigmas.copy(),
        "center_bounds": center_bounds_tf,
        "merged_seeds": merged_seeds_tf,
        "target_frame": tf
    }

# ==============================
# Map across frames (uses same tracking as above)
# ==============================

def map_peaks_over_frames(h5_path, peak_positions, *, anchor_peak0=ANCHOR_PEAK0):
    with h5py.File(h5_path, "r") as f:
        nframes = int(f["int"].shape[0])
        x_all = f["q"][:] if "q" in f else f["tth"][:]

    # Anchor
    anchor_centers = np.full(nframes, np.nan)
    seed_anchor = float(np.mean(peak_positions))
    _progress_bar(0, nframes, prefix="Anchoring")
    for fr in range(nframes):
        with h5py.File(h5_path, "r") as f:
            y = f["int"][fr, :]
        _, _, xw, yw = _window_data(x_all, y, [seed_anchor], window=WINDOW, graph_center=GRAPH_CENTER)
        sp = _fit_peak_single(xw, yw, seed_anchor)
        if sp is not None and np.isfinite(sp["center"]):
            anchor_centers[fr] = sp["center"]; seed_anchor = sp["center"]
        _progress_bar(fr+1, nframes, prefix="Anchoring")

    K = len(peak_positions)
    track_centers = np.array(sorted(peak_positions), float)
    track_sigmas  = np.full(K, np.nan)
    track_misses  = np.zeros(K, dtype=int)

    frames_list, centers, heights, fwhms, r2s, track_ids = [], [], [], [], [], []

    _progress_bar(0, nframes, prefix="Mapping")
    for fr in range(nframes):
        with h5py.File(h5_path, "r") as f:
            y = f["int"][fr, :]
        center, half, xw, yw = _window_data(x_all, y, track_centers, window=WINDOW, graph_center=GRAPH_CENTER)
        xmin, xmax = float(np.min(xw)), float(np.max(xw))
        dx = np.mean(np.diff(xw)) if len(xw) > 1 else WINDOW
        baseline = np.quantile(yw, BASELINE_QUANTILE)
        noise = robust_sigma(yw)

        merged_seeds = sorted(set(list(track_centers) + ([float(anchor_centers[fr])] if np.isfinite(anchor_centers[fr]) else [])))

        center_bounds = []
        for k, cx_prev in enumerate(track_centers):
            grow = min(MISS_TOL_MAX, (DRIFT_TOL_GROW ** track_misses[k]))
            cmin = max(xmin, cx_prev - DRIFT_TOL_NEG * grow)
            cmax = min(xmax, cx_prev + DRIFT_TOL_POS * grow)
            if cmin > cmax: cmin, cmax = cmax, cmin
            center_bounds.append((cmin, cmax))

        res = fit_single_frame(
            h5_path, fr, merged_seeds, plot=False, anchor_peak0=anchor_peak0,
            center_bounds_override=center_bounds
        )

        kept = res["rows"]  # [index, center, height, fwhm, amplitude]
        curr_centers = np.array([row[1] for row in kept], float)
        curr_sigmas  = np.array([row[3]/2.354820045 if np.isfinite(row[3]) else np.nan for row in kept], float)
        r2s.append(res["r2"])

        if curr_centers.size > 0:
            order = np.argsort(curr_centers)
            curr_centers = curr_centers[order]
            curr_sigmas  = curr_sigmas[order]
            mapping = _greedy_match(track_centers, curr_centers, ASSIGN_MAX_DIST)
        else:
            mapping = [-1]*K

        for k in range(K):
            j = mapping[k]
            if j >= 0:
                c_new = float(curr_centers[j])
                s_prev = track_sigmas[k]
                s_new  = _carry_sigma(s_prev if np.isfinite(s_prev) else curr_sigmas[j], dx, WINDOW)
                track_centers[k] = c_new
                track_sigmas[k]  = s_new
                track_misses[k]  = 0

                frames_list.append(fr)
                centers.append(c_new)
                # find corresponding kept row height
                hgt = [row[2] for row in kept if abs(row[1]-c_new) < 1e-12]
                heights.append(hgt[0] if len(hgt) else np.nan)
                fwhms.append(2.354820045 * s_new if np.isfinite(s_new) else np.nan)
                track_ids.append(k)
            else:
                track_misses[k] = min(track_misses[k] + 1, 10)

        _progress_bar(fr + 1, nframes, prefix="Mapping")

    if len(frames_list) == 0:
        print("No peaks kept over the dataset."); return None

    frames_arr  = np.array(frames_list, dtype=float)
    centers_arr = np.array(centers, dtype=float)
    heights_arr = np.array(heights, dtype=float)
    r2_arr      = np.array(r2s, dtype=float)

    plt.rcParams.update({"figure.dpi": 160, "savefig.dpi": 300,
                         "font.size": 16, "axes.labelsize": 18, "axes.titlesize": 20,
                         "xtick.labelsize": 14, "ytick.labelsize": 14})
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(frames_arr, centers_arr, c=heights_arr, cmap="plasma", s=24, alpha=0.9)
    cb = plt.colorbar(sc, ax=ax); cb.set_label("Peak height (a.u.)")
    ax.set_xlabel("Frame"); ax.set_ylabel("Peak center q (1/Å)")
    ax.set_title("Mapped peaks over frames (color = height)")
    ax.grid(alpha=0.3); plt.tight_layout(); plt.show()

    return {
        "frame": frames_arr, "center": centers_arr, "height": heights_arr,
        "fwhm": np.array(fwhms, dtype=float), "r2": r2_arr,
        "track_id": np.array(track_ids, dtype=int)
    }

# ==============================
# CLI
# ==============================

def main():
    parser = argparse.ArgumentParser(description="Multi-peak Gaussian fits with drift-bounded per-frame tracking.")
    parser.add_argument("h5", help="HDF5 file with 'q' or 'tth' and 'int' datasets")
    parser.add_argument("peaks", type=float, nargs='+', help="Initial peak q-positions (e.g., 3.025 3.012)")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--frame", type=int, help="Frame index to fit and show per-frame plot (uses tracking history up to this frame)")
    grp.add_argument("--map", action="store_true", help="Map kept peaks across ALL frames (with a progress bar)")
    parser.add_argument("--no-anchor", action="store_true", help="Do not anchor peak 0; let all centers float within the window")
    parser.add_argument("--center", type=float, default=None, help="Fixed graph center (q-units) for ALL frames")
    parser.add_argument("--window", type=float, default=None, help="Graph/fitting window width (q-units) for ALL frames")

    args = parser.parse_args()
    peak_positions = sorted(args.peaks)
    anchor = (not args.no_anchor)

    global GRAPH_CENTER, WINDOW
    if args.center is not None: GRAPH_CENTER = float(args.center)
    if args.window is not None: WINDOW = float(args.window)

    print(f"Peaks: {peak_positions}")
    print(f"Window: {WINDOW} q | Graph center: {('%.6f' % GRAPH_CENTER) if GRAPH_CENTER is not None else 'mean(peaks)'}")
    print(f"Anchor tol (peak 0): {ANCHOR_TOL} q | anchor={'off' if args.no_anchor else 'on'}")
    print(f"height_min: {HEIGHT_MIN} | height_min_sigma: {HEIGHT_MIN_SIGMA}·σ")
    print(f"Sigma bounds: [{MIN_SIGMA_ABS}, {min(MAX_SIGMA_ABS, MAX_SIGMA_FRAC * WINDOW)}] q")
    print(f"Selection: {'BIC' if _USE_BIC else 'AIC'} with ΔIC >= {_IC_IMPROVE} (edge penalty { _EDGE_PENALTY } if within { _EDGE_PENALTY_EPS } q)")
    print(f"Asymmetric center tol: neg={CENTER_TOL_NEG} / pos={CENTER_TOL_POS} q | MIN_CENTER_SEP={MIN_CENTER_SEP}")
    print(f"Per-frame drift: neg={DRIFT_TOL_NEG} / pos={DRIFT_TOL_POS} q (grow×{DRIFT_TOL_GROW}, cap {MISS_TOL_MAX}×)")

    if args.frame is not None:
        # IMPORTANT: ensure single-frame uses identical tracking up to that frame
        prep = _run_tracking_up_to(args.h5, peak_positions, args.frame, anchor_peak0=anchor,
                                   show_anchor_progress=True, show_map_progress=True)
        seeds = prep["merged_seeds"]
        bounds = prep["center_bounds"]
        _ = fit_single_frame(args.h5, prep["target_frame"], seeds, plot=True, anchor_peak0=anchor,
                             center_bounds_override=bounds)
    else:
        _ = map_peaks_over_frames(args.h5, peak_positions, anchor_peak0=anchor)

if __name__ == "__main__":
    main()


