#!/usr/bin/env python3
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
from lmfit.models import GaussianModel, LinearModel
import time
from math import erf, sqrt
try:
    from tqdm import tqdm
    _HAVE_TQDM = True
except Exception:
    _HAVE_TQDM = False

# ------------------------------
# Tunable constants (single-frame)
# ------------------------------

# Window (in q units) around the mean of specified peak positions
WINDOW = 0.25

# Sigma bounds
MIN_SIGMA_ABS = 0.0015     # tightened to prevent needle peaks
MAX_SIGMA_ABS = 0.025
MAX_SIGMA_FRAC = 0.25      # cap sigma to this fraction of WINDOW

# Anchor for peak 0 every frame (optional)
ANCHOR_TOL = 0.005         # q units around the first specified peak position
ANCHOR_PEAK0 = True        # set False to let all centers float freely

# Additional center drift limit for other peaks (prevents identity swapping)
CENTER_TOL = 0.020         # q units allowed drift from each guess for peaks i>0

# Peak separation (discourage duplicate/overlapping solutions)
MIN_SEP_Q = 0.006          # q units; try 0.004–0.010
MERGE_MIN_SEP_FRAC = 0.8   # proximity rule in units of mean sigma

# Pruning threshold: remove peaks (set amplitude to 0) if height < threshold
HEIGHT_MIN = 5.0           # absolute floor
HEIGHT_MIN_SIGMA = 4.0     # relative floor: K * robust_sigma(y)
PRUNE_SMALL = True

# Background controls (robust linear background)
BASELINE_QUANTILE = 0.20       # lower quantile baseline for sigma/height seeding
BKG_EXCLUDE_RADIUS = 0.010     # baseline min exclusion (will scale with sigma seeds)
BKG_TRIM_FRACTION = 0.30       # trimmed regression fraction
BKG_SLOPE_MAX_ABS = 2.0        # hard cap on background slope (intensity per q)

# Use robust loss in least-squares to reduce outlier impact
USE_ROBUST_LOSS = True

# Optional: per-peak sigma bounds (leave None to use global logic)
PEAK_SIGMA_MIN = None  # e.g., [0.001, 0.002]
PEAK_SIGMA_MAX = None  # e.g., [0.030, 0.060]

# --- Rescue (fallback) settings for post-solidification frames ---
RESCUE_ENABLED = True
RESCUE_R2_MIN = 0.85          # if first fit R² is below this, try rescue once
RESCUE_MIN_KEPT = 1           # or if kept peaks < this
RESCUE_EXPAND_WINDOW = 1.6    # multiply WINDOW during rescue (0.25 -> 0.40)
RESCUE_CENTER_TOL = 0.050     # temporarily allow centers to drift farther
RESCUE_MAX_SIGMA_FRAC = 0.30  # slightly looser broadening for the retry
RESEED_SPAN = 0.060           # search ± this (q) around each guess for local max

# --- Overlap-aware merge controls (after fit) ---
OVERLAP_COEF_MIN = 0.45   # consider merge if Gaussian shape overlap >= this
AIC_IMPROVE = 20         # extra peak must improve AIC by > this to be kept
MERGE_HEIGHT_FRAC = 0.9   # when heights similar, prefer dropping smaller amplitude

# ------------------------------
# Core utilities
# ------------------------------

def robust_sigma(y):
    med = np.median(y)
    return 1.4826 * np.median(np.abs(y - med)) + 1e-12

def compute_r2(y, yfit):
    ss_res = np.sum((y - yfit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2) + 1e-16
    return 1.0 - ss_res / ss_tot

def _window_data(x, yfull, peak_positions):
    center = np.mean(peak_positions)
    half = WINDOW / 2.0
    m = (x >= center - half) & (x <= center + half)
    xw, yw = x[m], yfull[m]
    mfin = np.isfinite(xw) & np.isfinite(yw)
    xw, yw = xw[mfin], yw[mfin]
    if xw.size < 5:
        raise ValueError("Too few points in window.")
    return center, half, xw, yw

def _local_height_sigma_seeds(xw, yw, baseline, cx, w=0.010):
    """
    Robust local seeds for sigma and height using a small neighborhood around cx.
    Uses upper-quantile intensity to avoid single-point noise/shoulders dominating.
    """
    m = np.abs(xw - cx) <= w
    if not np.any(m):
        sigma0 = max(np.mean(np.diff(xw)), MIN_SIGMA_ABS)
        height0 = max(np.max(yw) - baseline, robust_sigma(yw))
        return sigma0, height0

    xloc = xw[m]
    yloc = yw[m]
    ypk = np.quantile(yloc, 0.9)
    height0 = max(ypk - baseline, robust_sigma(yw))

    half = baseline + 0.5 * (ypk - baseline)
    above = yloc >= half
    if np.any(above):
        xl = np.min(xloc[above])
        xr = np.max(xloc[above])
        fwhm = max(xr - xl, np.mean(np.diff(xw)))
    else:
        fwhm = max(np.mean(np.diff(xw)), MIN_SIGMA_ABS)

    sigma0 = max(fwhm / 2.354820045, MIN_SIGMA_ABS)
    return sigma0, height0

def _robust_line_fit(x, y, max_iter=4, trim_frac=0.30):
    # Simple trimmed regression: fit, trim largest residuals, refit
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
    """
    Trimmed linear background seeded from off-peak points.
    Exclude around each center by max(exclude_radius, 2.5 * sigma_seed) if seeds provided.
    """
    if sigma_seeds is not None:
        radii = [max(exclude_radius, 2.5 * max(s, MIN_SIGMA_ABS)) for s in sigma_seeds]
    else:
        radii = [exclude_radius] * len(centers)

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

def _enforce_min_separation(params, n_peaks, min_sep):
    # keep ordering g0 <= g1 <= ... and enforce g(i) >= g(i-1) + min_sep
    for i in range(1, n_peaks):
        params[f"g{i}_center"].set(
            expr=f"max(g{i-1}_center + {min_sep:.8f}, {params[f'g{i}_center'].value})"
        )

def build_model(xw, yw, centers, baseline, center_bounds):
    """
    Build a model with a linear background and Gaussian peaks at given centers.
    """
    dx = np.mean(np.diff(xw)) if len(xw) > 1 else WINDOW
    min_sigma_global = max(1.00 * dx, MIN_SIGMA_ABS)  # tightened from 0.75*dx
    max_sigma_global = min(MAX_SIGMA_ABS, MAX_SIGMA_FRAC * WINDOW)

    # Precompute local seeds for each center
    sigma0_list, height0_list = [], []
    for cx in centers:
        sigma0_est, height0 = _local_height_sigma_seeds(xw, yw, baseline, cx, w=0.010)
        sigma0_clipped = np.clip(sigma0_est, min_sigma_global, max_sigma_global)
        sigma0_list.append(float(sigma0_clipped))
        height0_list.append(float(height0))

    # Robust background init (exclude scaled by sigma seeds)
    init_slope, init_intercept = _background_init(
        xw, yw, centers, BKG_EXCLUDE_RADIUS, sigma_seeds=sigma0_list
    )
    bkg = LinearModel(prefix="bkg_")
    model = bkg
    params = bkg.make_params(slope=init_slope, intercept=init_intercept)
    params["bkg_slope"].set(min=-BKG_SLOPE_MAX_ABS, max=BKG_SLOPE_MAX_ABS, value=init_slope, vary=True)
    params["bkg_intercept"].set(value=init_intercept, vary=True)

    # Add Gaussians
    for i, (cx, sigma0, height0) in enumerate(zip(centers, sigma0_list, height0_list)):
        gi = GaussianModel(prefix=f"g{i}_")
        model += gi

        # Per-peak sigma bounds if provided, else global
        min_sig_i = min_sigma_global if PEAK_SIGMA_MIN is None else max(min_sigma_global, PEAK_SIGMA_MIN[i])
        max_sig_i = max_sigma_global if PEAK_SIGMA_MAX is None else min(max_sigma_global, PEAK_SIGMA_MAX[i])

        amp0 = max(height0, 0.0) * sigma0 * np.sqrt(2 * np.pi)
        cmin, cmax = center_bounds[i]

        params.update(gi.make_params(center=np.clip(cx, cmin, cmax),
                                     sigma=sigma0,
                                     amplitude=max(amp0, 0.0)))

        params[f"g{i}_center"].set(min=cmin, max=cmax)
        params[f"g{i}_sigma"].set(min=min_sig_i, max=max_sig_i)
        params[f"g{i}_amplitude"].set(min=0.0)

    # Enforce a minimum separation between peak centers
    _enforce_min_separation(params, n_peaks=len(centers), min_sep=MIN_SEP_Q)

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
        peaks.append({
            "index": i, "center": ctr, "height": hgt,
            "fwhm": fwhm, "amplitude": amp, "sigma": sig
        })
        i += 1
    return peaks

def _gaussian_shape_overlap(mu1, s1, mu2, s2):
    """
    Approximate overlap coefficient (0..1) between two Gaussian shapes (ignores amplitude).
    Uses pooled sigma approximation; good enough for merge heuristics.
    """
    if not np.isfinite(s1) or not np.isfinite(s2) or s1 <= 0 or s2 <= 0:
        return 0.0
    d = abs(mu1 - mu2)
    sp = sqrt(s1**2 + s2**2)
    if sp <= 0:
        return 0.0
    z = - d / (2.0 * sp)
    Phi = 0.5 * (1.0 + erf(z / sqrt(2.0)))
    return max(0.0, min(1.0, 2.0 * Phi))

def _merge_close_peaks_with_aic(xw, yw, model, result, noise,
                                min_sep_q, merge_min_sep_frac,
                                overlap_coef_min, aic_improve, height_frac):
    """
    Decide which peaks to zero-out based on proximity OR strong shape overlap,
    and confirm by AIC: if removing the weaker peak does NOT improve AIC by >= aic_improve,
    we drop it.
    Returns: sorted list of peak indices to kill.
    """
    peaks = extract_peaks(result)
    n = len(peaks)
    to_kill = set()
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            di = abs(peaks[j]["center"] - peaks[i]["center"])
            pairs.append((di, i, j))
    pairs.sort(key=lambda t: t[0])

    for _, i, j in pairs:
        if i in to_kill or j in to_kill:
            continue
        pi, pj = peaks[i], peaks[j]
        ci, cj = pi["center"], pj["center"]
        si, sj = abs(pi["sigma"]), abs(pj["sigma"])
        hi, hj = pi["height"], pj["height"]

        d = abs(cj - ci)
        close_by_resolution = (d < min_sep_q)
        close_by_sigma = (d < merge_min_sep_frac * 0.5 * (si + sj))
        ovl = _gaussian_shape_overlap(ci, si, cj, sj)
        candidate = close_by_resolution or close_by_sigma or (ovl >= overlap_coef_min)
        if not candidate:
            continue

        # choose weaker to test-remove: by height; tie by amplitude
        drop = i if (hi < hj * height_frac or (abs(hi - hj) < 1e-12 and pi["amplitude"] < pj["amplitude"])) else j

        # Test fit with that peak removed (amplitude->0, center/sigma frozen)
        params_refit = result.params.copy()
        params_refit[f"g{drop}_amplitude"].set(value=0.0, vary=False)
        params_refit[f"g{drop}_center"].set(vary=False)
        params_refit[f"g{drop}_sigma"].set(vary=False)
        test = model.fit(yw, params_refit, x=xw, calc_covar=False, method="least_squares", max_nfev=600)

        # AIC decision: keep extra peak only if it improves AIC enough
        delta_aic = result.aic - test.aic   # positive => full model better
        if delta_aic < aic_improve:
            to_kill.add(drop)

    return sorted(to_kill)

def _local_argmax(xw, yw, cx, span):
    """Return x position of the max intensity within ±span of cx; fall back to cx."""
    m = (xw >= cx - span) & (xw <= cx + span)
    if not np.any(m):
        return float(cx)
    idx = np.argmax(yw[m])
    return float(xw[m][idx])

def _refit_with_rescue(x, yfull, peak_positions, frame, anchor_peak0,
                       window, center_tol, max_sigma_frac):
    """
    One-shot rescue: expand window, reseed centers to nearest local maxima,
    relax center/sigma tolerances, and refit.
    """
    center = float(np.mean(peak_positions))
    half = (window / 2.0)
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
        cx_new = _local_argmax(xw, yw, cx, RESEED_SPAN)
        reseeded.append(cx_new)
    reseeded = sorted(reseeded)

    center_bounds = []
    for i, cx in enumerate(reseeded):
        if anchor_peak0 and i == 0:
            cmin = max(xmin, cx - max(ANCHOR_TOL, min(RESEED_SPAN, center_tol)))
            cmax = min(xmax, cx + max(ANCHOR_TOL, min(RESEED_SPAN, center_tol)))
        else:
            cmin = max(xmin, cx - center_tol)
            cmax = min(xmax, cx + center_tol)
        if cmin > cmax:
            cmin, cmax = min(cmin, cmax), max(cmin, cmax)
        center_bounds.append((cmin, cmax))

    loss_kwargs = {"loss": "soft_l1", "f_scale": noise} if USE_ROBUST_LOSS else {}

    global MAX_SIGMA_FRAC
    old_max_sigma_frac = MAX_SIGMA_FRAC
    MAX_SIGMA_FRAC = max_sigma_frac
    try:
        model, params = build_model(xw, yw, reseeded, baseline, center_bounds)
        result = model.fit(yw, params, x=xw, calc_covar=False,
                           method="least_squares", max_nfev=800, **loss_kwargs)
        r2 = compute_r2(yw, result.best_fit)
    finally:
        MAX_SIGMA_FRAC = old_max_sigma_frac

    peaks = extract_peaks(result)
    thresh = max(HEIGHT_MIN, HEIGHT_MIN_SIGMA * noise)
    kept = [p for p in peaks if p["height"] >= thresh]

    return {
        "xw": xw, "yw": yw, "result": result, "r2": r2, "peaks": peaks,
        "kept": kept, "baseline": baseline, "noise": noise,
        "center": center, "half": half, "reseeded": reseeded
    }

def _format_eta(sec):
    if sec is None or sec == float("inf"):
        return "ETA --:--"
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"ETA {h:d}:{m:02d}:{s:02d}"
    return f"ETA {m:02d}:{s:02d}"

# ------------------------------
# Fit a single frame (and plot)
# ------------------------------

def fit_single_frame(h5_path, frame, peak_positions, plot=True, anchor_peak0=ANCHOR_PEAK0):
    # Load data directly (assume int is [nframes, nq] and x ascending)
    with h5py.File(h5_path, "r") as f:
        x = f["q"][:] if "q" in f else f["tth"][:]
        yfull = f["int"][frame, :]

    x = np.asarray(x, float)
    yfull = np.asarray(yfull, float)

    if x[0] > x[-1]:
        x = x[::-1]
        yfull = yfull[::-1]

    center, half, xw, yw = _window_data(x, yfull, peak_positions)
    xmin, xmax = float(np.min(xw)), float(np.max(xw))

    baseline = np.quantile(yw, BASELINE_QUANTILE)
    noise = robust_sigma(yw)

    # Center bounds
    center_bounds = []
    for i, cx in enumerate(peak_positions):
        if anchor_peak0 and i == 0:
            cmin = max(xmin, cx - ANCHOR_TOL)
            cmax = min(xmax, cx + ANCHOR_TOL)
        else:
            cmin = max(xmin, cx - CENTER_TOL)
            cmax = min(xmax, cx + CENTER_TOL)
        if cmin > cmax:
            cmin, cmax = min(cmin, cmax), max(cmin, cmax)
        center_bounds.append((cmin, cmax))

    loss_kwargs = {"loss": "soft_l1", "f_scale": noise} if USE_ROBUST_LOSS else {}

    model, params = build_model(xw, yw, peak_positions, baseline, center_bounds)
    result = model.fit(yw, params, x=xw, calc_covar=False, method="least_squares", max_nfev=800, **loss_kwargs)
    r2 = compute_r2(yw, result.best_fit)

    # Prune tiny peaks and refit
    peaks = extract_peaks(result)
    pruned_indices = []
    if PRUNE_SMALL:
        thresh0 = max(HEIGHT_MIN, HEIGHT_MIN_SIGMA * noise)
        pruned_indices = [p["index"] for p in peaks if p["height"] < thresh0]
        if len(pruned_indices) > 0:
            params_refit = result.params.copy()
            for i in pruned_indices:
                params_refit[f"g{i}_amplitude"].set(value=0.0, vary=False)
                params_refit[f"g{i}_center"].set(vary=False)
                params_refit[f"g{i}_sigma"].set(vary=False)
            result = model.fit(yw, params_refit, x=xw, calc_covar=False, method="least_squares", max_nfev=800, **loss_kwargs)
            r2 = compute_r2(yw, result.best_fit)
            peaks = extract_peaks(result)

    # Overlap + proximity + AIC-based merge (single pass), then refit
    to_kill = _merge_close_peaks_with_aic(
        xw, yw, model, result, noise,
        min_sep_q=MIN_SEP_Q,
        merge_min_sep_frac=MERGE_MIN_SEP_FRAC,
        overlap_coef_min=OVERLAP_COEF_MIN,
        aic_improve=AIC_IMPROVE,
        height_frac=MERGE_HEIGHT_FRAC
    )
    if to_kill:
        params_refit = result.params.copy()
        for i_drop in to_kill:
            params_refit[f"g{i_drop}_amplitude"].set(value=0.0, vary=False)
            params_refit[f"g{i_drop}_center"].set(vary=False)
            params_refit[f"g{i_drop}_sigma"].set(vary=False)
        result = model.fit(yw, params_refit, x=xw, calc_covar=False, method="least_squares", max_nfev=800, **loss_kwargs)
        r2 = compute_r2(yw, result.best_fit)
        peaks = extract_peaks(result)

    # Rescue path if initial fit likely failed
    did_rescue = False
    if RESCUE_ENABLED:
        kept_now = [p for p in peaks if p["height"] >= max(HEIGHT_MIN, HEIGHT_MIN_SIGMA * noise)]
        need_rescue = (r2 < RESCUE_R2_MIN) or (len(kept_now) < RESCUE_MIN_KEPT)

        if not need_rescue:
            hit_bounds = 0
            for i, (cmin, cmax) in enumerate(center_bounds):
                cval = result.params.get(f"g{i}_center", None)
                if cval is not None:
                    v = cval.value
                    if abs(v - cmin) < 1e-6 or abs(v - cmax) < 1e-6:
                        hit_bounds += 1
            need_rescue = hit_bounds >= max(1, len(peak_positions)//2)

        if need_rescue:
            did_rescue = True
            expanded_window = WINDOW * RESCUE_EXPAND_WINDOW
            rescue = _refit_with_rescue(
                x, yfull,
                peak_positions=peak_positions,
                frame=frame,
                anchor_peak0=anchor_peak0,
                window=expanded_window,
                center_tol=RESCUE_CENTER_TOL,
                max_sigma_frac=RESCUE_MAX_SIGMA_FRAC
            )
            if rescue is not None:
                xw = rescue["xw"]; yw = rescue["yw"]
                result = rescue["result"]; r2 = rescue["r2"]
                peaks = rescue["peaks"]
                baseline = rescue["baseline"]; noise = rescue["noise"]
                center = rescue["center"]; half = rescue["half"]

    bkg_slope = result.params["bkg_slope"].value
    bkg_intercept = result.params["bkg_intercept"].value

    # Kept peaks & table rows
    thresh = max(HEIGHT_MIN, HEIGHT_MIN_SIGMA * noise)
    kept = [p for p in peaks if p["height"] >= thresh]
    rows = [[p["index"], p["center"], p["height"], p["fwhm"], p["amplitude"]] for p in kept]

    if plot:
        plt.rcParams.update({
            "figure.dpi": 160, "savefig.dpi": 300,
            "font.size": 16, "axes.labelsize": 18, "axes.titlesize": 20,
            "xtick.labelsize": 14, "ytick.labelsize": 14,
        })
        fig, (ax, ax_tbl) = plt.subplots(2, 1, figsize=(10, 6.8), gridspec_kw={"height_ratios": [3, 1]})

        ax.plot(xw, yw, lw=1.8, label="Data")
        ax.plot(xw, result.best_fit, lw=2.2, label="Fit")

        comps = result.eval_components(x=xw)
        if "bkg_" in comps:
            ax.plot(xw, comps["bkg_"], ls="--", label="Background")

        for p in kept:
            i = p["index"]
            key = f"g{i}_"
            if key in comps:
                ax.plot(xw, comps[key], ls=":", alpha=0.8, label=f"Peak {i+1}")
            ax.axvline(result.params[f"g{i}_center"].value, alpha=0.25, ls="--")

        rescue_tag = " | rescue" if did_rescue else ""
        ax.set_xlabel("q (1/Å)")
        ax.set_ylabel("Intensity")
        ax.set_title(f"Frame {frame} | {len(kept)} kept peaks | R²={r2:.4f}{rescue_tag} | height_min=max({HEIGHT_MIN}, {HEIGHT_MIN_SIGMA}·σ)")
        ax.legend(loc="best")
        ax.grid(alpha=0.3)
        ax.set_xlim(center - half, center + half)

        ax_tbl.axis("off")
        cols = ["Peak #", "Center", "Height", "FWHM", "Amplitude"]
        table = ax_tbl.table(
            cellText=[[f"{r[0]}", f"{r[1]:.6g}", f"{r[2]:.6g}", f"{r[3]:.6g}", f"{r[4]:.6g}"] for r in rows],
            colLabels=cols, loc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.25)

        plt.tight_layout()
        plt.show()

    return {
        "frame": frame,
        "background": {"slope": bkg_slope, "intercept": bkg_intercept},
        "r2": r2,
        "rows": rows,
        "result": result,
        "peaks": peaks,
        "pruned_indices": pruned_indices,
    }

# ------------------------------
# Map mode (run all frames, collect kept peaks, scatter plot) + PROGRESS
# ------------------------------

def map_all_frames(h5_path, peak_positions, anchor_peak0=ANCHOR_PEAK0):
    """
    Runs fit_single_frame() for every frame (plot=False), collects all KEPT peaks,
    and builds a scatter map: frame index vs center (q), colored by height.
    Shows a progress bar with live ETA.
    """
    with h5py.File(h5_path, "r") as f:
        nframes = f["int"].shape[0]

    # Per-peak storage
    per_peak_frames = {i: [] for i in range(len(peak_positions))}
    per_peak_centers = {i: [] for i in range(len(peak_positions))}
    per_peak_heights = {i: [] for i in range(len(peak_positions))}
    per_peak_r2 = {i: [] for i in range(len(peak_positions))}

    start = time.perf_counter()

    if _HAVE_TQDM:
        pbar = tqdm(total=nframes, desc="Mapping frames", unit="frame")
        last_update_time = start
    else:
        print(f"Mapping {nframes} frames...")
        last_print_len = 0

    for fr in range(nframes):
        out = fit_single_frame(h5_path, fr, peak_positions, plot=False, anchor_peak0=anchor_peak0)
        kept = out["rows"]  # [index, center, height, fwhm, amplitude]
        r2 = out["r2"]

        for (idx, center, height, _, _) in kept:
            idx = int(idx)
            per_peak_frames[idx].append(fr)
            per_peak_centers[idx].append(center)
            per_peak_heights[idx].append(height)
            per_peak_r2[idx].append(r2)

        # progress UI
        done = fr + 1
        now = time.perf_counter()
        elapsed = now - start
        rate = done / elapsed if elapsed > 0 else 0.0
        remain = (nframes - done) / rate if rate > 0 else float("inf")

        if _HAVE_TQDM:
            if (now - last_update_time) >= 0.2 or done == nframes:
                pbar.set_postfix_str(_format_eta(remain))
                last_update_time = now
            pbar.update(1)
        else:
            pct = 100.0 * done / nframes
            bar_n = 24
            filled = int(bar_n * done / nframes)
            bar = "█" * filled + "·" * (bar_n - filled)
            msg = f"[{bar}] {pct:6.2f}%  {done}/{nframes}  {_format_eta(remain)}"
            print("\r" + msg + " " * max(0, last_print_len - len(msg)), end="", flush=True)
            last_print_len = len(msg)

    if _HAVE_TQDM:
        pbar.close()
    else:
        print()  # newline after fallback bar

    # --- Plot map ---
    plt.rcParams.update({
        "figure.dpi": 160, "savefig.dpi": 300,
        "font.size": 16, "axes.labelsize": 18, "axes.titlesize": 20,
        "xtick.labelsize": 14, "ytick.labelsize": 14,
    })
    fig, ax = plt.subplots(figsize=(10.5, 6.2))

    any_points = False
    for i in range(len(peak_positions)):
        if len(per_peak_frames[i]) == 0:
            continue
        any_points = True
        sc = ax.scatter(per_peak_frames[i], per_peak_centers[i],
                        c=per_peak_heights[i], s=16, alpha=0.9, label=f"Peak {i+1}")
    if any_points:
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Peak height (a.u.)")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Center q (1/Å)")
    ax.set_title("Peak Map: center vs frame (color = height)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()

# ------------------------------
# CLI
# ------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fit Gaussian peaks (linear background) for a single frame or build a map across frames"
    )
    parser.add_argument("h5", help="HDF5 file with 'q' or 'tth' and 'int' datasets")
    parser.add_argument("peaks", type=float, nargs='+',
                        help="Peak q-positions (e.g., 3.025 3.012)")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--frame", type=int,
                      help="Frame index to fit and show plot (single-frame mode)")
    mode.add_argument("--map", action="store_true",
                      help="Run over all frames and plot a scatter map (center vs frame, color=height)")

    parser.add_argument("--no-anchor", action="store_true",
                        help="Do not anchor peak 0; let all centers float within the window")

    args = parser.parse_args()
    peak_positions = sorted(args.peaks)

    print(f"Peaks: {peak_positions}")
    print(f"Window: {WINDOW} q | Anchor tol (peak 0): {ANCHOR_TOL} q | anchor={'off' if args.no_anchor else 'on'}")
    print(f"height_min: {HEIGHT_MIN} | height_min_sigma: {HEIGHT_MIN_SIGMA}*robust_sigma")
    print(f"Sigma bounds: [{MIN_SIGMA_ABS}, {min(MAX_SIGMA_ABS, MAX_SIGMA_FRAC * WINDOW)}] q")
    print(f"Min peak separation: {MIN_SEP_Q} q")
    print(f"Background: BASELINE_QUANTILE={BASELINE_QUANTILE}, EXCLUDE_RADIUS={BKG_EXCLUDE_RADIUS}, "
          f"TRIM_FRAC={BKG_TRIM_FRACTION}, SLOPE_CAP={BKG_SLOPE_MAX_ABS}, "
          f"ROBUST_LOSS={'on' if USE_ROBUST_LOSS else 'off'}")
    print(f"Overlap merge: overlap>= {OVERLAP_COEF_MIN}, ΔAIC> {AIC_IMPROVE}, height_frac= {MERGE_HEIGHT_FRAC}")
    if RESCUE_ENABLED:
        print(f"Rescue: R2<{RESCUE_R2_MIN} or kept<{RESCUE_MIN_KEPT} -> expand_window×{RESCUE_EXPAND_WINDOW}, "
              f"center_tol=±{RESCUE_CENTER_TOL}, MAX_SIGMA_FRAC={RESCUE_MAX_SIGMA_FRAC}, reseed±{RESEED_SPAN}")

    if args.map:
        map_all_frames(args.h5, peak_positions, anchor_peak0=(not args.no_anchor))
    else:
        fit_single_frame(
            args.h5, args.frame, peak_positions,
            plot=True, anchor_peak0=(not args.no_anchor)
        )

if __name__ == "__main__":
    main()




