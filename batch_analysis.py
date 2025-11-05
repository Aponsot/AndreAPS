#!/usr/bin/env python3
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
from lmfit.models import GaussianModel, LinearModel

# ------------------------------
# Tunable constants (single-frame) — unchanged
# ------------------------------

WINDOW = 0.25

MIN_SIGMA_ABS = 0.001      # q units
MAX_SIGMA_ABS = 0.025      # q units
MAX_SIGMA_FRAC = 0.25      # also cap sigma to this fraction of WINDOW

ANCHOR_TOL = 0.005         # q units around the first specified peak position
ANCHOR_PEAK0 = True        # set False to let all centers float freely

CENTER_TOL = 0.020         # q units allowed drift from each guess for peaks i>0

HEIGHT_MIN = 5.0           # absolute floor (kept)
HEIGHT_MIN_SIGMA = 3.0     # AND relative floor: K * robust_sigma(y)
PRUNE_SMALL = True

BASELINE_QUANTILE = 0.20
BKG_EXCLUDE_RADIUS = 0.010
BKG_TRIM_FRACTION = 0.30
BKG_SLOPE_MAX_ABS = 2.0

USE_ROBUST_LOSS = True

PEAK_SIGMA_MIN = None
PEAK_SIGMA_MAX = None

# --- Rescue (fallback) settings for post-solidification frames ---
RESCUE_ENABLED = True
RESCUE_R2_MIN = 0.85
RESCUE_MIN_KEPT = 1
RESCUE_EXPAND_WINDOW = 1.6
RESCUE_CENTER_TOL = 0.050
RESCUE_MAX_SIGMA_FRAC = 0.30
RESEED_SPAN = 0.060

# ------------------------------
# Internal sequential-fit constants (not exposed via CLI)
# ------------------------------
# Greedy residual-add; keep these hard-coded to avoid "new knobs" in the CLI.
_AIC_IMPROVE = 6.0           # require meaningful AIC gain to add another peak
_MAX_PEAKS = 16              # hard cap to avoid runaway
_RESIDUAL_PICK_SPAN = 0.030  # ±q span to snap to local residual max near each guess
_USE_GUESSES_FIRST = True    # add peaks near supplied guesses in descending seed height

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

def _window_data(x, yfull, peak_positions, window=None):
    window = WINDOW if window is None else window
    center = np.mean(peak_positions)
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

def _robust_line_fit(x, y, max_iter=4, trim_frac=0.30):
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

def _make_center_bounds(xmin, xmax, centers, anchor_peak0, anchor_tol, center_tol):
    bnds = []
    for i, cx in enumerate(centers):
        if anchor_peak0 and i == 0:
            cmin = max(xmin, cx - anchor_tol)
            cmax = min(xmax, cx + anchor_tol)
        else:
            cmin = max(xmin, cx - center_tol)
            cmax = min(xmax, cx + center_tol)
        if cmin > cmax:
            cmin, cmax = min(cmin, cmax), max(cmin, cmax)
        bnds.append((cmin, cmax))
    return bnds

def build_model(xw, yw, centers, baseline, center_bounds):
    dx = np.mean(np.diff(xw)) if len(xw) > 1 else WINDOW
    min_sigma_global = max(0.75 * dx, MIN_SIGMA_ABS)
    max_sigma_global = min(MAX_SIGMA_ABS, MAX_SIGMA_FRAC * WINDOW)

    sigma0_list, height0_list = [], []
    for cx in centers:
        sigma0_est, height0 = _local_height_sigma_seeds(xw, yw, baseline, cx, w=0.010)
        sigma0_clipped = np.clip(sigma0_est, min_sigma_global, max_sigma_global)
        sigma0_list.append(float(sigma0_clipped))
        height0_list.append(float(height0))

    init_slope, init_intercept = _background_init(
        xw, yw, centers, BKG_EXCLUDE_RADIUS, sigma_seeds=sigma0_list
    )
    bkg = LinearModel(prefix="bkg_")
    model = bkg
    params = bkg.make_params(slope=init_slope, intercept=init_intercept)
    params["bkg_slope"].set(min=-BKG_SLOPE_MAX_ABS, max=BKG_SLOPE_MAX_ABS, value=init_slope, vary=True)
    params["bkg_intercept"].set(value=init_intercept, vary=True)

    for i, (cx, sigma0, height0) in enumerate(zip(centers, sigma0_list, height0_list)):
        gi = GaussianModel(prefix=f"g{i}_")
        model += gi

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

def _local_argmax(xw, yw, cx, span):
    m = (xw >= cx - span) & (xw <= cx + span)
    if not np.any(m):
        return float(cx)
    idx = np.argmax(yw[m])
    return float(xw[m][idx])

# ------------------------------
# Sequential residual-add (no new CLI flags)
# ------------------------------

def _sequential_fit_single_frame(xw, yw, peak_positions, anchor_peak0, baseline, noise):
    """
    Greedy residual-add:
      1) fit background only
      2) add Gaussians one-by-one near supplied guesses (largest seed first)
      3) accept addition only if ΔAIC >= _AIC_IMPROVE and height >= noise-aware threshold
    """
    dx = np.mean(np.diff(xw)) if len(xw) > 1 else WINDOW
    min_sigma_global = max(0.75 * dx, MIN_SIGMA_ABS)
    max_sigma_global = min(MAX_SIGMA_ABS, MAX_SIGMA_FRAC * WINDOW)

    # background-only init (no exclusions)
    init_slope, init_intercept = _background_init(xw, yw, [], BKG_EXCLUDE_RADIUS, sigma_seeds=None)
    model = LinearModel(prefix="bkg_")
    params = model.make_params(slope=init_slope, intercept=init_intercept)
    params["bkg_slope"].set(min=-BKG_SLOPE_MAX_ABS, max=BKG_SLOPE_MAX_ABS, value=init_slope, vary=True)
    params["bkg_intercept"].set(value=init_intercept, vary=True)

    loss_kwargs = {"loss": "soft_l1", "f_scale": noise} if USE_ROBUST_LOSS else {}
    best_res = model.fit(yw, params, x=xw, calc_covar=False, method="least_squares", max_nfev=800, **loss_kwargs)
    best_aic = best_res.aic

    xmin, xmax = float(np.min(xw)), float(np.max(xw))
    peak_positions = list(sorted(peak_positions))
    center_bounds_guess = _make_center_bounds(xmin, xmax, peak_positions, anchor_peak0, ANCHOR_TOL, CENTER_TOL)

    # order additions by local seed height near guesses
    seeds = []
    for cx in peak_positions:
        sig0, h0 = _local_height_sigma_seeds(xw, yw, baseline, cx, w=0.010)
        seeds.append((cx, float(h0), float(np.clip(sig0, min_sigma_global, max_sigma_global))))
    seeds.sort(key=lambda t: t[1], reverse=True)

    n_added = 0
    used_positions = []
    height_thresh = max(HEIGHT_MIN, HEIGHT_MIN_SIGMA * noise)

    def _place_next_center(resid):
        if _USE_GUESSES_FIRST:
            for cx, h0, s0 in seeds:
                if cx in used_positions:
                    continue
                cx_new = _local_argmax(xw, resid + baseline, cx, span=max(_RESIDUAL_PICK_SPAN, 2*np.mean(np.diff(xw))))
                return cx, cx_new, h0, s0
        # fallback: global residual max (rare)
        j = int(np.argmax(resid))
        cx_new = float(xw[j])
        sig0, h0 = _local_height_sigma_seeds(xw, yw, baseline, cx_new, w=0.010)
        s0 = float(np.clip(sig0, min_sigma_global, max_sigma_global))
        return cx_new, cx_new, h0, s0

    while n_added < min(_MAX_PEAKS, max(1, len(peak_positions))):
        resid = yw - best_res.best_fit
        guess_cx, place_cx, h0, s0 = _place_next_center(resid)

        # bounds for the new peak
        if guess_cx in peak_positions:
            i_guess = peak_positions.index(guess_cx)
            cmin, cmax = center_bounds_guess[i_guess]
        else:
            cmin = max(xmin, place_cx - CENTER_TOL)
            cmax = min(xmax, place_cx + CENTER_TOL)

        amp0 = max(h0, 0.0) * s0 * np.sqrt(2*np.pi)

        gi = GaussianModel(prefix=f"g{n_added}_")
        new_model = best_res.model + gi
        new_params = best_res.params.copy()
        new_params.update(gi.make_params(center=np.clip(place_cx, cmin, cmax),
                                         sigma=s0,
                                         amplitude=max(amp0, 0.0)))
        new_params[f"g{n_added}_center"].set(min=cmin, max=cmax)
        if anchor_peak0 and n_added == 0 and (guess_cx in peak_positions) and (peak_positions.index(guess_cx) == 0):
            new_params[f"g{n_added}_center"].set(min=max(xmin, place_cx - ANCHOR_TOL),
                                                 max=min(xmax, place_cx + ANCHOR_TOL))
        new_params[f"g{n_added}_sigma"].set(min=min_sigma_global, max=max_sigma_global)
        new_params[f"g{n_added}_amplitude"].set(min=0.0)

        trial_res = new_model.fit(yw, new_params, x=xw, calc_covar=False, method="least_squares", max_nfev=800, **loss_kwargs)
        dAIC = best_aic - trial_res.aic

        # ensure the *new* peak is substantive
        trial_peaks = extract_peaks(trial_res)
        this_peak = next((p for p in trial_peaks if p["index"] == n_added), None)
        too_small = (this_peak is None) or (this_peak["height"] < height_thresh)

        if (dAIC < _AIC_IMPROVE) or too_small:
            break

        best_res = trial_res
        best_aic = trial_res.aic
        used_positions.append(guess_cx)
        n_added += 1

    # Optional pruning refit (your existing logic)
    final_res = best_res
    if PRUNE_SMALL:
        peaks_now = extract_peaks(final_res)
        pruned = [p["index"] for p in peaks_now if p["height"] < height_thresh]
        if len(pruned) > 0:
            refit_params = final_res.params.copy()
            for i in pruned:
                refit_params[f"g{i}_amplitude"].set(value=0.0, vary=False)
                refit_params[f"g{i}_center"].set(vary=False)
                refit_params[f"g{i}_sigma"].set(vary=False)
            final_res = final_res.model.fit(yw, refit_params, x=xw, calc_covar=False, method="least_squares", max_nfev=800, **loss_kwargs)

    return final_res

# ------------------------------
# Fit a single frame (and plot) — CLI unchanged
# ------------------------------

def fit_single_frame(h5_path, frame, peak_positions, plot=True, anchor_peak0=ANCHOR_PEAK0):
    # Load data
    with h5py.File(h5_path, "r") as f:
        x = f["q"][:] if "q" in f else f["tth"][:]
        yfull = f["int"][frame, :]

    x = np.asarray(x, float)
    yfull = np.asarray(yfull, float)

    # ascending safeguard
    if x[0] > x[-1]:
        x = x[::-1]
        yfull = yfull[::-1]

    center, half, xw, yw = _window_data(x, yfull, peak_positions)
    xmin, xmax = float(np.min(xw)), float(np.max(xw))

    baseline = np.quantile(yw, BASELINE_QUANTILE)
    noise = robust_sigma(yw)

    # bounds (for rescue bound-check only; main fit is sequential)
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

    # --- Sequential residual-add (replaces the previous "all-at-once") ---
    result = _sequential_fit_single_frame(xw, yw, peak_positions, anchor_peak0, baseline, noise)
    r2 = compute_r2(yw, result.best_fit)

    # Optional pruning already performed inside; extract peaks
    peaks = extract_peaks(result)

    # --- Rescue path if the initial sequential fit likely failed ---
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

    # Kept peaks & table
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
        "pruned_indices": [p["index"] for p in peaks if p["height"] < thresh],
    }

# ------------------------------
# Rescue helper (unchanged except factoring)
# ------------------------------

def _refit_with_rescue(x, yfull, peak_positions, frame, anchor_peak0,
                       window, center_tol, max_sigma_frac):
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

    center_bounds = _make_center_bounds(
        xmin, xmax, reseeded, anchor_peak0,
        max(ANCHOR_TOL, min(RESEED_SPAN, center_tol)),
        center_tol
    )

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
    kept = [p for p in peaks if p["height"] >= max(HEIGHT_MIN, HEIGHT_MIN_SIGMA * noise)]

    return {
        "xw": xw, "yw": yw, "result": result, "r2": r2, "peaks": peaks,
        "kept": kept, "baseline": baseline, "noise": noise,
        "center": center, "half": half
    }

# ------------------------------
# Original CLI — unchanged
# ------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fit Gaussian peaks (linear background) in a single frame"
    )
    parser.add_argument("h5", help="HDF5 file with 'q' or 'tth' and 'int' datasets")
    parser.add_argument("peaks", type=float, nargs='+',
                        help="Peak q-positions (e.g., 3.025 3.012)")
    parser.add_argument("--frame", type=int, required=True,
                        help="Frame index to fit and show plot")
    parser.add_argument("--no-anchor", action="store_true",
                        help="Do not anchor peak 0; let all centers float within the window")

    args = parser.parse_args()
    peak_positions = sorted(args.peaks)

    print(f"Peaks: {peak_positions}")
    print(f"Window: {WINDOW} q | Anchor tol (peak 0): {ANCHOR_TOL} q | anchor={'off' if args.no_anchor else 'on'}")
    print(f"height_min: {HEIGHT_MIN} | height_min_sigma: {HEIGHT_MIN_SIGMA}*robust_sigma")
    print(f"Sigma bounds: [{MIN_SIGMA_ABS}, {min(MAX_SIGMA_ABS, MAX_SIGMA_FRAC * WINDOW)}] q")
    print(f"Background: BASELINE_QUANTILE={BASELINE_QUANTILE}, EXCLUDE_RADIUS={BKG_EXCLUDE_RADIUS}, "
          f"TRIM_FRAC={BKG_TRIM_FRACTION}, SLOPE_CAP={BKG_SLOPE_MAX_ABS}, "
          f"ROBUST_LOSS={'on' if USE_ROBUST_LOSS else 'off'}")
    print(f"Center tol (non-anchored peaks): ±{CENTER_TOL} q")

    res = fit_single_frame(
        args.h5, args.frame, peak_positions,
        plot=True, anchor_peak0=(not args.no_anchor)
    )

if __name__ == "__main__":
    main()
