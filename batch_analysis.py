
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
from lmfit.models import GaussianModel, LinearModel

# ------------------------------
# Tunable constants (single-frame)
# ------------------------------

# Window (in q units) around the mean of specified peak positions
WINDOW = 0.25

# Sigma bounds
MIN_SIGMA_ABS = 0.001      # q units
MAX_SIGMA_ABS = 0.025      # q units
MAX_SIGMA_FRAC = 0.20      # also cap sigma to this fraction of WINDOW

# Anchor for peak 0 every frame (optional)
ANCHOR_TOL = 0.005         # q units around the first specified peak position
ANCHOR_PEAK0 = True        # set False to let all centers float freely

# Pruning threshold: remove peaks (set amplitude to 0) if height < HEIGHT_MIN
HEIGHT_MIN = 5.0
PRUNE_SMALL = True

# Background controls (robust linear background)
BASELINE_QUANTILE = 0.20       # lower quantile baseline for sigma/height seeding
BKG_EXCLUDE_RADIUS = 0.010     # exclude q around each peak when estimating background
BKG_TRIM_FRACTION = 0.30       # trimmed regression fraction
BKG_SLOPE_MAX_ABS = 2.0        # hard cap on background slope (intensity per q)

# Use robust loss in least-squares to reduce outlier impact
USE_ROBUST_LOSS = True


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

def _estimate_sigma0(xw, yw, baseline, cx):
    # Estimate initial sigma from a local FWHM around cx using half-maximum
    if xw.size < 3:
        return max(np.mean(np.diff(xw)), MIN_SIGMA_ABS)
    p = np.argmin(np.abs(xw - cx))
    ypk = yw[p]
    h = max(ypk - baseline, robust_sigma(yw))
    half_level = baseline + 0.5 * h
    # search left
    xl = xw[0]
    for i in range(p, 0, -1):
        if yw[i] <= half_level and yw[i-1] > half_level:
            t = (half_level - yw[i]) / (yw[i-1] - yw[i] + 1e-16)
            xl = xw[i] + t * (xw[i-1] - xw[i])
            break
    # search right
    xr = xw[-1]
    for i in range(p, len(xw)-1):
        if yw[i] > half_level and yw[i+1] <= half_level:
            t = (half_level - yw[i+1]) / (yw[i] - yw[i+1] + 1e-16)
            xr = xw[i+1] + t * (xw[i] - xw[i+1])
            break
    fwhm = max(xr - xl, np.mean(np.diff(xw)))
    return max(fwhm / 2.354820045, MIN_SIGMA_ABS)

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

def _background_init(xw, yw, centers, exclude_radius):
    mask = np.ones_like(xw, dtype=bool)
    for cx in centers:
        mask &= (np.abs(xw - cx) > exclude_radius)
    if mask.sum() >= max(5, int(0.2 * len(xw))):
        m, b = _robust_line_fit(xw[mask], yw[mask], trim_frac=BKG_TRIM_FRACTION)
    else:
        # Fallback if not enough off-peak points: use flat baseline near lower envelope
        m = 0.0
        b = np.quantile(yw, BASELINE_QUANTILE)
    # Bound slope reasonably
    m = float(np.clip(m, -BKG_SLOPE_MAX_ABS, BKG_SLOPE_MAX_ABS))
    return m, float(b)

def build_model(xw, yw, centers, baseline, center_bounds):
    """
    Build a model with a linear background and Gaussian peaks at given centers.
    Background slope is initialized robustly and capped to avoid runaway tilt.
    """
    dx = np.mean(np.diff(xw)) if len(xw) > 1 else WINDOW
    min_sigma = max(0.75 * dx, MIN_SIGMA_ABS)
    max_sigma = min(MAX_SIGMA_ABS, MAX_SIGMA_FRAC * WINDOW)

    # Robust background init
    init_slope, init_intercept = _background_init(xw, yw, centers, BKG_EXCLUDE_RADIUS)
    bkg = LinearModel(prefix="bkg_")
    model = bkg
    params = bkg.make_params(slope=init_slope, intercept=init_intercept)
    params["bkg_slope"].set(min=-BKG_SLOPE_MAX_ABS, max=BKG_SLOPE_MAX_ABS, value=init_slope, vary=True)
    params["bkg_intercept"].set(value=init_intercept, vary=True)

    for i, cx in enumerate(centers):
        gi = GaussianModel(prefix=f"g{i}_")
        model += gi

        # Initial estimates
        sigma0_est = _estimate_sigma0(xw, yw, baseline, cx)
        sigma0 = np.clip(sigma0_est, min_sigma, max_sigma)

        # Height/amplitude from initial sigma
        p = np.argmin(np.abs(xw - cx))
        ypk = yw[p]
        height0 = max(ypk - baseline, robust_sigma(yw))
        amp0 = height0 * sigma0 * np.sqrt(2 * np.pi)

        cmin, cmax = center_bounds[i]
        params.update(gi.make_params(center=np.clip(cx, cmin, cmax), sigma=sigma0, amplitude=max(amp0, 0.0)))

        params[f"g{i}_center"].set(min=cmin, max=cmax)
        params[f"g{i}_sigma"].set(min=min_sigma, max=max_sigma)
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

    # Simple ascending-q safeguard (no orientation detection)
    if x[0] > x[-1]:
        x = x[::-1]
        yfull = yfull[::-1]

    center, half, xw, yw = _window_data(x, yfull, peak_positions)
    xmin, xmax = float(np.min(xw)), float(np.max(xw))

    # Lower-quantile baseline for robust sigma/height seeding
    baseline = np.quantile(yw, BASELINE_QUANTILE)

    # Center bounds: free within window; optionally anchor peak 0
    center_bounds = []
    for i, cx in enumerate(peak_positions):
        if anchor_peak0 and i == 0:
            cmin = max(xmin, cx - ANCHOR_TOL)
            cmax = min(xmax, cx + ANCHOR_TOL)
        else:
            cmin, cmax = xmin, xmax
        if cmin > cmax:
            cmin, cmax = min(cmin, cmax), max(cmin, cmax)
        center_bounds.append((cmin, cmax))

    # Robust loss config
    loss_kwargs = {}
    if USE_ROBUST_LOSS:
        loss_kwargs = {"loss": "soft_l1", "f_scale": robust_sigma(yw)}

    # Build and fit
    model, params = build_model(xw, yw, peak_positions, baseline, center_bounds)
    result = model.fit(yw, params, x=xw, calc_covar=False, method="least_squares", max_nfev=800, **loss_kwargs)
    r2 = compute_r2(yw, result.best_fit)

    # Optional pruning and refit
    peaks = extract_peaks(result)
    pruned_indices = []
    if PRUNE_SMALL:
        pruned_indices = [p["index"] for p in peaks if p["height"] < HEIGHT_MIN]
        if len(pruned_indices) > 0:
            params_refit = result.params.copy()
            for i in pruned_indices:
                params_refit[f"g{i}_amplitude"].set(value=0.0, vary=False)
                params_refit[f"g{i}_center"].set(vary=False)
                params_refit[f"g{i}_sigma"].set(vary=False)
            result = model.fit(yw, params_refit, x=xw, calc_covar=False, method="least_squares", max_nfev=800, **loss_kwargs)
            r2 = compute_r2(yw, result.best_fit)
            peaks = extract_peaks(result)

    bkg_slope = result.params["bkg_slope"].value
    bkg_intercept = result.params["bkg_intercept"].value

    # Kept peaks for table
    kept = [p for p in peaks if p["height"] >= HEIGHT_MIN]
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

        ax.set_xlabel("q (1/Å)")
        ax.set_ylabel("Intensity")
        ax.set_title(f"Frame {frame} | {len(kept)} kept peaks | R²={r2:.4f} | height_min={HEIGHT_MIN}")
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
# Minimal CLI (single-frame only)
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
    print(f"height_min: {HEIGHT_MIN}")
    print(f"Sigma bounds: [{MIN_SIGMA_ABS}, {min(MAX_SIGMA_ABS, MAX_SIGMA_FRAC * WINDOW)}] q")
    print(f"Background: BASELINE_QUANTILE={BASELINE_QUANTILE}, EXCLUDE_RADIUS={BKG_EXCLUDE_RADIUS}, "
          f"TRIM_FRAC={BKG_TRIM_FRACTION}, SLOPE_CAP={BKG_SLOPE_MAX_ABS}, "
          f"ROBUST_LOSS={'on' if USE_ROBUST_LOSS else 'off'}")

    fit_single_frame(
        args.h5, args.frame, peak_positions,
        plot=True, anchor_peak0=(not args.no_anchor)
    )

if __name__ == "__main__":
    main()

