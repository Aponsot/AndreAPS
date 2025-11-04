import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
from lmfit.models import GaussianModel, LinearModel
from tqdm import tqdm

# ------------------------------
# Tunable constants (adjust here)
# ------------------------------

# Window (in q units) around the mean of specified peak positions
WINDOW = 0.20

# Sigma bounds reasoning:
# - min_sigma prevents “delta-like” peaks smaller than sampling resolution.
# - max_sigma prevents a single broad peak from absorbing background/overlaps.
MIN_SIGMA_ABS = 0.001      # absolute lower bound on sigma (q units)
MAX_SIGMA_ABS = 0.025      # absolute upper bound on sigma (q units)
MAX_SIGMA_FRAC = 0.20      # also cap sigma to this fraction of WINDOW

# Anchor for peak 0
ANCHOR_TOL = 0.005  # q units around the specified first peak position

# Quality threshold for adopting seeds to the next frame
R2_MIN = 0.6

# Pruning threshold: remove peaks (set amplitude to 0) if height < HEIGHT_MIN
HEIGHT_MIN = 5.0

# Plot controls
SHOW_EXAMPLE_FIT = True   # Show detailed fit for the first frame in map mode

# Scatter map styling
SCATTER_MARKER_SIZE = 12
SCATTER_CMAP = "plasma"
SCATTER_VMAX_PERCENTILE = 99  # clip color scale to this percentile of heights

# ------------------------------
# Background controls (robust linear background)
# ------------------------------

# Use a lower quantile as baseline to seed sigma/height robustly
BASELINE_QUANTILE = 0.20

# Robust background initialization excluding peak neighborhoods
BKG_EXCLUDE_RADIUS = 0.010      # q units to exclude around each peak center
BKG_TRIM_FRACTION = 0.30        # fraction of largest residuals trimmed in robust line fit
BKG_SLOPE_TOL = 0.5             # allowed deviation around robust slope (intensity per q)
BKG_SLOPE_MAX_ABS = 2.0         # hard cap for slope magnitude (intensity per q)

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

def build_model(xw, yw, centers, baseline, center_bounds=None, init_from=None):
    """
    Build a model with a linear background and Gaussian peaks at given centers.
    Sigma is constrained by data sampling and instrument bounds.
    Amplitude is non-negative. Center bounds come from center_bounds or default to window edges.
    Background slope is initialized robustly and bounded to avoid runaway tilt.
    """
    dx = np.mean(np.diff(xw)) if len(xw) > 1 else WINDOW
    # Data-informed sigma bounds
    min_sigma = max(0.75 * dx, MIN_SIGMA_ABS)
    max_sigma = min(MAX_SIGMA_ABS, MAX_SIGMA_FRAC * WINDOW)

    # Robust background initialization excluding peak neighborhoods
    init_slope, init_intercept = _background_init(xw, yw, centers, BKG_EXCLUDE_RADIUS)
    bkg = LinearModel(prefix="bkg_")
    model = bkg
    params = bkg.make_params(slope=init_slope, intercept=init_intercept)

    # Constrain slope around robust estimate; hard cap prevents runaway tilt
    smin = max(-BKG_SLOPE_MAX_ABS, init_slope - BKG_SLOPE_TOL)
    smax = min(BKG_SLOPE_MAX_ABS,  init_slope + BKG_SLOPE_TOL)
    params["bkg_slope"].set(min=smin, max=smax, value=init_slope, vary=True)
    # Intercept can vary freely (add bounds if your data requires)
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

        params.update(gi.make_params(center=cx, sigma=sigma0, amplitude=amp0))

        # Bounds for center: use provided bounds or default to window edges
        if center_bounds is None or center_bounds[i] is None:
            cmin = xw[0]
            cmax = xw[-1]
        else:
            cmin, cmax = center_bounds[i]
            cmin = max(xw[0], cmin)
            cmax = min(xw[-1], cmax)

        # Initial values: use provided seeds if available and inside bounds
        c_init = cx
        s_init = sigma0
        a_init = amp0
        if init_from is not None:
            if "center" in init_from and i < len(init_from["center"]) and init_from["center"][i] is not None:
                c_init = np.clip(init_from["center"][i], cmin, cmax)
            if "sigma" in init_from and i < len(init_from["sigma"]) and init_from["sigma"][i] is not None:
                s_init = float(np.clip(abs(init_from["sigma"][i]), min_sigma, max_sigma))
            if "amplitude" in init_from and i < len(init_from["amplitude"]) and init_from["amplitude"][i] is not None:
                a_seed = float(init_from["amplitude"][i])
                if a_seed > 0.0:
                    a_init = a_seed
                # else keep local estimate amp0

        params[f"g{i}_center"].set(min=cmin, max=cmax, value=c_init)
        # Sigma constrained by min/max
        params[f"g{i}_sigma"].set(min=min_sigma, max=max_sigma, value=s_init)
        # Non-negative amplitude
        params[f"g{i}_amplitude"].set(min=0.0, value=a_init)

    return model, params, (min_sigma, max_sigma)

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

def _window_data(x, yfull, peak_positions):
    center = np.mean(peak_positions)
    half = WINDOW / 2.0
    m = (x >= center - half) & (x <= center + half)
    xw, yw = x[m], yfull[m]
    mfin = np.isfinite(xw) & np.isfinite(yw)
    xw, yw = xw[mfin], yw[mfin]
    return center, half, xw, yw

def _param_at_bounds(result, name, frac_tol=0.05):
    """Check if a parameter is near its min/max bounds."""
    if name not in result.params:
        return False
    p = result.params[name]
    if (p.min is None) or (p.max is None):
        return False
    rng = max(p.max - p.min, 1e-12)
    rel = (p.value - p.min) / rng
    return (rel < frac_tol) or (rel > 1 - frac_tol)


# ------------------------------
# Fit one frame (seed-only logic; pruning)
# ------------------------------

def fit_peaks(h5_path, frame, peak_positions, plot=True, prev_solution=None,
              seed_only=True, anchor_each_frame=True):
    """
    Fit the specified peaks in a single frame using seed-only logic:
      - Centers free within the window for all peaks (no per-frame movement constraints).
      - Optionally anchor peak 0 center within [cx0 ± ANCHOR_TOL] every frame.
      - Use previous frame's best-fit parameters only to seed initial values.

    Robust linear background and pruning are applied.

    After fitting, prune peaks with height < HEIGHT_MIN by fixing amplitude to 0
    (and fixing center/sigma), then refit so pruned components do not affect other parameters.
    """
    with h5py.File(h5_path, "r") as f:
        x = f["q"][:] if "q" in f else f["tth"][:]
        yfull = f["int"][frame, :]

    x = np.asarray(x, float)
    yfull = np.asarray(yfull, float)

    center, half, xw, yw = _window_data(x, yfull, peak_positions)
    if xw.size < 5:
        raise ValueError("Too few points in window.")

    # Lower-quantile baseline for robust sigma/height seeding
    baseline = np.quantile(yw, BASELINE_QUANTILE)

    # Seed arrays from previous solution if available (used for initialization only)
    init_from = None
    if prev_solution is not None:
        centers_seed, sigmas_seed, amps_seed = [], [], []
        for i in range(len(peak_positions)):
            centers_seed.append(prev_solution.params[f"g{i}_center"].value
                                if f"g{i}_center" in prev_solution.params else peak_positions[i])
            sigmas_seed.append(prev_solution.params[f"g{i}_sigma"].value
                               if f"g{i}_sigma" in prev_solution.params else None)
            amps_seed.append(prev_solution.params[f"g{i}_amplitude"].value
                             if f"g{i}_amplitude" in prev_solution.params else None)
        init_from = {"center": centers_seed, "sigma": sigmas_seed, "amplitude": amps_seed}

    # Build center bounds: seed-only => free within window; optional anchor peak 0 every frame
    center_bounds = []
    for i, cx in enumerate(peak_positions):
        if anchor_each_frame and i == 0:
            cmin = max(xw[0], cx - ANCHOR_TOL)
            cmax = min(xw[-1], cx + ANCHOR_TOL)
        else:
            cmin, cmax = xw[0], xw[-1]
        center_bounds.append((cmin, cmax))

    # Robust loss configuration
    loss_kwargs = {}
    if USE_ROBUST_LOSS:
        loss_kwargs = {"loss": "soft_l1", "f_scale": robust_sigma(yw)}

    # Fit
    model, params, _ = build_model(
        xw, yw, peak_positions, baseline,
        center_bounds=center_bounds,
        init_from=init_from
    )
    result = model.fit(yw, params, x=xw, calc_covar=False,
                       method="least_squares", max_nfev=800, **loss_kwargs)
    r2 = compute_r2(yw, result.best_fit)

    # If background slope is stuck near its bounds, freeze it to the robust init and refit
    if _param_at_bounds(result, "bkg_slope", frac_tol=0.05):
        params2 = result.params.copy()
        params2["bkg_slope"].set(value=params2["bkg_slope"].value, vary=False)
        result = model.fit(yw, params2, x=xw, calc_covar=False,
                           method="least_squares", max_nfev=800, **loss_kwargs)
        r2 = compute_r2(yw, result.best_fit)

    # Prune small peaks by height and refit with amplitudes fixed to 0 (and center/sigma fixed)
    peaks = extract_peaks(result)
    pruned_indices = [p["index"] for p in peaks if p["height"] < HEIGHT_MIN]
    if len(pruned_indices) > 0:
        init_from_refit = {
            "center": [result.params[f"g{i}_center"].value for i in range(len(peak_positions))],
            "sigma":  [abs(result.params[f"g{i}_sigma"].value) for i in range(len(peak_positions))],
            "amplitude": [result.params[f"g{i}_amplitude"].value for i in range(len(peak_positions))]
        }
        model_refit, params_refit, _ = build_model(
            xw, yw, peak_positions, baseline,
            center_bounds=center_bounds,
            init_from=init_from_refit
        )
        # Fix pruned peaks to zero amplitude and freeze center/sigma to avoid wandering
        for i in pruned_indices:
            params_refit[f"g{i}_amplitude"].set(value=0.0, vary=False)
            params_refit[f"g{i}_center"].set(value=init_from_refit["center"][i], vary=False)
            params_refit[f"g{i}_sigma"].set(value=init_from_refit["sigma"][i], vary=False)

        result = model_refit.fit(yw, params_refit, x=xw, calc_covar=False,
                                 method="least_squares", max_nfev=800, **loss_kwargs)
        r2 = compute_r2(yw, result.best_fit)
        peaks = extract_peaks(result)
    else:
        peaks = extract_peaks(result)

    bkg_slope = result.params["bkg_slope"].value
    bkg_intercept = result.params["bkg_intercept"].value

    # Build rows for kept peaks only (height >= threshold)
    kept = [p for p in peaks if p["height"] >= HEIGHT_MIN]
    rows = [[p["index"], p["center"], p["height"], p["fwhm"], p["amplitude"]]
            for p in kept]

    if plot:
        plt.rcParams.update({
            "figure.dpi": 160, "savefig.dpi": 300,
            "font.size": 16, "axes.labelsize": 18, "axes.titlesize": 20,
            "xtick.labelsize": 14, "ytick.labelsize": 14,
        })
        fig, (ax, ax_tbl) = plt.subplots(2, 1, figsize=(10, 6.8),
                                         gridspec_kw={"height_ratios": [3, 1]})

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
            cellText=[[f"{r[0]}", f"{r[1]:.6g}", f"{r[2]:.6g}",
                      f"{r[3]:.6g}", f"{r[4]:.6g}"] for r in rows],
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
        "rows": rows,  # kept peaks only
        "result": result,
        "n_peaks": len(kept)
    }


# ------------------------------
# Sequential tracking (seed-only map)
# ------------------------------

def track_sequential(h5_path, peak_positions, anchor_every_frame=True):
    """
    Sequentially fit frames with seed-only logic:
      - Centers free within the window (no per-frame movement constraints).
      - Optionally anchor peak 0 every frame within ANCHOR_TOL.
      - Seed from the immediate prior frame only if the prior fit is reasonable (R2 >= R2_MIN).

    Returns tracking arrays for centers and heights (post-pruning).
    """
    with h5py.File(h5_path, "r") as f:
        x = f["q"][:] if "q" in f else f["tth"][:]
        int_ds = f["int"]
        nframes_total = int_ds.shape[0]
        frames = range(0, nframes_total)

        centers_tr = [[] for _ in peak_positions]
        heights_tr = [[] for _ in peak_positions]
        r2_tr = []

        prev_solution = None
        first = True

        for fr in tqdm(frames, desc="Seed-only tracking", unit="frame"):
            plot_this = SHOW_EXAMPLE_FIT and first
            try:
                out = fit_peaks(
                    h5_path, fr, peak_positions, plot=plot_this,
                    prev_solution=prev_solution,
                    seed_only=True,
                    anchor_each_frame=anchor_every_frame
                )
                result = out["result"]
                r2 = out["r2"]
                r2_tr.append(r2)

                # Extract centers/heights after pruning; peaks below threshold get height=0 and center from frozen values
                for i in range(len(peak_positions)):
                    ai = result.params[f"g{i}_amplitude"].value
                    si = abs(result.params[f"g{i}_sigma"].value)
                    hi = ai / (si * np.sqrt(2 * np.pi)) if si > 0 else 0.0
                    ci = result.params[f"g{i}_center"].value
                    centers_tr[i].append(ci)
                    heights_tr[i].append(max(hi, 0.0))

                # Seed next frame only if fit is reasonable
                prev_solution = result if r2 >= R2_MIN else None

            except Exception:
                r2_tr.append(np.nan)
                for i in range(len(peak_positions)):
                    centers_tr[i].append(np.nan)
                    heights_tr[i].append(0.0)
                prev_solution = None

            first = False

    return {
        "frames": list(frames),
        "centers": centers_tr,
        "heights": heights_tr,
        "r2": r2_tr,
        "peak_positions": peak_positions,
        "height_min": HEIGHT_MIN
    }

def plot_scatter_map(tracking):
    """
    Plot scatter: frame vs q-center, color = peak height.
    """
    frames = tracking["frames"]
    centers_tr = tracking["centers"]
    heights_tr = tracking["heights"]

    xs = []  # frame
    ys = []  # q center
    cs = []  # height

    for i in range(len(centers_tr)):
        for f_idx, fr in enumerate(frames):
            c = centers_tr[i][f_idx]
            h = heights_tr[i][f_idx]
            if np.isfinite(c) and h > 0.0:
                xs.append(fr)
                ys.append(c)
                cs.append(h)

    if len(xs) == 0:
        print("No peaks above height_min found to plot.")
        return

    vmax = np.percentile(cs, SCATTER_VMAX_PERCENTILE) if len(cs) > 20 else max(cs)
    plt.rcParams.update({
        "figure.dpi": 160, "savefig.dpi": 300,
        "font.size": 20, "axes.labelsize": 20, "axes.titlesize": 20,
        "xtick.labelsize": 16, "ytick.labelsize": 16,
    })

    plt.figure(figsize=(9, 5))
    sc = plt.scatter(xs, ys, c=cs, s=SCATTER_MARKER_SIZE, cmap=SCATTER_CMAP, vmin=0.0, vmax=vmax)
    cbar = plt.colorbar(sc)
    cbar.set_label("Peak height (a.u.)")
    plt.xlabel("Frame")
    plt.ylabel("q (1/Å)")
    plt.title(f"Peak map (height_min={HEIGHT_MIN}, anchor_tol={ANCHOR_TOL})")
    plt.tight_layout()
    plt.show()


# ------------------------------
# Minimal CLI
# ------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fit Gaussian peaks (linear background) at specified q-positions"
    )
    parser.add_argument("h5", help="HDF5 file with 'q' or 'tth' and 'int' datasets")
    parser.add_argument("peaks", type=float, nargs='+',
                        help="Peak q-positions (e.g., 3.025 3.012)")
    parser.add_argument("--frame", type=int, help="Fit a single frame and show plot")
    parser.add_argument("--map", action="store_true",
                        help="Sequential tracking across all frames (seed-only; anchors peak 0 every frame)")

    args = parser.parse_args()
    peak_positions = sorted(args.peaks)

    print(f"Peaks: {peak_positions}")
    print(f"Window: {WINDOW} q | Anchor tol (peak 0 each frame): {ANCHOR_TOL} q")
    print(f"R2_MIN: {R2_MIN} | height_min: {HEIGHT_MIN}")
    print(f"Sigma bounds: [{MIN_SIGMA_ABS}, {min(MAX_SIGMA_ABS, MAX_SIGMA_FRAC * WINDOW)}] q (data-informed)")
    print(f"Background: BASELINE_QUANTILE={BASELINE_QUANTILE}, EXCLUDE_RADIUS={BKG_EXCLUDE_RADIUS}, "
          f"TRIM_FRAC={BKG_TRIM_FRACTION}, SLOPE_TOL={BKG_SLOPE_TOL}, SLOPE_CAP={BKG_SLOPE_MAX_ABS}, "
          f"ROBUST_LOSS={'on' if USE_ROBUST_LOSS else 'off'}")

    if args.frame is not None:
        # Single-frame fit uses the same seed-only logic
        fit_peaks(args.h5, args.frame, peak_positions, plot=True,
                  prev_solution=None, seed_only=True, anchor_each_frame=True)
    elif args.map:
        tr = track_sequential(args.h5, peak_positions, anchor_every_frame=True)
        plot_scatter_map(tr)
    else:
        print("Specify --frame N to fit a single frame, or --map for sequential tracking across all frames.")

if __name__ == "__main__":
    main()
