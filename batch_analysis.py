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
WINDOW = 0.15

# Peak width constraints
MIN_SIGMA = 1e-6
MAX_SIGMA_FRAC = 0.22

# Seeding: allowed per-frame center movement (q units)
PER_FRAME_SHIFT = 0.015

# Quality threshold for accepting seeded result
R2_MIN = 0.6

# Pruning threshold: remove peaks (set amplitude to 0) if height < HEIGHT_MIN
HEIGHT_MIN = 5.0

# Amorphous/free movement ranges (disable per-frame center constraints in these frame ranges)
# Example: FREE_RANGES = [(400, 520), (780, 820)]
FREE_RANGES = []

# Plot controls
SHOW_EXAMPLE_FIT = True   # Show detailed fit for the first frame in map mode
SUMMARY_PLOTS = False     # Disable line summary plots; we use scatter colored by height instead

# Scatter map styling
SCATTER_MARKER_SIZE = 12
SCATTER_CMAP = "plasma"
SCATTER_VMAX_PERCENTILE = 99  # clip color scale to this percentile of heights


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

def build_model(xw, yw, centers, baseline, center_bounds=None, init_from=None):
    """
    Build a model with a linear background and Gaussian peaks at given centers,
    with constraints on center (from center_bounds), sigma, and non-negative amplitude.

    center_bounds: optional list of (min,max) per peak center.
    init_from: optional dict with 'center', 'sigma', 'amplitude' arrays for initial values.
    """
    dx = np.mean(np.diff(xw)) if len(xw) > 1 else WINDOW
    min_sigma = max(dx / 3.0, MIN_SIGMA)
    max_sigma = MAX_SIGMA_FRAC * WINDOW

    bkg = LinearModel(prefix="bkg_")
    model = bkg
    params = bkg.make_params(slope=0.0, intercept=baseline)

    for i, cx in enumerate(centers):
        gi = GaussianModel(prefix=f"g{i}_")
        model += gi

        # Initial estimates
        p = np.argmin(np.abs(xw - cx))
        ypk = yw[p]
        height0 = max(ypk - baseline, robust_sigma(yw))
        sigma0 = np.clip(dx, min_sigma, max_sigma)
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
                s_init = float(np.clip(init_from["sigma"][i], min_sigma, max_sigma))
            if "amplitude" in init_from and i < len(init_from["amplitude"]) and init_from["amplitude"][i] is not None:
                a_init = max(float(init_from["amplitude"][i]), 0.0)

        params[f"g{i}_center"].set(min=cmin, max=cmax, value=c_init)
        params[f"g{i}_sigma"].set(min=min_sigma, max=max_sigma, value=s_init)
        params[f"g{i}_amplitude"].set(min=0.0, value=a_init)

    return model, params

def extract_peaks(result):
    peaks = []
    i = 0
    while f"g{i}_center" in result.params:
        ctr = result.params[f"g{i}_center"].value
        sig = result.params[f"g{i}_sigma"].value
        amp = result.params[f"g{i}_amplitude"].value
        hgt = amp / (sig * np.sqrt(2 * np.pi)) if sig > 0 else 0.0
        fwhm = 2.354820045 * sig
        peaks.append({
            "index": i, "center": ctr, "height": hgt,
            "fwhm": fwhm, "amplitude": amp
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

def _detect_bounds_stuck(result, center_bounds, sigma_bounds, frac_tol=0.05):
    """Return True if any center/sigma is too close to bounds, suggesting instability."""
    def at_bound(val, low, high):
        rng = max(high - low, 1e-12)
        rel = (val - low) / rng
        return (rel < frac_tol) or (rel > 1 - frac_tol)

    n = len(center_bounds)
    hits = 0
    for i in range(n):
        cmin, cmax = center_bounds[i]
        smin, smax = sigma_bounds[i]
        cval = result.params[f"g{i}_center"].value
        sval = result.params[f"g{i}_sigma"].value
        if at_bound(cval, cmin, cmax):
            hits += 1
        if at_bound(sval, smin, smax):
            hits += 1
    return hits >= max(1, n // 2)

def in_any_range(frame, ranges):
    for a, b in ranges:
        if a <= frame <= b:
            return True
    return False


# ------------------------------
# Fit one frame (with pruning)
# ------------------------------

def fit_peaks(h5_path, frame, peak_positions, plot=True, prev_solution=None):
    """
    Fit the specified peaks in a single frame with:
      - Seeded per-frame bounds: center in [prev_center ± PER_FRAME_SHIFT], if available.
      - Free bounds within the window in frames within FREE_RANGES (amorphous zones).
    Fallback: if seeded fit looks poor or is stuck on bounds, refit with free bounds.

    After fitting, prune peaks with height < HEIGHT_MIN by fixing their amplitude to 0
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

    baseline = np.median(yw)
    noise = robust_sigma(yw)

    # Seed arrays from previous solution if available
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

    # Build bounds
    dx = np.mean(np.diff(xw)) if len(xw) > 1 else WINDOW
    min_sigma = max(dx / 3.0, MIN_SIGMA)
    max_sigma = MAX_SIGMA_FRAC * WINDOW
    sigma_bounds = [(min_sigma, max_sigma) for _ in peak_positions]

    # Center bounds for seeded attempt (or free in amorphous ranges)
    free_here = in_any_range(frame, FREE_RANGES)
    center_bounds_seed = []
    for i, cx in enumerate(peak_positions):
        if free_here or prev_solution is None or init_from is None:
            cmin, cmax = xw[0], xw[-1]  # free within window
        else:
            prev_c = init_from["center"][i]
            cmin = max(xw[0], prev_c - PER_FRAME_SHIFT)
            cmax = min(xw[-1], prev_c + PER_FRAME_SHIFT)
        center_bounds_seed.append((cmin, cmax))

    # First, seeded (or free) fit
    model_seed, params_seed = build_model(
        xw, yw, peak_positions, baseline,
        center_bounds=center_bounds_seed,
        init_from=init_from
    )
    result_seed = model_seed.fit(yw, params_seed, x=xw, calc_covar=False,
                                 method="least_squares", max_nfev=600)
    r2_seed = compute_r2(yw, result_seed.best_fit)
    peaks_seed = extract_peaks(result_seed)
    stuck_seed = _detect_bounds_stuck(result_seed, center_bounds_seed, sigma_bounds)

    # Fallback: if quality is poor or stuck on bounds, retry with free bounds (window only)
    use_fallback = (r2_seed < R2_MIN) or (not free_here and stuck_seed)
    if use_fallback:
        center_bounds_free = [(xw[0], xw[-1]) for _ in peak_positions]
        model_free, params_free = build_model(
            xw, yw, peak_positions, baseline,
            center_bounds=center_bounds_free,
            init_from=init_from
        )
        result_free = model_free.fit(yw, params_free, x=xw, calc_covar=False,
                                     method="least_squares", max_nfev=600)
        r2_free = compute_r2(yw, result_free.best_fit)
        peaks_free = extract_peaks(result_free)

        # Choose the better by R²
        if r2_free >= r2_seed:
            result = result_free
            peaks = peaks_free
            r2 = r2_free
            center_bounds_used = center_bounds_free
        else:
            result = result_seed
            peaks = peaks_seed
            r2 = r2_seed
            center_bounds_used = center_bounds_seed
    else:
        result = result_seed
        peaks = peaks_seed
        r2 = r2_seed
        center_bounds_used = center_bounds_seed

    # Prune small peaks by height and refit with amplitudes fixed to 0 (and center/sigma fixed)
    pruned_indices = [p["index"] for p in peaks if p["height"] < HEIGHT_MIN]
    if len(pruned_indices) > 0:
        init_from_refit = {
            "center": [result.params[f"g{i}_center"].value for i in range(len(peak_positions))],
            "sigma":  [result.params[f"g{i}_sigma"].value  for i in range(len(peak_positions))],
            "amplitude": [result.params[f"g{i}_amplitude"].value for i in range(len(peak_positions))]
        }
        model_refit, params_refit = build_model(
            xw, yw, peak_positions, baseline,
            center_bounds=center_bounds_used,
            init_from=init_from_refit
        )
        # Fix pruned peaks to zero amplitude and freeze center/sigma to avoid wandering
        for i in pruned_indices:
            params_refit[f"g{i}_amplitude"].set(value=0.0, vary=False)
            params_refit[f"g{i}_center"].set(value=init_from_refit["center"][i], vary=False)
            params_refit[f"g{i}_sigma"].set(value=init_from_refit["sigma"][i], vary=False)

        result = model_refit.fit(yw, params_refit, x=xw, calc_covar=False,
                                 method="least_squares", max_nfev=600)
        r2 = compute_r2(yw, result.best_fit)
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
        "noise": noise,
        "n_peaks": len(kept)
    }


# ------------------------------
# Sequential tracking (map mode)
# ------------------------------

def track_sequential(h5_path, peak_positions):
    """
    Sequentially fit frames with seeding from previous frame and small per-frame shift bounds.
    In frames within FREE_RANGES (amorphous zones), per-frame constraints are disabled.

    Returns tracking arrays for centers and heights (after pruning).
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
        last_good = None
        first = True

        for fr in tqdm(frames, desc="Sequential tracking", unit="frame"):
            plot_this = SHOW_EXAMPLE_FIT and first
            try:
                out = fit_peaks(
                    h5_path, fr, peak_positions, plot=plot_this,
                    prev_solution=(prev_solution or last_good)
                )
                result = out["result"]
                r2 = out["r2"]
                r2_tr.append(r2)

                # Extract centers/heights; mask pruned (height < threshold) as NaN center and 0 height
                for i in range(len(peak_positions)):
                    ai = result.params[f"g{i}_amplitude"].value
                    si = result.params[f"g{i}_sigma"].value
                    hi = ai / (si * np.sqrt(2 * np.pi)) if si > 0 else 0.0
                    if hi >= HEIGHT_MIN:
                        ci = result.params[f"g{i}_center"].value
                        centers_tr[i].append(ci)
                        heights_tr[i].append(hi)
                    else:
                        centers_tr[i].append(np.nan)
                        heights_tr[i].append(0.0)

                # Update seeds: adopt current if fit is reasonable
                if r2 >= R2_MIN:
                    prev_solution = result
                    last_good = result
                else:
                    prev_solution = last_good

            except Exception:
                r2_tr.append(np.nan)
                for i in range(len(peak_positions)):
                    centers_tr[i].append(np.nan)
                    heights_tr[i].append(0.0)
                prev_solution = last_good

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

    # Color clipping to avoid outliers dominating the color scale
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
    plt.title(f"Peak map (height_min={HEIGHT_MIN})")
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
    parser.add_argument("--map", action="store_true", help="Sequential tracking across all frames (scatter colored by height)")

    args = parser.parse_args()
    peak_positions = sorted(args.peaks)

    print(f"Peaks: {peak_positions}")
    print(f"Window: {WINDOW} q | Per-frame shift: {PER_FRAME_SHIFT} q | R2_MIN: {R2_MIN} | height_min: {HEIGHT_MIN}")
    if FREE_RANGES:
        print(f"Free movement ranges (amorphous): {FREE_RANGES}")

    if args.frame is not None:
        fit_peaks(args.h5, args.frame, peak_positions, plot=True)
    elif args.map:
        tr = track_sequential(args.h5, peak_positions)
        plot_scatter_map(tr)
    else:
        print("Specify --frame N to fit a single frame, or --map for sequential tracking across all frames.")

if __name__ == "__main__":
    main()


