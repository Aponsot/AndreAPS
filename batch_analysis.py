import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from lmfit.models import GaussianModel, LinearModel
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Window (in q units) used around the mean of specified peak positions
WINDOW = 0.15

# Peak width constraints
MIN_SIGMA = 1e-6
MAX_SIGMA_FRAC = 0.22

# Seeding constraints: allowed per-frame center movement (q units)
PER_FRAME_SHIFT_DEFAULT = 0.015

# Quality threshold for accepting seeded result
R2_MIN = 0.6  # minimum acceptable R² to adopt a seeded fit

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
        sigma0 = np.clip(dx, min_sigma, max_sigma)  # use local sampling for initial sigma
        amp0 = height0 * sigma0 * np.sqrt(2 * np.pi)

        params.update(gi.make_params(center=cx, sigma=sigma0, amplitude=amp0))

        # Bounds for center: use provided bounds or default to window edges
        if center_bounds is None or center_bounds[i] is None:
            cmin = xw[0]
            cmax = xw[-1]
        else:
            cmin, cmax = center_bounds[i]
            # Clip to data window to prevent out-of-range issues
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
    return hits >= max(1, n // 2)  # flag if half or more touch bounds

def fit_peaks(h5_path, frame, peak_positions, plot=True,
              prev_solution=None, per_frame_shift=None, free_center_bounds=False):
    """
    Fit the specified peaks with:
      - Seeded per-frame bounds: center in [prev_center ± per_frame_shift], if available.
      - Or free bounds within the data window when free_center_bounds=True (e.g., amorphous zones).
    Fallback: if seeded fit looks poor or is stuck on bounds, refit with free bounds.

    free_center_bounds=True allows centers to move freely within the window (no per-frame constraints).
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
            if f"g{i}_center" in prev_solution.params:
                centers_seed.append(prev_solution.params[f"g{i}_center"].value)
            else:
                centers_seed.append(peak_positions[i])
            if f"g{i}_sigma" in prev_solution.params:
                sigmas_seed.append(prev_solution.params[f"g{i}_sigma"].value)
            else:
                sigmas_seed.append(None)
            if f"g{i}_amplitude" in prev_solution.params:
                amps_seed.append(prev_solution.params[f"g{i}_amplitude"].value)
            else:
                amps_seed.append(None)
        init_from = {"center": centers_seed, "sigma": sigmas_seed, "amplitude": amps_seed}

    # Build bounds
    dx = np.mean(np.diff(xw)) if len(xw) > 1 else WINDOW
    min_sigma = max(dx / 3.0, MIN_SIGMA)
    max_sigma = MAX_SIGMA_FRAC * WINDOW
    sigma_bounds = [(min_sigma, max_sigma) for _ in peak_positions]

    # Center bounds for seeded attempt
    center_bounds_seed = []
    for i, cx in enumerate(peak_positions):
        if free_center_bounds or prev_solution is None or per_frame_shift is None or init_from is None:
            cmin, cmax = xw[0], xw[-1]  # free within window
        else:
            prev_c = init_from["center"][i]
            cmin = prev_c - per_frame_shift
            cmax = prev_c + per_frame_shift
            # Clip to window
            cmin = max(xw[0], cmin)
            cmax = min(xw[-1], cmax)
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
    use_fallback = (r2_seed < R2_MIN) or (not free_center_bounds and stuck_seed)
    if use_fallback:
        center_bounds_free = [(xw[0], xw[-1]) for _ in peak_positions]
        model_free, params_free = build_model(
            xw, yw, peak_positions, baseline,
            center_bounds=center_bounds_free,
            init_from=init_from  # still use initial values
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

    bkg_slope = result.params["bkg_slope"].value
    bkg_intercept = result.params["bkg_intercept"].value

    rows = [[p["index"], p["center"], p["height"], p["fwhm"], p["amplitude"]]
            for p in peaks]

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

        for i in range(len(peaks)):
            key = f"g{i}_"
            if key in comps:
                ax.plot(xw, comps[key], ls=":", alpha=0.7, label=f"Peak {i+1}")
            ax.axvline(result.params[f"g{i}_center"].value, alpha=0.25, ls="--")

        ax.set_xlabel("q (1/Å)")
        ax.set_ylabel("Intensity")
        ax.set_title(f"Frame {frame} | {len(peaks)} peaks | R²={r2:.4f}")
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
        "window": (center - half, center + half),
        "background": {"slope": bkg_slope, "intercept": bkg_intercept},
        "r2": r2,
        "rows": rows,
        "result": result,
        "x": xw, "y": yw, "yfit": result.best_fit,
        "noise": noise,
        "n_peaks": len(peaks),
        "bounds_used": center_bounds_used
    }

def _fit_one(args):
    h5_path, frame, peak_positions = args
    try:
        out = fit_peaks(h5_path, frame, peak_positions, plot=False)
        peaks = [(r[1], r[2]) for r in out.get("rows", [])]
        return frame, peaks
    except Exception:
        return frame, []

def peak_map_parallel(h5_path, peak_positions, marker_size=10):
    """
    Parallel map (no seeding): scatter of centers vs frame colored by height.
    """
    with h5py.File(h5_path, "r") as f:
        nframes = f["int"].shape[0]

    xs, ys, cs = [], [], []

    with ProcessPoolExecutor() as ex:
        futures = [ex.submit(_fit_one, (h5_path, fr, peak_positions))
                   for fr in range(nframes)]
        for fut in tqdm(as_completed(futures), total=nframes,
                        desc="Building peak map", unit="frame"):
            frame, peaks = fut.result()
            for q, height in peaks:
                xs.append(q)
                ys.append(frame)
                cs.append(height)

    if not xs:
        print("No peaks found.")
        return

    plt.rcParams.update({
        "figure.dpi": 160, "savefig.dpi": 300,
        "font.size": 20, "axes.labelsize": 20, "axes.titlesize": 20,
        "xtick.labelsize": 16, "ytick.labelsize": 16,
    })

    plt.figure(figsize=(9, 5))
    sc = plt.scatter(ys, xs, c=cs, s=marker_size, cmap="plasma")
    norm = Normalize(vmin=0.0, vmax=70, clip=True)
    cbar = plt.colorbar(sc, norm=norm)
    cbar.set_label("Peak height (a.u.)")
    plt.ylabel("q (1/Å)")
    plt.xlabel("Frame")
    plt.tight_layout()
    plt.show()

def parse_ranges(rng_str):
    """
    Parse ranges like "100:200,400:420" into a list of (start,end) inclusive tuples.
    """
    if not rng_str:
        return []
    out = []
    for part in rng_str.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            a, b = part.split(":")
            out.append((int(a), int(b)))
        else:
            # single frame
            f = int(part)
            out.append((f, f))
    return out

def in_any_range(frame, ranges):
    for a, b in ranges:
        if a <= frame <= b:
            return True
    return False

def track_sequential(h5_path, peak_positions,
                     start_frame=0, end_frame=None,
                     per_frame_shift=PER_FRAME_SHIFT_DEFAULT,
                     free_ranges=None,
                     plot_summary=True, show_example_fit=True):
    """
    Sequentially fit frames with seeding from previous frame and small per-frame shift bounds.
    In frames within free_ranges (amorphous zones), per-frame constraints are disabled
    and centers can move freely within the window.

    Produces summary plots: centers vs frame and heights vs frame.
    """
    free_ranges = free_ranges or []

    with h5py.File(h5_path, "r") as f:
        x = f["q"][:] if "q" in f else f["tth"][:]
        int_ds = f["int"]
        nframes_total = int_ds.shape[0]

        if end_frame is None:
            end_frame = nframes_total - 1
        start_frame = max(0, int(start_frame))
        end_frame = min(nframes_total - 1, int(end_frame))
        frames = range(start_frame, end_frame + 1)

        centers_tr = [[] for _ in peak_positions]
        heights_tr = [[] for _ in peak_positions]
        r2_tr = []

        prev_solution = None
        last_good = None

        # Optionally show a detailed fit for the first frame
        first = True

        for fr in tqdm(frames, desc="Sequential tracking", unit="frame"):
            plot_this = show_example_fit and first
            try:
                free_here = in_any_range(fr, free_ranges)
                out = fit_peaks(
                    h5_path, fr, peak_positions, plot=plot_this,
                    prev_solution=(prev_solution or last_good),
                    per_frame_shift=per_frame_shift,
                    free_center_bounds=free_here
                )
                result = out["result"]
                r2 = out["r2"]
                r2_tr.append(r2)

                # Extract centers/heights in a stable order
                for i in range(len(peak_positions)):
                    ci = result.params[f"g{i}_center"].value
                    ai = result.params[f"g{i}_amplitude"].value
                    si = result.params[f"g{i}_sigma"].value
                    hi = ai / (si * np.sqrt(2 * np.pi)) if si > 0 else 0.0
                    centers_tr[i].append(ci)
                    heights_tr[i].append(hi)

                # Update seeds: adopt current only if fit looks reasonable and not in free zone
                if (r2 >= R2_MIN) and not free_here:
                    prev_solution = result
                    last_good = result
                elif r2 >= R2_MIN and free_here:
                    # Allow seed update even in free zone if fit is good
                    prev_solution = result
                    last_good = result
                else:
                    # Keep last_good to avoid propagating a bad frame
                    prev_solution = last_good

            except Exception:
                r2_tr.append(np.nan)
                for i in range(len(peak_positions)):
                    centers_tr[i].append(np.nan)
                    heights_tr[i].append(np.nan)
                prev_solution = last_good

            first = False

    if plot_summary:
        # Summary plots: centers vs frame and heights vs frame
        plt.rcParams.update({
            "figure.dpi": 160, "savefig.dpi": 300,
            "font.size": 16, "axes.labelsize": 18, "axes.titlesize": 20,
            "xtick.labelsize": 14, "ytick.labelsize": 14,
        })
        fig, axs = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
        # Centers
        for i, pos in enumerate(peak_positions):
            axs[0].plot(list(frames), centers_tr[i], lw=2, label=f"Peak {i+1} (init {pos:.4f})")
        axs[0].set_ylabel("Center q (1/Å)")
        axs[0].grid(alpha=0.3)
        axs[0].legend(loc="best")
        # Heights
        for i, pos in enumerate(peak_positions):
            axs[1].plot(list(frames), heights_tr[i], lw=2, label=f"Peak {i+1}")
        axs[1].set_ylabel("Height (a.u.)")
        axs[1].set_xlabel("Frame")
        axs[1].grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    # Package results
    results = {
        "frames": list(frames),
        "centers": centers_tr,
        "heights": heights_tr,
        "r2": r2_tr,
        "peak_positions": peak_positions
    }
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Fit Gaussian peaks with linear background at specified q-positions"
    )
    parser.add_argument("h5", help="HDF5 file with 'q' and 'int' datasets")
    parser.add_argument("peaks", type=float, nargs='+',
                        help="Peak q-positions (e.g., 3.025 3.012)")
    parser.add_argument("--frame", type=int, help="Fit single frame and show plot")
    parser.add_argument("--map", action="store_true",
                        help="Generate peak map for all frames (no seeding)")
    parser.add_argument("--track", action="store_true",
                        help="Sequential tracking with seeding and small per-frame shifts")
    parser.add_argument("--start", type=int, default=0, help="Start frame for tracking")
    parser.add_argument("--end", type=int, help="End frame for tracking (inclusive)")
    parser.add_argument("--per-shift", type=float, default=PER_FRAME_SHIFT_DEFAULT,
                        help="Allowed per-frame center shift (q units), default=0.015")
    parser.add_argument("--free-ranges", type=str,
                        help="Comma-separated frame ranges with free movement (amorphous zones), e.g. '400:520,780:820'")
    parser.add_argument("--no-summary-plot", action="store_true",
                        help="Disable summary plots for tracking")
    parser.add_argument("--no-example-fit", action="store_true",
                        help="Do not show the detailed fit for the first frame in tracking")

    args = parser.parse_args()

    peak_positions = sorted(args.peaks)
    print(f"Fitting {len(peak_positions)} peak(s) at q = {peak_positions}")
    print("Center bounds: seeded ± per-frame shift; free within window in specified amorphous ranges.")

    if args.frame is not None:
        fit_peaks(args.h5, args.frame, peak_positions, plot=True)
    elif args.map:
        peak_map_parallel(args.h5, peak_positions)
    elif args.track:
        free_ranges = parse_ranges(args.free_ranges) if args.free_ranges else []
        track_sequential(args.h5, peak_positions,
                         start_frame=args.start,
                         end_frame=args.end,
                         per_frame_shift=args.per_shift,
                         free_ranges=free_ranges,
                         plot_summary=not args.no_summary_plot,
                         show_example_fit=not args.no_example_fit)
    else:
        print("Specify --frame N to fit a single frame, --map for all frames (no seeding), "
              "or --track for sequential seeded tracking")

if __name__ == "__main__":
    main()
