import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
from lmfit.models import GaussianModel, LinearModel

# --- Tunables for leniency during solidification ---
WINDOW = 0.30             # default fit window width (q or 2θ units)
MAX_JUMP = 0.08           # larger allowed center change per frame (same units as x)
CENTER_TOL = 0.04         # narrow bounds around seed center for first fit attempt
SEED_FRAMES = 30          # number of early frames to determine baseline center
MIN_POINTS = 5            # fewer points required to fit (more lenient)
SEED_COUNT = 5            # multi-start seeds around the guess
SEED_SPREAD = 0.3         # fraction of window half-width for seeding offsets
HEIGHT_SNR_MIN = 0.5      # lower minimal height-to-noise threshold (more lenient)
SIGMA_MIN_MULT = 0.25     # min sigma as multiple of mean dx (allows narrow peaks)
SIGMA_MAX_FRAC = 1.2      # max sigma as fraction of window width (allows thick peaks)
CONSEC_FAIL_EXPAND = 2    # after this many consecutive poor fits, expand window once

# --- Helpers ---
def robust_sigma(y):
    y = np.asarray(y, float)
    med = np.median(y)
    return 1.4826 * np.median(np.abs(y - med)) + 1e-12

def fwhm_to_sigma(fwhm):
    return fwhm / 2.354820045

def sigma_to_fwhm(sigma):
    return 2.354820045 * sigma

def moving_average(y, win):
    """Simple moving average with odd window length."""
    win = int(win)
    if win < 1:
        return y
    if win % 2 == 0:
        win += 1
    if win <= 1:
        return y
    kernel = np.ones(win, dtype=float) / win
    return np.convolve(y, kernel, mode="same")

def build_window(x, yfull, center, width):
    """Return xw, yw within [center - width/2, center + width/2], finite points only."""
    half = width / 2.0
    m = (x >= center - half) & (x <= center + half)
    xw, yw = x[m], yfull[m]
    mfin = np.isfinite(xw) & np.isfinite(yw)
    return xw[mfin], yw[mfin]

def initial_seeds(xw, yw, center_guess, n=SEED_COUNT, spread=SEED_SPREAD):
    """Propose multiple seeds around center_guess and include smoothed argmax."""
    half = (xw[-1] - xw[0]) / 2.0
    offsets = np.linspace(-spread * half, spread * half, n)
    seeds = np.clip(center_guess + offsets, xw[0], xw[-1])

    # Add a smoothed argmax as an extra candidate
    w = max(3, int(len(yw) * 0.05) | 1)  # force odd with | 1
    ysm = moving_average(yw, w) if len(yw) >= 3 else yw
    seed_argmax = xw[np.argmax(ysm)]
    seeds = np.unique(np.r_[seeds, seed_argmax])
    return seeds

def _make_gauss_linear_model(xw, yw, seed_center, narrow_bounds=True):
    """
    Build and initialize Gaussian + linear background model.
    If narrow_bounds=True, constrain center tightly around seed_center.
    """
    # Initial linear baseline via least squares
    try:
        bkg_slope_init, bkg_intercept_init = np.polyfit(xw, yw, 1)
    except Exception:
        bkg_slope_init, bkg_intercept_init = 0.0, float(np.median(yw))

    # Detrend for noise estimation
    y_trend = bkg_slope_init * xw + bkg_intercept_init
    y_detr = yw - y_trend
    noise = robust_sigma(y_detr)

    dx = float(np.mean(np.diff(xw))) if len(xw) > 1 else (xw[-1] - xw[0]) / max(len(xw), 1)
    min_sigma = max(dx * SIGMA_MIN_MULT, 1e-6)
    max_sigma = max(SIGMA_MAX_FRAC * (xw[-1] - xw[0]), min_sigma * 2.0)

    peak_idx = np.abs(xw - seed_center).argmin()
    height0 = max(yw[peak_idx] - (bkg_slope_init * xw[peak_idx] + bkg_intercept_init), HEIGHT_SNR_MIN * noise)
    sigma0 = fwhm_to_sigma((xw[-1] - xw[0]) / 3.0)  # width ~ window/3
    sigma0 = np.clip(sigma0, min_sigma, max_sigma)
    amp0 = max(height0 * sigma0 * np.sqrt(2 * np.pi), noise * sigma0 * np.sqrt(2 * np.pi))

    bkg = LinearModel(prefix="bkg_")
    gauss = GaussianModel(prefix="g_")
    model = bkg + gauss

    params = model.make_params(
        bkg_slope=float(bkg_slope_init),
        bkg_intercept=float(bkg_intercept_init),
        g_center=float(xw[peak_idx]),
        g_sigma=float(sigma0),
        g_amplitude=float(amp0),
    )

    if narrow_bounds:
        # Constrain center tightly near the suggested location
        cmin = float(max(xw[0], seed_center - CENTER_TOL))
        cmax = float(min(xw[-1], seed_center + CENTER_TOL))
    else:
        # Wider bounds within the window
        cmin = float(xw[0])
        cmax = float(xw[-1])

    params["g_center"].set(min=cmin, max=cmax)
    params["g_sigma"].set(min=float(min_sigma), max=float(max_sigma))
    params["g_amplitude"].set(min=0.0)

    return model, params, noise, min_sigma, max_sigma

def fit_peak_in_window_gauss(xw, yw, seed_center):
    """
    Fit Gaussian + linear background in a given window.
    Two-stage: try narrow bounds around the suggested center; if poor, widen bounds.
    Returns (result, ok_flag).
    """
    # Stage 1: narrow bounds near seed
    model, params, noise, min_sigma, max_sigma = _make_gauss_linear_model(xw, yw, seed_center, narrow_bounds=True)
    try:
        result = model.fit(yw, params, x=xw, nan_policy="omit")
    except Exception:
        result = None

    ok = False
    if result is not None:
        p = result.params
        sig = p["g_sigma"].value
        ctr = p["g_center"].value
        # Lenient acceptance: only require finite center and sensible sigma bounds
        if np.isfinite(ctr) and np.isfinite(sig) and (min_sigma <= sig <= max_sigma):
            ok = True

    # Stage 2: widen bounds if Stage 1 was not OK
    if not ok:
        model2, params2, noise2, min_sigma2, max_sigma2 = _make_gauss_linear_model(xw, yw, seed_center, narrow_bounds=False)
        try:
            result2 = model2.fit(yw, params2, x=xw, nan_policy="omit")
        except Exception:
            return None, False

        p2 = result2.params
        sig2 = p2["g_sigma"].value
        ctr2 = p2["g_center"].value
        if np.isfinite(ctr2) and np.isfinite(sig2) and (min_sigma2 <= sig2 <= max_sigma2):
            return result2, True
        else:
            return result2, False

    return result, True

def choose_best_fit(xw, yw, seeds):
    """Try multiple seeds, choose the fit with smallest reduced chi-square."""
    best = None
    best_redchi = np.inf
    best_ok = False
    for sc in seeds:
        res, ok = fit_peak_in_window_gauss(xw, yw, sc)
        if res is None:
            continue
        redchi = getattr(res, "redchi", np.inf)
        if redchi < best_redchi:
            best_redchi = redchi
            best = res
            best_ok = ok
    return best, best_ok

def fallback_center(xw, yw, seed_center):
    """Fallback center: lean towards the suggested location if data are too weak."""
    # Use small smoothing, but bias to seed_center if the signal is very weak
    w = max(3, int(len(yw) * 0.05) | 1)
    ysm = moving_average(yw, w) if len(yw) >= 3 else yw
    y0 = ysm - np.median(ysm)
    y0[y0 < 0] = 0
    if y0.sum() <= 0:
        return float(seed_center)
    centroid = float(np.sum(xw * y0) / np.sum(y0))
    # Blend centroid with seed to keep it near suggested location
    alpha = 0.7  # weight towards seed_center when weak
    return float(alpha * seed_center + (1 - alpha) * centroid)

def fit_single_peak_data(x, I, frame, center_guess, window=WINDOW):
    """
    Fit a single peak for one frame using Gaussian + linear background.
    Always returns a center (with confidence flag).
    """
    yfull = np.asarray(I[frame], float)
    x = np.asarray(x, float)

    xw, yw = build_window(x, yfull, center_guess, window)
    if len(xw) < MIN_POINTS:
        # Too few points; return suggested center with low confidence
        return {"center": float(center_guess), "fwhm": np.nan, "amplitude": np.nan, "ok": False}

    seeds = initial_seeds(xw, yw, center_guess)
    best, ok = choose_best_fit(xw, yw, seeds)

    if best is None:
        # fallback: estimate center and re-try once
        ctr_fb = fallback_center(xw, yw, center_guess)
        best, ok = fit_peak_in_window_gauss(xw, yw, ctr_fb)

    if best is None:
        # ultimate fallback: report suggested center
        return {"center": float(center_guess), "fwhm": np.nan, "amplitude": np.nan, "ok": False}

    p = best.params
    center_fit = float(p["g_center"].value)
    sigma_fit = float(p["g_sigma"].value)
    amplitude_fit = float(p["g_amplitude"].value)
    fwhm_fit = sigma_to_fwhm(sigma_fit) if np.isfinite(sigma_fit) else np.nan

    return {"center": center_fit, "fwhm": fwhm_fit, "amplitude": amplitude_fit, "ok": ok}

def robust_initial_center_data(x, I, initial_guess, nframes=SEED_FRAMES):
    """
    Determine robust baseline center using the first nframes with outlier rejection.
    """
    navail = min(nframes, I.shape[0])
    centers = []
    for frame in range(navail):
        res = fit_single_peak_data(x, I, frame, initial_guess)
        centers.append(res["center"])  # always has a value
    centers = np.array(centers, float)
    med = np.nanmedian(centers)
    mad = 1.4826 * np.nanmedian(np.abs(centers - med))
    good = np.abs(centers - med) <= (3 * mad if mad > 0 else np.inf)
    return float(np.nanmedian(centers[good])) if np.any(good) else float(med)

def process_dataset(h5_path, initial_guess):
    """
    Track peak center across all frames with sequential re-centering and robust zeroing.
    Returns diff_centers, failed_frames, nframes.
    """
    with h5py.File(h5_path, "r") as f:
        x = f["q"][:] if "q" in f else f["tth"][:]
        I = f["int"][:]
    nframes = I.shape[0]

    baseline_center = robust_initial_center_data(x, I, initial_guess, SEED_FRAMES)

    centers = np.full(nframes, np.nan, dtype=float)
    failed_frames = []
    center_prev = baseline_center
    window = WINDOW
    consec_fail = 0

    for frame in range(nframes):
        res = fit_single_peak_data(x, I, frame, center_prev, window=window)
        c = res["center"]
        ok = res["ok"]

        # Lenient jump control: accept larger shifts, but keep tracking
        if np.isfinite(c) and np.isfinite(center_prev):
            if abs(c - center_prev) > MAX_JUMP:
                # one-time wider bounds window refit around previous center
                res2 = fit_single_peak_data(x, I, frame, center_prev, window=min(2 * window, 2 * WINDOW))
                c2 = res2["center"]
                if np.isfinite(c2) and abs(c2 - center_prev) <= 1.5 * MAX_JUMP:
                    c = c2
                    ok = res2["ok"]

        centers[frame] = c
        center_prev = c if np.isfinite(c) else center_prev
        consec_fail = 0 if ok else consec_fail + 1
        if not ok:
            failed_frames.append(frame)

        # Adapt window after consecutive lower-confidence fits
        if consec_fail >= CONSEC_FAIL_EXPAND:
            window = min(2 * window, 2 * WINDOW)
        else:
            window = WINDOW

    # Robust zeroing against early frames
    early = centers[:min(SEED_FRAMES, nframes)]
    baseline = np.nanmedian(early) if np.any(np.isfinite(early)) else baseline_center
    diff_centers = centers - baseline

    return diff_centers, failed_frames, nframes

def main():
    ap = argparse.ArgumentParser(description="Track peak movement for 7 datasets (Gaussian + linear baseline, lenient).")
    ap.add_argument("--h5", nargs=7, required=True, help="7 HDF5 files with 'q' (or 'tth') and 'int'")
    ap.add_argument("--center", nargs=7, type=float, required=True, help="Initial guess for peak center for each dataset")
    args = ap.parse_args()

    # Plot style
    plt.rcParams.update({
        "figure.figsize": (6.5, 4.8),
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "font.size": 12,
        "axes.titlesize": 12,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.linewidth": 1.0,
        "axes.grid": False,
        "legend.frameon": False,
        "legend.fontsize": 12,
    })

    max_moves = []
    labels = []
    markers = ['o', 's', 'D', '^', 'v', 'p', 'X']
    
    dataset_iter = range(7)
    if use_tqdm and tqdm is not None:
        dataset_iter = tqdm(dataset_iter, desc="Datasets", leave=True)

    for i in range(7):
        diff_centers, failed_frames, nframes = process_dataset(args.h5[i], args.center[i])
        frames = np.arange(nframes)
        plt.scatter(frames, diff_centers, label=f"Beam Index {i}", s=12, alpha=0.7, marker=markers[i % len(markers)])

        max_move = float(np.nanmax(np.abs(diff_centers)))
        max_moves.append(max_move)
        labels.append(f"Beam Index {i}")

        if failed_frames:
            print(f"Note: Lower-confidence fits in dataset {i} at frames: {failed_frames}")

    plt.xlabel("Frame")
    plt.ylabel("Peak Center Differential q Movement (1/Å)")
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

    print("\nMax absolute peak movement per dataset:")
    for lab, mm in zip(labels, max_moves):
        print(f"  {lab}: {mm:.6g}")

if __name__ == "__main__":
    main()
