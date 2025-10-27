import argparse
import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
from lmfit.models import GaussianModel, LinearModel

# Optional progress bar
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

# --- Tunables ---
WINDOW = 0.30
MAX_JUMP = 0.08           # default max jump
MAX_JUMP_STRICT = 0.04    # stricter limit for noisy datasets
CENTER_TOL = 0.04
SEED_FRAMES = 30
MIN_POINTS = 5
SIGMA_MIN_MULT = 0.25
SIGMA_MAX_FRAC = 1.2
CONSEC_FAIL_EXPAND = 2
SMOOTH_WINDOW = 5         # temporal smoothing window (odd number)
OUTLIER_THRESHOLD = 3.0   # MAD multiplier for outlier detection

# --- Helpers ---
def robust_sigma(y):
    """Fast robust noise estimator."""
    med = np.median(y)
    return 1.4826 * np.median(np.abs(y - med)) + 1e-12

def sigma_to_fwhm(sigma):
    return 2.354820045 * sigma

def median_filter_1d(y, window=5):
    """Simple median filter for 1D array."""
    window = int(window)
    if window % 2 == 0:
        window += 1
    if window < 3 or len(y) < window:
        return y
    half = window // 2
    result = np.copy(y)
    for i in range(len(y)):
        start = max(0, i - half)
        end = min(len(y), i + half + 1)
        result[i] = np.nanmedian(y[start:end])
    return result

def build_window(x, yfull, center, width):
    """Extract window around center."""
    half = width / 2.0
    m = (x >= center - half) & (x <= center + half)
    xw, yw = x[m], yfull[m]
    mfin = np.isfinite(xw) & np.isfinite(yw)
    return xw[mfin], yw[mfin]

def fit_peak_single(xw, yw, seed_center, narrow=True):
    """
    Single Gaussian + linear fit with adaptive bounds.
    Returns (result, ok_flag).
    """
    # Quick baseline estimate
    try:
        bkg_slope, bkg_intercept = np.polyfit(xw, yw, 1)
    except Exception:
        bkg_slope, bkg_intercept = 0.0, np.median(yw)

    y_detr = yw - (bkg_slope * xw + bkg_intercept)
    noise = robust_sigma(y_detr)

    dx = np.mean(np.diff(xw)) if len(xw) > 1 else (xw[-1] - xw[0]) / len(xw)
    min_sigma = max(dx * SIGMA_MIN_MULT, 1e-6)
    max_sigma = max(SIGMA_MAX_FRAC * (xw[-1] - xw[0]), min_sigma * 2.0)

    # Initial guess at seed location
    peak_idx = np.abs(xw - seed_center).argmin()
    height0 = max(yw[peak_idx] - (bkg_slope * xw[peak_idx] + bkg_intercept), 0.5 * noise)
    sigma0 = np.clip((xw[-1] - xw[0]) / 7.0, min_sigma, max_sigma)
    amp0 = max(height0 * sigma0 * 2.5066, noise * sigma0 * 2.5066)

    # Build model
    model = LinearModel(prefix="bkg_") + GaussianModel(prefix="g_")
    params = model.make_params(
        bkg_slope=bkg_slope,
        bkg_intercept=bkg_intercept,
        g_center=xw[peak_idx],
        g_sigma=sigma0,
        g_amplitude=amp0,
    )

    # Adaptive center bounds
    if narrow:
        cmin = max(xw[0], seed_center - CENTER_TOL)
        cmax = min(xw[-1], seed_center + CENTER_TOL)
    else:
        cmin, cmax = xw[0], xw[-1]

    params["g_center"].set(min=cmin, max=cmax)
    params["g_sigma"].set(min=min_sigma, max=max_sigma)
    params["g_amplitude"].set(min=0.0)

    try:
        result = model.fit(yw, params, x=xw, nan_policy="omit")
        p = result.params
        sig = p["g_sigma"].value
        ctr = p["g_center"].value
        ok = np.isfinite(ctr) and np.isfinite(sig) and (min_sigma <= sig <= max_sigma)
        return result, ok
    except Exception:
        return None, False

def fit_single_peak(x, I, frame, center_guess, window=WINDOW):
    """
    Fit one frame. Returns dict with center, fwhm, amplitude, ok.
    """
    yfull = I[frame]
    xw, yw = build_window(x, yfull, center_guess, window)
    
    if len(xw) < MIN_POINTS:
        return {"center": center_guess, "fwhm": np.nan, "amplitude": np.nan, "ok": False}

    # Single seed from smoothed peak location
    if len(yw) >= 5:
        kernel = np.ones(5) / 5.0
        ysm = np.convolve(yw, kernel, mode='same')
    else:
        ysm = yw
    seed = xw[np.argmax(ysm)]

    # Try narrow bounds first
    result, ok = fit_peak_single(xw, yw, seed, narrow=True)
    
    # Fallback to wide bounds if needed
    if not ok:
        result, ok = fit_peak_single(xw, yw, seed, narrow=False)

    if result is None:
        return {"center": center_guess, "fwhm": np.nan, "amplitude": np.nan, "ok": False}

    p = result.params
    center_fit = p["g_center"].value
    sigma_fit = p["g_sigma"].value
    amplitude_fit = p["g_amplitude"].value
    fwhm_fit = sigma_to_fwhm(sigma_fit) if np.isfinite(sigma_fit) else np.nan

    return {"center": center_fit, "fwhm": fwhm_fit, "amplitude": amplitude_fit, "ok": ok}

def robust_initial_center(x, I, initial_guess, nframes=SEED_FRAMES, desc=None, show_progress=True):
    """Determine robust baseline center from early frames and assess variance."""
    navail = min(nframes, I.shape[0])
    centers = []

    iterator = range(navail)
    if show_progress and tqdm is not None:
        iterator = tqdm(iterator, desc=f"{desc}: seed", leave=False)

    for frame in iterator:
        res = fit_single_peak(x, I, frame, initial_guess)
        centers.append(res["center"])

    centers = np.array(centers)
    med = np.nanmedian(centers)
    mad = 1.4826 * np.nanmedian(np.abs(centers - med))
    good = np.abs(centers - med) <= (3 * mad if mad > 0 else np.inf)
    baseline = np.nanmedian(centers[good]) if np.any(good) else med
    
    # Return baseline and variance metric (MAD)
    return baseline, mad

def process_dataset(h5_path, initial_guess, desc=None, show_progress=True):
    """
    Track peak across all frames with adaptive constraints and smoothing.
    Returns diff_centers, failed_frames, nframes.
    """
    # Load data once
    with h5py.File(h5_path, "r") as f:
        x = f["q"][:] if "q" in f else f["tth"][:]
        I = f["int"][:]
    
    nframes = I.shape[0]
    
    # Get baseline center and assess dataset noise
    baseline_center, seed_mad = robust_initial_center(x, I, initial_guess, SEED_FRAMES, desc, show_progress)
    
    # Adaptive jump limit: tighten for noisy datasets
    # If seed MAD is high (>0.01), use stricter jump limit
    max_jump = MAX_JUMP_STRICT if seed_mad > 0.01 else MAX_JUMP
    
    if show_progress:
        print(f"{desc}: seed MAD={seed_mad:.5f}, using max_jump={max_jump:.4f}")

    centers = np.full(nframes, np.nan)
    failed_frames = []
    center_prev = baseline_center
    window = WINDOW
    consec_fail = 0

    # Track all frames
    iterator = range(nframes)
    if show_progress and tqdm is not None:
        iterator = tqdm(iterator, desc=f"{desc}: track", leave=False)

    for frame in iterator:
        res = fit_single_peak(x, I, frame, center_prev, window=window)
        c = res["center"]
        ok = res["ok"]

        # Jump control with adaptive limit
        if np.isfinite(c) and np.isfinite(center_prev):
            if abs(c - center_prev) > max_jump:
                # Retry with expanded window
                res2 = fit_single_peak(x, I, frame, center_prev, window=min(2 * window, 2 * WINDOW))
                c2 = res2["center"]
                if np.isfinite(c2) and abs(c2 - center_prev) <= 1.5 * max_jump:
                    c, ok = c2, res2["ok"]
                else:
                    # Reject large jump, keep previous center
                    c = center_prev
                    ok = False

        centers[frame] = c
        center_prev = c if np.isfinite(c) else center_prev
        consec_fail = 0 if ok else consec_fail + 1
        
        if not ok:
            failed_frames.append(frame)

        # Adaptive window
        window = min(2 * window, 2 * WINDOW) if consec_fail >= CONSEC_FAIL_EXPAND else WINDOW

    # Apply temporal smoothing to suppress spurious jumps
    centers_smooth = median_filter_1d(centers, window=SMOOTH_WINDOW)
    
    # Outlier detection and interpolation
    diff = centers_smooth - baseline_center
    diff_mad = 1.4826 * np.nanmedian(np.abs(diff - np.nanmedian(diff)))
    outliers = np.abs(diff - np.nanmedian(diff)) > OUTLIER_THRESHOLD * diff_mad
    
    if np.any(outliers):
        # Interpolate outliers from neighbors
        valid = ~outliers & np.isfinite(centers_smooth)
        if np.sum(valid) > 2:
            valid_idx = np.where(valid)[0]
            valid_vals = centers_smooth[valid]
            centers_smooth[outliers] = np.interp(
                np.where(outliers)[0], 
                valid_idx, 
                valid_vals
            )
            if show_progress:
                print(f"{desc}: interpolated {np.sum(outliers)} outlier frames")

    # Robust zeroing
    early = centers_smooth[:min(SEED_FRAMES, nframes)]
    baseline = np.nanmedian(early) if np.any(np.isfinite(early)) else baseline_center
    diff_centers = centers_smooth - baseline

    return diff_centers, failed_frames, nframes

def main():
    ap = argparse.ArgumentParser(description="Track peak movement for 7 datasets.")
    ap.add_argument("--h5", nargs=7, required=True, help="7 HDF5 files")
    ap.add_argument("--center", nargs=7, type=float, required=True, help="Initial peak centers")
    ap.add_argument("--no-progress", action="store_true", help="Disable progress bars")
    args = ap.parse_args()

    show_progress = not args.no_progress

    # Plot style
    plt.rcParams.update({
        "figure.figsize": (6.5, 4.8),
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "font.size": 12,
        "axes.labelsize": 14,
        "legend.fontsize": 12,
        "legend.frameon": False,
    })

    max_moves = []
    markers = ['o', 's', 'D', '^', 'v', 'p', 'X']

    dataset_iter = range(7)
    if show_progress and tqdm is not None:
        dataset_iter = tqdm(dataset_iter, desc="Datasets")

    for i in dataset_iter:
        desc = f"DS{i}"
        diff_centers, failed_frames, nframes = process_dataset(
            args.h5[i], args.center[i], desc, show_progress
        )
        
        frames = np.arange(nframes)
        plt.scatter(frames, diff_centers, label=f"Beam Index {i}", 
                   s=12, alpha=0.7, marker=markers[i])

        max_move = np.nanmax(np.abs(diff_centers))
        max_moves.append(max_move)

        if failed_frames and show_progress:
            print(f"Dataset {i}: {len(failed_frames)} low-confidence frames")

    plt.xlabel("Frame")
    plt.ylabel("Peak Center Differential q Movement (1/Å)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\nMax absolute peak movement per dataset:")
    for i, mm in enumerate(max_moves):
        print(f"  Beam Index {i}: {mm:.6g}")

if __name__ == "__main__":
    main()
