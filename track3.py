import argparse
import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
from lmfit.models import GaussianModel, LinearModel
from math import pi, sqrt
# Optional progress bar
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

# --- Tunables ---
WINDOW = 0.50
MAX_JUMP = 0.1           # default max jump
MAX_JUMP_STRICT = 0.1   # stricter limit for noisy datasets
FRAME_SKIP_JUMP = 0.22     # jump threshold to skip entire frame
MAX_TOTAL_MOVEMENT = 0.4  # hard cap on total movement from baseline
OUTLIER_SIGMA = 20     # sigma threshold for outlier removal
CENTER_TOL = 0.04
SEED_FRAMES = 30
MIN_POINTS = 5
SIGMA_MIN_MULT = 0.25
SIGMA_MAX_FRAC = 1.2
CONSEC_FAIL_EXPAND = 2
MAX_FRAMES = 200
h = 2 
k = 0 
l = 0
# thermal expansion polynomial  (Δa/a0 = c0 + c1 T + c2 T^2 + c3 T^3)
C0 = -0.358
C1 = 9.472e-3
C2 = 1.031e-6
C3 = -2.978e-10       # all units consistent with T in kelvin
# --- Helpers ---
def T_from_delta(delta):
    """
    Solve C3*T^3 + C2*T^2 + C1*T + (C0 - delta) = 0  for T (K).
    Returns the real root in the range 0–4000 K or np.nan if none.
    """
    import numpy as np
    coeffs = [C3, C2, C1, C0 - delta]      # highest power first -> np.roots
    roots = np.roots(coeffs)
    # keep real roots within a reasonable range
    real = roots[np.isreal(roots)].real
    good = real[(real > 0) & (real < 4000)]
    return good[0] if good.size else np.nan
def robust_sigma(y):
    """Fast robust noise estimator."""
    med = np.median(y)
    return 1.4826 * np.median(np.abs(y - med)) + 1e-12

def sigma_to_fwhm(sigma):
    return 2.354820045 * sigma

def build_window(x, yfull, center, width):
    """Extract window around center."""
    half = width / 2.0
    m = (x >= center - half) & (x <= center + half)
    xw, yw = x[m], yfull[m]
    mfin = np.isfinite(xw) & np.isfinite(yw)
    return xw[mfin], yw[mfin]

def filter_outliers(diff_centers, threshold_sigma=3.0, desc=None, show_progress=True):
    """
    Remove outliers using robust statistics (MAD-based).
    
    Parameters:
    -----------
    diff_centers : array
        Differential center positions
    threshold_sigma : float
        Number of sigma for outlier threshold (default: 3.0)
    desc : str
        Dataset description for logging
    show_progress : bool
        Whether to print removal info
    
    Returns:
    --------
    diff_centers_filtered : array
        Data with outliers set to NaN
    n_outliers : int
        Number of outliers removed
    """
    valid = np.isfinite(diff_centers)
    if not np.any(valid):
        return diff_centers, 0
    
    # Robust statistics using MAD
    med = np.nanmedian(diff_centers)
    mad = 1.4826 * np.nanmedian(np.abs(diff_centers - med))
    
    # Identify outliers
    outlier_mask = np.abs(diff_centers - med) > (threshold_sigma * mad)
    diff_centers_filtered = diff_centers.copy()
    diff_centers_filtered[outlier_mask] = np.nan
    
    n_outliers = np.sum(outlier_mask)
    if show_progress and n_outliers > 0:
        outlier_frames = np.where(outlier_mask)[0]
        print(f"{desc}: Removed {n_outliers} outlier frames (>{threshold_sigma}σ from median)")
        if n_outliers <= 5:
            print(f"  Outlier frames: {outlier_frames.tolist()}")
    
    return diff_centers_filtered, n_outliers

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
    a_0 = 2*pi / baseline * (sqrt(h**2 +k**2 + l**2))
    # Return baseline and variance metric (MAD)
    return baseline, mad, a_0

def process_dataset(h5_path, initial_guess, desc=None, show_progress=True, 
                   skip_jump_threshold=FRAME_SKIP_JUMP, max_total_movement=MAX_TOTAL_MOVEMENT,
                   outlier_sigma=OUTLIER_SIGMA, nframes_limit=MAX_FRAMES):
    """
    Track peak across up to nframes_limit frames with adaptive constraints.
    Returns diff_centers, fwhms, failed_frames, skipped_frames, nframes.
    """
    # Load data once
    with h5py.File(h5_path, "r") as f:
        x = f["q"][:] if "q" in f else f["tth"][:]
        I_full = f["int"][:]

    # Limit to first nframes_limit frames
    nframes = min(I_full.shape[0], nframes_limit)
    I = I_full[:nframes]

    # Get baseline center and assess dataset noise
    baseline_center, seed_mad , a_0 = robust_initial_center(x, I, initial_guess, SEED_FRAMES, desc, show_progress)
    
    # Adaptive jump limit: tighten for noisy datasets
    max_jump = MAX_JUMP_STRICT if seed_mad > 0.01 else MAX_JUMP
    
    if show_progress:
        print(f"{desc}: seed MAD={seed_mad:.5f}, using max_jump={max_jump:.4f} (processing {nframes} frames)")

    centers = np.full(nframes, np.nan)
    fwhms = np.full(nframes, np.nan)  # Track FWHM
    failed_frames = []
    skipped_frames = []
    center_prev = baseline_center
    window = WINDOW
    consec_fail = 0

    # Track limited frames
    iterator = range(nframes)
    if show_progress and tqdm is not None:
        iterator = tqdm(iterator, desc=f"{desc}: track", leave=False)

    for frame in iterator:
        res = fit_single_peak(x, I, frame, center_prev, window=window)
        c = res["center"]
        fwhm = res["fwhm"]  # Extract FWHM
        ok = res["ok"]

        # Check total movement from baseline (hard cap)
        if np.isfinite(c):
            total_movement = abs(c - baseline_center)
            if total_movement > max_total_movement:
                skipped_frames.append(frame)
                centers[frame] = np.nan
                fwhms[frame] = np.nan
                if show_progress and len(skipped_frames) <= 10:
                    print(f"{desc}: Skipping frame {frame}, total movement={total_movement:.5f} > {max_total_movement:.5f}")
                continue

        # Jump control with adaptive limit
        if np.isfinite(c) and np.isfinite(center_prev):
            jump = abs(c - center_prev)
            
            # Check if jump exceeds skip threshold
            if jump > skip_jump_threshold:
                skipped_frames.append(frame)
                centers[frame] = np.nan
                fwhms[frame] = np.nan
                if show_progress and len(skipped_frames) <= 10:
                    print(f"{desc}: Skipping frame {frame}, jump={jump:.5f} > {skip_jump_threshold:.5f}")
                continue
            
            # Existing jump handling for smaller jumps
            if jump > max_jump:
                # Retry with expanded window
                res2 = fit_single_peak(x, I, frame, center_prev, window=min(2 * window, 2 * WINDOW))
                c2 = res2["center"]
                jump2 = abs(c2 - center_prev) if np.isfinite(c2) else np.inf
                
                # Accept expanded window fit if it's better
                if jump2 <= 1.5 * max_jump:
                    c, fwhm, ok = c2, res2["fwhm"], res2["ok"]
                # Otherwise only reject if jump is REALLY extreme (>3x limit)
                elif jump > 3 * max_jump:
                    c = center_prev
                    ok = False

        centers[frame] = c
        fwhms[frame] = fwhm  # Store FWHM
        center_prev = c if np.isfinite(c) else center_prev
        consec_fail = 0 if ok else consec_fail + 1
        
        if not ok:
            failed_frames.append(frame)

        # Adaptive window
        window = min(2 * window, 2 * WINDOW) if consec_fail >= CONSEC_FAIL_EXPAND else WINDOW

    # Robust zeroing (no smoothing)
    early = centers[:min(SEED_FRAMES, nframes)]
    baseline = np.nanmedian(early) if np.any(np.isfinite(early)) else baseline_center
    diff_centers = centers - baseline

    # Apply outlier filter to centers
    diff_centers, n_outliers = filter_outliers(diff_centers, threshold_sigma=outlier_sigma, 
                                               desc=desc, show_progress=show_progress)
    
    # Apply same outlier mask to FWHM
    outlier_mask = ~np.isfinite(diff_centers) & np.isfinite(fwhms)
    fwhms[outlier_mask] = np.nan

    return diff_centers, fwhms, failed_frames, skipped_frames, nframes, a_0

def main():
    ap = argparse.ArgumentParser(description="Track peak movement for 7 datasets.")
    ap.add_argument("--h5", nargs=7, required=True, help="7 HDF5 files")
    ap.add_argument("--center", nargs=7, type=float, required=True, help="Initial peak centers")
    ap.add_argument("--no-progress", action="store_true", help="Disable progress bars")
    ap.add_argument("--skip-jump", type=float, default=FRAME_SKIP_JUMP, 
                    help=f"Skip frames with jumps > this value (default: {FRAME_SKIP_JUMP})")
    ap.add_argument("--max-movement", type=float, default=MAX_TOTAL_MOVEMENT,
                    help=f"Maximum total movement from baseline (default: {MAX_TOTAL_MOVEMENT})")
    ap.add_argument("--outlier-sigma", type=float, default=OUTLIER_SIGMA,
                    help=f"Sigma threshold for outlier removal (default: {OUTLIER_SIGMA})")
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
    
    # Create two figures
    fig1, ax1 = plt.subplots(figsize=(6.5, 4.8))
    fig2, ax2 = plt.subplots(figsize=(6.5, 4.8))
    fig3, ax3 = plt.subplots(figsize=(6.5, 4.8))
    fig4, ax4 = plt.subplots(figsize=(6.5, 4.8))

    dataset_iter = range(7)
    if show_progress and tqdm is not None:
        dataset_iter = tqdm(dataset_iter, desc="Datasets")

    for i in dataset_iter:
        desc = f"DS{i}"
        diff_centers, fwhms, failed_frames, skipped_frames, nframes, a_0 = process_dataset(
            args.h5[i], args.center[i], desc, show_progress, args.skip_jump,
            args.max_movement, args.outlier_sigma
        )
        
        frames = np.arange(nframes)
        
        # Plot 1: Peak position differential
        valid_mask = np.isfinite(diff_centers)
        ax1.scatter(frames[valid_mask], diff_centers[valid_mask], 
                   label=f"Beam Index {i}", s=12, alpha=0.7, marker=markers[i])
        a = 2*pi / diff_centers[valid_mask] * (sqrt(h**2 +k**+ l**2))# Example calculation (replace h,k,l as needed)
        
        # Δa/a0 for this dataset
        delta_a_over_a0 = (a - a_0)/ a_0           

        T_frame = np.vectorize(T_from_delta)(delta_a_over_a0)

# third panel you already made: Δa/a0
# ---- NEW fourth panel: Temperature vs frame ----  
        valid_T = np.isfinite(T_frame)
        ax4.scatter(frames[valid_T], T_frame[valid_T],
            s=12, alpha=0.7, marker=markers[i],
            label=f"Beam {i}")
        # Plot 2: FWHM
        valid_fwhm = np.isfinite(fwhms)
        ax2.scatter(frames[valid_fwhm], fwhms[valid_fwhm],
                   label=f"Beam Index {i}", s=12, alpha=0.7, marker=markers[i])
        ax3.scatter(frames[valid_mask], a/a_0, label=f"Beam Index {i} (a)", s=12, alpha=0.7, marker=markers[i])

        max_move = np.nanmax(np.abs(diff_centers))
        max_moves.append(max_move)

        if show_progress:
            if failed_frames:
                print(f"Dataset {i}: {len(failed_frames)} low-confidence frames")
            if skipped_frames:
                print(f"Dataset {i}: {len(skipped_frames)} frames skipped due to jumps > {args.skip_jump}")

    # Configure Plot 1: Position
    ax1.set_xlim(0, 200)
    ax1.set_ylim(-0.2, 0.025)
    ax1.set_xlabel("Frame")
    ax1.set_ylabel("Peak Center Differential q Movement (1/Å)")
    ax1.grid(True)
    ax1.legend()
    fig1.tight_layout()
    
    # Configure Plot 2: FWHM
    ax2.set_xlim(0, 200)
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("FWHM (1/Å)")
    ax2.grid(True)
    ax2.legend()
    fig2.tight_layout()

    ax3.set_xlim(0, 200)
    ax3.set_xlabel("Frame")
    ax3.set_ylabel("lattice parameter a (Å) shift")
    ax3.grid(True)
    ax3.legend()
    fig3.tight_layout()
    
    ax4.set_xlim(0, 200)
    ax4.set_xlabel("Frame")
    ax4.set_ylabel("Temperature (K)")
    ax4.grid(True)
    ax4.legend()
    fig4.tight_layout()

    
    plt.show()

    print("\nMax absolute peak movement per dataset:")
    for i, mm in enumerate(max_moves):
        print(f"  Beam Index {i}: {mm:.6g}")

if __name__ == "__main__":
    main()
