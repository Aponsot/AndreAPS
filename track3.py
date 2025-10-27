import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
from lmfit.models import GaussianModel, LinearModel

# --- Tunables ---
WINDOW = 0.30            # default fit window width (q or 2θ units)
MAX_JUMP = 0.03          # max allowed center change per frame (same units as x)
SEED_FRAMES = 30         # number of early frames to determine baseline center
MIN_POINTS = 8           # min points required in a window to fit
SEED_COUNT = 5           # number of seeds around the guess for multi-start
SEED_SPREAD = 0.4        # fraction of window half-width for seeding offsets
HEIGHT_SNR_MIN = 3.0     # minimal height-to-noise ratio to accept a fit
SIGMA_MIN_MULT = 0.5     # min sigma as multiple of mean dx
SIGMA_MAX_FRAC = 0.5     # max sigma as fraction of window width
CONSEC_FAIL_EXPAND = 2   # after this many consecutive failures, expand window once

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
    # choose smoothing window about ~5% of points, ensure odd and >=3
    w = max(3, int(len(yw) * 0.05) | 1)  # force odd with | 1
    ysm = moving_average(yw, w) if len(yw) >= 3 else yw
    seed_argmax = xw[np.argmax(ysm)]
    seeds = np.unique(np.r_[seeds, seed_argmax])
    return seeds

def fit_peak_in_window_gauss(xw, yw, seed_center):
    """
    Fit Gaussian + linear background in a given window.
    Returns (result, ok_flag).
    """
    # Initial linear baseline via least squares (for robust initial params)
    try:
        bkg_slope_init, bkg_intercept_init = np.polyfit(xw, yw, 1)
    except Exception:
        # Fallback if polyfit fails
        bkg_slope_init, bkg_intercept_init = 0.0, float(np.median(yw))

    # Detrend for noise estimation
    y_trend = bkg_slope_init * xw + bkg_intercept_init
    y_detr = yw - y_trend
    noise = robust_sigma(y_detr)

    dx = float(np.mean(np.diff(xw))) if len(xw) > 1 else (xw[-1] - xw[0]) / max(len(xw), 1)
    min_sigma = max(dx * SIGMA_MIN_MULT, 1e-6)
    max_sigma = max(SIGMA_MAX_FRAC * (xw[-1] - xw[0]), min_sigma * 2.0)

    peak_idx = np.abs(xw - seed_center).argmin()
    height0 = max(yw[peak_idx] - (bkg_slope_init * xw[peak_idx] + bkg_intercept_init), 3 * noise)
    sigma0 = fwhm_to_sigma((xw[-1] - xw[0]) / 3.0)  # width ~ window/3
    sigma0 = np.clip(sigma0, min_sigma, max_sigma)
    amp0 = max(height0 * sigma0 * np.sqrt(2 * np.pi), noise * sigma0 * np.sqrt(2 * np.pi))

    bkg = LinearModel(prefix="bkg_")
    gauss = GaussianModel(prefix="g_")
    model = bkg + gauss

    params = model.make_params(
        bkg_slope=bkg_slope_init,
        bkg_intercept=bkg_intercept_init,
        g_center=float(xw[peak_idx]),
        g_sigma=float(sigma0),
        g_amplitude=float(amp0),
    )
    params["g_center"].set(min=float(xw[0]), max=float(xw[-1]))
    params["g_sigma"].set(min=float(min_sigma), max=float(max_sigma))
    params["g_amplitude"].set(min=0.0)

    try:
        result = model.fit(yw, params, x=xw, nan_policy="omit")
    except Exception:
        return None, False

    # Validate fit: reasonable width and SNR
    p = result.params
    amp = p["g_amplitude"].value
    sig = p["g_sigma"].value
    ctr = p["g_center"].value

    height_fit = amp / (sig * np.sqrt(2 * np.pi)) if (sig is not None and sig > 0) else 0.0
    if (not np.isfinite(ctr)
        or not np.isfinite(sig)
        or sig < min_sigma
        or sig > max_sigma
        or height_fit < HEIGHT_SNR_MIN * noise):
        return result, False

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

def fallback_center(xw, yw):
    """Fallback center using smoothed argmax and centroid above median."""
    # Use a small moving average to denoise
    w = max(3, int(len(yw) * 0.05) | 1)
    ysm = moving_average(yw, w) if len(yw) >= 3 else yw
    y0 = ysm - np.median(ysm)
    y0[y0 < 0] = 0
    if y0.sum() <= 0:
        return xw[np.argmax(ysm)]
    return float(np.sum(xw * y0) / np.sum(y0))

def fit_single_peak_data(x, I, frame, center_guess, window=WINDOW):
    """
    Fit a single peak for one frame using Gaussian + linear background.
    Returns dict with center, fwhm, amplitude, ok flag.
    """
    yfull = np.asarray(I[frame], float)
    x = np.asarray(x, float)

    xw, yw = build_window(x, yfull, center_guess, window)
    if len(xw) < MIN_POINTS:
        return {"center": np.nan, "fwhm": np.nan, "amplitude": np.nan, "ok": False}

    seeds = initial_seeds(xw, yw, center_guess)
    best, ok = choose_best_fit(xw, yw, seeds)

    if best is None or not ok:
        # fallback: estimate center and re-try once
        ctr_fb = fallback_center(xw, yw)
        best, ok = fit_peak_in_window_gauss(xw, yw, ctr_fb)

    if best is None:
        return {"center": np.nan, "fwhm": np.nan, "amplitude": np.nan, "ok": False}

    p = best.params
    center_fit = float(p["g_center"].value)
    sigma_fit = float(p["g_sigma"].value)
    amplitude_fit = float(p["g_amplitude"].value)
    fwhm_fit = sigma_to_fwhm(sigma_fit)

    return {"center": center_fit, "fwhm": fwhm_fit, "amplitude": amplitude_fit, "ok": ok}

def robust_initial_center_data(x, I, initial_guess, nframes=SEED_FRAMES):
    """
    Determine robust baseline center using the first nframes with outlier rejection.
    """
    navail = min(nframes, I.shape[0])
    centers = []
    for frame in range(navail):
        res = fit_single_peak_data(x, I, frame, initial_guess)
        if res["ok"]:
            centers.append(res["center"])
    if not centers:
        return float(initial_guess)
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

        if ok and np.isfinite(c) and np.isfinite(center_prev):
            if abs(c - center_prev) > MAX_JUMP:
                # Try expanded window and re-fit once
                res2 = fit_single_peak_data(x, I, frame, center_prev, window=min(2 * window, 2 * WINDOW))
                c2 = res2["center"]
                if res2["ok"] and abs(c2 - center_prev) <= 1.5 * MAX_JUMP:
                    centers[frame] = c2
                    center_prev = c2
                    consec_fail = 0
                else:
                    centers[frame] = np.nan
                    failed_frames.append(frame)
                    consec_fail += 1
            else:
                centers[frame] = c
                center_prev = c
                consec_fail = 0
        else:
            centers[frame] = np.nan
            failed_frames.append(frame)
            consec_fail += 1

        # Adapt window after consecutive failures
        if consec_fail >= CONSEC_FAIL_EXPAND:
            window = min(2 * window, 2 * WINDOW)
        else:
            window = WINDOW

    # Robust zeroing against early successful frames
    early = centers[:min(SEED_FRAMES, nframes)]
    baseline = np.nanmedian(early) if np.any(np.isfinite(early)) else baseline_center
    diff_centers = centers - baseline

    return diff_centers, failed_frames, nframes

def main():
    ap = argparse.ArgumentParser(description="Track peak movement for 7 datasets (Gaussian + linear baseline).")
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

    for i in range(7):
        diff_centers, failed_frames, nframes = process_dataset(args.h5[i], args.center[i])
        frames = np.arange(nframes)
        plt.scatter(frames, diff_centers, label=f"Beam Index {i}", s=12, alpha=0.7, marker=markers[i % len(markers)])

        max_move = float(np.nanmax(np.abs(diff_centers)))
        max_moves.append(max_move)
        labels.append(f"Beam Index {i}")

        if failed_frames:
            print(f"Warning: Peak fitting failed for frames in dataset {i}: {failed_frames}")

    plt.xlabel("Frame")
    # If you prefer inverted y-range like before, use plt.ylim(-0.14, 0.0)
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
