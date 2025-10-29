import argparse
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
MAX_JUMP = 0.10           # default max jump (in q units)
MAX_JUMP_STRICT = 0.10    # stricter limit for noisy datasets
FRAME_SKIP_JUMP = 0.22    # jump threshold to skip entire frame
MAX_TOTAL_MOVEMENT = 0.40 # hard cap on total movement from baseline
OUTLIER_SIGMA = 20        # MAD-based outlier removal threshold (very permissive)
CENTER_TOL = 0.04
SEED_FRAMES = 30
MIN_POINTS = 5
SIGMA_MIN_MULT = 0.25
SIGMA_MAX_FRAC = 1.2
CONSEC_FAIL_EXPAND = 2
MAX_FRAMES = 200

# Reflection indices (confirm this matches your peak!)
h, k, l = 2, 0, 0

# thermal expansion polynomial (Δa/a0 = C0 + C1 T + C2 T^2 + C3 T^3), FRACTIONAL units
C0 = -0.358 / 100.0
C1 = 9.472e-3 / 100.0
C2 = 1.031e-6 / 100.0
C3 = -2.978e-10 / 100.0

# --- Helpers ---
def robust_sigma(y):
    med = np.median(y)
    return 1.4826 * np.median(np.abs(y - med)) + 1e-12

def sigma_to_fwhm(sigma):
    return 2.354820045 * sigma

def build_window(x, yfull, center, width):
    half = width / 2.0
    m = (x >= center - half) & (x <= center + half)
    xw, yw = x[m], yfull[m]
    mfin = np.isfinite(xw) & np.isfinite(yw)
    return xw[mfin], yw[mfin]

def fit_peak_single(xw, yw, seed_center, narrow=True):
    """
    Single Gaussian + linear baseline fit with adaptive bounds.
    Returns (result, ok_flag).
    """
    try:
        bkg_slope, bkg_intercept = np.polyfit(xw, yw, 1)
    except Exception:
        bkg_slope, bkg_intercept = 0.0, np.median(yw)

    y_detr = yw - (bkg_slope * xw + bkg_intercept)
    noise = robust_sigma(y_detr)

    dx = np.mean(np.diff(xw)) if len(xw) > 1 else (xw[-1] - xw[0]) / max(len(xw), 1)
    min_sigma = max(dx * SIGMA_MIN_MULT, 1e-6)
    max_sigma = max(SIGMA_MAX_FRAC * (xw[-1] - xw[0]), min_sigma * 2.0)

    peak_idx = np.abs(xw - seed_center).argmin()
    height0 = max(yw[peak_idx] - (bkg_slope * xw[peak_idx] + bkg_intercept), 0.5 * noise)
    sigma0 = np.clip((xw[-1] - xw[0]) / 7.0, min_sigma, max_sigma)
    amp0 = max(height0 * sigma0 * 2.5066, noise * sigma0 * 2.5066)

    model = LinearModel(prefix="bkg_") + GaussianModel(prefix="g_")
    params = model.make_params(
        bkg_slope=bkg_slope,
        bkg_intercept=bkg_intercept,
        g_center=xw[peak_idx],
        g_sigma=sigma0,
        g_amplitude=amp0,
    )

    # Center bounds
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
    """Fit one frame. Returns dict with center (absolute x), fwhm, amplitude, ok."""
    yfull = I[frame]
    xw, yw = build_window(x, yfull, center_guess, window)

    if len(xw) < MIN_POINTS:
        return {"center": center_guess, "fwhm": np.nan, "amplitude": np.nan, "ok": False}

    # Seed from smoothed peak location
    ysm = np.convolve(yw, np.ones(5) / 5.0, mode='same') if len(yw) >= 5 else yw
    seed = xw[np.argmax(ysm)]

    result, ok = fit_peak_single(xw, yw, seed, narrow=True)
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

    # Baseline lattice parameter a0 from baseline q
    G_norm = sqrt(h**2 + k**2 + l**2)
    a_0 = 2 * pi / baseline * G_norm
    return baseline, mad, a_0

def _load_q_and_I(h5_path, wavelength=None, energy_eV=None):
    """Load q-axis and intensity from HDF5. Convert tth->q if needed."""
    with h5py.File(h5_path, "r") as f:
        if "q" in f:
            x = f["q"][:]
        else:
            if "tth" not in f:
                raise ValueError("HDF5 must contain 'q' or 'tth'.")
            tth = f["tth"][:]
            # Determine wavelength
            if "lambda" in f:
                lam = float(f["lambda"][()])
            elif "wavelength" in f:
                lam = float(f["wavelength"][()])
            elif wavelength is not None:
                lam = float(wavelength)
            elif energy_eV is not None:
                lam = 12398.42 / float(energy_eV)  # Å (hc ≈ 12398.42 eV·Å)
            elif "energy" in f:
                lam = 12398.42 / float(f["energy"][()])  # Å
            else:
                raise ValueError("Need wavelength or energy to convert tth->q.")
            theta = np.deg2rad(tth / 2.0)
            x = (4.0 * pi / lam) * np.sin(theta)
        I_full = f["int"][:]
    return x, I_full

def process_dataset(h5_path, initial_guess, desc=None, show_progress=True,
                    skip_jump_threshold=FRAME_SKIP_JUMP, max_total_movement=MAX_TOTAL_MOVEMENT,
                    nframes_limit=MAX_FRAMES, wavelength=None, energy_eV=None):
    """
    Track peak across up to nframes_limit frames with adaptive constraints.
    Returns centers (absolute q), fwhms, failed_frames, skipped_frames, nframes, a_0.
    """
    # Load q-axis and intensities
    x, I_full = _load_q_and_I(h5_path, wavelength=wavelength, energy_eV=energy_eV)

    # Limit to first nframes_limit frames
    nframes = min(I_full.shape[0], nframes_limit)
    I = I_full[:nframes]

    # Baseline center and a0
    baseline_center, seed_mad, a_0 = robust_initial_center(x, I, initial_guess, SEED_FRAMES, desc, show_progress)

    # Adaptive jump limit: tighten for noisy datasets
    max_jump = MAX_JUMP_STRICT if seed_mad > 0.01 else MAX_JUMP
    if show_progress:
        print(f"{desc}: seed MAD={seed_mad:.5f}, using max_jump={max_jump:.4f} (processing {nframes} frames)")

    centers = np.full(nframes, np.nan)
    fwhms = np.full(nframes, np.nan)
    failed_frames = []
    skipped_frames = []
    center_prev = baseline_center
    window = WINDOW
    consec_fail = 0

    iterator = range(nframes)
    if show_progress and tqdm is not None:
        iterator = tqdm(iterator, desc=f"{desc}: track", leave=False)

    for frame in iterator:
        res = fit_single_peak(x, I, frame, center_prev, window=window)
        c = res["center"]
        fwhm = res["fwhm"]
        ok = res["ok"]

        if np.isfinite(c):
            total_movement = abs(c - baseline_center)
            if total_movement > max_total_movement:
                skipped_frames.append(frame)
                centers[frame] = np.nan
                fwhms[frame] = np.nan
                if show_progress and len(skipped_frames) <= 10:
                    print(f"{desc}: Skipping frame {frame}, total movement={total_movement:.5f} > {max_total_movement:.5f}")
                continue

        if np.isfinite(c) and np.isfinite(center_prev):
            jump = abs(c - center_prev)
            if jump > skip_jump_threshold:
                skipped_frames.append(frame)
                centers[frame] = np.nan
                fwhms[frame] = np.nan
                if show_progress and len(skipped_frames) <= 10:
                    print(f"{desc}: Skipping frame {frame}, jump={jump:.5f} > {skip_jump_threshold:.5f}")
                continue

            if jump > max_jump:
                res2 = fit_single_peak(x, I, frame, center_prev, window=min(2 * window, 2 * WINDOW))
                c2 = res2["center"]
                jump2 = abs(c2 - center_prev) if np.isfinite(c2) else np.inf
                if jump2 <= 1.5 * max_jump:
                    c, fwhm, ok = c2, res2["fwhm"], res2["ok"]
                elif jump > 3 * max_jump:
                    c = center_prev
                    ok = False

        centers[frame] = c
        fwhms[frame] = fwhm
        center_prev = c if np.isfinite(c) else center_prev
        consec_fail = 0 if ok else consec_fail + 1
        if not ok:
            failed_frames.append(frame)
        window = min(2 * window, 2 * WINDOW) if consec_fail >= CONSEC_FAIL_EXPAND else WINDOW

    return centers, fwhms, failed_frames, skipped_frames, nframes, a_0

# Fractional expansion helper (anchored inversion)
def f_poly(T):
    return C0 + C1*T + C2*T**2 + C3*T**3  # fractional Δa/a0

def r_T(T):
    return 1.0 + f_poly(T)

def T_from_delta_anchored(delta, T_ref=300.0):
    """
    Invert Δa/a0 measured relative to baseline a(T_ref):
    r(T) = (1 + delta) * r(T_ref), pick root near T_ref.
    """
    target = (1.0 + delta) * r_T(T_ref)
    coeffs = [C3, C2, C1, (1.0 + C0) - target]  # C3 T^3 + C2 T^2 + C1 T + (1+C0 - target) = 0
    roots = np.roots(coeffs)
    real = roots[np.isreal(roots)].real
    good = real[(real > 0) & (real < 4000)]
    if good.size == 0:
        return np.nan
    return good[np.argmin(np.abs(good - T_ref))]

def main():
    ap = argparse.ArgumentParser(description="Track absolute peak centers and compute lattice parameter and temperature.")
    ap.add_argument("--h5", nargs='+', required=True, help="HDF5 files (one per beam)")
    ap.add_argument("--center", nargs='+', type=float, required=True, help="Initial peak centers (in q or tth units)")
    ap.add_argument("--no-progress", action="store_true", help="Disable progress bars")
    ap.add_argument("--skip-jump", type=float, default=FRAME_SKIP_JUMP, help=f"Skip frames with jumps > this value (default: {FRAME_SKIP_JUMP})")
    ap.add_argument("--max-movement", type=float, default=MAX_TOTAL_MOVEMENT, help=f"Maximum total movement from baseline (default: {MAX_TOTAL_MOVEMENT})")
    ap.add_argument("--nframes", type=int, default=MAX_FRAMES, help="Max frames to process per dataset")
    ap.add_argument("--wavelength", type=float, help="X-ray wavelength in Å (used to convert tth->q if 'q' not present)")
    ap.add_argument("--energy", type=float, help="Photon energy in eV (alternative to wavelength for tth->q)")
    ap.add_argument("--T-ref", type=float, default=300.0, help="Reference temperature (K) to anchor inversion (default: 300 K)")
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

    markers = ['o', 's', 'D', '^', 'v', 'p', 'X']

    fig1, ax1 = plt.subplots(figsize=(6.5, 4.8))  # Absolute q center
    fig2, ax2 = plt.subplots(figsize=(6.5, 4.8))  # FWHM
    fig3, ax3 = plt.subplots(figsize=(6.5, 4.8))  # a/a0
    fig4, ax4 = plt.subplots(figsize=(6.5, 4.8))  # Temperature

    dataset_iter = range(len(args.h5))
    if show_progress and tqdm is not None:
        dataset_iter = tqdm(dataset_iter, desc="Datasets")

    for i in dataset_iter:
        desc = f"DS{i}"

        centers, fwhms, failed_frames, skipped_frames, nframes, a_0 = process_dataset(
            args.h5[i], args.center[i], desc, show_progress,
            args.skip_jump, args.max_movement, args.nframes,
            wavelength=args.wavelength, energy_eV=args.energy
        )

        frames = np.arange(nframes)
        valid_mask = np.isfinite(centers) & (centers > 0)

        # Plot 1: Absolute peak centers (q)
        ax1.scatter(frames[valid_mask], centers[valid_mask],
                    label=f"Beam {i}", s=12, alpha=0.7, marker=markers[i])

        # Compute lattice parameter a directly from absolute q
        G_norm = sqrt(h**2 + k**2 + l**2)
        q_valid = centers[valid_mask]
        a = (2.0 * pi / q_valid) * G_norm

        # Δa/a0 relative to baseline a_0
        delta_a_over_a0 = (a - a_0) / a_0

        # Temperature per frame, anchored to T_ref
        T_frame = np.array([T_from_delta_anchored(d, args.T_ref) for d in delta_a_over_a0])
        valid_T = np.isfinite(T_frame)

        # Plot 2: FWHM
        valid_fwhm = np.isfinite(fwhms)
        ax2.scatter(frames[valid_fwhm], fwhms[valid_fwhm],
                    label=f"Beam {i}", s=12, alpha=0.7, marker=markers[i])

        # Plot 3: a/a0
        ax3.scatter(frames[valid_mask], a / a_0,
                    label=f"Beam {i}", s=12, alpha=0.7, marker=markers[i])

        # Plot 4: Temperature (anchored absolute)
        ax4.scatter(frames[valid_mask][valid_T], T_frame[valid_T],
                    s=12, alpha=0.7, marker=markers[i], label=f"Beam {i}")

        if show_progress:
            if failed_frames:
                print(f"{desc}: {len(failed_frames)} low-confidence frames")
            if skipped_frames:
                print(f"{desc}: {len(skipped_frames)} frames skipped due to jumps > {args.skip_jump}")

            # Sanity prints
            if valid_mask.any():
                print(f"{desc}: a0={a_0:.5f} Å, Δa/a0 range {np.nanmin(delta_a_over_a0):.3e} .. {np.nanmax(delta_a_over_a0):.3e}")
            if valid_T.any():
                print(f"{desc}: T range {np.nanmin(T_frame):.1f} .. {np.nanmax(T_frame):.1f} K (anchored to {args.T_ref} K)")

    # Configure Plot 1: Absolute q center
    ax1.set_xlim(0, args.nframes)
    ax1.set_xlabel("Frame")
    ax1.set_ylabel("Peak center q (1/Å)")
    ax1.grid(True)
    ax1.legend()
    fig1.tight_layout()

    # Configure Plot 2: FWHM
    ax2.set_xlim(0, args.nframes)
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("FWHM (1/Å)")
    ax2.grid(True)
    ax2.legend()
    fig2.tight_layout()

    # Configure Plot 3: a/a0
    ax3.set_xlim(0, args.nframes)
    ax3.set_xlabel("Frame")
    ax3.set_ylabel("Lattice parameter ratio a/a0")
    ax3.grid(True)
    ax3.legend()
    fig3.tight_layout()

    # Configure Plot 4: Temperature
    ax4.set_xlim(0, args.nframes)
    ax4.set_xlabel("Frame")
    ax4.set_ylabel("Temperature (K)")
    ax4.grid(True)
    ax4.legend()
    fig4.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()
