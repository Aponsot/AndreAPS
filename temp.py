import argparse
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from math import pi, sqrt
from scipy.optimize import curve_fit

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

# --- Tunables ---
WINDOW = 0.50       # fitting window width (in q units)
MIN_POINTS = 5      # minimum points in window to try a fit
SEED_FRAMES = 30    # frames used to define baseline a0
MAX_FRAMES = 200    # max frames processed

# Frames of interest
MELT_FRAME = 59   # where melting happened (intercept point)
FIT_START  = 60    # start frame for exponential fit
FIT_END    = 200    # end frame for exponential fit (capped by available nframes)

# Reflection (set to your peak)
h, k, l = 2, 0, 0

# Thermal expansion polynomial (fractional units)
# Δa/a0 = C0 + C1*T + C2*T^2 + C3*T^3
C0 = -0.358 / 100.0
C1 =  9.472e-4 / 100.0
C2 =  1.031e-6 / 100.0
C3 = -2.978e-10 / 100.0

# Reference temperature (Kelvin) for anchoring (set to your baseline)
T_REF_K = 300.0

def f_poly_C(Tc):
    """Raw polynomial (fractional) with Tc in °C."""
    return C0 + C1*Tc + C2*(Tc**2) + C3*(Tc**3)

def f_rel_K(T_K, Tref_K=T_REF_K):
    """Fractional expansion relative to Kelvin reference."""
    Tc     = T_K     - 273.15
    Tc_ref = Tref_K  - 273.15
    return f_poly_C(Tc) - f_poly_C(Tc_ref)

def r_T_K(T_K, Tref_K=T_REF_K):
    """Lattice ratio r(T) = 1 + f_rel(T). r(Tref)=1 by construction."""
    return 1.0 + f_rel_K(T_K, Tref_K)

def T_from_delta_poly_lookup_K(delta_array, Tref_K=T_REF_K, Tmin_K=250.0, Tmax_K=3000.0, npts=7501):
    """
    Robust inversion in Kelvin:
    Solve r(T_K) = (1 + delta), since r(Tref_K)=1.
    Returns temperatures in Kelvin (NaN if target out of range).
    """
    Tgrid = np.linspace(Tmin_K, Tmax_K, npts)
    rgrid = r_T_K(Tgrid, Tref_K)

    # enforce monotonic for interp
    if not np.all(np.diff(rgrid) > 0):
        idx = np.argsort(rgrid)
        r_sorted = rgrid[idx]
        T_sorted = Tgrid[idx]
    else:
        r_sorted = rgrid
        T_sorted = Tgrid

    target = 1.0 + np.asarray(delta_array, dtype=float)
    T_out = np.full_like(target, np.nan, dtype=float)
    m = (target >= r_sorted[0]) & (target <= r_sorted[-1])
    T_out[m] = np.interp(target[m], r_sorted, T_sorted)
    return T_out

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

def fit_peak_single(xw, yw, seed_center):
    """
    Single Gaussian + linear baseline. Return absolute center (q) and FWHM.
    """
    if len(xw) < MIN_POINTS:
        return None

    try:
        bkg_slope, bkg_intercept = np.polyfit(xw, yw, 1)
    except Exception:
        bkg_slope, bkg_intercept = 0.0, np.median(yw)

    y_detr = yw - (bkg_slope * xw + bkg_intercept)
    noise = robust_sigma(y_detr)

    peak_idx = np.abs(xw - seed_center).argmin()
    height0 = max(yw[peak_idx] - (bkg_slope * xw[peak_idx] + bkg_intercept), 0.5 * noise)
    span = max(xw[-1] - xw[0], 1e-9)
    sigma0 = max(span / 7.0, 1e-6)
    amp0 = max(height0 * sigma0 * 2.5066, noise * sigma0 * 2.5066)

    # Build model
    from lmfit.models import GaussianModel, LinearModel
    model = LinearModel(prefix="bkg_") + GaussianModel(prefix="g_")
    params = model.make_params(
        bkg_slope=bkg_slope,
        bkg_intercept=bkg_intercept,
        g_center=xw[peak_idx],
        g_sigma=sigma0,
        g_amplitude=amp0,
    )
    params["g_sigma"].set(min=1e-6, max=max(span, 1.0))
    params["g_amplitude"].set(min=0.0)

    try:
        result = model.fit(yw, params, x=xw, nan_policy="omit")
        p = result.params
        center_fit = p["g_center"].value
        sigma_fit  = p["g_sigma"].value
        fwhm_fit   = sigma_to_fwhm(sigma_fit) if np.isfinite(sigma_fit) else np.nan
        return {"center": center_fit, "fwhm": fwhm_fit}
    except Exception:
        return None

def load_q_and_I_q_only(h5_path):
    """
    Load q and intensity from HDF5. Requires 'q' (1/Å) to be present.
    """
    with h5py.File(h5_path, "r") as f:
        if "q" not in f:
            raise ValueError(f"{h5_path} does not contain 'q'. This script expects q data already.")
        x = f["q"][:]
        I_full = f["int"][:]
    return x, I_full

def process_dataset(h5_path, initial_center, nframes_limit=MAX_FRAMES, show_progress=True):
    """
    Fit absolute peak centers per frame (in q). Compute baseline a0 from first SEED_FRAMES.
    Returns centers (q), fwhms, a0, frames_count.
    """
    x, I_full = load_q_and_I_q_only(h5_path)
    nframes = min(I_full.shape[0], nframes_limit)
    I = I_full[:nframes]

    centers = np.full(nframes, np.nan)
    fwhms   = np.full(nframes, np.nan)

    iterator = range(nframes)
    if show_progress and tqdm is not None:
        iterator = tqdm(iterator, desc=f"{h5_path}: fit", leave=False)

    seed = initial_center
    for frame in iterator:
        xw, yw = build_window(x, I[frame], seed, WINDOW)
        res = fit_peak_single(xw, yw, seed)
        if res is None:
            # fallback: widen window once
            xw2, yw2 = build_window(x, I[frame], seed, 2 * WINDOW)
            res = fit_peak_single(xw2, yw2, seed)
        if res is not None:
            centers[frame] = res["center"]
            fwhms[frame]   = res["fwhm"]
            seed = res["center"]

    # Baseline q0 and a0 from early frames
    valid = np.isfinite(centers)
    idx_valid = np.where(valid)[0]
    if idx_valid.size == 0:
        raise RuntimeError("No valid peak centers found.")
    early_idx = idx_valid[idx_valid < min(SEED_FRAMES, nframes)]
    q0 = np.nanmedian(centers[early_idx]) if early_idx.size > 0 else np.nanmedian(centers[valid])

    G_norm = sqrt(h**2 + k**2 + l**2)
    a0 = (2.0 * pi / q0) * G_norm

    return centers, fwhms, a0, nframes

# Bi-exponential model: fast + slow components
def exp_bi(x, c, A1, tau1, A2, tau2):
    return c + A1 * np.exp(-(x - FIT_START) / tau1) + A2 * np.exp(-(x - FIT_START) / tau2)

def main():
    ap = argparse.ArgumentParser(description="Raw temperature scatter with bi-exponential decay fits and extrapolated intercept at frame 58.")
    ap.add_argument("--h5", nargs='+', required=True, help="HDF5 files (one per dataset), must contain 'q' and 'int'")
    ap.add_argument("--center", nargs='+', type=float, required=True, help="Initial peak center guesses (in q units, 1/Å)")
    args = ap.parse_args()

    if len(args.h5) != len(args.center):
        raise ValueError("Number of --h5 files must match number of --center guesses.")

    # Plotting config
    plt.rcParams.update({"figure.dpi": 160, "savefig.dpi": 300, "font.size": 12})
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    # Plot styling (do not change calculations)
    Color = 'C6' # single hue for all datasets
    markers = ['o', 's', 'D', '^', 'v', 'p', 'X']  # different marker per dataset
    depths_um = [50, 100, 150]  # known depths

    dataset_iter = range(len(args.h5))
    if tqdm is not None:
        dataset_iter = tqdm(dataset_iter, desc="Datasets")

    for i in dataset_iter:
        centers, fwhms, a0, nframes = process_dataset(
            args.h5[i], args.center[i], nframes_limit=MAX_FRAMES, show_progress=True
        )
        frames = np.arange(nframes)
        valid_centers = np.isfinite(centers) & (centers > 0)

        if not np.any(valid_centers):
            print(f"DS{i}: No valid centers; skipping.")
            continue

        # Compute lattice parameter and fractional delta (full-length)
        G_norm = sqrt(h**2 + k**2 + l**2)
        a_full = np.full(nframes, np.nan)
        a_full[valid_centers] = (2.0 * pi / centers[valid_centers]) * G_norm
        delta_full = (a_full - a0) / a0

        # Convert to temperature (°C), aligned to frames
        T_full_K = np.full(nframes, np.nan)
        T_full_K[valid_centers] = T_from_delta_poly_lookup_K(delta_full[valid_centers], Tref_K=T_REF_K)
        T_full_C = T_full_K - 273.15

        # Scatter raw temperatures (small points, alpha ~0.8), same color, different marker
        finite_mask = np.isfinite(T_full_C)
        depth_label = depths_um[i % len(depths_um)]
        ax.scatter(frames[finite_mask], T_full_C[finite_mask],
                   s=8, alpha=0.4, color=Color, marker=markers[i % len(markers)])

        # Build fit window and fit bi-exponential decay
        max_frame_for_fit = min(nframes - 1, FIT_END)
        fit_mask = finite_mask & (frames >= FIT_START) & (frames <= max_frame_for_fit)
        fit_idx = np.where(fit_mask)[0]
        x_fit = frames[fit_idx]
        y_fit = T_full_C[fit_idx]

        print(f"DS{i}: fit window [{FIT_START}..{max_frame_for_fit}], finite points = {y_fit.size}")
        # Build fit window and fit bi-exponential decay (weighted to emphasize early times)
        max_frame_for_fit = min(nframes - 1, FIT_END)
        fit_mask = finite_mask & (frames >= FIT_START) & (frames <= max_frame_for_fit)
        fit_idx = np.where(fit_mask)[0]
        x_fit = frames[fit_idx]
        y_fit = T_full_C[fit_idx]

        print(f"DS{i}: fit window [{FIT_START}..{max_frame_for_fit}], finite points = {y_fit.size}")
        if y_fit.size >= 8 and np.ptp(x_fit) > 0:
            # Initial guesses
            last_n = max(3, min(10, y_fit.size))
            c0 = float(np.nanmedian(y_fit[-last_n:]))
            A_total = float(y_fit[0] - c0)
            # Ensure positive total amplitude (cooling)
            if A_total < 0:
                # If initial guess suggests heating (unlikely), flip sign to enforce cooling
                A_total = abs(A_total)

            # Split amplitude between fast and slow components
            A1_0 = 0.7 * A_total  # more weight to fast component
            A2_0 = 0.3 * A_total

            span = float(x_fit[-1] - x_fit[0])
            # Fast component should be small compared to span
            tau1_0 = max(2.0, span / 15.0)
            # Slow component noticeably larger
            tau2_0 = max(10.0, span / 3.0)

            # Bounds:
            # - c near late-time median (± 2*|A_total|)
            # - A1, A2 >= 0 (monotonic decay)
            # - tau1 small-ish; tau2 larger
            c_lo = c0 - 2.0 * abs(A_total)
            c_hi = c0 + 2.0 * abs(A_total)
            tau1_hi = max(5.0, span / 10.0)     # tighten upper bound for fast time constant
            tau2_lo = max(8.0, span / 6.0)     # ensure slow time constant is not too small

            # Weight early points more: sigma small near FIT_START -> higher weight
            # curve_fit minimizes sum((resid/sigma)^2), so smaller sigma increases weight.
            w_strength = 0.6  # increase to 1.0 for stronger emphasis on early points
            sigma = 1.0 / (1.0 + w_strength * (x_fit - FIT_START))

            try:
                popt, pcov = curve_fit(
                    exp_bi, x_fit, y_fit,
                    p0=(c0, A1_0, tau1_0, A2_0, tau2_0),
                    bounds=([c_lo, 0.0, 1e-3, 0.0, tau2_lo],
                            [c_hi, 1e6, tau1_hi, 1e6, 1e6]),
                    sigma=sigma,
                    absolute_sigma=True,
                    maxfev=10000
                )
                c_fit, A1_fit, tau1_fit, A2_fit, tau2_fit = popt

                # Extrapolate from MELT_FRAME onward
                x_line = np.arange(MELT_FRAME, nframes)
                y_line = exp_bi(x_line, *popt)
                line = ['-', '--', ':'][i % 3]  # different line style per dataset
                # Plot fit line (same color, label only depth)
                ax.plot(x_line, y_line, color=Color, linewidth=2.0,linestyle=line,  
                        label=f"{depth_label} μm")

                # Intercept at MELT_FRAME for console info
                T_melt = float(exp_bi(MELT_FRAME, *popt))
                print(f"Depth {depth_label} μm: T@{MELT_FRAME} = {T_melt:.1f} °C; τ_fast={tau1_fit:.1f}, τ_slow={tau2_fit:.1f}")
            except Exception as e:
                print(f"DS{i}: bi-exponential curve_fit failed: {e}")
        else:
            print(f"DS{i}: Not enough finite points to fit in [{FIT_START}..{max_frame_for_fit}].")
            # Initial guesses
            last_n = max(3, min(10, y_fit.size))
            c0 = float(np.nanmedian(y_fit[-last_n:]))
            A_total = float(y_fit[0] - c0)
            # Split amplitude between fast and slow components
            A1_0 = 0.5* A_total
            A2_0 = 0.5* A_total
            span = float(x_fit[-1] - x_fit[0])
            tau1_0 = max(1.0, span / 15.0)  # fast component
            tau2_0 = max(8.0, span / 3.0)    # slow component
    # Decorate single plot
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(f"Raw temperature scatter and bi-exponential decay fits (fit window {FIT_START}–{FIT_END}, intercept at {MELT_FRAME})")
    ax.axvline(MELT_FRAME, color='0.5', linestyle='--', linewidth=1.0)
    ax.axvline(FIT_START, color='0.6', linestyle=':', linewidth=1.0)

    # Legend: only shows which curve corresponds to which depth
    ax.legend(ncol=1, fontsize=10)

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
