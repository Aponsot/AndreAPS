import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
from lmfit.models import GaussianModel, LinearModel, ExponentialModel, ConstantModel
from math import pi, sqrt

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
MELT_FRAME = 58     # where melting happened
FIT_START  = 65     # start frame for exponential fit
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

# --- Polynomial thermal expansion (anchored inversion) ---
def f_poly(T):
    # fractional Δa/a0
    return C0 + C1*T + C2*T**2 + C3*T**3

def r_T(T):
    # lattice ratio r(T) = 1 + f(T)
    return 1.0 + f_poly(T)

def T_from_delta_poly_anchored(delta, T_ref=T_REF_K):
    """
    Anchored inversion: r(T) = (1 + delta) * r(T_ref)
    Solve C3*T^3 + C2*T^2 + C1*T + (1 + C0 - target) = 0,
    where target = (1 + delta) * r(T_ref).
    Returns the real root near T_ref.
    """
    target = (1.0 + delta) * r_T(T_ref)
    coeffs = [C3, C2, C1, (1.0 + C0) - target]
    roots = np.roots(coeffs)
    real = roots[np.isreal(roots)].real
    good = real[(real > 0) & (real < 4000)]
    if good.size == 0:
        return np.nan
    return good[np.argmin(np.abs(good - T_ref))]

def main():
    ap = argparse.ArgumentParser(description="Temperature plots and exponential fit (frames 65–200), extrapolated to frame 58.")
    ap.add_argument("--h5", nargs='+', required=True, help="HDF5 files (one per dataset), must contain 'q' and 'int'")
    ap.add_argument("--center", nargs='+', type=float, required=True, help="Initial peak center guesses (in q units, 1/Å)")
    args = ap.parse_args()

    if len(args.h5) != len(args.center):
        raise ValueError("Number of --h5 files must match number of --center guesses.")

    # Plotting config
    plt.rcParams.update({"figure.dpi": 160, "savefig.dpi": 300, "font.size": 12})
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    markers    = ['o', 's', 'D', '^', 'v', 'p', 'X']
    linestyles = ['-', '--', ':']  # different line style per dataset
    Color      = 'C1'              # single hue for all, as requested
    depths_um  = [50, 100, 150]    # labels for datasets

    dataset_iter = range(len(args.h5))
    if tqdm is not None:
        dataset_iter = tqdm(dataset_iter, desc="Datasets")

    for i in dataset_iter:
        centers, fwhms, a0, nframes = process_dataset(
            args.h5[i], args.center[i], nframes_limit=MAX_FRAMES, show_progress=True
        )

        frames = np.arange(nframes)
        valid = np.isfinite(centers) & (centers > 0)
        if not np.any(valid):
            print(f"DS{i}: No valid centers; skipping.")
            continue

        # Compute a per frame and fractional delta
        G_norm = sqrt(h**2 + k**2 + l**2)
        q_valid = centers[valid]
        a = (2.0 * pi / q_valid) * G_norm
        delta = (a - a0) / a0

        # Temperature from polynomial via lookup (anchored to T_REF)
        T_frame_K = T_from_delta_poly_lookup_K(delta, Tref_K=T_REF_K)
        T_frame_C = T_frame_K - 273.15

        # Left panel: scatter of temperature vs frame
        valid_T = np.isfinite(T_frame_C)
        ds_mask = valid.copy()
        ds_mask[valid] = valid_T  # apply temperature validity only where centers valid

        depth_label = depths_um[i % len(depths_um)]
        ax_left.scatter(frames[ds_mask], T_frame_C[valid_T], s=20, alpha=0.6,
                        marker=markers[i % len(markers)], color=Color,
                        label=f"depth {depth_label} um")

        # Right panel: exponential fit on frames FIT_START..FIT_END, extrapolate back to MELT_FRAME
        max_frame_for_fit = min(nframes - 1, FIT_END)
        fit_mask = valid.copy()
        fit_mask &= (frames >= FIT_START) & (frames <= max_frame_for_fit)
        # Also require finite temperatures
        fit_mask[valid] &= valid_T

        x_fit_data = frames[fit_mask]
        y_fit_data = T_frame_C[valid_T][(frames[valid] >= FIT_START) & (frames[valid] <= max_frame_for_fit)]
        # Guard against insufficient data
        if x_fit_data.size >= 5 and y_fit_data.size == x_fit_data.size:
            # Build model: y = c + amplitude * exp(decay * x), with decay < 0
            model = ConstantModel() + ExponentialModel()
            # Initial guesses: c ~ median of last few points, amplitude ~ first - c, decay negative
            last_n = max(3, min(10, x_fit_data.size))
            c0 = float(np.nanmedian(y_fit_data[-last_n:]))
            a0_exp = float(y_fit_data[0] - c0)
            decay0 = -0.02  # reasonable starting slope per frame

            params = model.make_params(c=c0, amplitude=a0_exp, decay=decay0)
            params['decay'].set(max=-1e-6)  # force decay to be negative (decay)
            # Fit
            try:
                result = model.fit(y_fit_data, params, x=x_fit_data, nan_policy="omit")
                p = result.params
                c_fit     = p['c'].value
                amp_fit   = p['amplitude'].value
                decay_fit = p['decay'].value
                tau_fit   = (-1.0 / decay_fit) if decay_fit < 0 else np.inf

                # Extrapolate and plot from MELT_FRAME to max_frame_for_fit
                x_line = np.arange(MELT_FRAME, max_frame_for_fit + 1)
                y_line = model.eval(params=p, x=x_line)
                ax_right.plot(x_line, y_line, linestyle=linestyles[i % len(linestyles)],
                              color=Color, linewidth=1.8, label=f"depth {depth_label} um (τ≈{tau_fit:.1f} frames)")
            except Exception as e:
                print(f"DS{i}: Exp fit failed: {e}")
        else:
            print(f"DS{i}: Not enough points for exp fit in frames {FIT_START}..{max_frame_for_fit}.")

        # Optional: overlay faint scatter on right to show raw points (can comment out)
        ax_right.scatter(frames[ds_mask], T_frame_C[valid_T], s=16, alpha=0.25,
                         marker=markers[i % len(markers)], color=Color)

        # Diagnostics
        dmin, dmax = float(np.nanmin(delta)), float(np.nanmax(delta))
        valid_idx = np.where(valid)[0]
        baseline_idx = valid_idx[valid_idx < min(SEED_FRAMES, nframes)]
        if baseline_idx.size > 0:
            a_baseline = (2.0 * pi / centers[baseline_idx]) * G_norm
            baseline_delta_median = float(np.nanmedian((a_baseline - a0) / a0))
        else:
            baseline_delta_median = np.nan
        print(f"DS{i}: a0={a0:.5f} Å, baseline median Δa/a0={baseline_delta_median:.3e}, range={dmin:.3e}..{dmax:.3e}")

        if np.any(ds_mask):
            Tmin_C = float(np.nanmin(T_frame_C[valid_T]))
            Tmax_C = float(np.nanmax(T_frame_C[valid_T]))
            print(f"DS{i}: Temperature (°C) range {Tmin_C:.1f} .. {Tmax_C:.1f} (anchored at {T_REF_K - 273.15:.1f} °C)")

    # Decorate plots
    for ax in (ax_left, ax_right):
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Frame")
        ax.axvline(MELT_FRAME, color='0.6', linestyle='--', linewidth=1.0)
        ax.axvline(FIT_START, color='0.7', linestyle=':', linewidth=1.0)
        ax.annotate("melt", xy=(MELT_FRAME, ax.get_ylim()[0]), xytext=(5, 5),
                    textcoords='offset points', fontsize=10, color='0.3')
        ax.annotate("fit start", xy=(FIT_START, ax.get_ylim()[0]), xytext=(5, 5),
                    textcoords='offset points', fontsize=10, color='0.4')

    ax_left.set_ylabel("Temperature (°C)")
    ax_left.set_title("Temperature vs Frame (scatter)")
    ax_left.legend()

    ax_right.set_ylabel("Temperature (°C)")
    ax_right.set_title(f"Exp. decay fit (frames {FIT_START}–{FIT_END}), extrapolated to frame {MELT_FRAME}")
    ax_right.legend()

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
