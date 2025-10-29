import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
from lmfit.models import GaussianModel, LinearModel
from math import pi, sqrt

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

# --- Simple tunables ---
WINDOW = 0.50         # fitting window width (in q units)
MIN_POINTS = 5        # minimum points in window to try a fit
SEED_FRAMES = 30      # frames used to define baseline a0
MAX_FRAMES = 200      # cap for frames processed

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
    Single Gaussian + linear baseline. Minimal bounds; return absolute center in q.
    """
    if len(xw) < MIN_POINTS:
        return None

    # simple baseline
    try:
        bkg_slope, bkg_intercept = np.polyfit(xw, yw, 1)
    except Exception:
        bkg_slope, bkg_intercept = 0.0, np.median(yw)

    y_detr = yw - (bkg_slope * xw + bkg_intercept)
    noise = robust_sigma(y_detr)

    # initial guesses
    peak_idx = np.abs(xw - seed_center).argmin()
    height0 = max(yw[peak_idx] - (bkg_slope * xw[peak_idx] + bkg_intercept), 0.5 * noise)
    span = xw[-1] - xw[0]
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
    # loose bounds
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

def process_dataset(h5_path, initial_center, h, k, l, nframes_limit=MAX_FRAMES, show_progress=True):
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

    # track using previous center as seed, starting from initial_center
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
            seed = res["center"]  # update seed
        # else keep NaN and seed unchanged

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

# Linear thermal expansion: delta = alpha * (T - Tref)
def T_from_delta_linear(delta, alpha, Tref):
    if alpha <= 0:
        return np.nan
    return Tref + (delta / alpha)

def main():
    ap = argparse.ArgumentParser(description="Minimal: q -> a -> Δa/a0 -> T (linear CTE), q-only input.")
    ap.add_argument("--h5", nargs='+', required=True, help="HDF5 files (one per dataset), must contain 'q' and 'int'")
    ap.add_argument("--center", nargs='+', type=float, required=True, help="Initial peak center guesses (in q units, 1/Å)")
    ap.add_argument("--hkl", type=str, default="2,0,0", help="Reflection indices 'h,k,l' (default 2,0,0)")
    ap.add_argument("--alpha", nargs='+', type=float, required=True,
                    help="Linear CTE α in 1/K (one value or one per dataset; typical 10e-6..25e-6)")
    ap.add_argument("--Tref", type=float, default=300.0, help="Reference temperature (K), default 300")
    ap.add_argument("--nframes", type=int, default=MAX_FRAMES, help="Max frames per dataset")
    ap.add_argument("--no-progress", action="store_true", help="Disable progress bars")
    args = ap.parse_args()

    h,k,l = [int(s) for s in args.hkl.split(",")]
    show_progress = not args.no_progress

    # Allow single alpha or per-dataset alphas
    alphas = args.alpha
    if len(alphas) == 1 and len(args.h5) > 1:
        alphas = alphas * len(args.h5)

    # Plotting
    plt.rcParams.update({"figure.dpi": 160, "savefig.dpi": 300, "font.size": 12})
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax_q, ax_fwhm, ax_ratio, ax_T = axes.ravel()
    markers = ['o', 's', 'D', '^', 'v', 'p', 'X']

    dataset_iter = range(len(args.h5))
    if show_progress and tqdm is not None:
        dataset_iter = tqdm(dataset_iter, desc="Datasets")

    for i in dataset_iter:
        centers, fwhms, a0, nframes = process_dataset(
            args.h5[i], args.center[i], h, k, l,
            nframes_limit=args.nframes, show_progress=show_progress
        )

        frames = np.arange(nframes)
        valid = np.isfinite(centers) & (centers > 0)
        if not np.any(valid):
            print(f"DS{i}: No valid centers; skipping.")
            continue

        # Plot absolute q centers
        ax_q.scatter(frames[valid], centers[valid], s=12, alpha=0.7, marker=markers[i % len(markers)], label=f"DS{i}")

        # Compute a per frame and fractional delta
        G_norm = sqrt(h**2 + k**2 + l**2)
        q_valid = centers[valid]
        a = (2.0 * pi / q_valid) * G_norm
        delta = (a - a0) / a0  # fractional Δa/a0 (should start near ~0)
        print(f"calculated a0={a0:.5f} Å for DS{i}")
        # Temperature via linear CTE
        alpha_i = alphas[i]
        T_frame = T_from_delta_linear(delta, alpha_i, args.Tref)

        # Plots
        v_fwhm = np.isfinite(fwhms)
        ax_fwhm.scatter(frames[v_fwhm], fwhms[v_fwhm], s=12, alpha=0.7, marker=markers[i % len(markers)], label=f"DS{i}")
        ax_ratio.scatter(frames[valid], (a / a0)-1, s=12, alpha=0.7, marker=markers[i % len(markers)], label=f"DS{i}")
        v_T = np.isfinite(T_frame)
        ax_T.scatter(frames[valid][v_T], T_frame[v_T], s=12, alpha=0.7, marker=markers[i % len(markers)], label=f"DS{i}")

        # Sanity prints
        dmin, dmax = float(np.nanmin(delta)), float(np.nanmax(delta))
        # Baseline: first SEED_FRAMES among valid frames
        valid_idx = np.where(valid)[0]
        baseline_idx = valid_idx[valid_idx < min(SEED_FRAMES, nframes)]
        if baseline_idx.size > 0:
            a_baseline = (2.0 * pi / centers[baseline_idx]) * G_norm
            baseline_delta_median = float(np.nanmedian((a_baseline - a0) / a0))
        else:
            baseline_delta_median = np.nan
        print(f"DS{i}: a0={a0:.5f} Å, baseline median Δa/a0={baseline_delta_median:.3e}, range={dmin:.3e}..{dmax:.3e}")
        if np.isfinite(dmax):
            Tmax_est = args.Tref + dmax / alpha_i
            print(f"DS{i}: With α={alpha_i:.3e} 1/K, estimated Tmax≈{Tmax_est:.1f} K")
        

    # Configure axes
    ax_q.set_xlabel("Frame");   ax_q.set_ylabel("Peak center q (1/Å)"); ax_q.grid(True);   ax_q.legend()
    ax_fwhm.set_xlabel("Frame"); ax_fwhm.set_ylabel("FWHM (1/Å)");       ax_fwhm.grid(True); ax_fwhm.legend()
    ax_ratio.set_xlabel("Frame"); ax_ratio.set_ylabel("a/a0 (ratio)");   ax_ratio.grid(True); ax_ratio.legend()
    ax_T.set_xlabel("Frame");     ax_T.set_ylabel("Temperature (K)");    ax_T.grid(True);     ax_T.legend()

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
