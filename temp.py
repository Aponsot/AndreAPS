#!/usr/bin/env python3
import argparse
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from math import pi, sqrt

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

# ------------------------------
# Tunables (edit defaults here)
# ------------------------------
WINDOW      = 0.50   # q-window width around seed center (1/Å)
MIN_POINTS  = 8      # min points in window for a fit
SEED_FRAMES = 30     # frames used for baseline (q0 -> a0)
MAX_FRAMES  = 400    # cap frames processed (set to your run length)

# ------------------------------
# Reflection: Tungsten bcc (110)
# q_hkl = (2π/a) * sqrt(h^2+k^2+l^2)  =>  a = (2π * sqrt(..))/q
# ------------------------------
H, K, L = 1, 1, 0

# ------------------------------
# Tungsten CTE polynomial (NIST SRM 737) for linear expansivity:
#   alpha(T) = (1/L_293) dL/dT  [1/K]
# The certificate provides piecewise cubic polynomials for:
#   (1/L_293)(dL/dT) * 1e6   over 80–1800 K
# Valid: 80–1800 K
# ------------------------------
T_REF_K = 293.0  # reference in SRM 737

def alpha_W(TK):
    """
    Tungsten linear expansivity α(T) [1/K] from NIST SRM 737 (80–1800 K).
    Piecewise fits for (α * 1e6):
      80–160 K:  -2.383 + 8.870e-2*T - 4.4114e-4*T^2 + 8.03e-7*T^3
      160–235 K:  0.225 + 3.982e-2*T - 1.3562e-4*T^2 + 1.667e-7*T^3
      235–330 K:  1.736 + 2.052e-2*T - 5.3494e-5*T^2 + 5.03e-8*T^3
      330–550 K:  3.353 + 5.816e-3*T - 8.9484e-6*T^2 + 5.26e-9*T^3
      550–1800 K: 4.1598 + 1.4179e-3*T - 9.5104e-7*T^2 + 4.176e-10*T^3
    """
    T = float(TK)
    if 80.0 <= T < 160.0:
        alpha_micro = (-2.383
                       + 8.870e-2*T
                       - 4.4114e-4*T*T
                       + 8.03e-7*T*T*T)
    elif 160.0 <= T < 235.0:
        alpha_micro = (0.225
                       + 3.982e-2*T
                       - 1.3562e-4*T*T
                       + 1.667e-7*T*T*T)
    elif 235.0 <= T < 330.0:
        alpha_micro = (1.736
                       + 2.052e-2*T
                       - 5.3494e-5*T*T
                       + 5.03e-8*T*T*T)
    elif 330.0 <= T < 550.0:
        alpha_micro = (3.353
                       + 5.816e-3*T
                       - 8.9484e-6*T*T
                       + 5.26e-9*T*T*T)
    elif 550.0 <= T <= 1800.0:
        alpha_micro = (4.1598
                       + 1.4179e-3*T
                       - 9.5104e-7*T*T
                       + 4.176e-10*T*T*T)
    else:
        return np.nan

    return alpha_micro * 1e-6  # [1/K]

def delta_from_T(TK, Tref_K=T_REF_K, n=2000):
    """
    Predict fractional change relative to Tref:
      delta = (a(T) - a(Tref))/a(Tref) ≈ ∫_{Tref}^{T} α(T') dT'
    (Numerical trapezoid integration)
    """
    T = float(TK)
    Tref = float(Tref_K)
    if not np.isfinite(T) or not np.isfinite(Tref):
        return np.nan
    if T == Tref:
        return 0.0

    lo, hi = (Tref, T) if T > Tref else (T, Tref)
    grid = np.linspace(lo, hi, int(n))
    a = np.array([alpha_W(t) for t in grid], dtype=float)
    if not np.all(np.isfinite(a)):
        return np.nan
    integ = np.trapz(a, grid)
    return integ if T > Tref else -integ

# ------------------------------
# Peak-fit helpers
# ------------------------------
def robust_sigma(y):
    y = np.asarray(y)
    med = np.median(y)
    return 1.4826 * np.median(np.abs(y - med)) + 1e-12

def build_window(x, y, center, width):
    half = 0.5 * width
    m = (x >= center - half) & (x <= center + half)
    xw, yw = x[m], y[m]
    m2 = np.isfinite(xw) & np.isfinite(yw)
    return xw[m2], yw[m2]

def fit_peak_lmfit(xw, yw, seed_center, model="pvoigt"):
    """
    Fit: linear background + (Gaussian or Pseudo-Voigt)
    Returns dict with center, fwhm_proxy, area
    """
    if len(xw) < MIN_POINTS:
        return None

    # background seed
    try:
        bkg_slope, bkg_intercept = np.polyfit(xw, yw, 1)
    except Exception:
        bkg_slope, bkg_intercept = 0.0, float(np.median(yw))

    y_bkg = bkg_slope * xw + bkg_intercept
    y_detr = yw - y_bkg
    noise = robust_sigma(y_detr)

    i0 = int(np.abs(xw - seed_center).argmin())
    x0 = float(xw[i0])
    height0 = max(float(yw[i0] - (bkg_slope * x0 + bkg_intercept)), 0.5 * noise)

    span = max(float(xw[-1] - xw[0]), 1e-9)
    sigma0 = max(span / 7.0, 1e-6)
    area0  = max(height0 * sigma0 * 2.5066, noise * sigma0 * 2.5066)

    from lmfit.models import LinearModel, GaussianModel, PseudoVoigtModel

    bkg = LinearModel(prefix="bkg_")
    if model == "gauss":
        peak = GaussianModel(prefix="p_")
        mdl = bkg + peak
        params = mdl.make_params(
            bkg_slope=bkg_slope,
            bkg_intercept=bkg_intercept,
            p_center=x0,
            p_sigma=sigma0,
            p_amplitude=area0,  # area in lmfit GaussianModel
        )
        params["p_sigma"].set(min=1e-6, max=max(span, 1.0))
        params["p_amplitude"].set(min=0.0)

    else:
        peak = PseudoVoigtModel(prefix="p_")
        mdl = bkg + peak
        params = mdl.make_params(
            bkg_slope=bkg_slope,
            bkg_intercept=bkg_intercept,
            p_center=x0,
            p_sigma=sigma0,
            p_amplitude=area0,   # area in lmfit PseudoVoigtModel
            p_fraction=0.5,      # 0=Gaussian, 1=Lorentzian
        )
        params["p_sigma"].set(min=1e-6, max=max(span, 1.0))
        params["p_amplitude"].set(min=0.0)
        params["p_fraction"].set(min=0.0, max=1.0)

    try:
        out = mdl.fit(yw, params, x=xw, nan_policy="omit")
        p = out.params
        center = float(p["p_center"].value)
        sigma  = float(p["p_sigma"].value)
        area   = float(p["p_amplitude"].value)

        # simple width proxy (good enough for debugging / QC)
        fwhm_proxy = 2.354820045 * sigma

        return {"center": center, "fwhm": fwhm_proxy, "area": area}
    except Exception:
        return None

# ------------------------------
# I/O + processing
# ------------------------------
def load_q_int(h5_path):
    with h5py.File(h5_path, "r") as f:
        if "q" not in f or "int" not in f:
            raise ValueError(f"{h5_path} must contain datasets 'q' and 'int'")
        q = f["q"][:]
        I = f["int"][:]
    return q, I

def process_dataset(h5_path, initial_center, model="pvoigt",
                    nframes_limit=MAX_FRAMES, show_progress=True):
    q, I_full = load_q_int(h5_path)
    nframes = min(I_full.shape[0], int(nframes_limit))
    I = I_full[:nframes]

    centers = np.full(nframes, np.nan)
    areas   = np.full(nframes, np.nan)

    it = range(nframes)
    if show_progress and tqdm is not None:
        it = tqdm(it, desc=os.path.basename(h5_path), leave=False)

    seed = float(initial_center)
    for fr in it:
        xw, yw = build_window(q, I[fr], seed, WINDOW)
        res = fit_peak_lmfit(xw, yw, seed, model=model)
        if res is None:
            xw2, yw2 = build_window(q, I[fr], seed, 2 * WINDOW)
            res = fit_peak_lmfit(xw2, yw2, seed, model=model)

        if res is not None:
            centers[fr] = res["center"]
            areas[fr]   = res["area"]
            seed = res["center"]

    valid = np.isfinite(centers) & (centers > 0)
    if not np.any(valid):
        raise RuntimeError(f"{h5_path}: no valid peak centers found.")

    idx = np.where(valid)[0]
    early = idx[idx < min(SEED_FRAMES, nframes)]
    q0 = float(np.nanmedian(centers[early])) if early.size else float(np.nanmedian(centers[valid]))

    Gnorm = sqrt(H*H + K*K + L*L)
    a0 = (2.0 * pi * Gnorm) / q0

    a = np.full(nframes, np.nan)
    a[valid] = (2.0 * pi * Gnorm) / centers[valid]
    delta = (a - a0) / a0

    return centers, areas, delta, q0, a0, nframes

# ------------------------------
# Main
# ------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Track W(110) peak center per frame and plot relative lattice shift (Δa/a0)."
    )
    ap.add_argument("--h5", nargs="+", required=True,
                    help="HDF5 files containing 'q' and 'int'")
    ap.add_argument("--center", nargs="+", type=float, required=True,
                    help="Initial peak center guesses (q, 1/Å), one per dataset (try ~2.8 for W110)")
    ap.add_argument("--model", choices=["gauss", "pvoigt"], default="pvoigt",
                    help="Peak model: gauss or pvoigt (pseudo-Voigt). Default: pvoigt")
    ap.add_argument("--max_frames", type=int, default=MAX_FRAMES)
    ap.add_argument("--window", type=float, default=WINDOW)
    ap.add_argument("--seed_frames", type=int, default=SEED_FRAMES)
    args = ap.parse_args()

    if len(args.h5) != len(args.center):
        raise ValueError("Number of --h5 files must match number of --center values.")

    global WINDOW, SEED_FRAMES
    WINDOW = float(args.window)
    SEED_FRAMES = int(args.seed_frames)

    plt.rcParams.update({"figure.dpi": 160, "savefig.dpi": 300, "font.size": 12})
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5))

    markers = ['o', 's', 'D', '^', 'v', 'p', 'X', '*', '<', '>']
    lines   = ['-', '--', ':', '-.']

    for i, (h5_path, c0) in enumerate(zip(args.h5, args.center)):
        centers, areas, delta, q0, a0, nframes = process_dataset(
            h5_path, c0, model=args.model, nframes_limit=args.max_frames, show_progress=True
        )

        frames = np.arange(nframes)
        m = np.isfinite(delta)
        label = os.path.splitext(os.path.basename(h5_path))[0]

        ax.plot(frames[m], delta[m],
                linestyle=lines[i % len(lines)],
                marker=markers[i % len(markers)],
                markersize=3,
                linewidth=1.5,
                label=label)

        print(f"[{i}] {label}: model={args.model}, q0={q0:.6f} 1/Å, a0={a0:.6f} Å, valid={m.sum()}/{nframes}")

    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Relative lattice shift Δa/a₀")
    ax.set_title(f"W(110) peak tracking ({args.model}) | window={WINDOW} | baseline frames={SEED_FRAMES}")
    ax.legend(fontsize=9, ncol=1)
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
