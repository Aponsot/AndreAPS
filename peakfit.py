import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.ndimage import gaussian_filter1d

from lmfit.models import PolynomialModel, GaussianModel, PseudoVoigtModel


# ----------------------------- helpers ----------------------------- #
def _sigma_from_fwhm(fwhm_pts: float, dq: float) -> float:
    """Convert FWHM in sample points to sigma in x-units using FWHM = 2.355*sigma."""
    return (fwhm_pts * dq) / 2.355


def fit_multi_peaks(
    q_limited,
    y_limited,
    peaks_idx,
    props,
    bg_degree=1,
    center_window=0.005,
    sigma_floor=1e-5,
    model_kind="pvoigt",  # 'pvoigt' or 'gauss'
):
    """
    Build & fit a background + sum(peaks) using good starting guesses from find_peaks.
    y_limited is the *raw* data in the fit window (no manual bg subtraction).
    """
    if len(peaks_idx) == 0:
        return None, {}

    dq = float(np.diff(q_limited).mean())
    # Background
    composite = PolynomialModel(degree=bg_degree, prefix="bg_")
    params = composite.make_params()
    params["bg_c0"].set(value=float(np.median(y_limited)), min=0)
    if bg_degree >= 1:
        params["bg_c1"].set(value=0)

    # Peak props
    fwhm_pts = props.get("widths", np.full_like(peaks_idx, 3.0, dtype=float))
    prominences = props.get("prominences", np.ones_like(peaks_idx, dtype=float))

    # Peak model type
    PeakModel = PseudoVoigtModel if model_kind.lower() in ("pv", "pvoigt") else GaussianModel

    # Add a component per detected peak
    for i, pidx in enumerate(peaks_idx):
        center0 = float(q_limited[pidx])

        sigma0 = float(_sigma_from_fwhm(float(fwhm_pts[i]), dq))
        sigma0 = max(sigma0, sigma_floor)

        # amplitude ~ area ≈ height * sigma * sqrt(2*pi) (height ~ prominence on bg-sub)
        height0 = float(prominences[i])
        amp0 = height0 * sigma0 * np.sqrt(2 * np.pi)

        g = PeakModel(prefix=f"g{i}_")
        composite += g

        params.update(g.make_params(center=center0, sigma=sigma0, amplitude=amp0))
        # Bounds to stop swapping:
        params[f"g{i}_center"].set(min=center0 - center_window, max=center0 + center_window)
        params[f"g{i}_sigma"].set(min=sigma0 * 0.25, max=sigma0 * 4)
        # Keep amplitude from exploding or flipping wildly
        params[f"g{i}_amplitude"].set(min=-abs(amp0) * 5, max=abs(amp0) * 5)

        # PseudoVoigt adds eta; start neutral mid-way if present
        if isinstance(g, PseudoVoigtModel):
            params[f"g{i}_fraction"].set(value=0.5, min=0, max=1)

    # Global fit
    result = composite.fit(y_limited, params, x=q_limited)
    comps = result.eval_components(x=q_limited)
    return result, comps


# ----------------------------- main routine ----------------------------- #
def peak_fit(h5_path, frame_number, peak_pos, window=0.1):
    with h5py.File(h5_path, "r") as f:
        int_val = f["int"][:]          # shape: (nframes, q)
        q = f["q"][:]                  # shape: (q,)
        cake_intensity_stack = f["cake_int"][:] if "cake_int" in f else None

    # Slices you wanted
    cake_slices = [0, 10, 19, 28]

    # Tunables
    sigma_smooth = 50                  # smoothing for detection baseline
    prom_cake = 2                      # min prominence for cake find_peaks
    prom_full = 2                      # min prominence for full data find_peaks
    model_kind = "gauss"              # 'pvoigt' or 'gauss'
    center_window = 0.005              # tighten if peaks are VERY close

    # Window the region around the target peak_pos
    q_min = peak_pos - window
    q_max = peak_pos + window
    mask = (q >= q_min) & (q <= q_max)
    q_limited = q[mask]

    if q_limited.size < 5:
        raise ValueError("Fit window too small or outside q-range.")

    # ---- FIGURE LAYOUT ----
    fig, axes = plt.subplots(
        2, 3, figsize=(15, 10), gridspec_kw={"width_ratios": [1, 1, 2]}
    )
    fig.suptitle(f"Peak Fit for Frame {frame_number}", fontsize=16)

    # ---- FULL AZIMUTHAL INTEGRATION ----
    y_full = int_val[frame_number, mask]

    # background for detection only
    background_full = gaussian_filter1d(y_full, sigma=sigma_smooth)
    data_bg_sub_full = y_full - background_full

    peaks_full, props_full = signal.find_peaks(
        data_bg_sub_full, prominence=prom_full, width=1
    )

    print(f"[FULL] Detected {len(peaks_full)} peaks at q:", q_limited[peaks_full])

    # Fit raw data with bg + multi-peak model
    result_full, comps_full = fit_multi_peaks(
        q_limited,
        y_full,
        peaks_full,
        props_full,
        bg_degree=1,
        center_window=center_window,
        model_kind=model_kind,
    )

    # Plot data + detection bg
    ax = axes[0, 2]
    ax.plot(q_limited, y_full, "--", label="Data")
    ax.plot(q_limited, background_full, "-", label="BG (pre-smooth)")
    ax.plot(q_limited[peaks_full], data_bg_sub_full[peaks_full], "x", label="Detected peaks")

    if result_full is not None:
        ax.plot(q_limited, result_full.best_fit, label="Total fit")
        # plot each Gaussian/PV component
        for name in sorted([k for k in comps_full.keys() if k.startswith("g")]):
            ax.plot(q_limited, comps_full[name], ":", label=name)
        # fitted bg component
        if "bg_" in comps_full:
            ax.plot(q_limited, comps_full["bg_"], "-", label="BG (fit)")

        # Print per-peak parameters
        print("\n[FULL] Fitted peak parameters:")
        for i in range(len(peaks_full)):
            c = result_full.params.get(f"g{i}_center").value
            s = result_full.params.get(f"g{i}_sigma").value
            a = result_full.params.get(f"g{i}_amplitude").value
            if model_kind == "pvoigt":
                eta = result_full.params.get(f"g{i}_fraction").value
                print(f"  g{i}: center={c:.6f}, sigma={s:.6g}, amp={a:.6g}, fraction(Lorentz)={eta:.3f}")
            else:
                print(f"  g{i}: center={c:.6f}, sigma={s:.6g}, amp={a:.6g}")

    ax.set_title("Full Azimuthal Integration")
    ax.set_xlabel("q")
    ax.set_ylabel("Intensity")
    ax.legend(loc="upper right", fontsize=8)

    # ---- CAKE SLICES ----
    if cake_intensity_stack is not None:
        for i, cs in enumerate(cake_slices):
            row, col = divmod(i, 2)
            y_cake = cake_intensity_stack[frame_number, cs, :][mask]

            bg_cake = gaussian_filter1d(y_cake, sigma=sigma_smooth)
            ysub = y_cake - bg_cake
            peaks_cake, props_cake = signal.find_peaks(ysub, prominence=prom_cake, width=1)

            axes[row, col].plot(q_limited, y_cake, "--", label="Cake data")
            axes[row, col].plot(q_limited, bg_cake, "-", label="BG (pre-smooth)")
            axes[row, col].plot(q_limited[peaks_cake], ysub[peaks_cake], "x", label="Detected peaks")
            axes[row, col].set_title(f"Cake slice {cs}")
            axes[row, col].set_xlabel("q")
            axes[row, col].set_ylabel("Intensity")
            axes[row, col].legend(loc="upper right", fontsize=8)

            print(f"[CAKE {cs}] Detected {len(peaks_cake)} peaks at q:", q_limited[peaks_cake])
    else:
        # If no cake data, blank the 2x2 spot
        for (r, c) in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            axes[r, c].axis("off")
            axes[r, c].text(0.5, 0.5, "No cake_int in file", ha="center", va="center")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def _parse_args():
    p = argparse.ArgumentParser(description="Multi-peak fitting around a target q.")
    p.add_argument("h5", type=str, help="Input h5 file containing processed data.")
    p.add_argument("frame_number", type=int, help="Frame number to process (0-indexed).")
    p.add_argument( "peak_pos", type=float, help="Center q to window around; peaks will be detected within this window.",)
    p.add_argument("--window", type=float, default=0.1, help="Half-window size in q.")
    return p.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    peak_fit(args.h5, args.frame_number, args.peak_pos, args.window)
