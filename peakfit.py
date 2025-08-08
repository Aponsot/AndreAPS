import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.ndimage import gaussian_filter1d
from lmfit.models import PolynomialModel, GaussianModel


# ----------------------------- helpers ----------------------------- #
def _sigma_from_fwhm(fwhm_pts: float, dq: float) -> float:
    """Convert FWHM in sample points to sigma in x-units using FWHM = 2.355*sigma."""
    return (fwhm_pts * dq) / 2.355
def _gaussian(x, mu, sig, amp):
    return amp * np.exp(-0.5*((x - mu)/sig)**2)

def _build_crude_model(q, peaks_idx, props, dq, floor_amp=1e-9):
    widths = props.get("widths", np.full_like(peaks_idx, 3.0, dtype=float))
    prominences = props.get("prominences", np.ones_like(peaks_idx, dtype=float))
    yhat = np.zeros_like(q, dtype=float)
    for i, pidx in enumerate(peaks_idx):
        mu = float(q[pidx])
        sigma = max((widths[i]*dq)/2.355, 1e-6)
        height = float(prominences[i])
        amp = max(height * sigma * np.sqrt(2*np.pi), floor_amp)  # area for Gaussian
        yhat += _gaussian(q, mu, sigma, amp)
    return yhat

def _augment_peaks_with_residual(q, y_bgsub, init_idx, init_props, dq,
                                 prom_base=2.0, add_frac=0.4, min_sep_sigma=0.8,
                                 max_iter=2):
    """
    y_bgsub is background-subtracted data used ONLY for detection.
    We keep orig raw data for fitting elsewhere.
    """
    peaks_idx = np.array(init_idx, dtype=int)
    widths = np.array(init_props.get("widths", np.full_like(peaks_idx, 3.0, dtype=float)), dtype=float)
    prominences = np.array(init_props.get("prominences", np.ones_like(peaks_idx, dtype=float)), dtype=float)

    for _ in range(max_iter):
        if peaks_idx.size == 0:
            mean_w_pts = 3.0
        else:
            mean_w_pts = float(np.clip(np.nanmean(widths), 2.0, 20.0))
        min_dist_pts = int(max(1, round(min_sep_sigma * mean_w_pts)))

        # crude model from current seeds
        crude = _build_crude_model(q, peaks_idx, {"widths": widths, "prominences": prominences}, dq)
        residual = y_bgsub - crude

        # detect on residual with lower threshold
        prom_add = max(0.3, add_frac) * prom_base
        cand_idx, cand_props = signal.find_peaks(residual, prominence=prom_add, width=1,
                                                 distance=min_dist_pts, rel_height=0.5)

        # keep only candidates sufficiently far from existing seeds
        new_idx = []
        new_w = []
        new_p = []
        for j, pidx in enumerate(cand_idx):
            qj = q[pidx]
            if peaks_idx.size and np.min(np.abs(q[peaks_idx] - qj)) < (min_dist_pts * dq):
                continue
            new_idx.append(pidx)
            new_w.append(cand_props["widths"][j])
            new_p.append(cand_props["prominences"][j])

        if not new_idx:
            break  # nothing else to add

        # merge
        peaks_idx = np.concatenate([peaks_idx, np.array(new_idx, dtype=int)])
        widths = np.concatenate([widths, np.array(new_w, dtype=float)])
        prominences = np.concatenate([prominences, np.array(new_p, dtype=float)])

        # sort by q
        order = np.argsort(q[peaks_idx])
        peaks_idx = peaks_idx[order]
        widths = widths[order]
        prominences = prominences[order]

    # return in scipy-like props dict
    props_out = {"widths": widths, "prominences": prominences}
    return peaks_idx, props_out

def _poisson_weights(y):
    return 1.0 / np.sqrt(np.clip(y, 1.0, None))


def _refit_overlapping_pairs(result, q, y, overlap_factor=2.5):
    """
    Automatically refit each adjacent peak pair that significantly overlaps.
    overlap_factor: separation threshold in units of avg sigma.
    """
    # gather IDs
    peak_ids = sorted({int(k.split('_')[0][1:]) for k in result.params if k.startswith('g') and k.endswith('_center')})
    if len(peak_ids) < 2:
        return 0

    centers = np.array([result.params[f"g{i}_center"].value for i in peak_ids])
    sigmas  = np.array([result.params[f"g{i}_sigma"].value  for i in peak_ids])

    ord_idx = np.argsort(centers)
    ids     = np.array(peak_ids)[ord_idx]
    centers = centers[ord_idx]
    sigmas  = sigmas[ord_idx]

    comps_all = result.model.eval_components(params=result.params, x=q)
    refined = 0

    for a, b in zip(range(len(ids)-1), range(1, len(ids))):
        i1, i2 = ids[a], ids[b]
        c1, c2 = centers[a], centers[b]
        s1, s2 = sigmas[a], sigmas[b]
        sep = abs(c2 - c1)
        s_avg = 0.5 * (s1 + s2)

        if sep >= overlap_factor * s_avg:
            continue  # not overlapped enough

        # residual = data - (everything except the pair)
        residual = y.copy()
        for name, comp in comps_all.items():
            if name not in (f"g{i1}_", f"g{i2}_"):
                residual -= comp

        # tight local window around the pair
        half_width = 3.0 * s_avg
        q_lo, q_hi = min(c1, c2) - half_width, max(c1, c2) + half_width
        loc = (q >= q_lo) & (q <= q_hi)
        if loc.sum() < 8:
            continue

        q_local = q[loc]
        y_local = residual[loc]

        # pair-only Gaussian refit (no background)
        pair_model = GaussianModel(prefix=f"g{i1}_") + GaussianModel(prefix=f"g{i2}_")
        p = pair_model.make_params()
        for j, cj, sj in ((i1, c1, s1), (i2, c2, s2)):
            aj = max(result.params[f"g{j}_amplitude"].value, 1e-9)
            p[f"g{j}_center"].set(value=cj, min=cj - 0.0015, max=cj + 0.0015)
            p[f"g{j}_sigma"].set(value=sj, min=0.5*sj, max=2.0*sj)
            p[f"g{j}_amplitude"].set(value=aj, min=0.3*aj, max=5.0*aj)

        w_local = _poisson_weights(y_local)
        refit = pair_model.fit(
            y_local, p, x=q_local, weights=w_local,
            method="least_squares", fit_kws={"loss": "soft_l1", "f_scale": 1.0}
        )

        # inject back
        for j in (i1, i2):
            for par in ("center", "sigma", "amplitude"):
                result.params[f"g{j}_{par}"].set(value=refit.params[f"g{j}_{par}"].value)

        # refresh components for next loop
        comps_all = result.model.eval_components(params=result.params, x=q)
        refined += 1

    return refined


def fit_multi_peaks(
    q_limited,
    y_limited,
    peaks_idx,
    props,
    bg_degree=1,
    center_window=0.005,
    sigma_floor=1e-5,
):
    """
    Build & fit a background + sum(Gaussians) using guesses from find_peaks.
    Adds: qshift + qscale; broad Gaussian "shoulder"; robust loss; auto pair refits.
    """
    if len(peaks_idx) == 0:
        return None, {}

    dq = float(np.diff(q_limited).mean())

    # Background polynomial
    composite = PolynomialModel(degree=bg_degree, prefix="bg_")
    params = composite.make_params()
    params["bg_c0"].set(value=float(np.median(y_limited)), min=0)
    if bg_degree >= 1:
        params["bg_c1"].set(value=0)

    # --- very broad Gaussian "shoulder" to soak up gentle curvature ---
    broad = GaussianModel(prefix='broad_')
    composite += broad
    span = (q_limited.max() - q_limited.min())
    sigma_broad0 = max(1e-6, 0.20 * span)
    params.update(broad.make_params(center=float(np.mean(q_limited)),
                                    sigma=sigma_broad0,
                                    amplitude=0.05 * float(np.trapz(y_limited, q_limited))))
    params['broad_sigma'].set(min=0.5*sigma_broad0, max=3*sigma_broad0)
    params['broad_amplitude'].set(min=0, max=float(np.trapz(y_limited, q_limited)))

    # Peak props
    fwhm_pts    = props.get("widths", np.full_like(peaks_idx, 3.0, dtype=float))
    prominences = props.get("prominences", np.ones_like(peaks_idx, dtype=float))

    # Global q-axis transformations
    params.add('qshift', value=0.0, min=-5e-3, max=5e-3)
    params.add('qscale', value=1.0, min=0.999, max=1.001)

    # Add a Gaussian for each detected peak
    for i, pidx in enumerate(peaks_idx):
        center0 = float(q_limited[pidx])
        sigma0  = max(float(_sigma_from_fwhm(float(fwhm_pts[i]), dq)), sigma_floor)

        g = GaussianModel(prefix=f"g{i}_")
        composite += g

        height0 = float(prominences[i])
        amp0 = max(height0 * sigma0 * np.sqrt(2*np.pi), 1e-9)

        params.update(g.make_params(center=center0, sigma=sigma0, amplitude=amp0))

        # tie centers to qscale/qshift + small per-peak wiggle
        params.add(f"g{i}_c0", value=center0, vary=False)
        params.add(f"g{i}_dcenter", value=0.0, min=-0.001, max=0.001)
        params[f"g{i}_center"].set(expr=f"qscale*(g{i}_c0) + qshift + g{i}_dcenter")

        # sane bounds
        params[f"g{i}_sigma"].set(min=0.3*sigma0, max=3.0*sigma0)
        params[f"g{i}_amplitude"].set(min=0.25*abs(amp0), max=10*abs(amp0))

    # Robust global fit
    w = _poisson_weights(y_limited)
    result = composite.fit(
        y_limited, params, x=q_limited, weights=w,
        method="least_squares", fit_kws={"loss": "soft_l1", "f_scale": 1.0}
    )

    # Second pass: auto refit overlapping adjacent pairs
    _refit_overlapping_pairs(result, q_limited, y_limited, overlap_factor=2.5)

    comps = result.model.eval_components(params=result.params, x=q_limited)
    return result, comps


# ----------------------------- main routine ----------------------------- #
def peak_fit(h5_path, frame_number, peak_pos, window=0.1):
    with h5py.File(h5_path, "r") as f:
        int_val = f["int"][:]          # (nframes, q)
        q = f["q"][:]                  # (q,)
        cake_intensity_stack = f["cake_int"][:] if "cake_int" in f else None

    cake_slices = [0, 10, 19, 28]

    sigma_smooth = 50
    prom_cake = 2
    prom_full = 2
    center_window = 0.0005

    # window region
    q_min, q_max = peak_pos - window, peak_pos + window
    mask = (q >= q_min) & (q <= q_max)
    q_limited = q[mask]
    if q_limited.size < 5:
        raise ValueError("Fit window too small or outside q-range.")

    # ---- FIGURE LAYOUT ----
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), gridspec_kw={"width_ratios": [1, 1, 2]})
    fig.suptitle(f"Peak Fit for Frame {frame_number}", fontsize=16)

    # ---- FULL AZIMUTHAL ----
    y_full = int_val[frame_number, mask]
    # background for detection only
    background_full = gaussian_filter1d(y_full, sigma=sigma_smooth)
    data_bg_sub_full = y_full - background_full

# initial detection
    peaks_full, props_full = signal.find_peaks(
        data_bg_sub_full, prominence=prom_full, width=1, rel_height=0.5
        )

# --- NEW: iterative residual detection to catch shoulders / hidden peaks ---
    dq = float(np.diff(q_limited).mean())
    peaks_full, props_full = _augment_peaks_with_residual(
        q_limited, data_bg_sub_full, peaks_full, props_full, dq,
        prom_base=prom_full, add_frac=0.45,   # candidate threshold ~45% of base prom
        min_sep_sigma=0.8, max_iter=2         # tune if needed
    )

    print(f"[FULL] Seeds after augmentation: {len(peaks_full)} @", q_limited[peaks_full])
    print(f"[FULL] Detected {len(peaks_full)} peaks at q:", q_limited[peaks_full])

    result_full, comps_full = fit_multi_peaks(
        q_limited, y_full, peaks_full, props_full,
        bg_degree=1, center_window=center_window
    )

    # dense eval for smooth curves
    q_dense = np.linspace(q_limited.min(), q_limited.max(), len(q_limited) * 5)
    best_fit_dense = result_full.model.eval(params=result_full.params, x=q_dense)
    comps_dense = result_full.model.eval_components(params=result_full.params, x=q_dense)

    # Plot
    ax = axes[0, 2]
    ax.plot(q_limited, y_full, "--", label="Data")
    ax.plot(q_limited, background_full, "-", label="BG (pre-smooth)")
    ax.plot(q_limited[peaks_full], data_bg_sub_full[peaks_full], "x", label="Detected peaks")
    ax.plot(q_dense, best_fit_dense, "-", label="Total fit (dense)")
    for name in sorted([k for k in comps_dense.keys() if k.startswith('g')]):
        ax.plot(q_dense, comps_dense[name], ":", alpha=0.85, label=name)
    if "bg_" in comps_dense:
        ax.plot(q_dense, comps_dense["bg_"], "-", label="BG (fit)")
    if "broad_" in comps_dense:
        ax.plot(q_dense, comps_dense["broad_"], "--", alpha=0.7, label="Broad")

    print("\n[FULL] Fitted peak parameters:")
    for i in range(len(peaks_full)):
        c = result_full.params.get(f"g{i}_center").value
        s = result_full.params.get(f"g{i}_sigma").value
        a = result_full.params.get(f"g{i}_amplitude").value
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
        for (r, c) in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            axes[r, c].axis("off")
            axes[r, c].text(0.5, 0.5, "No cake_int in file", ha="center", va="center")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# ----------------------------- CLI ----------------------------- #
def _parse_args():
    p = argparse.ArgumentParser(description="Multi-peak fitting around a target q.")
    p.add_argument("h5", type=str, help="Input h5 file containing processed data.")
    p.add_argument("frame_number", type=int, help="Frame number to process (0-indexed).")
    p.add_argument("peak_pos", type=float, help="Center q to window around; peaks detected within this window.")
    p.add_argument("--window", type=float, default=0.1, help="Half-window size in q.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    peak_fit(args.h5, args.frame_number, args.peak_pos, args.window)
