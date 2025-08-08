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
    return amp * np.exp(-0.5 * ((x - mu) / sig) ** 2)


def _build_crude_model(q, peaks_idx, props, dq, floor_amp=1e-9):
    widths = props.get("widths", np.full_like(peaks_idx, 3.0, dtype=float))
    prominences = props.get("prominences", np.ones_like(peaks_idx, dtype=float))
    yhat = np.zeros_like(q, dtype=float)
    for i, pidx in enumerate(peaks_idx):
        mu = float(q[pidx])
        sigma = max((widths[i] * dq) / 2.355, 1e-6)
        height = float(prominences[i])
        amp = max(height * sigma * np.sqrt(2 * np.pi), floor_amp)  # area for Gaussian
        yhat += _gaussian(q, mu, sigma, amp)
    return yhat

def _augment_peaks_with_residual(
    q,
    y_bgsub,
    init_idx,
    init_props,
    dq,
    prom_base=2.0,
    add_frac=0.45,
    min_sep_sigma=0.8,
    max_iter=2,
    # --- NEW knobs ---
    mad_k=3.5,              # residual must exceed k * noise (MAD) to be considered
    corr_min=0.6,           # min normalized correlation with a Gaussian template
    aic_drop=2.0,           # require AIC to drop by at least this to accept a new peak
    width_bounds_pts=(2, 25) # acceptable FWHM in sample points for residual peaks
):
    """
    Iteratively detect missed shoulders on the residual of a crude Gaussian sum.
    Adds robust gating: noise-aware threshold, shape correlation, and AIC test.
    """
    peaks_idx = np.array(init_idx, dtype=int)
    widths = np.array(
        init_props.get("widths", np.full_like(peaks_idx, 3.0, dtype=float)),
        dtype=float,
    )
    prominences = np.array(
        init_props.get("prominences", np.ones_like(peaks_idx, dtype=float)),
        dtype=float,
    )

    def _noise_mad(x):
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        # robust sigma ~ 1.4826 * MAD
        return 1.4826 * mad

    def _local_mask_away_from_peaks(q, centers, sigmas, span=3.0):
        if len(centers) == 0:
            return np.ones_like(q, dtype=bool)
        mask = np.ones_like(q, dtype=bool)
        for c, s in zip(centers, sigmas):
            mask &= ~((q > c - span*s) & (q < c + span*s))
        return mask

    def _gauss_template_corr(x, y, mu, sig):
        # build unit-energy Gaussian template on x, correlate with y in a local window
        t = np.exp(-0.5*((x - mu)/max(sig, 1e-6))**2)
        t /= np.sqrt((t**2).sum() + 1e-12)
        y0 = y - y.mean()
        den = np.sqrt((y0**2).sum()) + 1e-12
        return float((t * y0).sum() / den)

    def _aic(n, rss, k):
        # AIC = n*ln(rss/n) + 2k (constant terms ignored)
        rss = max(rss, 1e-18)
        return n * np.log(rss / max(n,1)) + 2 * k

    for _ in range(max_iter):
        # crude model from current seeds
        crude = _build_crude_model(q, peaks_idx, {"widths": widths, "prominences": prominences}, dq)
        residual = y_bgsub - crude

        # estimate noise from residual far from current peaks
        centers = q[peaks_idx] if peaks_idx.size else np.array([])
        sigmas_now = (widths * dq) / 2.355 if widths.size else np.array([])
        away = _local_mask_away_from_peaks(q, centers, sigmas_now, span=3.0)
        sigma_noise = _noise_mad(residual[away]) if away.any() else _noise_mad(residual)
        thr_abs = mad_k * sigma_noise

        # baseline thresholds for residual detection
        mean_w_pts = float(np.clip(np.nanmean(widths) if widths.size else 3.0, *width_bounds_pts))
        min_dist_pts = int(max(1, round(min_sep_sigma * mean_w_pts)))
        prom_add = max(0.3, add_frac) * prom_base

        cand_idx, cand_props = signal.find_peaks(
            residual, prominence=prom_add, width=1, distance=min_dist_pts, rel_height=0.5
        )

        new_idx, new_w, new_p = [], [], []
        for j, pidx in enumerate(cand_idx):
            qj = q[pidx]
            # spacing from existing peaks
            if peaks_idx.size and np.min(np.abs(q[peaks_idx] - qj)) < (min_dist_pts * dq):
                continue

            # residual height must exceed robust noise threshold
            if residual[pidx] < thr_abs:
                continue

            # width gate (from find_peaks width in pts)
            w_pts = float(cand_props["widths"][j])
            if not (width_bounds_pts[0] <= w_pts <= width_bounds_pts[1]):
                continue

            # local correlation with Gaussian template
            sig_j = max((w_pts * dq) / 2.355, 1e-6)
            # take a small local window
            loc = (q > qj - 3*sig_j) & (q < qj + 3*sig_j)
            if loc.sum() < 7:
                continue
            corr = _gauss_template_corr(q[loc], residual[loc], qj, sig_j)
            if corr < corr_min:
                continue

            # --------- AIC gate: does adding this peak help enough? ---------
            # Model0: crude in the local window
            y0 = residual[loc]  # residual we are trying to explain (pos bump)
            n = int(loc.sum())
            rss0 = float(np.sum((y0 - 0.0)**2))  # residual vs zero (since residual already subtracted crude)

            # one-Gaussian fit on local residual (area param only; center/width fixed)
            # best amp in LS sense for fixed mu,sigma is linear:
            gtpl = _gaussian(q[loc], qj, sig_j, 1.0)
            amp_opt = float(np.dot(gtpl, y0) / (np.dot(gtpl, gtpl) + 1e-12))
            y1 = amp_opt * gtpl
            rss1 = float(np.sum((y0 - y1)**2))

            aic0 = _aic(n, rss0, k=0)   # no extra params
            aic1 = _aic(n, rss1, k=1)   # +1 for amplitude
            if (aic0 - aic1) < aic_drop:
                continue
            # ---------------------------------------------------------------

            new_idx.append(pidx)
            new_w.append(w_pts)
            new_p.append(cand_props["prominences"][j])

        if not new_idx:
            break

        peaks_idx = np.concatenate([peaks_idx, np.array(new_idx, dtype=int)])
        widths = np.concatenate([widths, np.array(new_w, dtype=float)])
        prominences = np.concatenate([prominences, np.array(new_p, dtype=float)])

        order = np.argsort(q[peaks_idx])
        peaks_idx = peaks_idx[order]
        widths = widths[order]
        prominences = prominences[order]

    return peaks_idx, {"widths": widths, "prominences": prominences}


def _poisson_weights(y):
    return 1.0 / np.sqrt(np.clip(y, 1.0, None))


def _refit_overlapping_pairs(result, q, y, overlap_factor=2.5):
    """
    Automatically refit each adjacent peak pair that significantly overlaps.
    overlap_factor: separation threshold in units of avg sigma.
    """
    peak_ids = sorted(
        {int(k.split("_")[0][1:]) for k in result.params if k.startswith("g") and k.endswith("_center")}
    )
    if len(peak_ids) < 2:
        return 0

    centers = np.array([result.params[f"g{i}_center"].value for i in peak_ids])
    sigmas = np.array([result.params[f"g{i}_sigma"].value for i in peak_ids])

    ord_idx = np.argsort(centers)
    ids = np.array(peak_ids)[ord_idx]
    centers = centers[ord_idx]
    sigmas = sigmas[ord_idx]

    comps_all = result.model.eval_components(params=result.params, x=q)
    refined = 0

    for a, b in zip(range(len(ids) - 1), range(1, len(ids))):
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
            p[f"g{j}_sigma"].set(value=sj, min=0.5 * sj, max=2.0 * sj)
            p[f"g{j}_amplitude"].set(value=aj, min=0.3 * aj, max=5.0 * aj)

        w_local = _poisson_weights(y_local)
        refit = pair_model.fit(
            y_local,
            p,
            x=q_local,
            weights=w_local,
            method="least_squares",
            fit_kws={"loss": "soft_l1", "f_scale": 1.0},
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
    center_window=0.005,  # kept for compatibility (centers are tied via qshift/qscale+dcenter)
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

    # Very broad Gaussian "shoulder" to soak up gentle curvature
    broad = GaussianModel(prefix="broad_")
    composite += broad
    span = (q_limited.max() - q_limited.min())
    sigma_broad0 = max(1e-6, 0.20 * span)
    params.update(
        broad.make_params(
            center=float(np.mean(q_limited)),
            sigma=sigma_broad0,
            amplitude=0.05 * float(np.trapz(y_limited, q_limited)),
        )
    )
    params["broad_sigma"].set(min=0.5 * sigma_broad0, max=3 * sigma_broad0)
    params["broad_amplitude"].set(min=0, max=float(np.trapz(y_limited, q_limited)))

    # Peak props
    fwhm_pts = props.get("widths", np.full_like(peaks_idx, 3.0, dtype=float))
    prominences = props.get("prominences", np.ones_like(peaks_idx, dtype=float))

    # Global q-axis transformations
    params.add("qshift", value=0.0, min=-5e-3, max=5e-3)
    params.add("qscale", value=1.0, min=0.999, max=1.001)

    # Add a Gaussian for each detected peak
    for i, pidx in enumerate(peaks_idx):
        center0 = float(q_limited[pidx])
        sigma0 = max(float(_sigma_from_fwhm(float(fwhm_pts[i]), dq)), sigma_floor)

        g = GaussianModel(prefix=f"g{i}_")
        composite += g

        height0 = float(prominences[i])
        amp0 = max(height0 * sigma0 * np.sqrt(2 * np.pi), 1e-9)

        params.update(g.make_params(center=center0, sigma=sigma0, amplitude=amp0))

        # tie centers to qscale/qshift + small per-peak wiggle
        params.add(f"g{i}_c0", value=center0, vary=False)
        params.add(f"g{i}_dcenter", value=0.0, min=-0.001, max=0.001)
        params[f"g{i}_center"].set(expr=f"qscale*(g{i}_c0) + qshift + g{i}_dcenter")

        # sane bounds
        params[f"g{i}_sigma"].set(min=0.3 * sigma0, max=3.0 * sigma0)
        params[f"g{i}_amplitude"].set(min=0.25 * abs(amp0), max=10 * abs(amp0))

    # Robust global fit
    w = _poisson_weights(y_limited)
    result = composite.fit(
        y_limited,
        params,
        x=q_limited,
        weights=w,
        method="least_squares",
        fit_kws={"loss": "soft_l1", "f_scale": 1.0},
    )

    # Second pass: auto refit overlapping adjacent pairs
    _refit_overlapping_pairs(result, q_limited, y_limited, overlap_factor=2.5)

    comps = result.model.eval_components(params=result.params, x=q_limited)
    return result, comps


# ----------------------------- main routine ----------------------------- #
def peak_fit(h5_path, frame_number, peak_pos, window=0.1):
    with h5py.File(h5_path, "r") as f:
        int_val = f["int"][:]  # (nframes, q)
        q = f["q"][:]  # (q,)
        cake_intensity_stack = f["cake_int"][:] if "cake_int" in f else None

    cake_slices = [0, 10, 19, 28]

    # Tunables
    prom_cake = 2
    prom_full = 2
    center_window = 0.0005  # maintained but centers are tied via qshift/qscale+dcenter

    # Window region
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

    # Detection-only background (adaptive smoothing)
    sigma_smooth_detect = max(3, int(0.08 * len(q_limited)))  # ~8% of window length
    background_full = gaussian_filter1d(y_full, sigma=sigma_smooth_detect)
    data_bg_sub_full = y_full - background_full

    # Initial detection with minimum spacing
    dq = float(np.diff(q_limited).mean())
    w_pts_guess = 3.0
    min_dist_pts = int(max(1, round(0.8 * w_pts_guess)))

    peaks_full, props_full = signal.find_peaks(
        data_bg_sub_full,
        prominence=prom_full,
        width=1,
        distance=min_dist_pts,
        rel_height=0.5,
    )

    # Augment via crude-model residual (shoulders)
    peaks_full, props_full = _augment_peaks_with_residual(
        q_limited,
        data_bg_sub_full,
        peaks_full,
        props_full,
        dq,
        prom_base=prom_full,
        add_frac=0.45,
        min_sep_sigma=0.8,
        max_iter=2,
    )

    print(f"[FULL] after augmentation: {len(peaks_full)} peaks @ {q_limited[peaks_full]}")

    # Fit (global + automatic pair refits)
    result_full, comps_full = fit_multi_peaks(
        q_limited, y_full, peaks_full, props_full, bg_degree=1, center_window=center_window
    )

    # ---- Second-pass augmentation using the fitted model residual ----
    model_fit = result_full.model.eval(params=result_full.params, x=q_limited)
    residual2 = (y_full - background_full) - (model_fit - background_full)  # residual on bg-sub scale

    # detect small leftovers with lower threshold
    mean_w_pts = float(np.clip(np.nanmean(props_full.get("widths", [w_pts_guess])), 2.0, 20.0))
    min_dist_pts2 = int(max(1, round(0.8 * mean_w_pts)))
    cand2_idx, cand2_props = signal.find_peaks(
        residual2,
        prominence=max(0.35, 0.4) * prom_full,
        width=1,
        distance=min_dist_pts2,
        rel_height=0.5,
    )

    # Merge candidates not too close to existing peaks; then refit
    if cand2_idx.size:
        existing_q = q_limited[peaks_full] if len(peaks_full) else np.array([])
        keep = []
        for j, pidx in enumerate(cand2_idx):
            qj = q_limited[pidx]
            if existing_q.size and np.min(np.abs(existing_q - qj)) < (0.8 * min_dist_pts2 * dq):
                continue
            keep.append(j)

        if keep:
            peaks_full = np.concatenate([np.asarray(peaks_full, int), cand2_idx[keep].astype(int)])
            widths = np.concatenate([np.asarray(props_full.get("widths")), cand2_props["widths"][keep]])
            promin = np.concatenate([np.asarray(props_full.get("prominences")), cand2_props["prominences"][keep]])
            order = np.argsort(q_limited[peaks_full])
            peaks_full = peaks_full[order]
            props_full = {"widths": widths[order], "prominences": promin[order]}

            print(f"[FULL] +model residual added {len(keep)} new peaks. Re-fitting...")

            result_full, comps_full = fit_multi_peaks(
                q_limited, y_full, peaks_full, props_full, bg_degree=1, center_window=center_window
            )

    # ---- Dense evaluation for smooth curves ----
    q_dense = np.linspace(q_limited.min(), q_limited.max(), len(q_limited) * 5)
    best_fit_dense = result_full.model.eval(params=result_full.params, x=q_dense)
    comps_dense = result_full.model.eval_components(params=result_full.params, x=q_dense)

    # Plot
    ax = axes[0, 2]
    ax.plot(q_limited, y_full, "--", label="Data")
    ax.plot(q_limited, background_full, "-", label="BG (detect)")
    ax.plot(q_limited[peaks_full], data_bg_sub_full[peaks_full], "x", label="Detected peaks")
    ax.plot(q_dense, best_fit_dense, "-", label="Total fit (dense)")
    for name in sorted([k for k in comps_dense.keys() if k.startswith("g")]):
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

    # ---- CAKE SLICES (unchanged detection here; can add augmentation if desired) ----
    if cake_intensity_stack is not None:
        sigma_smooth_detect_cake = max(3, int(0.08 * len(q_limited)))
        for i, cs in enumerate(cake_slices):
            row, col = divmod(i, 2)
            y_cake = cake_intensity_stack[frame_number, cs, :][mask]
            bg_cake = gaussian_filter1d(y_cake, sigma=sigma_smooth_detect_cake)
            ysub = y_cake - bg_cake
            peaks_cake, props_cake = signal.find_peaks(ysub, prominence=prom_cake, width=1)
            axes[row, col].plot(q_limited, y_cake, "--", label="Cake data")
            axes[row, col].plot(q_limited, bg_cake, "-", label="BG (detect)")
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
