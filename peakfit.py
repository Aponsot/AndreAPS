#!/usr/bin/env python3
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.ndimage import gaussian_filter1d, percentile_filter
from lmfit.models import PolynomialModel, GaussianModel


# ----------------------------- helpers ----------------------------- #
def _sigma_from_fwhm(fwhm_pts: float, dq: float) -> float:
    """FWHM -> sigma in x-units via FWHM = 2.355*sigma."""
    return (fwhm_pts * dq) / 2.355


def fit_metrics(result, x, y, weights=None):
    yhat = result.model.eval(params=result.params, x=x)
    resid = y - yhat
    n = len(y)
    k = sum(p.vary for p in result.params.values())  # #free params

    if weights is not None:
        w = np.asarray(weights, float)
        sse = float(np.sum((w * resid) ** 2))
        ybar = float(np.sum((w ** 2) * y) / np.sum(w ** 2))
        sst = float(np.sum((w * (y - ybar)) ** 2))
    else:
        sse = float(np.sum(resid ** 2))
        ybar = float(np.mean(y))
        sst = float(np.sum((y - ybar) ** 2))

    r2 = 1.0 - sse / max(sst, 1e-18)
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / max(n - k - 1, 1)

    red_chisq = sse / max(n - k, 1)
    aic = result.aic
    bic = result.bic
    rmse = np.sqrt(sse / n)
    max_abs = float(np.max(np.abs(resid)))

    return dict(r2=r2, adj_r2=adj_r2, red_chisq=red_chisq, aic=aic, bic=bic,
                rmse=rmse, max_abs=max_abs, n=n, k=k)


def _poisson_weights(y):
    return 1.0 / np.sqrt(np.clip(y, 1.0, None))


def _augment_once_residual(q, y_bgsub, peaks_idx, props, dq,
                           add_frac=0.45, min_sep_sigma=0.8):
    """
    One simple residual pass: build a crude Gaussian sum from seeds,
    detect extra small peaks on residual, merge if reasonably separated.
    """
    if len(peaks_idx) == 0:
        return peaks_idx, props

    widths = np.array(props.get("widths", np.full_like(peaks_idx, 3.0, dtype=float)), float)
    promin = np.array(props.get("prominences", np.ones_like(peaks_idx, dtype=float)), float)

    # crude model
    yhat = np.zeros_like(q, float)
    for i, pidx in enumerate(peaks_idx):
        mu = float(q[pidx])
        sig = max(_sigma_from_fwhm(widths[i], dq), 1e-6)
        amp = float(promin[i]) * sig * np.sqrt(2 * np.pi)  # area ≈ prominence * sigma * sqrt(2π)
        yhat += amp * np.exp(-0.5 * ((q - mu) / sig) ** 2)

    residual = y_bgsub - yhat

    mean_w_pts = float(np.clip(np.nanmean(widths) if widths.size else 3.0, 2.0, 20.0))
    min_dist_pts = int(max(1, round(min_sep_sigma * mean_w_pts)))

    prom_base = float(np.median(promin)) if promin.size else 2.0
    prom_add = add_frac * max(0.5, prom_base)

    cand_idx, cand_props = signal.find_peaks(
        residual, prominence=prom_add, width=1, distance=min_dist_pts, rel_height=0.5
    )

    if cand_idx.size == 0:
        return peaks_idx, props

    # keep candidates not too close to existing ones
    keep = []
    for j, pidx in enumerate(cand_idx):
        qj = q[pidx]
        if len(peaks_idx) and np.min(np.abs(q[peaks_idx] - qj)) < (min_dist_pts * dq):
            continue
        keep.append(j)

    if not keep:
        return peaks_idx, props

    new_idx = cand_idx[keep].astype(int)
    new_w = cand_props["widths"][keep]
    new_p = cand_props["prominences"][keep]

    peaks_idx = np.concatenate([np.asarray(peaks_idx, int), new_idx])
    widths = np.concatenate([widths, np.asarray(new_w, float)])
    promin = np.concatenate([promin, np.asarray(new_p, float)])

    order = np.argsort(q[peaks_idx])
    return peaks_idx[order], {"widths": widths[order], "prominences": promin[order]}


def fit_multi_peaks(q, y, peaks_idx, props, bg_degree=1):
    """
    Robust fit: background poly + sum of Gaussians.
    Centers tied to global qshift+qscale with small per-peak wiggle.
    Also tries quadratic background and keeps it if AIC improves.
    """
    if len(peaks_idx) == 0:
        return None, {}

    dq = float(np.diff(q).mean())

    # Background (linear by default)
    composite = PolynomialModel(degree=bg_degree, prefix="bg_")
    params = composite.make_params()
    params["bg_c0"].set(value=float(np.median(y)), min=0)
    if bg_degree >= 1:
        params["bg_c1"].set(value=0)

    # Global q-axis drift/scale
    params.add("qshift", value=0.0, min=-5e-3, max=5e-3)
    params.add("qscale", value=1.0, min=0.999, max=1.001)

    fwhm_pts = np.asarray(props.get("widths", np.full_like(peaks_idx, 3.0, dtype=float)), float)
    prominences = np.asarray(props.get("prominences", np.ones_like(peaks_idx, dtype=float)), float)

    for i, pidx in enumerate(peaks_idx):
        center0 = float(q[pidx])
        sigma0 = max(_sigma_from_fwhm(float(fwhm_pts[i]), dq), 1e-6)

        g = GaussianModel(prefix=f"g{i}_")
        composite += g

        height0 = float(prominences[i])
        amp0 = max(height0 * sigma0 * np.sqrt(2 * np.pi), 1e-9)  # area

        params.update(g.make_params(center=center0, sigma=sigma0, amplitude=amp0))
        # tie: center = qscale*base + qshift + tiny per-peak wiggle
        params.add(f"g{i}_c0", value=center0, vary=False)

        # widen the wiggle to ~3 samples or ~0.0018, whichever larger
        dwig = max(3 * dq, 0.0018)
        params.add(f"g{i}_dcenter", value=0.0, min=-dwig, max=dwig)
        params[f"g{i}_center"].set(expr=f"qscale*(g{i}_c0) + qshift + g{i}_dcenter")

        # looser width bounds for melted frames
        params[f"g{i}_sigma"].set(min=0.25 * sigma0, max=4.0 * sigma0)
        params[f"g{i}_amplitude"].set(min=0.2 * abs(amp0), max=10 * abs(amp0))

    # Robust global fit (linear bg)
    w = _poisson_weights(y)
    result = composite.fit(
        y, params, x=q, weights=w,
        method="least_squares", fit_kws={"loss": "soft_l1", "f_scale": 1.0}
    )
    comps = result.model.eval_components(params=result.params, x=q)

    # --- Try quadratic background; keep if AIC improves meaningfully ---
    best_result, best_comps, best_aic = result, comps, result.aic
    if bg_degree == 1:
        comp2 = PolynomialModel(degree=2, prefix="bg2_")
        m2 = comp2
        p2 = comp2.make_params()
        p2["bg2_c0"].set(value=float(np.median(y)), min=0)
        p2["bg2_c1"].set(value=0)
        p2["bg2_c2"].set(value=0)

        # Re-add Gaussians seeded from the linear-bg solution
        for i, _ in enumerate(peaks_idx):
            g = GaussianModel(prefix=f"g{i}_")
            m2 += g
            p2.update(g.make_params(
                center=best_result.params[f"g{i}_center"].value,
                sigma=best_result.params[f"g{i}_sigma"].value,
                amplitude=best_result.params[f"g{i}_amplitude"].value
            ))
            # keep center ties and bounds
            p2.add(f"g{i}_c0", value=best_result.params[f"g{i}_c0"].value, vary=False)
            p2.add(f"g{i}_dcenter",
                   value=best_result.params[f"g{i}_dcenter"].value,
                   min=best_result.params[f"g{i}_dcenter"].min,
                   max=best_result.params[f"g{i}_dcenter"].max)
            p2[f"g{i}_center"].set(expr=f"qscale*(g{i}_c0) + qshift + g{i}_dcenter")

        # global transforms
        p2.add("qshift", value=best_result.params["qshift"].value,
               min=best_result.params["qshift"].min, max=best_result.params["qshift"].max)
        p2.add("qscale", value=best_result.params["qscale"].value,
               min=best_result.params["qscale"].min, max=best_result.params["qscale"].max)

        r2 = m2.fit(y, p2, x=q, weights=w,
                    method="least_squares", fit_kws={"loss": "soft_l1", "f_scale": 1.0})
        if (best_aic - r2.aic) > 2.0:  # worth the extra term
            best_result = r2
            best_comps = m2.eval_components(params=r2.params, x=q)

    return best_result, best_comps


# --- detection background helpers --- #
def detect_background_floor(y_win, pct=15, win_frac=0.05, smooth_sigma=2):
    """
    Percentile-floor baseline for PEAK DETECTION ONLY.
    pct: lower percentile (smaller => flatter).
    win_frac: window size as fraction of length.
    smooth_sigma: tiny Gaussian smooth to de-block the percentile output.
    """
    n = len(y_win)
    win = max(7, int(win_frac * n) | 1)  # odd, >=7
    bg = percentile_filter(y_win, percentile=pct, size=win)
    if smooth_sigma and smooth_sigma > 0:
        bg = gaussian_filter1d(bg, sigma=smooth_sigma)
    return bg


def detect_background_adaptive(y_win, floor_pct=14, floor_win_frac=0.05,
                               floor_smooth=2, gauss_frac=0.03,
                               target_frac=0.55, max_iter=12):
    """
    Blend between floor and Gaussian so that ~target_frac of points lie
    BELOW the baseline (keeps a true floor while allowing mild slope post-melt).
    """
    n = len(y_win)
    # floor
    bg_floor = detect_background_floor(y_win, pct=floor_pct,
                                       win_frac=floor_win_frac,
                                       smooth_sigma=floor_smooth)
    # gentle Gaussian
    sg = max(3, min(12, int(gauss_frac * n)))
    bg_gauss = gaussian_filter1d(y_win, sigma=sg)

    # binary search for alpha in [0,1]
    lo, hi = 0.0, 1.0
    for _ in range(max_iter):
        alpha = 0.5 * (lo + hi)
        bg = alpha * bg_gauss + (1 - alpha) * bg_floor
        frac_below = float(np.mean(y_win <= bg))
        if frac_below < target_frac:   # baseline too low -> raise toward gauss
            lo = alpha
        else:                          # baseline too high -> lower toward floor
            hi = alpha
    alpha = 0.5 * (lo + hi)
    bg = alpha * bg_gauss + (1 - alpha) * bg_floor
    return bg, dict(alpha=alpha, frac_below=float(np.mean(y_win <= bg)),
                    sg=sg, floor_pct=floor_pct, win_frac=floor_win_frac)


def _robust_sigma(x):
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))


# -------- shoulder seeding via derivatives (NEW) -------- #
def _derivative_shoulder_seeds(q, y_det, dq, w_guess_pts, min_dist_pts, existing_peaks):
    """
    Seed extra components using slope/curvature cues:
    - y1: first derivative (large |y1| can indicate shoulder edges)
    - y2: second derivative (strong negative curvature points)
    Returns indices for additional seeds that are not near existing peaks.
    """
    n = len(y_det)
    if n < 7:
        return np.array([], dtype=int), {}

    # Savitzky–Golay window for derivatives (odd, <= n)
    wlen = max(7, int(round(2.5 * w_guess_pts)) | 1)
    max_wlen = n if (n % 2 == 1) else (n - 1)
    wlen = min(wlen, max_wlen)
    if wlen < 5:
        return np.array([], dtype=int), {}

    y1 = signal.savgol_filter(y_det, window_length=wlen, polyorder=3, deriv=1, delta=dq, mode="interp")
    y2 = signal.savgol_filter(y_det, window_length=wlen, polyorder=3, deriv=2, delta=dq, mode="interp")

    sig1 = _robust_sigma(y1)
    sig2 = _robust_sigma(y2)
    # slope extremes and curvature minima
    slope_idx, _ = signal.find_peaks(np.abs(y1), prominence=max(4.0*sig1, 1e-12), distance=min_dist_pts)
    curv_idx,  _ = signal.find_peaks(-y2,    prominence=max(3.5*sig2, 1e-12), distance=min_dist_pts)
    # union
    cand = np.unique(np.concatenate([slope_idx, curv_idx]))
    if cand.size == 0:
        return cand.astype(int), {"wlen": wlen, "sig1": sig1, "sig2": sig2, "n_cand": 0}

    # Filter: keep only where intensity is positive enough and not too close to existing peaks
    ysig = _robust_sigma(y_det)
    thr = max(3.5 * ysig, 0.25 * float(np.max(y_det)) if np.max(y_det) > 0 else 0.0)

    keep = []
    for idx in cand:
        # intensity threshold
        if y_det[idx] < thr:
            continue
        # not too close to an existing max-peak
        if len(existing_peaks) and np.min(np.abs(q[existing_peaks] - q[idx])) < (1.0*w_guess_pts*dq):
            continue
        keep.append(idx)

    out = np.array(keep, dtype=int)
    return out, {"wlen": wlen, "sig1": sig1, "sig2": sig2, "n_cand": len(cand), "n_kept": len(out)}


# ----------------------------- main ----------------------------- #
def peak_fit(h5_path, frame_number, peak_pos, window=0.1, augment=False, use_derivatives=True):
    with h5py.File(h5_path, "r") as f:
        Int = f["int"][:]  # (nframes, q)
        q = f["q"][:]      # (q,)
        cake = f["cake_int"][:] if "cake_int" in f else None

    # Window
    q_min, q_max = peak_pos - window, peak_pos + window
    mask = (q >= q_min) & (q <= q_max)
    q_win = q[mask]
    y_win = Int[frame_number, mask]
    if q_win.size < 5:
        raise ValueError("Fit window too small or outside q-range.")

    # Figure (2 rows x 3 columns; right column is wider by width_ratios)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), gridspec_kw={"width_ratios": [1, 1, 2]})
    fig.suptitle(f"Peak Fit for Frame {frame_number}", fontsize=16)

    # --- Detection (background-sub only) ---
    dq = float(np.diff(q_win).mean())

    # Adaptive baseline (floor + gaussian blend picked per frame)
    bg_detect, bg_info = detect_background_adaptive(
        y_win,
        floor_pct=14,
        floor_win_frac=0.05,
        floor_smooth=2,
        gauss_frac=0.03,
        target_frac=0.55
    )
    y_det = y_win - bg_detect
    print(f"[bg] alpha={bg_info['alpha']:.2f} frac_below={bg_info['frac_below']:.2f} "
          f"sg={bg_info['sg']} floor_pct={bg_info['floor_pct']} win_frac={bg_info['win_frac']}")

    # Adaptive thresholds for detection (handles melting)
    sig = _robust_sigma(y_det)
    K = 3.6  # slightly easier picks post-melt
    prom_full = max(1.0, K * sig)

    w_guess_pts = 3.0
    wmin = max(1, int(0.6 * w_guess_pts))
    wmax = int(20 * w_guess_pts)  # allow broader peaks post-melt
    min_dist_pts = int(max(1, round(0.6 * w_guess_pts)))

    # Primary maxima
    peaks, props = signal.find_peaks(
        y_det,
        prominence=prom_full,
        width=(wmin, wmax),
        distance=min_dist_pts,
        rel_height=0.5,
    )
    print(f"[detect] primary maxima: sig={sig:.3g} prom>={prom_full:.3g} width∈[{wmin},{wmax}] N={len(peaks)}")

    # ---- NEW: derivative-based shoulder seeds for initial fit ----
    deriv_idx = np.array([], dtype=int)
    if use_derivatives:
        deriv_idx, dinfo = _derivative_shoulder_seeds(q_win, y_det, dq, w_guess_pts, min_dist_pts, peaks)
        if deriv_idx.size:
            print(f"[deriv] kept {dinfo['n_kept']}/{dinfo['n_cand']} derivative seeds "
                  f"(wlen={dinfo['wlen']} sig1={dinfo['sig1']:.3g} sig2={dinfo['sig2']:.3g})")
            # Merge derivative seeds into peaks & props
            peaks = np.concatenate([peaks.astype(int), deriv_idx.astype(int)])
            # widths: no strong estimate for shoulders -> seed with w_guess_pts
            widths = np.asarray(props.get("widths", np.full(0, w_guess_pts)), float)
            promin = np.asarray(props.get("prominences", np.full(0, max(1.0, 2.5 * sig))), float)

            new_w = np.full(deriv_idx.size, w_guess_pts, dtype=float)

            # simple local prominence proxy for seeds (shoulders may not be true maxima)
            def _local_prom(i, L=int(round(2.0 * w_guess_pts))):
                lo = max(0, i - L); hi = min(len(y_det), i + L + 1)
                baseline = np.min(y_det[lo:hi]) if hi > lo else 0.0
                return max(0.0, float(y_det[i] - baseline))

            new_p = np.array([_local_prom(i) for i in deriv_idx], dtype=float)
            new_p[new_p <= 0] = max(1.0, 2.5 * sig)

            if widths.size:
                widths = np.concatenate([widths, new_w])
                promin = np.concatenate([promin, new_p])
            else:
                widths = new_w
                promin = new_p

            order = np.argsort(q_win[peaks])
            peaks = peaks[order]
            props = {"widths": widths[order], "prominences": promin[order]}
        else:
            print("[deriv] no derivative-based seeds added.")

    # Optionally augment once from residuals
    if augment and len(peaks) > 0:
        peaks_try, props_try = _augment_once_residual(
            q_win, y_det, peaks, props, dq, add_frac=0.40, min_sep_sigma=0.8
        )
        if len(peaks_try) > len(peaks):
            peaks, props = peaks_try, props_try
            print(f"[detect] after residual augment: {len(peaks)} @ {q_win[peaks]}")

    # --- Fit ---
    result, comps = fit_multi_peaks(q_win, y_win, peaks, props, bg_degree=1)
    w = 1.0 / np.sqrt(np.clip(y_win, 1.0, None))  # same weights used in fit
    m = fit_metrics(result, q_win, y_win, weights=w)
    print(f"[METRICS] R2={m['r2']:.6f}  adjR2={m['adj_r2']:.6f}  redχ²={m['red_chisq']:.3g}  "
          f"AIC={m['aic']:.2f}  BIC={m['bic']:.2f}  RMSE={m['rmse']:.3g}  max|res|={m['max_abs']:.3g}")

    # --- If R² still low, try a tiny residual add & refit once ---
    TARGET = 0.98
    if m['r2'] < TARGET and len(peaks) > 0:
        yhat = result.model.eval(params=result.params, x=q_win)
        resid_bg = (y_win - bg_detect) - (yhat - bg_detect)
        sigR = _robust_sigma(resid_bg)
        cand, cprops = signal.find_peaks(
            resid_bg, prominence=2.5 * sigR, width=1,
            distance=int(max(1, round(0.6 * w_guess_pts)))
        )
        keep = []
        for j, pidx in enumerate(cand):
            if len(peaks) and np.min(np.abs(q_win[peaks] - q_win[pidx])) < (0.6 * w_guess_pts * dq):
                continue
            keep.append(j)
        if keep:
            peaks = np.concatenate([np.asarray(peaks, int), cand[keep].astype(int)])
            widths = np.concatenate([np.asarray(props.get("widths", np.full_like(peaks, 3.0))),
                                     cprops["widths"][keep]])
            promin = np.concatenate([np.asarray(props.get("prominences", np.ones_like(peaks))),
                                     cprops["prominences"][keep]])
            order = np.argsort(q_win[peaks]); peaks = peaks[order]
            props = {"widths": widths[order], "prominences": promin[order]}
            result, comps = fit_multi_peaks(q_win, y_win, peaks, props, bg_degree=1)
            m = fit_metrics(result, q_win, y_win, weights=_poisson_weights(y_win))
            print(f"[refit] R2→{m['r2']:.6f}")

    # Dense plotting
    q_dense = np.linspace(q_win.min(), q_win.max(), len(q_win) * 5)
    best_fit_dense = result.model.eval(params=result.params, x=q_dense)
    comps_dense = result.model.eval_components(params=result.params, x=q_dense)

    # --- Right top: Full Azimuthal Integration ---
    ax = axes[0, 2]
    ax.plot(q_win, y_win, "--", label="Data")
    ax.plot(q_win, bg_detect, "-", label="BG (detect)")
    ax.plot(q_win[peaks], y_det[peaks], "x", label="Detected peaks")
    if use_derivatives and deriv_idx.size:
        ax.plot(q_win[deriv_idx], y_det[deriv_idx], "^", ms=7, label="Derivative seeds")
    ax.plot(q_dense, best_fit_dense, "-", label="Total fit (dense)")
    for name in sorted(k for k in comps_dense if k.startswith("g")):
        ax.plot(q_dense, comps_dense[name], ":", alpha=0.85, label=name)
    if "bg_" in comps_dense:
        ax.plot(q_dense, comps_dense["bg_"], "-", label="BG (fit)")

    ax.set_title("Full Azimuthal Integration")
    ax.set_xlabel("q"); ax.set_ylabel("Intensity")
    ax.legend(loc="upper right", fontsize=8)

    # --- Left 2x2: Cake previews (simple) ---
    if cake is not None:
        slices = [0, 10, 19, 28]
        sigma_smooth_detect = max(3, min(12, int(0.03 * len(q_win))))  # for preview only
        for i, cs in enumerate(slices):
            r, c = divmod(i, 2)
            y_c = cake[frame_number, cs, :][mask]
            bg_c = gaussian_filter1d(y_c, sigma=sigma_smooth_detect)
            ysub = y_c - bg_c
            pk_c, _ = signal.find_peaks(ysub, prominence=prom_full, width=(wmin, wmax))
            axes[r, c].plot(q_win, y_c, "--", label="Cake data")
            axes[r, c].plot(q_win, bg_c, "-", label="BG (detect)")
            axes[r, c].plot(q_win[pk_c], ysub[pk_c], "x", label="Peaks")
            axes[r, c].set_title(f"Cake slice {cs}")
            axes[r, c].set_xlabel("q"); axes[r, c].set_ylabel("Intensity")
            axes[r, c].legend(loc="upper right", fontsize=8)
    else:
        for (r, c) in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            axes[r, c].axis("off"); axes[r, c].text(0.5, 0.5, "No cake_int", ha="center", va="center")

    # --- Bottom-right: Table with metrics + peak centers ---
    table_ax = axes[1, 2]
    table_ax.axis("off")

    table_data = [
        ["R²", f"{m['r2']:.6f}"],
        ["Adj R²", f"{m['adj_r2']:.6f}"],
        ["Reduced χ²", f"{m['red_chisq']:.3g}"],
        ["AIC", f"{m['aic']:.2f}"],
        ["BIC", f"{m['bic']:.2f}"],
        ["RMSE", f"{m['rmse']:.3g}"],
        ["Max |res|", f"{m['max_abs']:.3g}"],
        ["# Peaks", f"{len(peaks)}"],
        ["", ""],  # spacer
    ]
    for i in range(len(peaks)):
        c = result.params[f"g{i}_center"].value
        table_data.append([f"Peak {i} center", f"{c:.6f}"])

    table = table_ax.table(cellText=table_data,
                           colLabels=["Metric / Peak", "Value"],
                           cellLoc="center",
                           loc="center", fontsize=12)
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.25)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# ----------------------------- CLI ----------------------------- #
def _parse_args():
    p = argparse.ArgumentParser(description="Lean multi-peak Gaussian fitting with derivative-based shoulder seeding.")
    p.add_argument("h5", type=str)
    p.add_argument("frame_number", type=int)
    p.add_argument("peak_pos", type=float)
    p.add_argument("--window", type=float, default=0.1, help="Half-window in q.")
    p.add_argument("--augment", action="store_true", help="Enable one-pass residual augmentation.")
    p.add_argument("--no-deriv", action="store_true", help="Disable derivative-based shoulder seeding.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    peak_fit(args.h5, args.frame_number, args.peak_pos, args.window,
             augment=args.augment)
