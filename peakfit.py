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
    Minimal, robust fit: background poly + sum of Gaussians.
    Centers tied to global qshift+qscale with tiny per-peak wiggle.
    """
    if len(peaks_idx) == 0:
        return None, {}

    dq = float(np.diff(q).mean())

    # Background
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
        params.add(f"g{i}_dcenter", value=0.0, min=-0.001, max=0.001)
        params[f"g{i}_center"].set(expr=f"qscale*(g{i}_c0) + qshift + g{i}_dcenter")

        params[f"g{i}_sigma"].set(min=0.3 * sigma0, max=3.0 * sigma0)
        params[f"g{i}_amplitude"].set(min=0.2 * abs(amp0), max=10 * abs(amp0))

    # Robust global fit
    w = _poisson_weights(y)
    result = composite.fit(
        y, params, x=q, weights=w,
        method="least_squares", fit_kws={"loss": "soft_l1", "f_scale": 1.0}
    )

    comps = result.model.eval_components(params=result.params, x=q)
    return result, comps


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


def detect_background_adaptive(y_win, floor_pct=15, floor_win_frac=0.05,
                               floor_smooth=2, gauss_frac=0.03,
                               target_frac=0.60, max_iter=12):
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

    # binary search for alpha in [0,1] so frac(y <= alpha*gauss + (1-alpha)*floor) ≈ target_frac
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


# ----------------------------- main ----------------------------- #
def peak_fit(h5_path, frame_number, peak_pos, window=0.1, augment=False):
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

    # Figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), gridspec_kw={"width_ratios": [1, 1, 2]})
    fig.suptitle(f"Peak Fit for Frame {frame_number}", fontsize=16)

    # --- Detection (background-sub only) ---
    dq = float(np.diff(q_win).mean())

    # Adaptive baseline (floor + gaussian blend picked per frame)
    bg_detect, bg_info = detect_background_adaptive(
        y_win,
        floor_pct=15,       # try 10–20
        floor_win_frac=0.05,# try 0.04–0.07
        floor_smooth=2,
        gauss_frac=0.03,    # gaussian sigma ≈ 3% of window length
        target_frac=0.60    # ~60% of points under baseline
    )
    y_det = y_win - bg_detect
    print(f"[bg] alpha={bg_info['alpha']:.2f} frac_below={bg_info['frac_below']:.2f} "
          f"sg={bg_info['sg']} floor_pct={bg_info['floor_pct']} win_frac={bg_info['win_frac']}")

    # Adaptive thresholds for detection (handles melting)
    sig = _robust_sigma(y_det)
    K = 4.0  # noise multiplier for prominence; try 3.5–5.0 if needed
    prom_full = max(1.0, K * sig)

    w_guess_pts = 3.0
    wmin = max(1, int(0.6 * w_guess_pts))
    wmax = int(16 * w_guess_pts)  # allow broader peaks post-melt
    min_dist_pts = int(max(1, round(0.7 * w_guess_pts)))

    peaks, props = signal.find_peaks(
        y_det,
        prominence=prom_full,
        width=(wmin, wmax),
        distance=min_dist_pts,
        rel_height=0.5,
    )
    print(f"[detect] sig={sig:.3g} prom>={prom_full:.3g} width∈[{wmin},{wmax}] N={len(peaks)}")

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

    # Dense plotting
    q_dense = np.linspace(q_win.min(), q_win.max(), len(q_win) * 5)
    best_fit_dense = result.model.eval(params=result.params, x=q_dense)
    comps_dense = result.model.eval_components(params=result.params, x=q_dense)

    ax = axes[0, 2]
    ax.plot(q_win, y_win, "--", label="Data")
    ax.plot(q_win, bg_detect, "-", label="BG (detect)")
    ax.plot(q_win[peaks], y_det[peaks], "x", label="Detected peaks")
    ax.plot(q_dense, best_fit_dense, "-", label="Total fit (dense)")
    for name in sorted(k for k in comps_dense if k.startswith("g")):
        ax.plot(q_dense, comps_dense[name], ":", alpha=0.85, label=name)
    if "bg_" in comps_dense:
        ax.plot(q_dense, comps_dense["bg_"], "-", label="BG (fit)")

    ax.set_title("Full Azimuthal Integration")
    ax.set_xlabel("q"); ax.set_ylabel("Intensity")
    ax.legend(loc="upper right", fontsize=8)

    # --- Cake previews (simple) ---
    if cake is not None:
        slices = [0, 10, 19, 28]
        for i, cs in enumerate(slices):
            r, c = divmod(i, 2)
            y_c = cake[frame_number, cs, :][mask]
            # reuse same detect sigma for preview baseline
            sigma_smooth_detect = max(3, min(12, int(0.03 * len(q_win))))
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

    # Print compact params
    print("\n[FIT] Peaks:")
    for i in range(len(peaks)):
        c = result.params[f"g{i}_center"].value
        s = result.params[f"g{i}_sigma"].value
        a = result.params[f"g{i}_amplitude"].value
        print(f"  g{i}: center={c:.6f}, sigma={s:.6g}, area={a:.6g}")

    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.show()


# ----------------------------- CLI ----------------------------- #
def _parse_args():
    p = argparse.ArgumentParser(description="Lean multi-peak Gaussian fitting with adaptive detection baseline.")
    p.add_argument("h5", type=str)
    p.add_argument("frame_number", type=int)
    p.add_argument("peak_pos", type=float)
    p.add_argument("--window", type=float, default=0.1, help="Half-window in q.")
    p.add_argument("--augment", action="store_true", help="Enable one-pass residual augmentation.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    peak_fit(args.h5, args.frame_number, args.peak_pos, args.window, augment=args.augment)
