#!/usr/bin/env python3
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.ndimage import gaussian_filter1d  # for noise estimate
from lmfit.models import PolynomialModel, GaussianModel


# ===================== TUNABLE PEAK-FIT PARAMS =====================
# Set these to refine detection & fit behavior.
W_GUESS_PTS       = 3.0   # nominal FWHM in *points*
HP_SIGMA_MULT     = 2.0   # high-pass smoothing = HP_SIGMA_MULT * W_GUESS_PTS
K_PROM            = 4.0   # noise multiplier for prominence threshold
MIN_PROM_ABS_FRAC = 0.03  # 3% of local dynamic range as absolute floor
PROM_ABS_FLOOR    = 1.0   # absolute minimum prominence (counts)

WMIN_FRAC         = 0.6   # min width (points) = WMIN_FRAC * W_GUESS_PTS
WMAX_MULT         = 8.0   # max width (points) = WMAX_MULT * W_GUESS_PTS
MIN_DIST_FRAC     = 0.8   # min distance between peaks (points) = MIN_DIST_FRAC * W_GUESS_PTS

MERGE_FRAC        = 0.75  # merge peaks closer than MERGE_FRAC * W_GUESS_PTS (keep higher prom)
MAX_PEAKS         = 4     # hard cap on number of peaks to fit

SLOPE_LIMIT_SCALE = 2.0   # |bg_c1| <= SLOPE_LIMIT_SCALE * (yspan/qspan)

SIGMA_MIN_FRAC    = 0.20  # lower bound on sigma = SIGMA_MIN_FRAC * sigma0
SIGMA_MAX_MULT    = 3.0   # upper bound on sigma = SIGMA_MAX_MULT * sigma0
AMP_MIN_FRAC      = 0.10  # min amplitude = AMP_MIN_FRAC * initial amplitude
AMP_MAX_MULT      = 10.0  # max amplitude = AMP_MAX_MULT * initial amplitude

REFIT_TARGET_R2   = 0.98  # if R^2 below this, try one residual-based augment+refit
# ==================================================================


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


def _estimate_noise(y_det, hp_sigma_pts=6):
    """Estimate noise via high-pass residual: y_det - gaussian_smooth(y_det)."""
    if len(y_det) < 7:
        return _robust_sigma(y_det), y_det
    smooth = gaussian_filter1d(y_det, sigma=hp_sigma_pts)
    resid = y_det - smooth
    return _robust_sigma(resid), resid


def _prune_and_merge_peaks(q, y_det, peaks, props, dq,
                           w_guess_pts=W_GUESS_PTS, merge_frac=MERGE_FRAC,
                           max_peaks=MAX_PEAKS):
    """Merge too-close peaks, then cap by prominence."""
    if len(peaks) <= 1:
        return peaks, props

    order = np.argsort(peaks)
    peaks = peaks[order]
    widths = np.asarray(props.get("widths", np.full_like(peaks, w_guess_pts)), float)[order]
    promin = np.asarray(props.get("prominences", np.ones_like(peaks)), float)[order]

    keep = [0]
    for i in range(1, len(peaks)):
        prev = keep[-1]
        if (peaks[i] - peaks[prev]) < int(max(1, merge_frac * w_guess_pts)):
            # keep the more prominent
            if promin[i] > promin[prev]:
                keep[-1] = i
        else:
            keep.append(i)
    peaks = peaks[keep]
    widths = widths[keep]
    promin = promin[keep]

    if len(peaks) > max_peaks:
        idx = np.argsort(promin)[-max_peaks:]
        idx.sort()
        peaks = peaks[idx]
        widths = widths[idx]
        promin = promin[idx]

    return peaks, {"widths": widths, "prominences": promin}


def _fit_linear_only(q, y):
    """Fit only the linear background (used when no peaks are detected)."""
    bg = PolynomialModel(degree=1, prefix="bg_")
    p = bg.make_params()
    p["bg_c0"].set(value=float(np.median(y)), min=0)
    p["bg_c1"].set(value=0.0)
    w = _poisson_weights(y)
    res = bg.fit(y, p, x=q, weights=w, method="least_squares",
                 fit_kws={"loss": "soft_l1", "f_scale": 1.0})
    comps = res.model.eval_components(params=res.params, x=q)
    return res, comps


def _augment_once_residual(q, y_bgsub, peaks_idx, props, dq,
                           add_frac=0.45, min_sep_sigma=0.8):
    """Simple residual pass to add small peaks if reasonably separated."""
    if len(peaks_idx) == 0:
        return peaks_idx, props

    widths = np.array(props.get("widths", np.full_like(peaks_idx, W_GUESS_PTS, dtype=float)), float)
    promin = np.array(props.get("prominences", np.ones_like(peaks_idx, dtype=float)), float)

    # crude model of current peaks
    yhat = np.zeros_like(q, float)
    for i, pidx in enumerate(peaks_idx):
        mu = float(q[pidx])
        sig = max(_sigma_from_fwhm(widths[i], dq), 1e-6)
        amp = float(promin[i]) * sig * np.sqrt(2 * np.pi)
        yhat += amp * np.exp(-0.5 * ((q - mu) / sig) ** 2)

    residual = y_bgsub - yhat

    mean_w_pts = float(np.clip(np.nanmean(widths) if widths.size else W_GUESS_PTS, 2.0, 20.0))
    min_dist_pts = int(max(1, round(min_sep_sigma * mean_w_pts)))

    prom_base = float(np.median(promin)) if promin.size else 2.0
    prom_add = add_frac * max(0.5, prom_base)

    cand_idx, cand_props = signal.find_peaks(
        residual, prominence=prom_add, width=1, distance=min_dist_pts, rel_height=0.5
    )
    if cand_idx.size == 0:
        return peaks_idx, props

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


# ---------------- STRICT LINEAR BACKGROUND FOR FIT ---------------- #
def fit_multi_peaks(q, y, peaks_idx, props):
    """Linear background (degree=1) + sum of Gaussians. No higher-degree terms."""
    if len(peaks_idx) == 0:
        return None, {}

    dq = float(np.diff(q).mean())

    # Linear background with bounded slope
    composite = PolynomialModel(degree=1, prefix="bg_")
    params = composite.make_params()
    params["bg_c0"].set(value=float(np.median(y)), min=0)
    params["bg_c1"].set(value=0.0)

    qspan = max(q.max() - q.min(), 1e-9)
    yspan = max(np.percentile(y, 95) - np.percentile(y, 5), 1.0)
    smax = SLOPE_LIMIT_SCALE * yspan / qspan
    params["bg_c1"].set(min=-smax, max=+smax)

    fwhm_pts = np.asarray(props.get("widths", np.full_like(peaks_idx, W_GUESS_PTS, dtype=float)), float)
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
        dwig = max(3 * dq, 0.0018)
        params.add(f"g{i}_dcenter", value=0.0, min=-dwig, max=dwig)
        params[f"g{i}_center"].set(expr=f"qscale*(g{i}_c0) + qshift + g{i}_dcenter")

        # width & amplitude bounds (tighter than before)
        params[f"g{i}_sigma"].set(min=SIGMA_MIN_FRAC * sigma0, max=SIGMA_MAX_MULT * sigma0)
        params[f"g{i}_amplitude"].set(min=AMP_MIN_FRAC * abs(amp0), max=AMP_MAX_MULT * abs(amp0))

    # Global q-axis drift/scale for peak centers
    params.add("qshift", value=0.0, min=-5e-3, max=5e-3)
    params.add("qscale", value=1.0, min=0.999, max=1.001)

    # Fit with Poisson-like weights, robust loss
    w = _poisson_weights(y)
    result = composite.fit(y, params, x=q, weights=w,
                           method="least_squares",
                           fit_kws={"loss": "soft_l1", "f_scale": 1.0})
    comps = result.model.eval_components(params=result.params, x=q)
    return result, comps


# -------------- ROBUST LINEAR BASELINE FOR DETECTION -------------- #
def detect_background_linear(q_win, y_win, frac_keep=0.60, clip_sigma=2.5, max_iter=10):
    """
    Robust straight-line baseline used ONLY for detection & plotting.
    Iterative sigma-clipped OLS, keeping the lowest-residual fraction to
    avoid peaks pulling the line upward.
    """
    q = np.asarray(q_win, float)
    y = np.asarray(y_win, float)
    n = y.size
    if n < 3:
        a = float(np.median(y)); b = 0.0
        return a + b*q, {"a": a, "b": b, "iters": 0, "kept": n}

    mask = np.ones(n, dtype=bool)
    kept = n
    a = float(np.median(y)); b = 0.0
    for it in range(max_iter):
        X = np.vstack([np.ones(mask.sum()), q[mask]]).T
        beta, *_ = np.linalg.lstsq(X, y[mask], rcond=None)
        a, b = float(beta[0]), float(beta[1])
        bg = a + b*q
        resid = y - bg
        rsel = resid[mask]
        med = np.median(rsel)
        sig = 1.4826 * np.median(np.abs(rsel - med)) if rsel.size > 0 else 0.0
        if sig == 0.0:
            break
        below = y <= (bg + clip_sigma * sig)
        idx = np.where(below)[0]
        if idx.size < 5:
            break
        order = np.argsort(resid[idx])
        k = max(5, int(frac_keep * idx.size))
        newmask = np.zeros(n, dtype=bool)
        newmask[idx[order[:k]]] = True
        if newmask.sum() == mask.sum() and np.all(newmask == mask):
            kept = newmask.sum()
            mask = newmask
            break
        kept = newmask.sum()
        mask = newmask
    bg = a + b*q
    return bg, {"a": a, "b": b, "iters": it + 1, "kept": int(kept)}


def _robust_sigma(x):
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))


# ----------------------------- main ----------------------------- #
def peak_fit(h5_path, frame_number, peak_pos, window=0.1, augment=False, show_detect_bg=True):
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

    # Figure (2 rows x 3 columns; right column is wider)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), gridspec_kw={"width_ratios": [1, 1, 2]})
    fig.suptitle(f"Peak Fit for Frame {frame_number}", fontsize=16)

    # --- Strict linear baseline for detection ---
    dq = float(np.diff(q_win).mean())
    bg_detect, bg_info = detect_background_linear(q_win, y_win, frac_keep=0.60, clip_sigma=2.5, max_iter=10)
    y_det = y_win - bg_detect
    print(f"[bg-detect-linear] a={bg_info['a']:.3g}  b={bg_info['b']:.3g}  iters={bg_info['iters']}  kept={bg_info['kept']}")

    # --- Conservative detection on high-pass residual ---
    hp_sigma_pts = int(round(HP_SIGMA_MULT * W_GUESS_PTS))
    sig_hp, hp_resid = _estimate_noise(y_det, hp_sigma_pts=max(3, hp_sigma_pts))
    dyn = max(y_det.max() - y_det.min(), 1.0)
    prom_full = max(MIN_PROM_ABS_FRAC * dyn, K_PROM * sig_hp, PROM_ABS_FLOOR)

    wmin = max(1, int(WMIN_FRAC * W_GUESS_PTS))
    wmax = int(WMAX_MULT * W_GUESS_PTS)
    min_dist_pts = int(max(1, round(MIN_DIST_FRAC * W_GUESS_PTS)))

    peaks, props = signal.find_peaks(
        y_det,
        prominence=prom_full,
        width=(wmin, wmax),
        distance=min_dist_pts,
        rel_height=0.5,
    )

    peaks, props = _prune_and_merge_peaks(
        q_win, y_det, peaks, props, dq,
        w_guess_pts=W_GUESS_PTS, merge_frac=MERGE_FRAC, max_peaks=MAX_PEAKS
    )
    print(f"[detect] hp_sig={sig_hp:.3g} prom>={prom_full:.3g} width∈[{wmin},{wmax}] N={len(peaks)}")

    # If none, fit linear only and plot
    if len(peaks) == 0:
        result, comps = _fit_linear_only(q_win, y_win)
    else:
        # Optionally augment once from residuals (conservative seeds now)
        if augment:
            peaks_try, props_try = _augment_once_residual(
                q_win, y_det, peaks, props, dq, add_frac=0.40, min_sep_sigma=0.8
            )
            if len(peaks_try) > len(peaks):
                peaks, props = peaks_try, props_try
                print(f"[detect] after residual augment: {len(peaks)} @ {q_win[peaks]}")

        # --- Fit (strict linear BG) ---
        result, comps = fit_multi_peaks(q_win, y_win, peaks, props)

        # Optional one-pass refit if R² is low
        w = _poisson_weights(y_win)
        m = fit_metrics(result, q_win, y_win, weights=w)
        if m['r2'] < REFIT_TARGET_R2 and len(peaks) > 0:
            yhat = result.model.eval(params=result.params, x=q_win)
            resid_bg = (y_win - bg_detect) - (yhat - bg_detect)
            sigR = _robust_sigma(resid_bg)
            cand, cprops = signal.find_peaks(
                resid_bg, prominence=2.5 * sigR, width=1,
                distance=int(max(1, round(0.6 * W_GUESS_PTS)))
            )
            keep = []
            for j, pidx in enumerate(cand):
                if len(peaks) and np.min(np.abs(q_win[peaks] - q_win[pidx])) < (0.6 * W_GUESS_PTS * dq):
                    continue
                keep.append(j)
            if keep:
                peaks = np.concatenate([np.asarray(peaks, int), cand[keep].astype(int)])
                widths = np.concatenate([np.asarray(props.get("widths", np.full_like(peaks, W_GUESS_PTS))),
                                         cprops["widths"][keep]])
                promin = np.concatenate([np.asarray(props.get("prominences", np.ones_like(peaks))),
                                         cprops["prominences"][keep]])
                order = np.argsort(q_win[peaks]); peaks = peaks[order]
                props = {"widths": widths[order], "prominences": promin[order]}
                result, comps = fit_multi_peaks(q_win, y_win, peaks, props)

    # Metrics (final)
    w = _poisson_weights(y_win)
    m = fit_metrics(result, q_win, y_win, weights=w)
    print(f"[METRICS] R2={m['r2']:.6f}  adjR2={m['adj_r2']:.6f}  redχ²={m['red_chisq']:.3g}  "
          f"AIC={m['aic']:.2f}  BIC={m['bic']:.2f}  RMSE={m['rmse']:.3g}  max|res|={m['max_abs']:.3g}")

    # Dense plotting
    q_dense = np.linspace(q_win.min(), q_win.max(), len(q_win) * 5)
    best_fit_dense = result.model.eval(params=result.params, x=q_dense)
    comps_dense = result.model.eval_components(params=result.params, x=q_dense)

    # --- Right top: Full Azimuthal Integration ---
    ax = axes[0, 2]
    ax.plot(q_win, y_win, "--", label="Data")
    if show_detect_bg:
        ax.plot(q_win, bg_detect, "-", label="BG (detect, linear)")
    ax.plot(q_win[peaks] if len(peaks) else [], (y_det[peaks] if len(peaks) else []), "x", label="Detected peaks")
    ax.plot(q_dense, best_fit_dense, "-", label="Total fit (dense)")
    if "bg_" in comps_dense:
        ax.plot(q_dense, comps_dense["bg_"], "-", label="BG (fit, linear)")
    for name in sorted(k for k in comps_dense if k.startswith("g")):
        ax.plot(q_dense, comps_dense[name], ":", alpha=0.85, label=name)
    ax.set_title("Full Azimuthal Integration")
    ax.set_xlabel("q"); ax.set_ylabel("Intensity")
    ax.legend(loc="upper right", fontsize=8)

    # --- Left 2x2: Cake previews (linear baseline) ---
    if cake is not None:
        slices = [0, 10, 19, 28]
        for i, cs in enumerate(slices):
            r, c = divmod(i, 2)
            y_c = cake[frame_number, cs, :][mask]
            bg_c, _ = detect_background_linear(q_win, y_c, frac_keep=0.60, clip_sigma=2.5, max_iter=8)
            ysub = y_c - bg_c
            pk_c, _ = signal.find_peaks(ysub, prominence=prom_full, width=(wmin, wmax))
            axes[r, c].plot(q_win, y_c, "--", label="Cake data")
            axes[r, c].plot(q_win, bg_c, "-", label="BG (detect, linear)")
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
        ["# Peaks", f"{len([k for k in comps_dense if k.startswith('g')])}"],
        ["", ""],
    ]
    # Add centers if any peaks
    gkeys = sorted(k for k in comps_dense if k.startswith("g"))
    for i, kname in enumerate(gkeys):
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
    p = argparse.ArgumentParser(description="Multi-peak Gaussian fitting with strictly linear background.")
    p.add_argument("h5", type=str)
    p.add_argument("frame_number", type=int)
    p.add_argument("peak_pos", type=float)
    p.add_argument("--window", type=float, default=0.1, help="Half-window in q.")
    p.add_argument("--augment", action="store_true", help="Enable one-pass residual augmentation.")
    p.add_argument("--hide-detect-bg", action="store_true", help="Do not plot the detection baseline.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    peak_fit(args.h5, args.frame_number, args.peak_pos, args.window,
             augment=args.augment, show_detect_bg=not args.hide_detect_bg)
