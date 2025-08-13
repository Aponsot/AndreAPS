#!/usr/bin/env python3
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.ndimage import gaussian_filter1d, percentile_filter
from lmfit.models import PolynomialModel, GaussianModel
import warnings

# ----------------------------- TUNABLES ----------------------------- #
MIN_WIDTH_PTS = 2          # absolute min peak width (in sample points)
MAX_WIDTH_PTS = 25         # absolute max peak width (in sample points)  <-- caps “chonky” peaks
WIGGLE_SAMPLES = 3         # allowed center wiggle in samples
K_SIGMA = 3.6              # detection threshold multiplier
TARGET_R2 = 0.98           # optional refit trigger

# Quiet common warnings (we also disable covariance below)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="lmfit")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="uncertainties")

# ----------------------------- helpers ----------------------------- #
def _sigma_from_fwhm(fwhm_pts: float, dq: float) -> float:
    """FWHM -> sigma in x-units via FWHM = 2.355*sigma."""
    return (fwhm_pts * dq) / 2.355

def fit_metrics(result, x, y, weights=None):
    yhat = result.model.eval(params=result.params, x=x)
    resid = y - yhat
    n = len(y)
    # count only params that actually vary and appear in the model
    k = sum(p.vary for p in result.params.values() if p in result.eval_components(x=x, **result.best_values))

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
    # AIC/BIC from lmfit are fine even with calc_covar=False
    aic = result.aic
    bic = result.bic
    rmse = np.sqrt(sse / n)
    max_abs = float(np.max(np.abs(resid)))

    return dict(r2=r2, adj_r2=adj_r2, red_chisq=red_chisq, aic=aic, bic=bic,
                rmse=rmse, max_abs=max_abs, n=n, k=k)

def _poisson_weights(y):
    return 1.0 / np.sqrt(np.clip(y, 1.0, None))

def _robust_sigma(x):
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))

# --- strict linear background for detection (sigma-clipped LSQ) ---
def _linear_bg_strict(q, y, frac_keep=0.60, clip_sigma=2.5, max_iter=10):
    q = np.asarray(q, float); y = np.asarray(y, float); n = y.size
    if n < 3:
        a = float(np.median(y)); b = 0.0
        return a + b*q, dict(a=a, b=b, kept=n, iters=0)
    mask = np.ones(n, bool); a = float(np.median(y)); b = 0.0; kept = n
    for it in range(max_iter):
        X = np.vstack([np.ones(mask.sum()), q[mask]]).T
        beta, *_ = np.linalg.lstsq(X, y[mask], rcond=None)
        a, b = float(beta[0]), float(beta[1])
        bg = a + b*q
        resid = y - bg
        rsel = resid[mask]
        sig = 1.4826 * np.median(np.abs(rsel - np.median(rsel))) if rsel.size else 0.0
        if sig == 0.0: break
        below = y <= (bg + clip_sigma*sig)
        idx = np.where(below)[0]
        if idx.size < 5: break
        order = np.argsort(resid[idx])
        k = max(5, int(frac_keep * idx.size))
        newmask = np.zeros(n, bool); newmask[idx[order[:k]]] = True
        if newmask.sum()==mask.sum() and np.all(newmask==mask):
            kept = newmask.sum(); break
        mask = newmask; kept = mask.sum()
    return a + b*q, dict(a=a, b=b, kept=kept, iters=it+1)

def fit_multi_peaks(q, y, peaks_idx, props, bg_degree=1, dq=None):
    """
    Linear background + sum of Gaussians (no derivative seeding).
    Centers tied to global qshift+qscale with small per-peak wiggle.
    """
    dq = float(np.diff(q).mean()) if dq is None else float(dq)

    # Background
    composite = PolynomialModel(degree=bg_degree, prefix="bg_")
    params = composite.make_params()
    params["bg_c0"].set(value=float(np.median(y)), min=0.0)
    if bg_degree >= 1:
        params["bg_c1"].set(value=0.0)

    # Add Gaussians if any peaks
    if len(peaks_idx) > 0:
        # global transforms (only when peaks exist)
        params.add("qshift", value=0.0, min=-5e-3, max=5e-3)
        params.add("qscale", value=1.0, min=0.999, max=1.001)

        fwhm_pts = np.asarray(props.get("widths", np.full_like(peaks_idx, 3.0, dtype=float)), float)
        prominences = np.asarray(props.get("prominences", np.ones_like(peaks_idx, dtype=float)), float)

        # absolute sigma bounds from global limits
        sig_min_abs = _sigma_from_fwhm(MIN_WIDTH_PTS, dq)
        sig_max_abs = _sigma_from_fwhm(MAX_WIDTH_PTS, dq)

        for i, pidx in enumerate(peaks_idx):
            center0 = float(q[pidx])

            # clamp initial sigma to absolute bounds
            sigma0 = max(_sigma_from_fwhm(float(fwhm_pts[i]), dq), 1e-9)
            sigma0 = min(max(sigma0, sig_min_abs), sig_max_abs)

            g = GaussianModel(prefix=f"g{i}_")
            composite += g

            height0 = float(prominences[i])
            amp0 = max(height0 * sigma0 * np.sqrt(2 * np.pi), 1e-9)  # area

            params.update(g.make_params(center=center0, sigma=sigma0, amplitude=amp0))
            params.add(f"g{i}_c0", value=center0, vary=False)

            # wiggle in x
            dwig = max(WIGGLE_SAMPLES * dq, 0.0018)
            params.add(f"g{i}_dcenter", value=0.0, min=-dwig, max=dwig)
            params[f"g{i}_center"].set(expr=f"qscale*(g{i}_c0) + qshift + g{i}_dcenter")

            # width/amplitude bounds: intersect relative and absolute caps
            params[f"g{i}_sigma"].set(min=max(0.25 * sigma0, sig_min_abs),
                                      max=min(4.0 * sigma0,  sig_max_abs))
            params[f"g{i}_amplitude"].set(min=0.2 * abs(amp0), max=10 * abs(amp0))

    # Fit (disable covariance to avoid warnings when model is light/degenerate)
    w = _poisson_weights(y)
    result = composite.fit(
        y, params, x=q, weights=w,
        method="least_squares", fit_kws={"loss": "soft_l1", "f_scale": 1.0},
        calc_covar=False
    )
    comps = result.model.eval_components(params=result.params, x=q)
    return result, comps

# ----------------------------- main ----------------------------- #
def peak_fit(h5_path, frame_number, peak_pos, window=0.1):
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

    # --- Detection (background-sub only, STRICT LINEAR) ---
    dq = float(np.diff(q_win).mean())
    bg_detect, _bginfo = _linear_bg_strict(q_win, y_win)
    y_det = y_win - bg_detect

    sig = _robust_sigma(y_det)
    prom_full = max(1.0, K_SIGMA * sig)

    # use absolute caps for detection widths
    wmin = max(1, MIN_WIDTH_PTS)
    wmax = max(MIN_WIDTH_PTS+1, MAX_WIDTH_PTS)
    min_dist_pts = max(1, int(0.5 * MIN_WIDTH_PTS))

    peaks, props = signal.find_peaks(
        y_det,
        prominence=prom_full,
        width=(wmin, wmax),
        distance=min_dist_pts,
        rel_height=0.5,
    )
    print(f"[detect] sig={sig:.3g} prom>={prom_full:.3g} width∈[{wmin},{wmax}] N={len(peaks)}")

    # --- Fit (linear BG only) ---
    result, comps = fit_multi_peaks(q_win, y_win, peaks, props, bg_degree=1, dq=dq)
    w = _poisson_weights(y_win)
    m = fit_metrics(result, q_win, y_win, weights=w)
    print(f"[METRICS] R2={m['r2']:.6f}  adjR2={m['adj_r2']:.6f}  redχ²={m['red_chisq']:.3g}  "
          f"AIC={m['aic']:.2f}  BIC={m['bic']:.2f}  RMSE={m['rmse']:.3g}  max|res|={m['max_abs']:.3g}")

    # --- Optional one-shot residual peek (still respects width caps) ---
    if (m['r2'] < TARGET_R2) and (len(peaks) > 0):
        yhat = result.model.eval(params=result.params, x=q_win)
        resid_bg = (y_win - bg_detect) - (yhat - bg_detect)
        sigR = _robust_sigma(resid_bg)
        cand, cprops = signal.find_peaks(
            resid_bg,
            prominence=max(1.0, 2.5 * sigR),
            width=(wmin, wmax),
            distance=min_dist_pts,
            rel_height=0.5,
        )
        keep = []
        for j, pidx in enumerate(cand):
            if len(peaks) and np.min(np.abs(q_win[peaks] - q_win[pidx])) < (0.6 * MIN_WIDTH_PTS * dq):
                continue
            keep.append(j)
        if keep:
            peaks = np.concatenate([np.asarray(peaks, int), cand[keep].astype(int)])
            widths = np.concatenate([np.asarray(props.get("widths", np.full(len(peaks), MIN_WIDTH_PTS))),
                                     cprops["widths"][keep]])
            promin = np.concatenate([np.asarray(props.get("prominences", np.ones(len(peaks)))),
                                     cprops["prominences"][keep]])
            order = np.argsort(q_win[peaks]); peaks = peaks[order]
            props = {"widths": widths[order], "prominences": promin[order]}
            result, comps = fit_multi_peaks(q_win, y_win, peaks, props, bg_degree=1, dq=dq)
            m = fit_metrics(result, q_win, y_win, weights=_poisson_weights(y_win))
            print(f"[refit] R2→{m['r2']:.6f}")

    # Dense plotting
    q_dense = np.linspace(q_win.min(), q_win.max(), len(q_win) * 5)
    best_fit_dense = result.model.eval(params=result.params, x=q_dense)
    comps_dense = result.model.eval_components(params=result.params, x=q_dense)

    # --- Right top: Full Azimuthal Integration ---
    ax = axes[0, 2]
    ax.plot(q_win, y_win, "--", label="Data")
    ax.plot(q_win, bg_detect, "-", label="BG (detect, linear)")
    if len(peaks) > 0:
        ax.plot(q_win[peaks], y_det[peaks], "x", label="Detected peaks")
    for name in sorted(k for k in comps_dense if k.startswith("g")):
        ax.plot(q_dense, comps_dense[name], ":", alpha=0.85, label=name)
    if "bg_" in comps_dense:
        ax.plot(q_dense, comps_dense["bg_"], "-", label="BG (fit, linear)")

    ax.set_title("Full Azimuthal Integration")
    ax.set_xlabel("q"); ax.set_ylabel("Intensity")
    ax.legend(loc="upper right", fontsize=8)

    # --- Left 2x2: Cake previews (simple) ---
    if cake is not None:
        slices = [0, 10, 19, 28]
        sigma_smooth_detect = max(3, min(12, int(0.03 * len(q_win))))  # preview only
        for i, cs in enumerate(slices):
            r, c = divmod(i, 2)
            y_c = cake[frame_number, cs, :][mask]
            bg_c = gaussian_filter1d(y_c, sigma=sigma_smooth_detect)
            ysub = y_c - bg_c
            pk_c, _ = signal.find_peaks(ysub, prominence=prom_full, width=(wmin, wmax), distance=min_dist_pts)
            axes[r, c].plot(q_win, y_c, "--", label="Cake data")
            axes[r, c].plot(q_win, bg_c, "-", label="BG (detect)")
            if len(pk_c) > 0:
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
        ["", ""],
    ]
    for i in range(len(peaks)):
        c = result.params.get(f"g{i}_center", None)
        if c is not None:
            table_data.append([f"Peak {i} center", f"{c.value:.6f}"])

    table = table_ax.table(cellText=table_data,
                           colLabels=["Metric / Peak", "Value"],
                           cellLoc="center",
                           loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.25)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# ----------------------------- CLI ----------------------------- #
def _parse_args():
    p = argparse.ArgumentParser(description="Lean multi-peak Gaussian fitting (strict linear BG, no derivative seeding).")
    p.add_argument("h5", type=str)
    p.add_argument("frame_number", type=int)
    p.add_argument("peak_pos", type=float)
    p.add_argument("--window", type=float, default=0.1, help="Half-window in q.")
    return p.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    peak_fit(args.h5, args.frame_number, args.peak_pos, args.window)
