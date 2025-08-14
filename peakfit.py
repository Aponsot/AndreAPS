#!/usr/bin/env python3
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.signal as signal
from lmfit.models import PolynomialModel, GaussianModel

# --- publication plotting defaults ---
def set_pub_style(scale=1.1):
    base = 14 * scale
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 300,         # crisp in work docs
        "font.size": base,
        "axes.titlesize": base * 1.3,
        "axes.labelsize": base * 1.15,
        "xtick.labelsize": base,
        "ytick.labelsize": base,
        "legend.fontsize": base * 0.95,
        "lines.linewidth": 1.8,
        "lines.markersize": 6,
        "figure.autolayout": True,  # reduce clipping
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.35,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    })

set_pub_style(1.1)

# ----------------------------- helpers ----------------------------- #
def _sigma_from_fwhm(fwhm_pts: float, dq: float) -> float:
    """FWHM -> sigma in x-units via FWHM = 2.355*sigma."""
    return (fwhm_pts * dq) / 2.355

def fit_metrics(result, x, y, weights=None):
    yhat = result.model.eval(params=result.params, x=x)
    resid = y - yhat
    n = len(y)
    k = sum(1 for p in result.params.values() if p.vary)  # free params

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

def _robust_sigma(x):
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))

# --- strict linear background for detection (sigma-clipped LSQ) ---
def _linear_bg_strict(q, y, frac_keep=0.60, clip_sigma=2.5, max_iter=10):
    q = np.asarray(q, float); y = np.asarray(y, float); n = y.size
    if n < 3:
        a = float(np.median(y)); b = 0.0
        return a + b*q, dict(a=a, b=b, kept=n, iters=0)
    mask = np.ones(n, bool); a = float(np.median(y)); b = 0.0
    for it in range(max_iter):
        X = np.vstack([np.ones(mask.sum()), q[mask]]).T
        beta, *_ = np.linalg.lstsq(X, y[mask], rcond=None)
        a, b = float(beta[0]), float(beta[1])
        bg = a + b*q
        resid = y - bg
        rsel = resid[mask]
        sig = 1.4826 * np.median(np.abs(rsel - np.median(rsel))) if rsel.size else 0.0
        if sig == 0.0:
            break
        below = y <= (bg + clip_sigma*sig)
        idx = np.where(below)[0]
        if idx.size < 5:
            break
        order = np.argsort(resid[idx])
        k = max(5, int(frac_keep * idx.size))
        newmask = np.zeros(n, bool); newmask[idx[order[:k]]] = True
        if newmask.sum()==mask.sum() and np.all(newmask==mask):
            break
        mask = newmask
    return a + b*q, dict(a=a, b=b, kept=mask.sum(), iters=it+1)

def fit_multi_peaks(q, y, peaks_idx, props, bg_degree=1):
    """
    Linear background + sum of Gaussians.
    Centers tied to global qshift+qscale with small per-peak wiggle.
    """
    dq = float(np.diff(q).mean())

    # Background (strictly linear if bg_degree=1)
    composite = PolynomialModel(degree=bg_degree, prefix="bg_")
    params = composite.make_params()
    params["bg_c0"].set(value=float(np.median(y)), min=0.0)
    if bg_degree >= 1:
        params["bg_c1"].set(value=0.0)

    # global transforms (harmless if no peaks)
    params.add("qshift", value=0.0, min=-5e-3, max=5e-3)
    params.add("qscale", value=1.0, min=0.999, max=1.001)

    # Add Gaussians if any peaks
    if len(peaks_idx) > 0:
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

            # wiggle ~3 samples or ~0.0018 (whichever larger)
            dq_samp = max(3 * dq, 0.0018)
            params.add(f"g{i}_dcenter", value=0.0, min=-dq_samp, max=dq_samp)
            params[f"g{i}_center"].set(expr=f"qscale*(g{i}_c0) + qshift + g{i}_dcenter")

            # width/amplitude bounds
            params[f"g{i}_sigma"].set(min=0.25 * sigma0, max=4.0 * sigma0)
            params[f"g{i}_amplitude"].set(min=0.2 * abs(amp0), max=10 * abs(amp0))

    # Fit
    w = _poisson_weights(y)
    result = composite.fit(
        y, params, x=q, weights=w,
        method="least_squares", fit_kws={"loss": "soft_l1", "f_scale": 1.0}
    )
    comps = result.model.eval_components(params=result.params, x=q)
    return result, comps

# ----------------------------- main ----------------------------- #
def peak_fit(h5_path, frame_number, peak_pos, window=0.1):
    with h5py.File(h5_path, "r") as f:
        q = f["q"][:]                       # (q,)
        # read only the needed frame, not the whole dataset
        y_full = f["int"][frame_number, :]  # (q,)

    # Window
    q_min, q_max = peak_pos - window, peak_pos + window
    mask = (q >= q_min) & (q <= q_max)
    q_win = q[mask]
    y_win = y_full[mask]
    if q_win.size < 5:
        raise ValueError("Fit window too small or outside q-range.")

    # Figure: top = azimuthal curve; bottom = table
    fig, (ax, table_ax) = plt.subplots(
        2, 1, figsize=(12, 9),
        gridspec_kw={"height_ratios": [3.0, 1.2]}
    )
    fig.suptitle(f"Peak Fit for Frame {frame_number}", fontsize=16)

    # --- Detection (background-sub only, STRICT LINEAR) ---
    dq = float(np.diff(q_win).mean())
    bg_detect, _ = _linear_bg_strict(q_win, y_win)
    y_det = y_win - bg_detect

    # thresholds for detection
    sig = _robust_sigma(y_det)
    K = 1.5
    prom_full = max(1.0, K * sig)

    w_guess_pts = 6.0
    wmin = max(1, int(0.6 * w_guess_pts))
    wmax = int(20 * w_guess_pts)
    min_dist_pts = 1

    peaks, props = signal.find_peaks(
        y_det,
        prominence=prom_full,
        width=(wmin, wmax),
        distance=min_dist_pts,
        rel_height=0.2,
    )

    # --- Fit (linear BG only) ---
    result, _ = fit_multi_peaks(q_win, y_win, peaks, props, bg_degree=1)
    w = _poisson_weights(y_win)
    m = fit_metrics(result, q_win, y_win, weights=w)

    # Dense evaluation for smooth curves
    q_dense = np.linspace(q_win.min(), q_win.max(), max(len(q_win) * 5, 400))
    best_fit_dense = result.model.eval(params=result.params, x=q_dense)
    comps_dense = result.model.eval_components(params=result.params, x=q_dense)

    # --- TOP: Full Azimuthal Integration with peaks + fit ---
    ax.plot(q_win, y_win, "-", label="Data", zorder=2)
    # fitted background (not the detection BG)
    if "bg_" in comps_dense:
        ax.plot(q_dense, comps_dense["bg_"], "--", linewidth=1.6, alpha=0.9, label="Fitted BG", zorder=3)
    ax.plot(q_dense, best_fit_dense, "-", linewidth=2.4, alpha=0.95, label="Total fit", zorder=4)

    # peak markers at true data level
    if len(peaks) > 0:
        ax.plot(q_win[peaks], y_win[peaks], "x", label="Detected peaks", zorder=5)

    # individual Gaussians
    for name in sorted(k for k in comps_dense if k.startswith("g")):
        ax.plot(q_dense, comps_dense[name], ":", alpha=0.85, label=name, zorder=3)

    ax.set_title("Full Azimuthal Integration")
    ax.set_xlabel("q")
    ax.set_ylabel("Intensity")

    # de-duplicate legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right", ncol=2)

    # --- BOTTOM: Table with metrics + peak centers ---
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
        c = result.params.get(f"g{i}_center")
        if c is not None:
            table_data.append([f"Peak {i} center", f"{c.value:.6f}"])

    table = table_ax.table(
        cellText=table_data,
        colLabels=["Metric / Peak", "Value"],
        cellLoc="center",
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.25)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# ----------------------------- CLI ----------------------------- #
def _parse_args():
    p = argparse.ArgumentParser(
        description="Full-azimuth peak detection + Gaussian fitting (linear background) with a summary table."
    )
    p.add_argument("h5", type=str)
    p.add_argument("frame_number", type=int)
    p.add_argument("peak_pos", type=float)
    p.add_argument("--window", type=float, default=0.1, help="Half-window in q.")
    return p.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    peak_fit(args.h5, args.frame_number, args.peak_pos, args.window)
