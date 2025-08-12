#!/usr/bin/env python3
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.ndimage import gaussian_filter1d
from lmfit.models import PolynomialModel, GaussianModel


# ===================== TUNABLE PEAK-FIT PARAMS =====================
# Detection & noise
W_GUESS_PTS       = 6.0   # nominal FWHM in *points*
HP_SIGMA_MULT     = 2.0   # high-pass smoothing = HP_SIGMA_MULT * W_GUESS_PTS
K_PROM            = 4.0   # noise multiplier for prominence threshold
MIN_PROM_ABS_FRAC = 0.03  # min prominence as fraction of local dynamic range (after BG removal)
PROM_ABS_FLOOR    = 1.0   # absolute minimum prominence (counts)

# Detection width/distance (in points, scaled by W_GUESS_PTS)
WMIN_FRAC         = 0.6
WMAX_MULT         = 10.0
MIN_DIST_FRAC     = 0.8
MAX_INIT_PEAKS    = 4     # cap initial seeds BEFORE fitting (prune/merge still applies)

# Exactly how many peaks to fit?
# - If None: fit exactly the number of seeds found (after pruning/merging/cap).
# - If an integer (e.g., 2): force exactly that many peaks.
EXACT_N_PEAKS     = None

# Gaussian bounds (relative to seed)
SIGMA_MIN_FRAC    = 0.20  # lower bound on sigma = SIGMA_MIN_FRAC * sigma_seed
SIGMA_MAX_MULT    = 4.0   # upper bound on sigma = SIGMA_MAX_MULT * sigma_seed
AMP_MIN_FRAC      = 0.05  # min amplitude (AREA) = AMP_MIN_FRAC * area_seed
AMP_MAX_MULT      = 20.0  # max amplitude (AREA) = AMP_MAX_MULT * area_seed

# Centers: per-peak, not globally tied
CENTER_WIGGLE_PTS = 8.0   # each center can move ± CENTER_WIGGLE_PTS * dq from its seed

# Linear background (fit) slope guard
SLOPE_LIMIT_SCALE = 2.0   # |bg_c1| <= SLOPE_LIMIT_SCALE * (yspan/qspan)

# Optional: one-shot residual augment (adds at most one extra peak)
USE_RESIDUAL_AUGMENT = False
REFIT_TARGET_R2      = 0.985
# ==================================================================


# ----------------------------- helpers ----------------------------- #
def _sigma_from_fwhm_pts(fwhm_pts: float, dq: float) -> float:
    """FWHM (in samples) -> sigma in q-units."""
    return (fwhm_pts * dq) / 2.355

def _fwhm_from_sigma(sigma: float) -> float:
    return 2.355 * sigma

def _poisson_weights(y):
    return 1.0 / np.sqrt(np.clip(y, 1.0, None))

def _robust_sigma(x):
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))

def robust_linear_bg(q, y, frac_keep=0.60, clip_sigma=2.5, max_iter=10):
    """Iterative sigma-clipped straight line (used for detection baseline only)."""
    q = np.asarray(q, float); y = np.asarray(y, float); n = y.size
    if n < 3:
        a = float(np.median(y)); b = 0.0
        return a + b*q, dict(a=a, b=b, kept=n, iters=0)
    mask = np.ones(n, dtype=bool)
    a = float(np.median(y)); b = 0.0; kept = n
    for it in range(max_iter):
        X = np.vstack([np.ones(mask.sum()), q[mask]]).T
        beta, *_ = np.linalg.lstsq(X, y[mask], rcond=None)
        a, b = float(beta[0]), float(beta[1])
        bg = a + b*q
        resid = y - bg
        rsel = resid[mask]
        sig = 1.4826 * np.median(np.abs(rsel - np.median(rsel))) if rsel.size else 0.0
        if sig == 0.0: break
        below = y <= (bg + clip_sigma * sig)
        idx = np.where(below)[0]
        if idx.size < 5: break
        order = np.argsort(resid[idx])
        k = max(5, int(frac_keep * idx.size))
        newmask = np.zeros(n, dtype=bool); newmask[idx[order[:k]]] = True
        if newmask.sum() == mask.sum() and np.all(newmask == mask):
            kept = newmask.sum(); break
        mask = newmask; kept = mask.sum()
    return (a + b*q), dict(a=a, b=b, kept=kept, iters=it+1)

def estimate_noise(y_det, hp_sigma_pts):
    """Noise from high-pass residual: y_det - smooth(y_det)."""
    hp_sigma_pts = max(3, int(round(hp_sigma_pts)))
    if y_det.size < 7:
        return _robust_sigma(y_det), y_det
    smooth = gaussian_filter1d(y_det, sigma=hp_sigma_pts)
    resid = y_det - smooth
    return _robust_sigma(resid), resid

def prune_merge_sort(q, peaks, props, peak_pos, w_guess_pts):
    """Sort by proximity to peak_pos then by prominence; merge too-close; enforce MAX_INIT_PEAKS."""
    if peaks.size == 0:
        return np.array([], dtype=int), {"widths": np.array([]), "prominences": np.array([])}

    order = np.argsort(np.abs(q[peaks] - peak_pos) + 1e-9 * (-props["prominences"]))
    peaks = peaks[order]
    widths = props["widths"][order].astype(float)
    promin = props["prominences"][order].astype(float)

    # merge peaks closer than MERGE_FRAC * W_GUESS_PTS (keep higher prominence)
    keep = [0]
    for i in range(1, len(peaks)):
        prev = keep[-1]
        if (peaks[i] - peaks[prev]) < int(max(1, MERGE_FRAC * w_guess_pts)):
            if promin[i] > promin[prev]:
                keep[-1] = i
        else:
            keep.append(i)

    peaks = peaks[keep]; widths = widths[keep]; promin = promin[keep]
    # cap initial seeds
    if len(peaks) > MAX_INIT_PEAKS:
        peaks  = peaks[:MAX_INIT_PEAKS]
        widths = widths[:MAX_INIT_PEAKS]
        promin = promin[:MAX_INIT_PEAKS]

    return peaks, {"widths": widths, "prominences": promin}

def seeds_from_detection(q, y_det, dq, peak_pos):
    """Return seeds: list of (center_q, sigma_seed, area_seed) and detection info."""
    dyn = max(float(y_det.max() - y_det.min()), 1.0)
    hp_sigma_pts = HP_SIGMA_MULT * W_GUESS_PTS
    sig_hp, _ = estimate_noise(y_det, hp_sigma_pts=hp_sigma_pts)
    prom_thr = max(MIN_PROM_ABS_FRAC * dyn, K_PROM * sig_hp, PROM_ABS_FLOOR)

    wmin = max(1, int(WMIN_FRAC * W_GUESS_PTS))
    wmax = int(WMAX_MULT * W_GUESS_PTS)
    mindist = int(max(1, round(MIN_DIST_FRAC * W_GUESS_PTS)))

    peaks, props = signal.find_peaks(
        y_det, prominence=prom_thr, width=(wmin, wmax), distance=mindist, rel_height=0.5
    )
    peaks, props = prune_merge_sort(q, peaks, props, peak_pos, W_GUESS_PTS)

    seeds = []
    for i, idx in enumerate(peaks):
        fwhm_pts = float(props["widths"][i]) if "widths" in props and len(props["widths"])>i else W_GUESS_PTS
        sigma_seed = _sigma_from_fwhm_pts(max(fwhm_pts, 1.0), dq)
        height_est = max(float(y_det[idx]), 1.0)
        area_seed = height_est * sigma_seed * np.sqrt(2*np.pi)  # GaussianModel amplitude = area
        seeds.append((float(q[idx]), sigma_seed, area_seed))

    # Fallback if none
    if not seeds:
        sigma0 = _sigma_from_fwhm_pts(W_GUESS_PTS, dq)
        i0 = int(np.nanargmin(np.abs(q - peak_pos)))
        height0 = max(float(y_det[i0]), 1.0)
        area0 = height0 * sigma0 * np.sqrt(2*np.pi)
        seeds = [(float(q[i0]), sigma0, area0)]

    return seeds, dict(prom_thr=prom_thr, wmin=wmin, wmax=wmax, mindist=mindist)

def build_and_fit(q, y, seeds, npeaks):
    """Linear background + exactly npeaks Gaussians. Returns (result, components)."""
    dq = float(np.diff(q).mean())
    # linear background with guarded slope
    bg = PolynomialModel(degree=1, prefix="bg_")
    model = bg
    params = bg.make_params()
    params["bg_c0"].set(value=float(np.median(y)), min=0.0)
    params["bg_c1"].set(value=0.0)
    qspan = max(q.max()-q.min(), 1e-12)
    yspan = max(np.percentile(y,95)-np.percentile(y,5), 1.0)
    smax = SLOPE_LIMIT_SCALE * yspan / qspan
    params["bg_c1"].set(min=-smax, max=+smax)

    # ensure we have exactly npeaks seeds: split strongest if too few, trim if too many
    seeds = list(seeds)
    if len(seeds) > npeaks:
        # keep first npeaks (already sorted by proximity/prominence)
        seeds = seeds[:npeaks]
    while len(seeds) < npeaks:
        j = int(np.argmax([a for _,_,a in seeds]))
        c, s, a = seeds.pop(j)
        offset = 0.40 * _fwhm_from_sigma(s)
        seeds += [(c - offset/2, s, a/2), (c + offset/2, s, a/2)]

    for i, (c0, s0, a0) in enumerate(seeds):
        g = GaussianModel(prefix=f"g{i}_")
        model += g
        params.update(g.make_params(center=c0, sigma=max(1e-6, s0), amplitude=max(1e-9, a0)))
        # per-peak center wiggle
        wig = CENTER_WIGGLE_PTS * dq
        params[f"g{i}_center"].set(min=c0 - wig, max=c0 + wig)
        # sigma & area bounds relative to seed
        params[f"g{i}_sigma"].set(min=max(1e-6, SIGMA_MIN_FRAC * s0),
                                  max=max(2e-6, SIGMA_MAX_MULT * s0))
        params[f"g{i}_amplitude"].set(min=max(1e-9, AMP_MIN_FRAC * abs(a0)),
                                      max=max(1e-8, AMP_MAX_MULT * abs(a0)))

    # fit
    w = _poisson_weights(y)
    result = model.fit(y, params, x=q, weights=w,
                       method="least_squares",
                       fit_kws={"loss": "soft_l1", "f_scale": 1.0})
    comps = model.eval_components(params=result.params, x=q)
    return result, comps

def fit_metrics(result, x, y, weights=None):
    yhat = result.model.eval(params=result.params, x=x)
    resid = y - yhat
    n = len(y)
    k = sum(p.vary for p in result.params.values())
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
    rmse = np.sqrt(sse / max(n, 1))
    max_abs = float(np.max(np.abs(resid)))
    return dict(r2=r2, adj_r2=adj_r2, red_chisq=red_chisq, aic=aic, bic=bic,
                rmse=rmse, max_abs=max_abs, n=n, k=k)


# ----------------------------- main ----------------------------- #
def peak_fit(h5_path, frame_number, peak_pos, window=0.1):
    # load
    with h5py.File(h5_path, "r") as f:
        Int = f["int"][:]  # (nframes, q)
        q = f["q"][:]      # (q,)

    # window
    q_min, q_max = peak_pos - window, peak_pos + window
    mask = (q >= q_min) & (q <= q_max)
    q_win = q[mask]
    y_win = Int[int(frame_number), mask].astype(float)
    if q_win.size < max(20, int(4 * W_GUESS_PTS)):
        raise ValueError("Fit window too small or outside q-range.")

    dq = float(np.diff(q_win).mean())

    # detection baseline (linear) and bg-sub for seeding
    bg_det, _ = robust_linear_bg(q_win, y_win, frac_keep=0.60, clip_sigma=2.5, max_iter=10)
    y_det = y_win - bg_det

    # seeds from detection
    seeds, detinfo = seeds_from_detection(q_win, y_det, dq, peak_pos)
    seeds_used = len(seeds)

    # decide npeaks to fit (no model selection)
    if EXACT_N_PEAKS is None:
        npeaks = seeds_used
    else:
        npeaks = int(EXACT_N_PEAKS)

    # fit exactly npeaks
    result, comps = build_and_fit(q_win, y_win, seeds, npeaks)

    # optional: one-shot residual augment (adds at most one) — disabled by default
    if USE_RESIDUAL_AUGMENT and EXACT_N_PEAKS is None:
        w = _poisson_weights(y_win)
        m_now = fit_metrics(result, q_win, y_win, weights=w)
        if m_now["r2"] < REFIT_TARGET_R2:
            yhat = result.model.eval(params=result.params, x=q_win)
            resid = y_win - yhat
            sigR = _robust_sigma(resid)
            cand, props = signal.find_peaks(resid, prominence=2.5*sigR, width=1)
            if cand.size:
                centers_now = np.array([result.params[f"g{i}_center"].value for i in range(npeaks)], float)
                for p in cand:
                    if np.min(np.abs(q_win[p] - centers_now)) > (0.6 * W_GUESS_PTS * dq):
                        s0 = _sigma_from_fwhm_pts(W_GUESS_PTS, dq)
                        a0 = max(resid[p], 1.0) * s0 * np.sqrt(2*np.pi)
                        seeds2 = [(result.params[f"g{i}_center"].value,
                                   max(1e-6, result.params[f"g{i}_sigma"].value),
                                   max(1e-9, result.params[f"g{i}_amplitude"].value)) for i in range(npeaks)]
                        seeds2.append((float(q_win[p]), s0, a0))
                        result, comps = build_and_fit(q_win, y_win, seeds2, npeaks+1)
                        npeaks += 1
                        break  # add only one

    # metrics
    w = _poisson_weights(y_win)
    metrics = fit_metrics(result, q_win, y_win, weights=w)

    # ----------- plotting -----------
    fig, ax = plt.subplots(2, 1, figsize=(11, 8), gridspec_kw={"height_ratios": [3, 2]})

    # top: data + fit + components
    ax0 = ax[0]
    ax0.plot(q_win, y_win, "k.", ms=3, label="Data")

    # fitted BG (linear)
    if "bg_" in comps:
        ax0.plot(q_win, comps["bg_"], "-", lw=1.2, label="BG (fit, linear)")

    # total fit (dense)
    q_dense = np.linspace(q_win.min(), q_win.max(), len(q_win)*5)
    y_dense = result.model.eval(params=result.params, x=q_dense)
    ax0.plot(q_dense, y_dense, "-", lw=1.6, label=f"Total fit ({npeaks} Gaussian{'s' if npeaks>1 else ''})")

    # individual Gaussians
    comp_dense = result.model.eval_components(params=result.params, x=q_dense)
    for k in sorted([k for k in comp_dense if k.startswith("g")]):
        ax0.plot(q_dense, comp_dense[k], ":", lw=1.0, label=k)

    # mark centers
    centers = [result.params[f"g{i}_center"].value for i in range(npeaks)]
    ax0.plot(centers, [np.interp(c, q_dense, y_dense) for c in centers], "x", ms=8, label="Peak centers")

    ax0.set_xlabel("q"); ax0.set_ylabel("Intensity")
    ax0.set_title(f"Frame {frame_number} | seeds used: {seeds_used} | npeaks fit: {npeaks}")
    ax0.legend(loc="best", fontsize=8)

    # bottom: metrics + per-peak params
    ax1 = ax[1]; ax1.axis("off")
    rows = [
        ["Metric", "Value"],
        ["R²", f"{metrics['r2']:.6f}"],
        ["Adj R²", f"{metrics['adj_r2']:.6f}"],
        ["Reduced χ²", f"{metrics['red_chisq']:.3g}"],
        ["AIC", f"{metrics['aic']:.2f}"],
        ["BIC", f"{metrics['bic']:.2f}"],
        ["RMSE", f"{metrics['rmse']:.3g}"],
        ["Max |res|", f"{metrics['max_abs']:.3g}"],
        ["# Peaks", f"{npeaks}"],
        ["Detection prom≥", f"{detinfo['prom_thr']:.3g}"],
        ["Width range (pts)", f"[{detinfo['wmin']},{detinfo['wmax']}]"],
        ["Min dist (pts)", f"{detinfo['mindist']}"],
        ["", ""],
    ]
    for i in range(npeaks):
        c = result.params[f"g{i}_center"].value
        s = result.params[f"g{i}_sigma"].value
        a = result.params[f"g{i}_amplitude"].value  # AREA
        height = a / (np.sqrt(2*np.pi) * s)
        rows += [
            [f"Peak {i} center", f"{c:.6f}"],
            [f"Peak {i} height", f"{height:.3g}"],
            [f"Peak {i} sigma",  f"{s:.6g}"],
            [f"Peak {i} FWHM",   f"{_fwhm_from_sigma(s):.6g}"],
            [f"Peak {i} area",   f"{a:.3g}"],
            ["", ""],
        ]
    table = ax1.table(cellText=rows, colLabels=None, loc="center", cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1.1, 1.18)

    plt.tight_layout()
    plt.show()


# ----------------------------- CLI ----------------------------- #
def _parse_args():
    p = argparse.ArgumentParser(description="Linear-BG Gaussian peak fitting (no model selection).")
    p.add_argument("h5", type=str)
    p.add_argument("frame_number", type=int)
    p.add_argument("peak_pos", type=float)
    p.add_argument("--window", type=float, default=0.1, help="Half-window in q around peak_pos")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    peak_fit(args.h5, args.frame_number, args.peak_pos, args.window)
