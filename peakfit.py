#!/usr/bin/env python3
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.ndimage import gaussian_filter1d
from lmfit.models import PolynomialModel, GaussianModel

# ===================== TUNABLE PEAK-FIT PARAMS =====================
# Sampling / width prior
W_GUESS_PTS       = 6.0   # nominal FWHM in *points*; set ~ your median points/FWHM

# Detection sensitivity
K_PROM            = 4.0   # noise multiplier (higher = stricter)
MIN_PROM_ABS_FRAC = 0.03  # floor as fraction of dynamic range after BG-sub
PROM_ABS_FLOOR    = 1.0   # absolute minimum prominence in counts

# Multi-scale candidate generation (captures overlaps)
SCALES_FWHM       = [0.6, 1.0, 1.6, 2.3]  # widths relative to W_GUESS_PTS for matched smoothing
WMIN_FRAC         = 0.4   # min width (points) = WMIN_FRAC * W_GUESS_PTS
WMAX_MULT         = 10.0  # max width (points) = WMAX_MULT * W_GUESS_PTS
MIN_DIST_FRAC_INIT= 0.5   # NMS radius in points = MIN_DIST_FRAC_INIT * W_GUESS_PTS
VALLEY_FRAC       = 0.15  # require ≥15% valley depth between neighbors

MAX_CANDIDATES    = 12    # hard cap on initial candidates

# Gaussian bounds relative to seeds
SIGMA_MIN_FRAC    = 0.20
SIGMA_MAX_MULT    = 4.0
AMP_MIN_FRAC      = 0.05  # lmfit Gaussian amplitude = AREA
AMP_MAX_MULT      = 20.0

# Center wiggle per peak (independent centers; no global shift/scale)
CENTER_WIGGLE_PTS = 8.0   # ± points each center may move from its seed

# Linear background slope guard
SLOPE_LIMIT_SCALE = 2.0   # |bg_c1| <= SLOPE_LIMIT_SCALE * (yspan/qspan)

# Backward pruning by BIC
PRUNE_BIC_DROP    = 2.0   # require ΔBIC ≤ -2.0 to keep a removal (lower is better)
PRUNE_MAX_STEPS   = 10    # safety cap
# ==================================================================

# ----------------------------- helpers ----------------------------- #
def _sigma_from_fwhm_pts(fwhm_pts: float, dq: float) -> float:
    return (fwhm_pts * dq) / 2.355
def _fwhm_from_sigma(sigma: float) -> float:
    return 2.355 * sigma
def _poisson_weights(y):
    return 1.0 / np.sqrt(np.clip(y, 1.0, None))
def _robust_sigma(x):
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))

def robust_linear_bg(q, y, frac_keep=0.60, clip_sigma=2.5, max_iter=10):
    """Iterative sigma-clipped straight line (detection baseline only)."""
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

def estimate_noise(y_det, w_guess_pts):
    """Noise from high-pass residual: y_det - gaussian_smooth(y_det) with σ≈W_GUESS_PTS/2."""
    hp_sigma_pts = max(3.0, 0.5 * w_guess_pts)  # in points
    if y_det.size < 7:
        return _robust_sigma(y_det)
    smooth = gaussian_filter1d(y_det, sigma=hp_sigma_pts)
    resid = y_det - smooth
    return _robust_sigma(resid)

def nms_candidates(q, score_idx, score_vals, radius_pts):
    """Non-maximum suppression on indices using a points-radius."""
    order = np.argsort(-score_vals)  # high score first
    picked = []
    for o in order:
        idx = int(score_idx[o])
        if all(abs(idx - p) > radius_pts for p in picked):
            picked.append(idx)
    return np.array(picked, int)

def multiscale_candidates(q, y_det, dq, peak_pos):
    """Build a generous candidate set from multi-scale matched filtering, valley gating, and NMS."""
    n = y_det.size
    dyn = max(float(y_det.max() - y_det.min()), 1.0)
    sig_noise = estimate_noise(y_det, W_GUESS_PTS)
    prom_thr = max(MIN_PROM_ABS_FRAC * dyn, K_PROM * sig_noise, PROM_ABS_FLOOR)

    wmin = max(1, int(WMIN_FRAC * W_GUESS_PTS))
    wmax = int(WMAX_MULT * W_GUESS_PTS)

    cand_idx = []
    cand_prom = []
    cand_width = []
    # collect from each scale
    for sc in SCALES_FWHM:
        sigma_pts = max(1.0, (W_GUESS_PTS * sc) / 2.355)
        y_s = gaussian_filter1d(y_det, sigma=sigma_pts)
        # adaptive, same thresholds across scales
        pks, props = signal.find_peaks(
            y_s, prominence=prom_thr, width=(wmin, wmax), rel_height=0.5
        )
        for i, p in enumerate(pks):
            cand_idx.append(int(p))
            cand_prom.append(float(props["prominences"][i]) if "prominences" in props else float(y_s[p]))
            cand_width.append(float(props["widths"][i]) if "widths" in props else W_GUESS_PTS)

    if not cand_idx:
        # fallback: single candidate at closest point to peak_pos
        i0 = int(np.argmin(np.abs(q - peak_pos)))
        return [i0], [W_GUESS_PTS], prom_thr

    cand_idx = np.array(cand_idx, int)
    cand_prom = np.array(cand_prom, float)
    cand_width = np.array(cand_width, float)

    # NMS by points radius
    radius_pts = max(1, int(round(MIN_DIST_FRAC_INIT * W_GUESS_PTS)))
    keep_idx = nms_candidates(q, cand_idx, cand_prom, radius_pts)

    # fallback: keep the single strongest candidate if NMS returned none
    if len(keep_idx) == 0:
        keep_idx = np.array([int(cand_idx[np.argmax(cand_prom)])])

    # >>> FIX: apply the SAME mask to all arrays
    mask_keep = np.isin(cand_idx, keep_idx)
    cand_idx   = cand_idx[mask_keep]
    cand_prom  = cand_prom[mask_keep]
    cand_width = cand_width[mask_keep]

    # sort by proximity to peak_pos, tie-break by higher prominence
    order = np.argsort(np.abs(q[cand_idx] - peak_pos) + 1e-9 * (-cand_prom))
    cand_idx   = cand_idx[order]
    cand_prom  = cand_prom[order]
    cand_width = cand_width[order]
    # valley gating between neighbors (to keep true splits)
    def valley_ok(i, j):
        a, b = (i, j) if i < j else (j, i)
        if b - a < 2:
            return False
        valley = float(np.min(y_det[a:b+1]))
        h_i, h_j = float(y_det[i]), float(y_det[j])
        depth = min(h_i, h_j) - valley
        return depth >= (VALLEY_FRAC * max(1.0, min(h_i, h_j)))

    # always keep the strongest near peak_pos, then allow neighbors that pass valley test
    kept = []
    for idx in cand_idx:
        if len(kept) == 0:
            kept.append(idx)
            continue
        # allow if it forms a real valley with the closest kept peak
        nearest = kept[np.argmin(np.abs(np.array(kept) - idx))]
        if valley_ok(idx, nearest):
            kept.append(idx)

    # cap candidates
    if len(kept) > MAX_CANDIDATES:
        kept = kept[:MAX_CANDIDATES]

    # provide a matching width per kept idx (fallback to W_GUESS_PTS)
    widths = []
    for k in kept:
        # find a width we measured for this index
        matches = np.where(cand_idx == k)[0]
        widths.append(float(cand_width[matches[0]]) if matches.size else W_GUESS_PTS)

    return kept, widths, prom_thr

def seeds_from_candidates(q, y_det, dq, idx_list, width_pts_list):
    """Convert candidate indices to (center_q, sigma_seed, area_seed)."""
    seeds = []
    for idx, wpts in zip(idx_list, width_pts_list):
        sigma_seed = _sigma_from_fwhm_pts(max(1.0, wpts), dq)
        height = max(float(y_det[idx]), 1.0)
        area = height * sigma_seed * np.sqrt(2*np.pi)  # Gaussian AREA
        seeds.append((float(q[idx]), sigma_seed, area))
    return seeds

def build_and_fit(q, y, seeds):
    """Linear background + all seeds Gaussians. Returns (result, components)."""
    dq = float(np.diff(q).mean())
    # linear background (degree 1) with slope bound
    bg = PolynomialModel(degree=1, prefix="bg_")
    model = bg
    params = bg.make_params()
    params["bg_c0"].set(value=float(np.median(y)), min=0.0)
    params["bg_c1"].set(value=0.0)
    qspan = max(q.max()-q.min(), 1e-12)
    yspan = max(np.percentile(y,95)-np.percentile(y,5), 1.0)
    smax = SLOPE_LIMIT_SCALE * yspan / qspan
    params["bg_c1"].set(min=-smax, max=+smax)

    # add Gaussians
    for i, (c0, s0, a0) in enumerate(seeds):
        g = GaussianModel(prefix=f"g{i}_")
        model += g
        params.update(g.make_params(center=c0, sigma=max(1e-6, s0), amplitude=max(1e-9, a0)))
        wig = CENTER_WIGGLE_PTS * dq
        params[f"g{i}_center"].set(min=c0 - wig, max=c0 + wig)
        params[f"g{i}_sigma"].set(min=max(1e-6, SIGMA_MIN_FRAC * s0),
                                  max=max(2e-6, SIGMA_MAX_MULT * s0))
        params[f"g{i}_amplitude"].set(min=max(1e-9, AMP_MIN_FRAC * abs(a0)),
                                      max=max(1e-8, AMP_MAX_MULT * abs(a0)))

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

def prune_by_bic(q, y, seeds):
    """Start with all seeds; iteratively remove one Gaussian if BIC improves by ≥ PRUNE_BIC_DROP."""
    if not seeds:
        return build_and_fit(q, y, [(float(q[len(q)//2]), _sigma_from_fwhm_pts(W_GUESS_PTS, float(np.diff(q).mean())), 1.0)])

    # initial fit
    result, comps = build_and_fit(q, y, seeds)
    w = _poisson_weights(y)
    m = fit_metrics(result, q, y, weights=w)
    current = dict(seeds=seeds, result=result, comps=comps, metrics=m)

    for step in range(PRUNE_MAX_STEPS):
        n_now = len(current["seeds"])
        if n_now <= 1:
            break
        # evaluate removing each peak once; choose the best BIC
        best_drop = None
        best_state = None
        for i in range(n_now):
            seeds_try = current["seeds"][:i] + current["seeds"][i+1:]
            res_try, comp_try = build_and_fit(q, y, seeds_try)
            m_try = fit_metrics(res_try, q, y, weights=w)
            # ΔBIC = new - old (we want negative to improve)
            dBIC = m_try["bic"] - current["metrics"]["bic"]
            if (best_drop is None) or (m_try["bic"] < best_state["metrics"]["bic"]):
                best_drop = (i, dBIC)
                best_state = dict(seeds=seeds_try, result=res_try, comps=comp_try, metrics=m_try)
        # accept removal only if BIC improved by at least PRUNE_BIC_DROP
        if best_drop is None or best_drop[1] > -PRUNE_BIC_DROP:
            break
        current = best_state  # keep pruned model

    return current["result"], current["comps"], current["metrics"]
# ----------------------------- main ----------------------------- #
def peak_fit(h5_path, frame_number, peak_pos, window=0.1):
    with h5py.File(h5_path, "r") as f:
        Int = f["int"][:]  # (nframes, q)
        q = f["q"][:]      # (q,)

    # window
    q_min, q_max = float(peak_pos - window), float(peak_pos + window)
    mask = (q >= q_min) & (q <= q_max)
    q_win = q[mask]
    y_win = Int[int(frame_number), mask].astype(float)
    if q_win.size < max(20, int(4 * W_GUESS_PTS)):
        raise ValueError("Fit window too small or outside q-range.")

    dq = float(np.diff(q_win).mean())

    # detection baseline (strict linear), build BG-sub for seeding
    bg_det, bg_info = robust_linear_bg(q_win, y_win, frac_keep=0.60, clip_sigma=2.5, max_iter=10)
    y_det = y_win - bg_det

    # multi-scale candidates → seeds
    idx_list, width_pts_list, prom_thr = multiscale_candidates(q_win, y_det, dq, peak_pos)
    seeds = seeds_from_candidates(q_win, y_det, dq, idx_list, width_pts_list)

    # prune by BIC (fluid number of peaks)
    result, comps, metrics = prune_by_bic(q_win, y_win, seeds)

    # ----------- plotting -----------
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(11, 8), gridspec_kw={"height_ratios": [3, 2]})

    ax0.plot(q_win, y_win, "k.", ms=3, label="Data")

    # fitted BG (linear)
    if "bg_" in comps:
        ax0.plot(q_win, comps["bg_"], "-", lw=1.2, label="BG (fit, linear)")

    # total fit
    q_dense = np.linspace(q_win.min(), q_win.max(), len(q_win)*5)
    y_dense = result.model.eval(params=result.params, x=q_dense)
    ax0.plot(q_dense, y_dense, "-", lw=1.6, label="Total fit")

    # components
    comp_dense = result.model.eval_components(params=result.params, x=q_dense)
    gnames = sorted([k for k in comp_dense if k.startswith("g")])
    for k in gnames:
        ax0.plot(q_dense, comp_dense[k], ":", lw=1.0, label=k)

    # mark fitted centers
    centers = [result.params[f"g{i}_center"].value for i in range(len(gnames))]
    ax0.plot(centers, [np.interp(c, q_dense, y_dense) for c in centers], "x", ms=8, label="Centers")

    ax0.set_xlabel("q"); ax0.set_ylabel("Intensity")
    ax0.set_title(f"Frame {frame_number} | seeds init: {len(seeds)} | fitted peaks: {len(gnames)}")
    ax0.legend(loc="best", fontsize=8)

    # metrics table
    ax1.axis("off")
    rows = [
        ["R²", f"{metrics['r2']:.6f}"],
        ["Adj R²", f"{metrics['adj_r2']:.6f}"],
        ["Reduced χ²", f"{metrics['red_chisq']:.3g}"],
        ["AIC", f"{metrics['aic']:.2f}"],
        ["BIC", f"{metrics['bic']:.2f}"],
        ["RMSE", f"{metrics['rmse']:.3g}"],
        ["Max |res|", f"{metrics['max_abs']:.3g}"],
        ["Det. prom≥", f"{prom_thr:.3g}"],
        ["", ""],
    ]
    for i in range(len(gnames)):
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
    p = argparse.ArgumentParser(description="Fluid Gaussian peak fitting with linear background and BIC pruning.")
    p.add_argument("h5", type=str)
    p.add_argument("frame_number", type=int)
    p.add_argument("peak_pos", type=float)
    p.add_argument("--window", type=float, default=0.1, help="Half-window in q around peak_pos")
    return p.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    peak_fit(args.h5, args.frame_number, args.peak_pos, args.window)
