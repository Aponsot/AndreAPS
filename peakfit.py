#!/usr/bin/env python3
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from lmfit.models import GaussianModel, PolynomialModel


# ===== MINIMAL KNOBS =====
WIDTH_PTS    = 6.0   # typical points across a peak FWHM
SENSITIVITY  = 0.6   # 0 (strict) … 1 (eager) for detection
FIT_WIN_MULT = 1.8   # local fit half-window ≈ FIT_WIN_MULT * FWHM (per peak)
# =========================


# --------- small helpers (boring but robust) ---------
def _robust_sigma(x):
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))

def _odd(k):  # ensure odd integer >= 5
    k = int(max(5, round(k)))
    return k + 1 if (k % 2 == 0) else k

def linear_bg_strict(q, y, frac_keep=0.60, clip_sigma=2.5, max_iter=10):
    """Strict degree-1 baseline via sigma-clipped least squares."""
    q = np.asarray(q, float); y = np.asarray(y, float); n = y.size
    if n < 3:
        a = float(np.median(y)); b = 0.0
        return a + b*q, {"a": a, "b": b, "iters": 0, "kept": n}
    mask = np.ones(n, bool); kept = n; a = float(np.median(y)); b = 0.0
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
        k = max(5, int(frac_keep*idx.size))
        newmask = np.zeros(n, bool); newmask[idx[order[:k]]] = True
        if newmask.sum()==mask.sum() and np.all(newmask==mask):
            kept = newmask.sum(); break
        mask = newmask; kept = mask.sum()
    return a + b*q, {"a": a, "b": b, "iters": it+1, "kept": kept}


# --------- detection: derivatives only (no SciPy find_peaks) ---------
def detect_peaks_derivative(q, y, width_pts=WIDTH_PTS, sensitivity=SENSITIVITY, hint=None):
    dq = float(np.median(np.diff(q))) if len(q) > 1 else 1.0

    # derive windows/thresholds from 2 knobs
    win_smooth = _odd(width_pts * (1.2 - 0.4*sensitivity))
    win_deriv  = _odd(width_pts * (1.4 - 0.6*sensitivity))
    poly = 3 if win_deriv >= 7 else 2

    K2 = 4.5 - 2.5*sensitivity       # curvature multiplier (eager → smaller)
    MIN_FRAC = 0.06 - 0.04*sensitivity  # amp floor fraction of (y-bg) dyn range
    PROM_FLOOR = 1.0
    min_sep_pts = max(1, int(round((0.6 - 0.25*sensitivity) * width_pts)))
    valley_frac = 0.15 + 0.05*(0.5 - sensitivity)

    # strict linear BG
    bg, _ = linear_bg_strict(q, y)
    y_det = y - bg

    # Savitzky–Golay amplitude line + derivatives
    wl = min(win_smooth, len(y_det)//2*2-1)
    wd = min(win_deriv,  len(y_det)//2*2-1)
    wl = wl if wl >= 5 else 5
    wd = wd if wd >= 5 else 5
    y_line = savgol_filter(y_det, window_length=wl, polyorder=min(poly,3))
    dy  = savgol_filter(y_det, window_length=wd, polyorder=poly, deriv=1, delta=dq)
    d2y = savgol_filter(y_det, window_length=wd, polyorder=poly, deriv=2, delta=dq)

    # candidates: dy +→− and d2y<0
    sgn = np.sign(dy)
    cross = np.where((sgn[:-1] > 0) & (sgn[1:] <= 0))[0] + 1
    cand = cross[(d2y[cross] < 0)]
    if cand.size == 0:
        i0 = int(np.argmax(y_det))
        return dict(idx=np.array([i0]), q=np.array([q[i0]]), y_det=y_det, bg=bg,
                    width_pts_est=np.array([width_pts], float), info=dict(min_sep_pts=min_sep_pts))

    # thresholds
    sig2 = _robust_sigma(d2y)
    dyn  = max(float(y_det.max() - y_det.min()), 1.0)
    thr_curv = max(K2 * sig2, 1e-12)
    thr_amp  = max(MIN_FRAC * dyn, PROM_FLOOR)

    scores = (-d2y[cand]).astype(float)
    amps   = y_det[cand].astype(float)
    keep = (scores >= thr_curv) & (amps >= thr_amp)
    cand = cand[keep]; scores = scores[keep]
    if cand.size == 0:
        i0 = int(np.argmax(y_det))
        return dict(idx=np.array([i0]), q=np.array([q[i0]]), y_det=y_det, bg=bg,
                    width_pts_est=np.array([width_pts], float), info=dict(min_sep_pts=min_sep_pts))

    # simple NMS by points
    order = np.argsort(-scores)
    picked = []
    for o in order:
        ii = int(cand[o])
        if all(abs(ii - p) > min_sep_pts for p in picked):
            picked.append(ii)

    # valley check on smoothed line
    def valley_ok(i, j):
        a,b = (i,j) if i<j else (j,i)
        if b-a < 2: return False
        valley = float(np.min(y_line[a:b+1]))
        hi, hj = float(y_line[i]), float(y_line[j])
        return (min(hi,hj) - valley) >= valley_frac*max(1.0, min(hi,hj))

    kept = []
    for i in picked:
        if not kept:
            kept.append(i)
            continue
        nearest = kept[np.argmin(np.abs(np.array(kept)-i))]
        if valley_ok(i, nearest):
            kept.append(i)

    kept = np.array(sorted(kept), int)
    # crude width estimate from curvature: σ ≈ sqrt(H / -y''); FWHM_pts = 2.355σ / dq
    H  = np.maximum(y_line[kept], 1e-12)
    C2 = np.maximum(-d2y[kept], 1e-12)
    sigma_est = np.sqrt(H / C2)             # in q-units
    fwhm_est  = 2.355 * sigma_est
    width_pts_est = fwhm_est / max(dq, 1e-12)

    return dict(idx=kept, q=q[kept].astype(float), y_det=y_det, bg=bg,
                width_pts_est=width_pts_est.astype(float),
                info=dict(min_sep_pts=min_sep_pts, thr_curv=thr_curv, thr_amp=thr_amp))


# --------- per-peak local fit (1 Gaussian + linear BG) ---------
def fit_peak_local(q, y, i_center, width_pts_est, fit_win_mult=FIT_WIN_MULT):
    """
    Fit one peak in a small neighborhood:
      model = Gaussian(AREA amplitude) + linear background (degree 1).
    Returns (result, slice_mask).
    """
    dq = float(np.median(np.diff(q))) if len(q) > 1 else 1.0
    # convert width estimate to sigma (q-units), pick a local half-width
    fwhm_pts = max(width_pts_est, 2.0)                  # keep sane
    sigma_q  = (fwhm_pts * dq) / 2.355
    half_q   = fit_win_mult * 2.355 * sigma_q           # ≈ fit over ±(fit_win_mult * FWHM)

    q0 = float(q[i_center])
    m = (q >= (q0 - half_q)) & (q <= (q0 + half_q))
    if m.sum() < 7:
        # expand slightly if too tight
        k = max(7, int(2.5 * fwhm_pts))
        a = max(0, i_center - k); b = min(len(q)-1, i_center + k)
        m = np.zeros_like(q, bool); m[a:b+1] = True

    q_loc = q[m]; y_loc = y[m]

    # model: linear BG + Gaussian
    bg = PolynomialModel(degree=1, prefix="bg_")
    g  = GaussianModel(prefix="g_")
    model = bg + g

    # initial params
    p = model.make_params()
    # BG
    p["bg_c0"].set(value=float(np.median(y_loc)), min=0.0)
    p["bg_c1"].set(value=0.0)
    # Gaussian (lmfit Gaussian amplitude = AREA)
    height0 = max(float(y_loc[np.argmax(y_loc)] - np.median(y_loc)), 1.0)
    area0   = height0 * sigma_q * np.sqrt(2*np.pi)
    p["g_center"].set(value=q0, min=q_loc.min(), max=q_loc.max())
    p["g_sigma"].set(value=max(1e-9, sigma_q), min=0.25*sigma_q, max=3.5*sigma_q)
    p["g_amplitude"].set(value=max(1e-9, area0), min=0.05*area0, max=20*area0)

    # light Poisson-ish weights
    w = 1.0 / np.sqrt(np.clip(y_loc, 1.0, None))
    result = model.fit(y_loc, p, x=q_loc, weights=w,
                       method="least_squares",
                       fit_kws={"loss": "soft_l1", "f_scale": 1.0})
    return result, m


# ----------------------------- main ----------------------------- #
def run(h5_path, frame_number, peak_pos, window=0.1, save_png=False):
    # load
    with h5py.File(h5_path, "r") as f:
        Int = f["int"][:]           # (nframes, nq)
        q   = f["q"][:].astype(float)

    # window
    q_min, q_max = float(peak_pos - window), float(peak_pos + window)
    mwin = (q >= q_min) & (q <= q_max)
    q_win = q[mwin]; y_win = Int[int(frame_number), mwin].astype(float)
    if q_win.size < max(20, int(4*WIDTH_PTS)):
        raise ValueError("Window too small or outside q-range.")

    # detect
    det = detect_peaks_derivative(q_win, y_win, WIDTH_PTS, SENSITIVITY, hint=peak_pos)
    idxs = det["idx"]; widths_pts = det["width_pts_est"]

    # per-peak fits
    fits = []
    for i, (ix, wpts) in enumerate(zip(idxs, widths_pts)):
        res, mask = fit_peak_local(q_win, y_win, int(ix), float(wpts))
        # summarize
        c  = res.params["g_center"].value
        s  = res.params["g_sigma"].value
        a  = res.params["g_amplitude"].value   # AREA
        h  = a / (np.sqrt(2*np.pi) * s)        # HEIGHT
        bic = res.bic; aic = res.aic
        # compute simple weighted R² on the local slice
        q_loc = q_win[mask]; y_loc = y_win[mask]
        yhat  = res.model.eval(params=res.params, x=q_loc)
        wloc  = 1.0 / np.sqrt(np.clip(y_loc, 1.0, None))
        resid = y_loc - yhat
        sse = float(np.sum((wloc * resid) ** 2))
        ybar = float(np.sum((wloc**2) * y_loc) / np.sum(wloc**2))
        sst = float(np.sum((wloc * (y_loc - ybar)) ** 2))
        r2  = 1.0 - sse / max(sst, 1e-18)
        fits.append(dict(result=res, mask=mask, center=c, sigma=s, fwhm=2.355*s,
                         area=a, height=h, bic=bic, aic=aic, r2=r2))

    # ---------- plot ----------
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(11, 8), gridspec_kw={"height_ratios": [3, 2]})
    ax0.plot(q_win, y_win, "k.", ms=3, label="Data")

    # show detection BG (what detector saw)
    ax0.plot(q_win, det["bg"], "-", lw=1.0, label="BG (linear, detect)")

    # draw each local fitted Gaussian component on dense grid
    q_dense = np.linspace(q_win.min(), q_win.max(), len(q_win) * 5)
    y_sum = np.zeros_like(q_dense)
    for j, f in enumerate(fits):
        # evaluate only the Gaussian part
        g = GaussianModel(prefix=f"g{j}_")
        # fake a tiny model to eval component easily
        amp = f["area"]; sig = f["sigma"]; cen = f["center"]
        y_g = amp * np.exp(-0.5 * ((q_dense - cen)/sig)**2) / (1.0)  # lmfit gaussian area form handled via model; here manual: area * N(0,σ)
        # area form of Gaussian height = area/(sqrt(2π)σ); profile = height*exp...
        y_g = f["height"] * np.exp(-0.5 * ((q_dense - cen)/sig)**2)
        y_sum += y_g
        ax0.plot(q_dense, y_g, ":", lw=1.0, label=f"g{j}")

        # mark center
        ax0.plot([f["center"]], [np.interp(f["center"], q_dense, y_g + np.interp(f["center"], q_win, det["bg"]))],
                 "x", ms=8)

    ax0.plot(q_dense, y_sum + np.interp(q_dense, q_win, det["bg"]), "-", lw=1.6, label="Sum of fitted Gaussians + BG")

    ax0.set_xlabel("q"); ax0.set_ylabel("Intensity")
    ax0.set_title(f"Frame {frame_number} | detected={len(idxs)} | fitted={len(fits)}")
    ax0.legend(loc="best", fontsize=8)

    # table of per-peak fit params
    ax1.axis("off")
    rows = [["Peak", "Center", "Height", "Sigma", "FWHM", "Area", "Local R²"]]
    for j, f in enumerate(fits):
        rows.append([
            f"{j}",
            f"{f['center']:.6f}",
            f"{f['height']:.3g}",
            f"{f['sigma']:.6g}",
            f"{f['fwhm']:.6g}",
            f"{f['area']:.3g}",
            f"{f['r2']:.4f}",
        ])
    table = ax1.table(cellText=rows, colLabels=None, loc="center", cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1.1, 1.18)

    plt.tight_layout()
    if save_png:
        import os
        base = os.path.splitext(os.path.basename(h5_path))[0]
        out = f"{base}_f{frame_number}_q{peak_pos:.5f}_w{window:.3f}_localfits.png"
        fig.savefig(out, dpi=220, bbox_inches="tight"); print(f"[saved] {out}")
    plt.show()

    return dict(detected=len(idxs), fitted=len(fits), fits=fits)


# ---------- CLI ----------
def _parse_args():
    p = argparse.ArgumentParser(description="Derivative-detect + per-peak local Gaussian fits (linear BG).")
    p.add_argument("h5", type=str)
    p.add_argument("frame_number", type=int)
    p.add_argument("peak_pos", type=float)
    p.add_argument("--window", type=float, default=0.1)
    p.add_argument("--save-png", action="store_true")
    return p.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    run(args.h5, args.frame_number, args.peak_pos, args.window, save_png=args.save_png)
