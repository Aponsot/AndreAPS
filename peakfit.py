#!/usr/bin/env python3
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter  # derivatives + smoothing

# ======== ONLY TWO KNOBS ========
WIDTH_PTS   = 6.0   # your typical points across a peak FWHM in this window
SENSITIVITY = 0.6   # 0 (strict) … 1 (eager). Higher = more/closer peaks
# ================================

# fixed internals (kept simple)
VALLEY_FRAC_BASE = 0.15    # baseline valley depth requirement
MAX_CAND         = 24      # hard cap for candidates before NMS

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
        k = max(5, int(frac_keep * idx.size))
        newmask = np.zeros(n, bool); newmask[idx[order[:k]]] = True
        if newmask.sum()==mask.sum() and np.all(newmask==mask):
            kept = newmask.sum(); break
        mask = newmask; kept = mask.sum()
    return a + b*q, {"a": a, "b": b, "iters": it+1, "kept": kept}

def nms_by_points(idxs, scores, radius_pts):
    """Non-maximum suppression in sample space."""
    order = np.argsort(-scores)
    kept = []
    for o in order:
        i = int(idxs[o])
        if all(abs(i - k) > radius_pts for k in kept):
            kept.append(i)
    return np.array(kept, int)

def valley_ok(yline, i, j, frac):
    """Require a real valley of fractional depth between peaks at i and j."""
    a, b = (i, j) if i < j else (j, i)
    if b - a < 2:  # too close to evaluate a valley
        return False
    valley = float(np.min(yline[a:b+1]))
    h_i, h_j = float(yline[i]), float(yline[j])
    return (min(h_i, h_j) - valley) >= frac * max(1.0, min(h_i, h_j))

def detect_peaks_derivative(q, y, hint=None):
    """
    Derivative-based peak detection:
      - strict linear BG
      - Savitzky–Golay 1st/2nd derivatives
      - zero-crossings of dy ( +→− ) with negative curvature (d2y<0)
      - NMS + valley gating
    Returns dict with arrays: idx, q, height_bgsub, width_pts_est, snr, curvature.
    """
    # 0) derive all thresholds from the two knobs
    #    (higher sensitivity → longer windows are *smaller* and thresholds looser)
    dq = float(np.median(np.diff(q))) if len(q) > 1 else 1.0
    win_smooth = _odd(WIDTH_PTS * (1.2 - 0.4*SENSITIVITY))      # amplitude line
    win_deriv  = _odd(WIDTH_PTS * (1.4 - 0.6*SENSITIVITY))      # derivatives line
    poly = 3 if win_deriv >= 7 else 2

    # curvature threshold (in y'' units); eager => lower K2
    K2 = 4.5 - 2.5*SENSITIVITY     # 4.5 → 2.0 as SENS 0→1
    # amplitude floor (fraction of dynamic range of y-bg)
    MIN_FRAC = 0.06 - 0.04*SENSITIVITY   # 6% → 2%
    PROM_FLOOR = 1.0
    # min separation for NMS (in points)
    min_sep_pts = max(1, int(round((0.6 - 0.25*SENSITIVITY) * WIDTH_PTS)))
    # valley fraction
    valley_frac = VALLEY_FRAC_BASE + 0.05*(0.5 - SENSITIVITY)   # ~0.175 at SENS=0.25

    # 1) strict linear BG and bg-sub
    bg, _ = linear_bg_strict(q, y)
    y_det = y - bg

    # 2) smooth amplitude and compute derivatives (Savitzky–Golay)
    y_line = savgol_filter(y_det, window_length=min(win_smooth, len(y_det)//2*2-1), polyorder=min(poly, 3))
    dy  = savgol_filter(y_det, window_length=min(win_deriv, len(y_det)//2*2-1), polyorder=poly, deriv=1, delta=dq)
    d2y = savgol_filter(y_det, window_length=min(win_deriv, len(y_det)//2*2-1), polyorder=poly, deriv=2, delta=dq)

    # 3) candidate indices: zero-crossing of dy from + → − and negative curvature
    sgn = np.sign(dy)
    cross = np.where((sgn[:-1] > 0) & (sgn[1:] <= 0))[0] + 1  # zero-crossings at i where dy[i-1]>0, dy[i]<=0
    if cross.size == 0:
        # fallback: take global max of y_det within window
        i0 = int(np.argmax(y_det))
        return dict(idx=np.array([i0]), q=np.array([q[i0]]), height_bgsub=np.array([float(y_det[i0])]),
                    width_pts_est=np.array([WIDTH_PTS], float), snr=np.array([1.0]), curvature=np.array([float(-d2y[i0])]),
                    y_det=y_det, bg=bg, thr_curv=np.nan, thr_amp=np.nan)

    cand = cross[(d2y[cross] < 0)]  # must be a maximum
    if cand.size == 0:
        i0 = int(np.argmax(y_det))
        return dict(idx=np.array([i0]), q=np.array([q[i0]]), height_bgsub=np.array([float(y_det[i0])]),
                    width_pts_est=np.array([WIDTH_PTS], float), snr=np.array([1.0]), curvature=np.array([float(-d2y[i0])]),
                    y_det=y_det, bg=bg, thr_curv=np.nan, thr_amp=np.nan)

    # 4) scores and thresholds
    # curvature noise: robust sigma of d2y away from peaks → approximate by global
    sig2 = _robust_sigma(d2y)
    dyn  = max(float(y_det.max() - y_det.min()), 1.0)
    thr_curv = max(K2 * sig2, 1e-12)
    thr_amp  = max(MIN_FRAC * dyn, PROM_FLOOR)

    scores = (-d2y[cand]).astype(float)
    amps   = y_det[cand].astype(float)

    mask = (scores >= thr_curv) & (amps >= thr_amp)
    cand = cand[mask]; scores = scores[mask]; amps = amps[mask]
    if cand.size == 0:
        i0 = int(np.argmax(y_det))
        return dict(idx=np.array([i0]), q=np.array([q[i0]]), height_bgsub=np.array([float(y_det[i0])]),
                    width_pts_est=np.array([WIDTH_PTS], float), snr=np.array([float(y_det[i0]/max(_robust_sigma(y_det),1e-9))]),
                    curvature=np.array([float(-d2y[i0])]), y_det=y_det, bg=bg, thr_curv=thr_curv, thr_amp=thr_amp)

    # 5) NMS and valley gating
    if cand.size > MAX_CAND:
        take = np.argsort(-scores)[:MAX_CAND]
        cand, scores = cand[take], scores[take]

    keep = nms_by_points(cand, scores, min_sep_pts)

    # valley check against nearest kept
    kept = []
    for i in keep[np.argsort(-scores[np.isin(cand, keep)])]:
        if not kept:
            kept.append(int(i)); continue
        nearest = kept[np.argmin(np.abs(np.array(kept) - i))]
        if valley_ok(y_line, int(i), int(nearest), valley_frac):
            kept.append(int(i))

    if not kept:
        i0 = int(np.argmax(y_det))
        kept = [i0]

    kept = np.array(kept, int)

    # 6) width estimate from curvature (Gaussian: y''(μ) ≈ -H/σ² ⇒ σ ≈ sqrt(H / -y''))
    H  = np.maximum(y_line[kept], 1e-12)
    C2 = np.maximum(-d2y[kept], 1e-12)
    sigma_est = np.sqrt(H / C2)                 # in x-units (same as q)
    fwhm_est  = 2.355 * sigma_est
    width_pts = fwhm_est / max(dq, 1e-12)
    snr = H / max(_robust_sigma(y_det - y_line), 1e-9)

    return dict(
        idx=kept,
        q=q[kept].astype(float),
        height_bgsub=H.astype(float),
        width_pts_est=width_pts.astype(float),
        snr=snr.astype(float),
        curvature=C2.astype(float),
        y_det=y_det,
        bg=bg,
        thr_curv=thr_curv,
        thr_amp=thr_amp,
        info=dict(win_smooth=int(win_smooth), win_deriv=int(win_deriv), min_sep_pts=int(min_sep_pts))
    )

def main(h5_path, frame_number, peak_pos, window=0.1, save_png=False):
    # load
    with h5py.File(h5_path, "r") as f:
        Int = f["int"][:]  # (nframes, nq)
        q   = f["q"][:].astype(float)

    # window
    q_min, q_max = float(peak_pos - window), float(peak_pos + window)
    m = (q >= q_min) & (q <= q_max)
    q_win = q[m]; y_win = Int[int(frame_number), m].astype(float)
    if q_win.size < max(20, int(4*WIDTH_PTS)):
        raise ValueError("Window too small or outside q-range.")

    det = detect_peaks_derivative(q_win, y_win, hint=peak_pos)

    # print compact table
    print(f"[detect-deriv] peaks={len(det['idx'])} | thr_curv={det['thr_curv']:.3g}  thr_amp≥{det['thr_amp']:.3g} "
          f"| windows(smooth,deriv)=({det['info']['win_smooth']},{det['info']['win_deriv']}) "
          f"| min_sep_pts={det['info']['min_sep_pts']}  | knobs: WIDTH_PTS={WIDTH_PTS}, SENSITIVITY={SENSITIVITY}")
    print(" idx   q_pos         height_bgsub   width_pts_est   curvature     SNR")
    for ix, qq, hh, ww, cc, ss in zip(det["idx"], det["q"], det["height_bgsub"],
                                      det["width_pts_est"], det["curvature"], det["snr"]):
        print(f"{ix:4d}  {qq: .6f}    {hh: .3g}        {ww: .2f}        {cc: .3g}    {ss: .2f}")

    # plot
    fig, ax = plt.subplots(1, 1, figsize=(11, 4.6))
    ax.plot(q_win, y_win, "k.", ms=3, label="Data")
    ax.plot(q_win, det["bg"], "-", lw=1.0, label="BG (linear)")
    ax.plot(q_win, det["y_det"], "-", lw=1.0, alpha=0.55, label="Data - BG")

    # mark peaks
    for qpk, hpk in zip(det["q"], det["height_bgsub"]):
        ax.axvline(qpk, color="tab:red", ls="--", lw=1.0, alpha=0.85)
        ax.plot(qpk, hpk + np.interp(qpk, q_win, det["bg"]), "x", ms=8, color="tab:red")

    ax.set_xlabel("q"); ax.set_ylabel("Intensity")
    ax.set_title(f"Frame {frame_number} | derivative peaks: {len(det['idx'])}")
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()

    if save_png:
        import os
        base = os.path.splitext(os.path.basename(h5_path))[0]
        out = f"{base}_f{frame_number}_q{peak_pos:.5f}_w{window:.3f}_deriv_peaks.png"
        fig.savefig(out, dpi=220, bbox_inches="tight")
        print(f"[saved] {out}")

    plt.show()
    return det

# ---------- CLI ----------
def _parse_args():
    p = argparse.ArgumentParser(description="Derivative-only peak detection (strict linear BG).")
    p.add_argument("h5", type=str)
    p.add_argument("frame_number", type=int)
    p.add_argument("peak_pos", type=float)
    p.add_argument("--window", type=float, default=0.1)
    p.add_argument("--save-png", action="store_true")
    return p.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    main(args.h5, args.frame_number, args.peak_pos, args.window, save_png=args.save_png)
