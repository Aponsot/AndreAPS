#!/usr/bin/env python3
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.ndimage import gaussian_filter1d

# ========= ONLY TWO TUNABLES =========
WIDTH_PTS    = 6.0   # typical points across a peak FWHM in your window
SENSITIVITY  = 0.6   # 0 (conservative) … 1 (eager). Higher = find more shoulders
# =====================================

# fixed, sane internals (don’t touch unless you must)
VALLEY_FRAC = 0.15            # require ≥15% valley depth to accept close neighbors
MS_SCALES   = [1.0, 1.6]      # matched-filter scales (× WIDTH_PTS)
MAX_CAND    = 12

def robust_sigma(x):
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))

def linear_bg_strict(q, y, frac_keep=0.60, clip_sigma=2.5, max_iter=10):
    q = np.asarray(q, float); y = np.asarray(y, float)
    n = y.size
    if n < 3:
        a = float(np.median(y)); b = 0.0
        return a + b*q, dict(a=a,b=b,kept=n,iters=0)
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
    return a + b*q, dict(a=a,b=b,kept=kept,iters=it+1)

def nms(idx, scores, radius_pts):
    order = np.argsort(-scores)
    picked = []
    for o in order:
        i = int(idx[o])
        if all(abs(i - p) > radius_pts for p in picked):
            picked.append(i)
    return np.array(picked, int)

def valley_ok(y_line, i, j, frac=VALLEY_FRAC):
    a,b = (i,j) if i<j else (j,i)
    if b-a < 2: return False
    valley = float(np.min(y_line[a:b+1]))
    h_i, h_j = float(y_line[i]), float(y_line[j])
    return (min(h_i,h_j) - valley) >= frac * max(1.0, min(h_i,h_j))

def detect_peaks(q, y, hint=None):
    # 0) derive thresholds from the single SENSITIVITY knob
    #    (higher sensitivity → lower thresholds)
    K_PROM    = 5.0 - 2.0*SENSITIVITY      # 5 → 3 as SENS goes 0→1
    MIN_FRAC  = 0.06 - 0.04*SENSITIVITY    # 6% → 2%
    PROM_FLOOR= 1.0

    # 1) strict linear BG, bg-sub signal
    bg, _ = linear_bg_strict(q, y)
    y_det = y - bg

    # 2) noise & thresholds
    hp_sigma_pts = max(3.0, 0.5*WIDTH_PTS)
    smooth = gaussian_filter1d(y_det, sigma=hp_sigma_pts) if len(y_det)>7 else y_det
    sig = robust_sigma(y_det - smooth)
    dyn = max(float(y_det.max() - y_det.min()), 1.0)
    prom_thr = max(K_PROM*sig, MIN_FRAC*dyn, PROM_FLOOR)

    wmin = max(1, int(round(0.4*WIDTH_PTS)))
    wmax = int(round(10.0*WIDTH_PTS))
    nms_rad = max(1, int(round(0.5*WIDTH_PTS)))

    # 3) multi-scale matched filtering → raw candidates
    c_idx, c_prom, c_w = [], [], []
    for sc in MS_SCALES:
        sigma_pts = max(1.0, (WIDTH_PTS*sc)/2.355)
        y_s = gaussian_filter1d(y_det, sigma=sigma_pts)
        pks, props = signal.find_peaks(y_s, prominence=prom_thr, width=(wmin,wmax), rel_height=0.5)
        if pks.size:
            prom = props["prominences"] if "prominences" in props else np.ones_like(pks, float)
            wid  = props["widths"] if "widths" in props else np.full_like(pks, WIDTH_PTS, float)
            c_idx.append(pks.astype(int)); c_prom.append(prom.astype(float)); c_w.append(wid.astype(float))
    if not c_idx:
        # fallback: one at the hint (or global max of bg-sub)
        i0 = int(np.argmin(np.abs(q - hint))) if hint is not None else int(np.argmax(y_det))
        return dict(idx=np.array([i0]), q=np.array([q[i0]]), height_bgsub=np.array([float(y_det[i0])]),
                    width_pts=np.array([WIDTH_PTS], float), prom=np.array([float(y_det[i0])]), snr=np.array([float(y_det[i0]/max(sig,1e-9))]),
                    y_det=y_det, bg=bg, prom_thr=prom_thr)

    c_idx  = np.concatenate(c_idx)
    c_prom = np.concatenate(c_prom)
    c_w    = np.concatenate(c_w)

    # 4) non-maximum suppression
    keep = nms(c_idx, c_prom, nms_rad)
    mask = np.isin(c_idx, keep)
    c_idx, c_prom, c_w = c_idx[mask], c_prom[mask], c_w[mask]

    # 5) sort (if hint given, prioritize closeness; else by prominence)
    if hint is not None:
        order = np.argsort(np.abs(q[c_idx] - hint) + 1e-9*(-c_prom))
    else:
        order = np.argsort(-c_prom)
    c_idx, c_prom, c_w = c_idx[order], c_prom[order], c_w[order]

    # 6) valley gating on a lightly smoothed bg-sub trace
    y_line = gaussian_filter1d(y_det, sigma=max(1.0, WIDTH_PTS/2.355))
    kept = []
    for i in c_idx:
        if not kept:
            kept.append(int(i)); continue
        nearest = kept[np.argmin(np.abs(np.array(kept)-i))]
        if valley_ok(y_line, int(i), int(nearest)):
            kept.append(int(i))
        # If there is no clear valley, skip (prevents splitting noise shoulders)

    if len(kept) > MAX_CAND:
        kept = kept[:MAX_CAND]

    kept = np.array(kept, int)
    # attach width/prom from nearest candidate entries
    width_pts = np.zeros_like(kept, float)
    prom_vals = np.zeros_like(kept, float)
    for j, k in enumerate(kept):
        jj = int(np.argmin(np.abs(c_idx - k)))
        width_pts[j] = float(c_w[jj]); prom_vals[j] = float(c_prom[jj])

    heights = y_det[kept].astype(float)
    snr = heights / max(sig, 1e-9)

    return dict(idx=kept, q=q[kept].astype(float), height_bgsub=heights,
                width_pts=width_pts, prom=prom_vals, snr=snr,
                y_det=y_det, bg=bg, prom_thr=prom_thr)

def main(h5_path, frame_number, peak_pos, window=0.1, save_png=False):
    with h5py.File(h5_path, "r") as f:
        Int = f["int"][:]       # (nframes, nq)
        q   = f["q"][:].astype(float)

    q_min, q_max = float(peak_pos - window), float(peak_pos + window)
    m = (q >= q_min) & (q <= q_max)
    q_win = q[m]; y_win = Int[int(frame_number), m].astype(float)
    if q_win.size < max(20, int(4*WIDTH_PTS)):
        raise ValueError("Window too small or outside q-range.")

    det = detect_peaks(q_win, y_win, hint=peak_pos)

    # print compact table
    print(f"[detect] prom≥{det['prom_thr']:.3g}  |  peaks: {len(det['idx'])}  |  knobs: WIDTH_PTS={WIDTH_PTS}, SENSITIVITY={SENSITIVITY}")
    print(" idx   q_pos         height_bgsub   width_pts   SNR")
    for ix, qq, hh, ww, ss in zip(det["idx"], det["q"], det["height_bgsub"], det["width_pts"], det["snr"]):
        print(f"{ix:4d}  {qq: .6f}    {hh: .3g}       {ww: .2f}     {ss: .2f}")

    # plot
    fig, ax = plt.subplots(1, 1, figsize=(11, 4.5))
    ax.plot(q_win, y_win, "k.", ms=3, label="Data")
    ax.plot(q_win, det["bg"], "-", lw=1.0, label="BG (linear)")
    ax.plot(q_win, det["y_det"], "-", lw=1.0, alpha=0.55, label="Data - BG")

    for qpk, hpk in zip(det["q"], det["height_bgsub"]):
        ax.axvline(qpk, color="tab:red", ls="--", lw=1.0, alpha=0.85)
        ax.plot(qpk, hpk + np.interp(qpk, q_win, det["bg"]), "x", ms=8, color="tab:red")

    ax.set_xlabel("q"); ax.set_ylabel("Intensity")
    ax.set_title(f"Frame {frame_number} | detected peaks: {len(det['idx'])}")
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()

    if save_png:
        import os
        base = os.path.splitext(os.path.basename(h5_path))[0]
        out = f"{base}_f{frame_number}_q{peak_pos:.5f}_w{window:.3f}_peaks.png"
        fig.savefig(out, dpi=220, bbox_inches="tight")
        print(f"[saved] {out}")

    plt.show()
    return det

# ---------- CLI ----------
def _parse_args():
    p = argparse.ArgumentParser(description="Minimal peak detection (strict linear BG).")
    p.add_argument("h5", type=str)
    p.add_argument("frame_number", type=int)
    p.add_argument("peak_pos", type=float)
    p.add_argument("--window", type=float, default=0.1)
    p.add_argument("--save-png", action="store_true")
    return p.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    main(args.h5, args.frame_number, args.peak_pos, args.window, save_png=args.save_png)
