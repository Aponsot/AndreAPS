#!/usr/bin/env python3
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
from lmfit.models import GaussianModel, LinearModel

# ------------------------ helpers ------------------------ #
def robust_sigma(y):
    y = np.asarray(y, float)
    med = np.median(y)
    return 1.4826 * np.median(np.abs(y - med)) + 1e-12

def fwhm_to_sigma(fwhm):
    return fwhm / 2.354820045

def compute_r2(y, yfit):
    y = np.asarray(y, float); yfit = np.asarray(yfit, float)
    ss_res = np.sum((y - yfit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2) + 1e-16
    return 1.0 - ss_res / ss_tot

def smooth_ma(y, win=7):
    n = max(1, int(win))
    if n % 2 == 0: n += 1
    if n <= 1: return y.copy()
    k = np.ones(n, float) / n
    ypad = np.pad(y, (n//2, n//2), mode="edge")
    return np.convolve(ypad, k, mode="valid")

def derivative_maxima(x, y, smooth_win=7, min_sep_pts=3):
    ys = smooth_ma(y, smooth_win)
    dx = float(np.mean(np.diff(x))) if len(x) > 1 else 1.0
    dy = np.gradient(ys, dx)
    d2y = np.gradient(dy, dx)
    # + to - zero-crossings
    sgn = np.sign(dy)
    cand = np.where((sgn[:-1] > 0) & (sgn[1:] <= 0))[0] + 1
    cand = cand[d2y[cand] < 0]
    if cand.size == 0:
        return cand
    # NMS by height
    order = np.argsort(ys[cand])[::-1]
    kept = []; taken = np.zeros(len(ys), dtype=bool)
    for idx in cand[order]:
        if not taken[max(0, idx-min_sep_pts): idx+min_sep_pts+1].any():
            kept.append(idx)
            taken[max(0, idx-min_sep_pts): idx+min_sep_pts+1] = True
    kept.sort()
    return np.array(kept, int)

def curvature_minima(x, y, smooth_win=7, min_sep_pts=3, kappa=1.0):
    """
    Seed centers from minima of the second derivative (most negative curvature).
    kappa is a threshold in units of robust_sigma(d2y).
    """
    ys = smooth_ma(y, smooth_win)
    dx = float(np.mean(np.diff(x))) if len(x) > 1 else 1.0
    d1 = np.gradient(ys, dx)
    d2 = np.gradient(d1, dx)
    # local minima in d2
    m = (d2[1:-1] < d2[:-2]) & (d2[1:-1] < d2[2:])
    idx = np.where(m)[0] + 1
    if idx.size == 0:
        return idx
    # keep only “sufficiently negative” curvature
    s2 = robust_sigma(d2)
    idx = idx[d2[idx] < -kappa * s2]
    if idx.size == 0:
        return idx
    # NMS by |d2| magnitude
    order = np.argsort(d2[idx])  # most negative first
    kept = []; taken = np.zeros(len(d2), dtype=bool)
    for i in idx[order]:
        if not taken[max(0, i-min_sep_pts): i+min_sep_pts+1].any():
            kept.append(i)
            taken[max(0, i-min_sep_pts): i+min_sep_pts+1] = True
    kept.sort()
    return np.array(kept, int)

def build_model_from_centers(xw, yw, centers, min_sigma, max_sigma, baseline):
    bkg = LinearModel(prefix="bkg_")
    model = bkg
    params = bkg.make_params(slope=0.0, intercept=baseline)
    # add one Gaussian per center
    for i, cx in enumerate(centers):
        gi = GaussianModel(prefix=f"g{i}_")
        model += gi
        # initial sigma from half-height crossings around nearest data point
        p = np.argmin(np.abs(xw - cx))
        # simple half-height-based width estimate
        ypk = yw[p]; h = max(ypk - baseline, 1e-12); yhalf = baseline + 0.5*h
        li = p;  ri = p
        while li > 0 and yw[li] > yhalf: li -= 1
        while ri < len(yw)-1 and yw[ri] > yhalf: ri += 1
        fwhm0 = max((ri - li), 3) * np.mean(np.diff(xw)) if len(xw) > 1 else max_sigma
        sigma0 = np.clip(fwhm_to_sigma(fwhm0), min_sigma, max_sigma)

        height0 = max(ypk - baseline, robust_sigma(yw))
        amp0 = height0 * sigma0 * np.sqrt(2*np.pi)

        params.update(gi.make_params(center=cx, sigma=sigma0, amplitude=amp0))
        params[f"g{i}_center"].set(min=xw[0], max=xw[-1], value=cx)
        params[f"g{i}_sigma"].set(min=min_sigma, max=max_sigma, value=sigma0)
        params[f"g{i}_amplitude"].set(min=0.0, value=amp0)
    return model, params

# ------------------------ core ------------------------ #
def fit_peaks_curvature_residual(
    h5_path, frame, center, window=0.1, plot=True,
    smooth_win=7, min_sep_pts=3, min_height_sigma=1.5,
    max_sigma_frac=0.22,           # cap σ to avoid one super-wide peak
    residual_add_iters=2,          # try adding up to N extra peaks
    residual_snr=1.5,              # require residual peak > N*noise
    aic_improve=4.0                # keep new peak only if AIC drops by this
):
    with h5py.File(h5_path, "r") as f:
        x = f["q"][:] if "q" in f else f["tth"][:]
        I = f["int"][:]

    yfull = np.asarray(I[frame], float)
    x = np.asarray(x, float)

    half = window/2.0
    m = (x >= center-half) & (x <= center+half)
    xw, yw = x[m], yfull[m]
    mfin = np.isfinite(xw) & np.isfinite(yw)
    xw, yw = xw[mfin], yw[mfin]
    if xw.size < 5:
        raise ValueError("Too few points in window.")

    dx = float(np.mean(np.diff(xw))) if len(xw) > 1 else window
    baseline = np.median(yw)
    noise = robust_sigma(yw)

    # ---- seed centers: derivative maxima + curvature minima ----
    pk = derivative_maxima(xw, yw, smooth_win=smooth_win, min_sep_pts=min_sep_pts)
    cm = curvature_minima(xw, yw, smooth_win=smooth_win, min_sep_pts=min_sep_pts, kappa=1.0)

    seeds = np.unique(np.r_[pk, cm]).tolist()
    if not seeds:
        seeds = [np.abs(xw - center).argmin()]  # fallback single seed
    centers = [float(xw[i]) for i in seeds]
    centers.sort()

    # ---- bounds to discourage a single huge σ ----
    min_sigma = max(dx/3.0, 1e-6)
    max_sigma = max_sigma_frac * window

    # ---- initial fit ----
    model, params = build_model_from_centers(xw, yw, centers, min_sigma, max_sigma, baseline)
    result = model.fit(yw, params, x=xw)
    best_aic = result.aic

    # ---- iteratively add peaks from positive residual if justified ----
    for _ in range(residual_add_iters):
        res = yw - result.best_fit
        rnoise = robust_sigma(res)
        i_new = int(np.argmax(res))
        if res[i_new] < residual_snr * rnoise:
            break  # no strong leftover bump

        # add a new center at residual max and refit; keep only if AIC improves
        centers_plus = centers + [float(xw[i_new])]
        centers_plus.sort()
        model2, params2 = build_model_from_centers(xw, yw, centers_plus, min_sigma, max_sigma, baseline)
        result2 = model2.fit(yw, params2, x=xw)
        if result2.aic < best_aic - aic_improve:
            result = result2
            best_aic = result2.aic
            centers = centers_plus
        else:
            break

    # ---- collect kept peaks (prominence-like filtering) ----
    rows = []
    for i in range(len(centers)):
        amp = result.params[f"g{i}_amplitude"].value
        ctr = result.params[f"g{i}_center"].value
        sig = result.params[f"g{i}_sigma"].value
        hgt = amp / (sig * np.sqrt(2*np.pi))
        fwhm = 2*np.sqrt(2*np.log(2)) * sig
        if hgt >= (min_height_sigma * noise):
            rows.append([i, ctr, hgt, fwhm, amp])

    bkg_slope = result.params["bkg_slope"].value
    bkg_intercept = result.params["bkg_intercept"].value
    r2 = compute_r2(yw, result.best_fit)

    # ---- plot (same style you liked) ----
    if plot:
        plt.rcParams.update({
            "figure.dpi": 160, "savefig.dpi": 300,
            "font.size": 16, "axes.labelsize": 18, "axes.titlesize": 20,
            "xtick.labelsize": 14, "ytick.labelsize": 14,
        })
        fig, (ax, ax_tbl) = plt.subplots(2, 1, figsize=(10, 6.8),
                                         gridspec_kw={"height_ratios":[3,1]})
        ax.plot(xw, yw, lw=1.8, label="Data")
        ax.plot(xw, result.best_fit, lw=2.2, label="Fit")
        comps = result.eval_components(x=xw)
        if "bkg_" in comps:
            ax.plot(xw, comps["bkg_"], ls="--", label="Background")
        for i in range(len(centers)):
            key = f"g{i}_"
            if key in comps:
                ax.plot(xw, comps[key], ls=":", label=f"Peak {i}")
                ax.axvline(result.params[f"g{i}_center"].value, alpha=0.25)
        ax.set_xlabel("q or 2θ"); ax.set_ylabel("Intensity")
        ax.set_title(
            f"Frame {frame} | window [{center-half:.5f}, {center+half:.5f}] | "
            f"R²={r2:.4f} | bkg: y={bkg_slope:.3g}·x+{bkg_intercept:.3g} | AIC={best_aic:.1f}"
        )
        ax.legend(loc="best"); ax.grid(alpha=0.3)

        ax_tbl.axis("off")
        cols = ["Peak #", "Center", "Height", "FWHM", "Amplitude"]
        table = ax_tbl.table(
            cellText=[[f"{r[0]}", f"{r[1]:.6g}", f"{r[2]:.6g}", f"{r[3]:.6g}", f"{r[4]:.6g}"] for r in rows] or [["—"]*5],
            colLabels=cols, loc="center"
        )
        table.auto_set_font_size(False); table.set_fontsize(12); table.scale(1, 1.25)
        plt.tight_layout(); plt.show()

    return {
        "frame": frame,
        "window": (center - half, center + half),
        "background": {"slope": bkg_slope, "intercept": bkg_intercept},
        "r2": r2,
        "rows": rows,
        "result": result,
        "x": xw, "y": yw, "yfit": result.best_fit,
        "noise": noise,
        "aic": best_aic,
    }

# ------------------------ CLI ------------------------ #
def main():
    ap = argparse.ArgumentParser(description="Curvature + residual peak seeding with Gaussian(s) + linear background.")
    ap.add_argument("h5"); ap.add_argument("frame", type=int); ap.add_argument("center", type=float)
    ap.add_argument("--window", type=float, default=0.1)
    ap.add_argument("--smooth-win", type=int, default=7)
    ap.add_argument("--min-sep-pts", type=int, default=3)
    ap.add_argument("--min-height-sigma", type=float, default=1.5)
    ap.add_argument("--max-sigma-frac", type=float, default=0.22)
    ap.add_argument("--residual-add-iters", type=int, default=2)
    ap.add_argument("--residual-snr", type=float, default=1.5)
    ap.add_argument("--aic-improve", type=float, default=4.0)
    args = ap.parse_args()

    out = fit_peaks_curvature_residual(
        args.h5, args.frame, args.center, window=args.window,
        smooth_win=args.smooth_win, min_sep_pts=args.min_sep_pts, min_height_sigma=args.min_height_sigma,
        max_sigma_frac=args.max_sigma_frac, residual_add_iters=args.residual_add_iters,
        residual_snr=args.residual_snr, aic_improve=args.aic_improve
    )

    print(f"\nFrame {out['frame']}  window: {out['window'][0]:.6f}..{out['window'][1]:.6f}")
    print(f"Background: slope={out['background']['slope']:.6g}, intercept={out['background']['intercept']:.6g}")
    print(f"Noise: {out['noise']:.6g}  R^2: {out['r2']:.6g}  AIC: {out['aic']:.2f}")
    for r in out["rows"]:
        print(f"Peak {r[0]}: center={r[1]:.6g}, height={r[2]:.6g}, FWHM={r[3]:.6g}, amplitude={r[4]:.6g}")

if __name__ == "__main__":
    main()
