#!/usr/bin/env python3
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
from lmfit.models import GaussianModel, LinearModel

# ========================== TUNABLES ==========================
WINDOW = 0.15            # fit window width in x-units (q or 2θ)
SMOOTH_WIN = 9       # moving-average window (odd int)
MIN_SEP_PTS = 2      # min pts between derivative maxima / curvature seeds
MIN_HEIGHT_SIGMA = 10  # keep peaks with height >= N * noise (MAD)
MAX_SIGMA_FRAC = 0.22   # cap σ as fraction of WINDOW to avoid 1 huge Gaussian
RESIDUAL_ADD_ITERS = 1  # try adding up to N peaks from positive residual
RESIDUAL_SNR = 1.8      # residual bump must exceed N * residual noise
AIC_IMPROVE = 6.0       # require this much AIC drop to keep an added peak

# --- post-fit merge rules to avoid tiny extra peaks next to a main one ---
MERGE_MIN_SEP_FRAC = 5   # if centers closer than this * avg FWHM, consider merge
MERGE_HEIGHT_FRAC = 0.58   # merge if smaller peak height < this * larger height
MERGE_AIC_TOL = 100 # allow slight AIC increase when simplifying the model
# ===============================================================



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
    sgn = np.sign(dy)
    cand = np.where((sgn[:-1] > 0) & (sgn[1:] <= 0))[0] + 1
    cand = cand[d2y[cand] < 0]
    if cand.size == 0:
        return cand
    order = np.argsort(ys[cand])[::-1]
    kept = []; taken = np.zeros(len(ys), dtype=bool)
    for idx in cand[order]:
        if not taken[max(0, idx-min_sep_pts): idx+min_sep_pts+1].any():
            kept.append(idx)
            taken[max(0, idx-min_sep_pts): idx+min_sep_pts+1] = True
    kept.sort()
    return np.array(kept, int)

def curvature_minima(x, y, smooth_win=7, min_sep_pts=3, kappa=1.0):
    ys = smooth_ma(y, smooth_win)
    dx = float(np.mean(np.diff(x))) if len(x) > 1 else 1.0
    d1 = np.gradient(ys, dx)
    d2 = np.gradient(d1, dx)
    m = (d2[1:-1] < d2[:-2]) & (d2[1:-1] < d2[2:])
    idx = np.where(m)[0] + 1
    if idx.size == 0:
        return idx
    s2 = robust_sigma(d2)
    idx = idx[d2[idx] < -kappa * s2]
    if idx.size == 0:
        return idx
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

    for i, cx in enumerate(centers):
        gi = GaussianModel(prefix=f"g{i}_")
        model += gi

        p = np.argmin(np.abs(xw - cx))
        ypk = yw[p]; h = max(ypk - baseline, 1e-12); yhalf = baseline + 0.5*h
        li = p;  ri = p
        while li > 0 and yw[li] > yhalf: li -= 1
        while ri < len(yw)-1 and yw[ri] > yhalf: ri += 1
        fwhm0 = max((ri - li), 3) * np.mean(np.diff(xw)) if len(xw) > 1 else max_sigma
        sigma0 = float(np.clip(fwhm_to_sigma(fwhm0), min_sigma, max_sigma))

        height0 = max(ypk - baseline, robust_sigma(yw))
        amp0 = height0 * sigma0 * np.sqrt(2*np.pi)

        params.update(gi.make_params(center=cx, sigma=sigma0, amplitude=amp0))
        params[f"g{i}_center"].set(min=xw[0], max=xw[-1], value=cx)
        params[f"g{i}_sigma"].set(min=min_sigma, max=max_sigma, value=sigma0)
        params[f"g{i}_amplitude"].set(min=0.0, value=amp0)

    return model, params

def collect_stats(result):
    """Return list of dicts [{index, center, height, fwhm, amplitude}]"""
    out = []
    i = 0
    while True:
        c = f"g{i}_center"
        s = f"g{i}_sigma"
        a = f"g{i}_amplitude"
        if c not in result.params: break
        ctr = result.params[c].value
        sig = result.params[s].value
        amp = result.params[a].value
        hgt = amp / (sig * np.sqrt(2*np.pi)) if sig > 0 else 0.0
        fwhm = 2*np.sqrt(2*np.log(2)) * sig
        out.append({"index": i, "center": ctr, "height": hgt, "fwhm": fwhm, "amplitude": amp})
        i += 1
    return out


# ------------------------ core ------------------------ #
def fit_peaks(h5_path, frame, center, plot=True):
    with h5py.File(h5_path, "r") as f:
        x = f["q"][:] if "q" in f else f["tth"][:]
        yfull = f["int"][frame, :]      # only the needed row
    x = np.asarray(x, float)
    yfull = np.asarray(yfull, float)

    half = WINDOW/2.0
    m = (x >= center-half) & (x <= center+half)
    xw, yw = x[m], yfull[m]
    mfin = np.isfinite(xw) & np.isfinite(yw)
    xw, yw = xw[mfin], yw[mfin]
    if xw.size < 5:
        raise ValueError("Too few points in window.")

    dx = float(np.mean(np.diff(xw))) if len(xw) > 1 else WINDOW
    baseline = np.median(yw)
    noise = robust_sigma(yw)

    # ---- seed centers: derivative maxima + curvature minima ----
    pk = derivative_maxima(xw, yw, smooth_win=SMOOTH_WIN, min_sep_pts=MIN_SEP_PTS)
    cm = curvature_minima(xw, yw, smooth_win=SMOOTH_WIN, min_sep_pts=MIN_SEP_PTS, kappa=1.0)
    seeds = np.unique(np.r_[pk, cm]).tolist()
    if not seeds:
        seeds = [np.abs(xw - center).argmin()]
    centers = [float(xw[i]) for i in seeds]; centers.sort()

    # ---- bounds to discourage a single huge σ ----
    min_sigma = max(dx/3.0, 1e-6)
    max_sigma = MAX_SIGMA_FRAC * WINDOW

    # ---- initial fit ----
    model, params = build_model_from_centers(xw, yw, centers, min_sigma, max_sigma, baseline)
    result = model.fit(
    yw, params, x=xw,
    calc_covar=False,
    method="least_squares",
    max_nfev=600
)
    best_aic = result.aic

    # ---- residual-based addition with AIC guard ----
    for _ in range(RESIDUAL_ADD_ITERS):
        res = yw - result.best_fit
        rnoise = robust_sigma(res)
        i_new = int(np.argmax(res))
        if res[i_new] < RESIDUAL_SNR * rnoise:
            break
        centers_plus = sorted(centers + [float(xw[i_new])])
        model2, params2 = build_model_from_centers(xw, yw, centers_plus, min_sigma, max_sigma, baseline)
        result2 = model2.fit(yw, params2, x=xw)
        if result2.aic < best_aic - AIC_IMPROVE:
            result = result2; best_aic = result2.aic; centers = centers_plus
        else:
            break

    # ---- post-fit merge of tiny, too-close neighbors ----
    changed = True
    while changed:
        changed = False
        stats = collect_stats(result)
        if len(stats) < 2: break
        stats_sorted = sorted(stats, key=lambda s: s["center"])
        for i in range(len(stats_sorted)-1):
            a, b = stats_sorted[i], stats_sorted[i+1]
            d = abs(b["center"] - a["center"])
            fwhm_avg = 0.5 * (a["fwhm"] + b["fwhm"])
            # if very close AND one is much smaller -> drop the smaller and refit
            if d < MERGE_MIN_SEP_FRAC * max(fwhm_avg, 1e-12):
                small, big = (a, b) if a["height"] <= b["height"] else (b, a)
                if small["height"] < MERGE_HEIGHT_FRAC * max(big["height"], 1e-12):
                    keep_centers = [s["center"] for s in stats if s["index"] != small["index"]]
                    model3, params3 = build_model_from_centers(xw, yw, keep_centers, min_sigma, max_sigma, baseline)
                    result3 = model3.fit(yw, params3, x=xw)
                    # accept merge if AIC not substantially worse
                    if result3.aic <= result.aic + MERGE_AIC_TOL:
                        result = result3; centers = keep_centers; changed = True
                        break

    # ---- collect kept peaks with noise threshold ----
    rows = []
    stats = collect_stats(result)
    for s in stats:
        if s["height"] >= (MIN_HEIGHT_SIGMA * noise):
            rows.append([s["index"], s["center"], s["height"], s["fwhm"], s["amplitude"]])

    bkg_slope = result.params["bkg_slope"].value
    bkg_intercept = result.params["bkg_intercept"].value
    r2 = compute_r2(yw, result.best_fit)

    # ---- plotting ----
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
        # components
        i = 0
        while f"g{i}_" in "".join(result.params.keys()):
            key = f"g{i}_"
            if key + "center" in result.params:
                if key in comps: ax.plot(xw, comps[key], ls=":")
                ax.axvline(result.params[key+"center"].value, alpha=0.25)
            i += 1
        ax.set_xlabel("q (1/Å)")
        ax.set_ylabel("Intensity")
        ax.set_title(f"Frame {frame} | R²={r2:.4f}")
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
        "aic": result.aic,
    }
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def _fit_one(args):
    """Worker for parallel execution: returns (frame, [(q_center, height), ...])."""
    h5_path, frame, center = args
    try:
        out = fit_peaks(h5_path, frame, center, plot=False)
        peaks = [(r[1], r[2]) for r in out.get("rows", [])]  # (q, height)
        return frame, peaks
    except Exception:
        return frame, []

def peak_map_for_all_frames_parallel(h5_path, center, marker_size=14):
    """
    Parallel version: runs fit_peaks() across frames with a progress bar.
    x = q (1/Å), y = frame index, color = fitted peak height (a.u.).
    """
    import os
    import h5py
    import numpy as np
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    # Avoid BLAS oversubscription inside workers (helps a lot on multi-core)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    # how many frames?
    with h5py.File(h5_path, "r") as f:
        nframes = f["int"].shape[0]

    xs, ys, cs = [], [], []

    workers = 4

    frames = range(nframes)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_fit_one, (h5_path, fr, center)) for fr in frames]
        for fut in tqdm(as_completed(futures), total=nframes,
                        desc="Building peak map (parallel)", unit="frame"):
            frame, peaks = fut.result()
            for q, height in peaks:
                xs.append(q); ys.append(frame); cs.append(height)

    if not xs:
        print("No peaks found across frames in the specified window.")
        return

    # --- same publication style as your sequential plot ---
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "font.size": 20,
        "axes.labelsize": 20,
        "axes.titlesize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
    })
    plt.figure(figsize=(9, 5))
    sc = plt.scatter(xs, ys, c=cs, s=marker_size, cmap="viridis")
    cbar = plt.colorbar(sc); cbar.set_label("Peak height (a.u.)")
    plt.xlabel("q (1/Å)")
    plt.ylabel("Frame")
    plt.tight_layout()
    plt.show()


# ------------------------ CLI (minimal) ------------------------ #
def main():
    ap = argparse.ArgumentParser(description="Peak fitting (derivative+curvature seeds, residual growth, merge pruning).")
    ap.add_argument("h5", help="HDF5 with 'q' (or 'tth') and 'int'")
    ap.add_argument("center", type=float, help="Center of the 0.1-wide window")
    args = ap.parse_args()

    
    peak_map_for_all_frames_parallel(args.h5, args.center)
    

