#!/usr/bin/env python3
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.signal as signal
from lmfit.models import GaussianModel, PolynomialModel


# ===================== TUNABLES (edit here) =====================
W_FWHM_PTS      = 6.0    # nominal points-per-FWHM in your data window
K_PROM          = 4.0    # detection strictness (higher = fewer peaks)
MIN_PROM_FRAC   = 0.03   # min prominence as fraction of (y - linear BG) dynamic range
PROM_ABS_FLOOR  = 1.0    # absolute minimum prominence (counts)
MIN_DIST_FRAC   = 0.6    # min distance between detected peaks (in points) = MIN_DIST_FRAC*W_FWHM_PTS
MAX_INIT_PEAKS  = 8      # cap on number of seeds taken into the fit (keeps things sane)

# Optional: force an exact number of Gaussians (None = use detected count)
EXACT_N_PEAKS   = None   # e.g., set to 2 or 4 if you know the count; leave None for fluid

# (light) bounds to keep parameters reasonable but not boxed-in
CENTER_WIGGLE_PTS = 10.0  # each center can move ± this many points from its seed during fit
SIGMA_MIN_FRAC    = 0.25  # sigma lower bound = SIGMA_MIN_FRAC * sigma_seed
SIGMA_MAX_MULT    = 4.0   # sigma upper bound = SIGMA_MAX_MULT * sigma_seed
AMP_MIN_FRAC      = 0.05  # area lower bound   = AMP_MIN_FRAC   * area_seed
AMP_MAX_MULT      = 20.0  # area upper bound   = AMP_MAX_MULT   * area_seed
# ===============================================================


def robust_sigma(x: np.ndarray) -> float:
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))


def seed_peaks(q, y, peak_pos, w_fwhm_pts):
    """
    Seed peak centers on a background-subtracted signal using a simple linear BG
    and scipy.find_peaks with an adaptive prominence threshold.
    """
    # simple linear background (for detection only)
    X = np.vstack([np.ones_like(q), q]).T
    (c0, c1), *_ = np.linalg.lstsq(X, y, rcond=None)
    bg = c0 + c1 * q
    y_det = y - bg

    # noise & thresholds
    # use a gentle high-pass to estimate noise without overcomplicating
    if len(y_det) > 7:
        sm = signal.windows.gaussian(M=7, std=1.5)
        sm = sm / sm.sum()
        smooth = np.convolve(y_det, sm, mode="same")
        resid = y_det - smooth
    else:
        resid = y_det
    sig = robust_sigma(resid)
    dyn = max(float(y_det.max() - y_det.min()), 1.0)
    prom = max(K_PROM * sig, MIN_PROM_FRAC * dyn, PROM_ABS_FLOOR)

    # distances in points
    min_dist_pts = max(1, int(round(MIN_DIST_FRAC * w_fwhm_pts)))
    # allow a wide width range for detection; lmfit will refine
    wmin = 1
    wmax = max(3, int(round(4.0 * w_fwhm_pts)))

    peaks, props = signal.find_peaks(y_det, prominence=prom, width=(wmin, wmax), distance=min_dist_pts)

    # sort candidates by closeness to requested peak_pos, then by prominence
    if peaks.size:
        order = np.argsort(np.abs(q[peaks] - peak_pos) + 1e-6 * (-props["prominences"]))
        peaks = peaks[order]
        props = {k: props[k][order] for k in props}

    # cap total seeds
    if len(peaks) > MAX_INIT_PEAKS:
        peaks = peaks[:MAX_INIT_PEAKS]
        for k in ["widths", "prominences"]:
            props[k] = props[k][:len(peaks)]

    # fallback: ensure at least one seed near peak_pos
    if len(peaks) == 0:
        i0 = int(np.argmin(np.abs(q - peak_pos)))
        peaks = np.array([i0], dtype=int)
        props = {"widths": np.array([w_fwhm_pts], float), "prominences": np.array([max(y_det[i0], 1.0)], float)}

    return peaks, props, bg, y_det, dict(prom_used=prom, min_dist_pts=min_dist_pts)


def build_linear_plus_gaussians(q, y, peaks_idx, props, w_fwhm_pts):
    """
    Build a linear background + sum of Gaussians model with reasonable initial params
    and mild bounds (centers can move, widths/amplitudes can scale).
    """
    dq = float(np.mean(np.diff(q)))
    sigma_seed = (w_fwhm_pts * dq) / 2.355

    poly = PolynomialModel(degree=1, prefix="bg_")
    model = poly
    params = poly.make_params()
    params["bg_c0"].set(value=float(np.median(y)), min=0.0)
    params["bg_c1"].set(value=0.0)  # keep linear; no extra degrees

    for i, pidx in enumerate(peaks_idx):
        g = GaussianModel(prefix=f"g{i}_")
        model += g

        c0 = float(q[pidx])
        # use detected width if present; otherwise nominal
        if "widths" in props and len(props["widths"]) > i:
            wpts = float(props["widths"][i])
            s0 = max(1e-9, (wpts * dq) / 2.355)
        else:
            s0 = sigma_seed

        # lmfit Gaussian amplitude is AREA
        height0 = max(float(y[pidx]), 1.0)
        area0 = height0 * s0 * np.sqrt(2.0 * np.pi)

        params.update(g.make_params(center=c0, sigma=s0, amplitude=area0))

        # mild, symmetric bounds
        wig = CENTER_WIGGLE_PTS * dq
        params[f"g{i}_center"].set(min=c0 - wig, max=c0 + wig)
        params[f"g{i}_sigma"].set(min=max(1e-9, SIGMA_MIN_FRAC * s0),
                                  max=max(2e-9, SIGMA_MAX_MULT * s0))
        params[f"g{i}_amplitude"].set(min=max(1e-9, AMP_MIN_FRAC * abs(area0)),
                                      max=max(2e-9, AMP_MAX_MULT * abs(area0)))

    return model, params


def fit_and_summarize(q, y, model, params):
    # light Poisson-ish weights help when counts change across the window
    w = 1.0 / np.sqrt(np.clip(y, 1.0, None))
    result = model.fit(y, params, x=q, weights=w, method="least_squares",
                       fit_kws={"loss": "soft_l1", "f_scale": 1.0})

    # metrics
    yhat = result.best_fit
    resid = y - yhat
    n, k = len(y), sum(p.vary for p in result.params.values())
    sse = float(np.sum((w * resid) ** 2))
    ybar = float(np.sum((w**2) * y) / np.sum(w**2))
    sst = float(np.sum((w * (y - ybar)) ** 2))
    r2 = 1.0 - sse / max(sst, 1e-18)
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / max(n - k - 1, 1)
    red_chisq = sse / max(n - k, 1)
    rmse = np.sqrt(sse / n)
    return result, dict(r2=r2, adj_r2=adj_r2, red_chisq=red_chisq, rmse=rmse,
                        aic=result.aic, bic=result.bic, max_abs=float(np.max(np.abs(resid)))), resid


def peak_fit(h5_path, frame_number, peak_pos, window=0.1):
    # --- load
    with h5py.File(h5_path, "r") as f:
        Int = f["int"][:]   # shape: (nframes, nq)
        q   = f["q"][:].astype(float)

    # --- window
    q_min, q_max = float(peak_pos - window), float(peak_pos + window)
    mask = (q >= q_min) & (q <= q_max)
    q_win = q[mask]
    y_win = Int[int(frame_number), mask].astype(float)
    if q_win.size < max(20, int(4 * W_FWHM_PTS)):
        raise ValueError("Fit window too small or outside q-range.")

    # --- seeding
    peaks, props, bg_det, y_det, detinfo = seed_peaks(q_win, y_win, peak_pos, W_FWHM_PTS)

    # choose how many to fit (fluid by default)
    if EXACT_N_PEAKS is None:
        npeaks = len(peaks)
    else:
        npeaks = int(EXACT_N_PEAKS)
        # ensure we have that many seeds: trim or pad by splitting strongest
        if len(peaks) > npeaks:
            peaks = peaks[:npeaks]
            for k in ["widths", "prominences"]:
                props[k] = props[k][:npeaks]
        while len(peaks) < npeaks:
            # duplicate the strongest (by prominence) with a tiny offset
            j = int(np.argmax(props["prominences"]))
            p = int(peaks[j])
            off = max(1, int(round(W_FWHM_PTS * 0.3)))
            new = min(len(q_win)-1, p + off)
            peaks = np.append(peaks, new)
            for k in ["widths", "prominences"]:
                props[k] = np.append(props[k], props[k][j])

    # --- model + fit
    model, params = build_linear_plus_gaussians(q_win, y_win, peaks, props, W_FWHM_PTS)
    result, metrics, resid = fit_and_summarize(q_win, y_win, model, params)

    # --- plotting (top: fit; bottom: metrics & per-peak)
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(11, 8), gridspec_kw={"height_ratios": [3, 2]})
    ax0.plot(q_win, y_win, "k.", ms=3, label="Data")

    # fitted BG (linear)
    comps_dense = result.eval_components(x=q_win)
    if "bg_" in comps_dense:
        ax0.plot(q_win, comps_dense["bg_"], "-", lw=1.2, label="BG (fit, linear)")

    # total fit (dense)
    q_dense = np.linspace(q_win.min(), q_win.max(), len(q_win) * 5)
    y_dense = result.eval(x=q_dense)
    ax0.plot(q_dense, y_dense, "-", lw=1.6, label=f"Total fit ({len(peaks)} Gaussian{'s' if len(peaks)>1 else ''})")

    # components
    comp_dense = result.model.eval_components(params=result.params, x=q_dense)
    gnames = sorted([k for k in comp_dense if k.startswith("g")])
    for k in gnames:
        ax0.plot(q_dense, comp_dense[k], ":", lw=1.0, label=k)

    # mark fitted centers
    centers = []
    for i in range(len(gnames)):
        centers.append(result.params[f"g{i}_center"].value)
    ax0.plot(centers, [np.interp(c, q_dense, y_dense) for c in centers], "x", ms=8, label="Centers")

    ax0.set_xlabel("q"); ax0.set_ylabel("Intensity")
    ax0.set_title(f"Frame {frame_number} | seeds used: {len(peaks)} | det prom≥ {detinfo['prom_used']:.3g}")
    ax0.legend(loc="best", fontsize=8)

    # table
    ax1.axis("off")
    rows = [
        ["R²", f"{metrics['r2']:.6f}"],
        ["Adj R²", f"{metrics['adj_r2']:.6f}"],
        ["Reduced χ²", f"{metrics['red_chisq']:.3g}"],
        ["AIC", f"{metrics['aic']:.2f}"],
        ["BIC", f"{metrics['bic']:.2f}"],
        ["RMSE", f"{metrics['rmse']:.3g}"],
        ["Max |res|", f"{metrics['max_abs']:.3g}"],
        ["# Peaks fit", f"{len(gnames)}"],
        ["", ""],
    ]
    for i in range(len(gnames)):
        c = result.params[f"g{i}_center"].value
        s = result.params[f"g{i}_sigma"].value
        a = result.params[f"g{i}_amplitude"].value  # AREA for lmfit Gaussian
        height = a / (np.sqrt(2*np.pi) * s)
        rows += [
            [f"Peak {i} center", f"{c:.6f}"],
            [f"Peak {i} height", f"{height:.3g}"],
            [f"Peak {i} sigma",  f"{s:.6g}"],
            [f"Peak {i} FWHM",   f"{2.355*s:.6g}"],
            [f"Peak {i} area",   f"{a:.3g}"],
            ["", ""],
        ]
    table = ax1.table(cellText=rows, colLabels=None, loc="center", cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1.1, 1.18)

    plt.tight_layout()
    plt.show()


# ----------------------------- CLI ----------------------------- #
def _parse_args():
    p = argparse.ArgumentParser(description="Linear-background Gaussian peak fitting (simple & fluid).")
    p.add_argument("h5", type=str)
    p.add_argument("frame_number", type=int)
    p.add_argument("peak_pos", type=float)
    p.add_argument("--window", type=float, default=0.1, help="Half-window in q around peak_pos")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    peak_fit(args.h5, args.frame_number, args.peak_pos, args.window)
