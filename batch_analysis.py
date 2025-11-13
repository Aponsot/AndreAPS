#!/usr/bin/env python3
import argparse, os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from lmfit.models import LinearModel
from lmfit import Model, Parameters
from numpy import sqrt
from scipy.special import erf

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

# ------------------------------
HALF_WINDOW   = 0.20
MIN_POINTS    = 8
PEAK_HEIGHT_MIN = 6000          # admission + reporting floor (height at center)

# Sigma bounds (per component)
SIGMA_MIN_FIT = 0.0005
SIGMA_MAX_FIT = 0.15

# Asymmetric per-seed drift (relative to each seed)
DRIFT_NEG = 0.11
DRIFT_POS = 0.020

# Component geometry/acceptance
MIN_SEP      = 0.0020           # min separation between component centers
AIC_IMPROVE  = 1.0              # require at least this ΔAIC improvement to accept split

# NEW: residual-based split trigger (absolute threshold)
# Set > 0 to enable: split if max|resid| >= RESIDUAL_SPLIT_THRESH
RESIDUAL_SPLIT_THRESH = 15

SPLIT_DELTA_SIGMA_FRAC = 0.6    # child offset ≈ frac * parent_sigma (clamped by MIN_SEP)

# Plot title time scaling (0/None -> use frame index)
SEC_PER_FRAME = 0.004

# Mapping frame controls
MAP_FRAME_START = 0      # inclusive start frame for map
MAP_FRAME_END   = None   # inclusive end frame; None -> last frame
MAP_STEP        = 2     # step between frames (e.g., 1, 5, 10)

# Save map arrays to HDF5 for later temperature analysis
SAVE_MAP_TO_H5 = False   # set True to write <inputbasename>_peakmap.h5

# Pixel-integrated Gaussian
# ------------------------------
def bin_edges_from_centers(x):
    x = np.asarray(x, float)
    if x.size < 2:
        dx = 1.0
        return np.array([x[0]-0.5*dx, x[0]+0.5*dx], float)
    mids = 0.5 * (x[:-1] + x[1:])
    edges = np.empty(x.size + 1, float)
    edges[1:-1] = mids
    edges[0]  = x[0] - (mids[0] - x[0])
    edges[-1] = x[-1] + (x[-1] - mids[-1])
    return edges

def pixint_gauss(x, amplitude, center, sigma):
    sigma = max(float(sigma), 1e-12)  # area-normalized amplitude
    edges = bin_edges_from_centers(x)
    t1 = (edges[1:] - center) / (sigma * sqrt(2.0))
    t0 = (edges[:-1] - center) / (sigma * sqrt(2.0))
    return amplitude * 0.5 * (erf(t1) - erf(t0))

# ------------------------------
# Helpers
# ------------------------------
def parse_centers(s: str):
    vals = [float(v) for v in s.split(",") if v.strip()]
    if not vals:
        raise ValueError("No centers parsed from --centers.")
    return np.array(vals, float)

def sigma_to_fwhm(sigma):
    return 2.354820045 * sigma

def robust_sigma(y):
    y = np.asarray(y, float)
    med = np.median(y)
    return 1.4826 * np.median(np.abs(y - med)) + 1e-12

def r2_score(y_true, y_fit):
    y_true = np.asarray(y_true, float); y_fit = np.asarray(y_fit, float)
    ss_res = np.sum((y_true - y_fit)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2) + 1e-12
    return 1.0 - ss_res/ss_tot

def load_q_and_I(h5_path):
    with h5py.File(h5_path, "r") as f:
        if "q" in f:
            x = np.asarray(f["q"][:], float)
        elif "tth" in f:
            x = np.asarray(f["tth"][:], float)
        else:
            raise ValueError("HDF5 must contain 'q' or 'tth'.")
        I_full = np.asarray(f["int"][:], float)

    if I_full.ndim == 1:
        I_full = I_full[None, :]
    elif I_full.ndim > 2:
        I_full = I_full.mean(axis=tuple(range(1, I_full.ndim)))

    if I_full.shape[1] != x.shape[0]:
        raise ValueError(f"Shape mismatch: int.shape={I_full.shape}, x.shape={x.shape}")
    return x, I_full

def window_mask(x, centers, halfwidth):
    lo = float(np.min(centers) - halfwidth)
    hi = float(np.max(centers) + halfwidth)
    return (x >= lo) & (x <= hi)

def _gaussian_y(x, amp_area, cen, sig):
    return pixint_gauss(x, amp_area, cen, sig)

def nearest_seed_bounds(center_value, seeds):
    seed = float(seeds[np.argmin(np.abs(seeds - center_value))])
    return seed - DRIFT_NEG, seed + DRIFT_POS

def clamp_center_param(params, name, center_value, seeds):
    lo, hi = nearest_seed_bounds(center_value, seeds)
    params[name].set(min=lo, max=hi)

# NEW: global min-sep enforcement on reported peaks
def apply_min_sep_mask(centers, heights, base_mask):
    """
    Enforce MIN_SEP on the peaks we plan to report.

    centers, heights: 1D arrays
    base_mask: boolean mask (e.g., height >= PEAK_HEIGHT_MIN)
    Returns: new mask with MIN_SEP enforced (drops weaker in too-close pairs).
    """
    centers = np.asarray(centers, float)
    heights = np.asarray(heights, float)
    mask = np.asarray(base_mask, bool) & np.isfinite(centers) & np.isfinite(heights)

    idx = np.where(mask)[0]
    if idx.size <= 1:
        return mask

    # sort by center
    centers_sub = centers[idx]
    heights_sub = heights[idx]
    order = np.argsort(centers_sub)
    idx_sorted = idx[order]

    keep = [idx_sorted[0]]
    for i in idx_sorted[1:]:
        # compare to all kept peaks; if too close, keep the higher peak
        too_close_to = [j for j in keep if abs(centers[i] - centers[j]) < MIN_SEP]
        if not too_close_to:
            keep.append(i)
        else:
            # compare to the closest offending one
            j = min(too_close_to, key=lambda jj: abs(centers[i] - centers[jj]))
            if heights[i] > heights[j]:
                keep[keep.index(j)] = i  # replace weaker with stronger

    newmask = np.zeros_like(mask)
    newmask[keep] = True
    return newmask

# Model build / metrics
# ------------------------------
def _build_seed_model(xw, yw, seeds):
    # quick linear background guess
    try:
        bkg_slope, bkg_intercept = np.polyfit(xw, yw, 1)
    except Exception:
        bkg_slope, bkg_intercept = 0.0, float(np.median(yw))

    model = LinearModel(prefix="bkg_")
    params = model.make_params(bkg_slope=bkg_slope, bkg_intercept=bkg_intercept)

    # initial sigma scales with span and number of peaks
    span = max(xw[-1] - xw[0], 1e-9)
    sigma0_base = max(span / (7.0 * len(seeds)), 1e-6)
    sigma0 = float(np.clip(sigma0_base, SIGMA_MIN_FIT, SIGMA_MAX_FIT))

    for i, c_seed in enumerate(seeds):
        g = Model(pixint_gauss, prefix=f"g{i}_")
        model = model + g

        # area from (height≈(y - bkg) at nearest bin) * sigma * sqrt(2π)
        idx = np.abs(xw - c_seed).argmin()
        y_at_seed = yw[idx]
        y_bkg = bkg_slope * xw[idx] + bkg_intercept
        height0 = max(y_at_seed - y_bkg, np.std(yw) * 0.5)
        amp0 = max(height0 * sigma0 * np.sqrt(2.0 * np.pi), 0.0)

        params.update(g.make_params(amplitude=amp0, center=c_seed, sigma=sigma0))
        params[f"g{i}_sigma"].set(min=SIGMA_MIN_FIT, max=SIGMA_MAX_FIT)
        params[f"g{i}_amplitude"].set(min=0.0)
        params[f"g{i}_center"].set(min=c_seed - DRIFT_NEG, max=c_seed + DRIFT_POS)
    return model, params

def _extract_metrics(result, xw):
    bkg_slope = result.params.get("bkg_slope").value if "bkg_slope" in result.params else 0.0
    bkg_intercept = result.params.get("bkg_intercept").value if "bkg_intercept" in result.params else 0.0
    bkg_line = bkg_slope * xw + bkg_intercept

    centers, sigmas, amps = [], [], []
    i = 0
    while f"g{i}_center" in result.params:
        centers.append(result.params[f"g{i}_center"].value)
        sigmas.append (result.params[f"g{i}_sigma"].value)
        amps.append   (result.params[f"g{i}_amplitude"].value)
        i += 1
    centers = np.asarray(centers, float)
    sigmas  = np.asarray(sigmas,  float)
    amps    = np.asarray(amps,    float)

    heights = np.full_like(centers, np.nan, float)
    fwhm    = np.full_like(centers, np.nan, float)
    peak_at_center = np.full_like(centers, np.nan, float)
    for j in range(centers.size):
        if np.isfinite(sigmas[j]) and sigmas[j] > 0:
            heights[j] = amps[j] / (sigmas[j] * np.sqrt(2.0 * np.pi))   # area -> height
            fwhm[j] = sigma_to_fwhm(sigmas[j])
        peak_at_center[j] = bkg_slope * centers[j] + bkg_intercept + (heights[j] if np.isfinite(heights[j]) else 0.0)

    comps = []
    for a, c, s in zip(amps, centers, sigmas):
        if np.all(np.isfinite([a, c, s])) and s > 0:
            comps.append(_gaussian_y(xw, a, c, s))
        else:
            comps.append(np.full_like(xw, np.nan, float))
    return bkg_line, centers, sigmas, amps, heights, fwhm, peak_at_center, comps

def _sum_components(xw, result):
    comps = []
    i = 0
    while f"g{i}_center" in result.params:
        c = result.params[f"g{i}_center"].value
        s = result.params[f"g{i}_sigma"].value
        a = result.params[f"g{i}_amplitude"].value
        if np.all(np.isfinite([a, c, s])) and s > 0:
            comps.append(_gaussian_y(xw, a, c, s))
        i += 1
    return np.sum(np.vstack(comps), axis=0) if comps else np.zeros_like(xw, float)

# Simplified FORCE-SPLIT (honors seed cap and min-sep)
# ------------------------------
def _try_split_once(xw, yw, result, seed_cap, seeds):
    # respect seed cap
    n_now = 0
    while f"g(n_now)_center" in result.params:
        n_now += 1
    n_now = 0
    while f"g{n_now}_center" in result.params:
        n_now += 1
    if n_now == 0 or n_now >= seed_cap:
        return result, False

    # residual-based trigger (now absolute threshold)
    bkg_line, centers, sigmas, amps, heights, _, _, _ = _extract_metrics(result, xw)
    resid_vec = yw - (bkg_line + _sum_components(xw, result))
    resid_max = float(np.max(np.abs(resid_vec))) if resid_vec.size else 0.0

    if RESIDUAL_SPLIT_THRESH <= 0 or resid_max < RESIDUAL_SPLIT_THRESH:
        return result, False

    # split tallest peak
    j_main = int(np.nanargmax(heights))
    c0 = float(centers[j_main]); s0 = max(float(sigmas[j_main]), 1e-12); a0 = max(float(amps[j_main]), 1e-12)

    delta = max(MIN_SEP, SPLIT_DELTA_SIGMA_FRAC * s0)
    c1, c2 = c0 - 0.5*delta, c0 + 0.5*delta
    s1 = s2 = float(np.clip(s0, SIGMA_MIN_FIT, SIGMA_MAX_FIT))
    a1 = a2 = 0.5 * a0

    # rebuild params: drop main, add two children
    params_new = Parameters()
    model_new = LinearModel(prefix="bkg_")
    for nm in ["bkg_slope", "bkg_intercept"]:
        if nm in result.params:
            p = result.params[nm]
            params_new.add(nm, value=p.value, min=p.min, max=p.max, vary=p.vary)
        else:
            params_new.add(nm, value=0.0)

    next_idx = 0
    j = 0
    while f"g{j}_center" in result.params:
        if j != j_main:
            gk = Model(pixint_gauss, prefix=f"g{next_idx}_")
            model_new = model_new + gk
            ccur = result.params[f"g{j}_center"].value
            scur = result.params[f"g{j}_sigma"].value
            acur = result.params[f"g{j}_amplitude"].value
            params_new.update(gk.make_params(center=ccur, sigma=scur, amplitude=acur))
            params_new[f"g{next_idx}_sigma"].set(min=SIGMA_MIN_FIT, max=SIGMA_MAX_FIT)
            params_new[f"g{next_idx}_amplitude"].set(min=0.0)
            clamp_center_param(params_new, f"g{next_idx}_center", ccur, seeds)
            next_idx += 1
        j += 1

    # add two children
    g1 = Model(pixint_gauss, prefix=f"g{next_idx}_")
    model_new = model_new + g1
    params_new.update(g1.make_params(center=c1, sigma=s1, amplitude=a1))
    params_new[f"g{next_idx}_sigma"].set(min=SIGMA_MIN_FIT, max=SIGMA_MAX_FIT)
    params_new[f"g{next_idx}_amplitude"].set(min=0.0)
    clamp_center_param(params_new, f"g{next_idx}_center", c1, seeds)
    next_idx += 1

    g2 = Model(pixint_gauss, prefix=f"g{next_idx}_")
    model_new = model_new + g2
    params_new.update(g2.make_params(center=c2, sigma=s2, amplitude=a2))
    params_new[f"g{next_idx}_sigma"].set(min=SIGMA_MIN_FIT, max=SIGMA_MAX_FIT)
    params_new[f"g{next_idx}_amplitude"].set(min=0.0)
    clamp_center_param(params_new, f"g{next_idx}_center", c2, seeds)

    trial = model_new.fit(yw, params_new, x=xw, nan_policy="omit")

    # accept if ΔAIC improves and min-sep holds
    if not (trial.aic <= result.aic - AIC_IMPROVE):
        return result, False

    _, centers_t, _, _, _, _, _, _ = _extract_metrics(trial, xw)
    if centers_t.size >= 2:
        diffs = np.abs(np.subtract.outer(centers_t, centers_t))
        diffs += np.eye(centers_t.size) * 1e9
        if np.min(diffs) < MIN_SEP:
            return result, False

    # still within cap?
    n_after = 0
    while f"g{n_after}_center" in trial.params:
        n_after += 1
    if n_after > seed_cap:
        return result, False

    return trial, True

# Fit a single frame
# ------------------------------
def fit_frame(x, y, seeds, halfwidth):
    m = window_mask(x, seeds, halfwidth)
    if not np.any(m):
        return {"success": False}
    xw, yw = x[m], y[m]
    if xw.size < MIN_POINTS:
        return {"success": False}

    base_model, params = _build_seed_model(xw, yw, seeds)
    try:
        result = base_model.fit(yw, params, x=xw, nan_policy="omit")
    except Exception:
        return {"success": False}

    # height prune + quick refit
    bkg_line, c_all, s_all, a_all, h_all, w_all, p_all, comps = _extract_metrics(result, xw)
    keep = np.isfinite(h_all) & (h_all >= PEAK_HEIGHT_MIN)
    if keep.size and not np.all(keep):
        params_new = Parameters()
        model_new = LinearModel(prefix="bkg_")
        for nm in ["bkg_slope", "bkg_intercept"]:
            if nm in result.params:
                p = result.params[nm]
                params_new.add(nm, value=p.value, min=p.min, max=p.max, vary=p.vary)
            else:
                params_new.add(nm, value=0.0)
        next_idx = 0
        j = 0
        while f"g{j}_center" in result.params:
            if keep[j]:
                gk = Model(pixint_gauss, prefix=f"g{next_idx}_")
                model_new = model_new + gk
                ccur = result.params[f"g{j}_center"].value
                scur = result.params[f"g{j}_sigma"].value
                acur = result.params[f"g{j}_amplitude"].value
                params_new.update(gk.make_params(center=ccur, sigma=scur, amplitude=acur))
                params_new[f"g{next_idx}_sigma"].set(min=SIGMA_MIN_FIT, max=SIGMA_MAX_FIT)
                params_new[f"g{next_idx}_amplitude"].set(min=0.0)
                clamp_center_param(params_new, f"g{next_idx}_center", ccur, seeds)
                next_idx += 1
            j += 1
        result = model_new.fit(yw, params_new, x=xw, nan_policy="omit")
        bkg_line, c_all, s_all, a_all, h_all, w_all, p_all, comps = _extract_metrics(result, xw)

    # one optional split attempt (keeps seeds cap)
    result, _ = _try_split_once(xw, yw, result, seed_cap=len(seeds), seeds=seeds)
    bkg_line, c_all, s_all, a_all, h_all, w_all, p_all, comps = _extract_metrics(result, xw)

    # reporting mask: height floor + MIN_SEP across all reported peaks
    base_valid = np.isfinite(h_all) & (h_all >= PEAK_HEIGHT_MIN)
    valid = apply_min_sep_mask(c_all, h_all, base_valid)

    centers_out = c_all.copy(); centers_out[~valid] = np.nan
    fwhm_out    = w_all.copy(); fwhm_out[~valid]    = np.nan
    height_out  = h_all.copy(); height_out[~valid]  = np.nan
    area_out    = a_all.copy(); area_out[~valid]    = np.nan
    peakfit_out = p_all.copy(); peakfit_out[~valid] = np.nan

    comp_sum = bkg_line + _sum_components(xw, result)

    return {
        "success": True,
        "xw": xw, "yw": yw, "yfit": result.best_fit, "bkg": bkg_line,
        "centers": centers_out, "fwhm": fwhm_out, "height_fit": height_out,
        "area_fit": area_out, "peak_fit": peakfit_out, "components": comps,
        "comp_sum": comp_sum, "r2": r2_score(yw, result.best_fit)
    }

# Visual style
# ------------------------------
def apply_pub_style():
    plt.rcParams.update({
        "figure.figsize": (6, 5),
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "font.size": 12,
        "axes.labelsize": 14,
        "legend.fontsize": 11,
        "legend.frameon": False,
        "axes.linewidth": 1.15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "axes.grid": False,
    })

def style_axes(ax, light_grid=True):
    for side in ("top","right","bottom","left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.15)
    ax.minorticks_on()
    if light_grid:
        ax.grid(True, which="major", alpha=0.12, linestyle="-", linewidth=0.6)

# Main (CLI)
# ------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Pixel-integrated Gaussian tracker (linear bkg, height-prune, optional residual-based split, seed-bounded centers)."
    )
    ap.add_argument("--h5", required=True, help="HDF5 with 'q' (or 'tth') and 'int'")
    ap.add_argument("--centers", required=True, help="Comma-separated initial peak centers (e.g., 2.975,3.124)")
    ap.add_argument("--frame", type=int, default=None, help="Fit a single frame index. Omit to track all frames.")
    args = ap.parse_args()

    seeds0 = parse_centers(args.centers)
    x, I_full = load_q_and_I(args.h5)
    nframes = I_full.shape[0]

    # -------- Single frame --------
    if args.frame is not None:
        if not (0 <= args.frame < nframes):
            raise ValueError(f"--frame {args.frame} is out of range [0, {nframes-1}]")
        y = I_full[args.frame]
        res = fit_frame(x, y, seeds0, HALF_WINDOW)
        if not res["success"]:
            print("Fit failed for the requested frame.")
            return

        # Terminal table
        vis = np.isfinite(res["centers"])
        centers_v = res["centers"][vis]
        fwhm_v    = res["fwhm"][vis]
        hfit_v    = res["height_fit"][vis]

        print("PEAKS (kept >= height floor and MIN_SEP):")
        print("Idx\tCenter\t\tFWHM\t\tHeight")
        for i, (c, w, h) in enumerate(zip(centers_v, fwhm_v, hfit_v), start=1):
            print(f"{i}\t{c:.6f}\t{w:.6f}\t{h:.6f}")
        print(f"R2 = {res['r2']:.4f}\n")

        # Plot
        apply_pub_style()
        from matplotlib.gridspec import GridSpec
        plt.rcParams.update({"figure.figsize": (7, 5.2)})

        fig = plt.figure()
        gs = GridSpec(2, 1, height_ratios=[3.0, 1.2], hspace=0.50)

        ax = fig.add_subplot(gs[0]); style_axes(ax, light_grid=True)
        ax.plot(res["xw"], res["yw"],   lw=1.2, color="tab:blue",   label="Data")
        ax.plot(res["xw"], res["yfit"], lw=1.8, color="tab:orange", label="Total fit")
        ax.plot(res["xw"], res["bkg"],  "--",  lw=1.2, color="tab:green", alpha=0.7, label="Linear bkg")

        for comp in res["components"]:
            ax.plot(res["xw"], comp, lw=1.0, linestyle="--", alpha=0.7)

        ax.plot(res["xw"], res["comp_sum"], lw=1.0, color="k", alpha=0.35)

        for c in centers_v:
            ax.axvline(c, linestyle="--", alpha=0.35, lw=0.9, color="0.4")

        lo = float(np.min(seeds0) - HALF_WINDOW)
        hi = float(np.max(seeds0) + HALF_WINDOW)
        ax.set_xlim(lo, hi)

        title_prefix = (f"{args.frame*float(SEC_PER_FRAME):.1f} s | "
                        if (SEC_PER_FRAME is not None and SEC_PER_FRAME > 0)
                        else f"Frame {args.frame} | ")
        ax.set_title(title_prefix + f"R²={res['r2']:.4f}", pad=6)
        ax.set_xlabel("q (Å⁻¹)")
        ax.set_ylabel("Intensity (a.u.)")
        ax.legend(loc="upper right", ncol=1, fontsize=10)

        axr = fig.add_subplot(gs[1]); style_axes(axr, light_grid=False)
        resid = res["yw"] - res["yfit"]
        axr.plot(res["xw"], resid, lw=1.0, color="0.2")
        axr.axhline(0.0, color="0.25", lw=0.8)
        axr.set_xlabel("q (1/Å)")
        axr.set_ylabel("Residual")

        fig.tight_layout()
        plt.show()
        return

    # -------- Mapping (subset of frames with step) --------
    start = int(MAP_FRAME_START)
    if start < 0:
        start = 0
    if start >= nframes:
        raise ValueError(f"MAP_FRAME_START {MAP_FRAME_START} is >= total frames {nframes}")

    if MAP_FRAME_END is None:
        end = nframes - 1
    else:
        end = int(MAP_FRAME_END)
        if end < 0:
            end = nframes - 1
        if end >= nframes:
            end = nframes - 1

    if end < start:
        raise ValueError(f"MAP_FRAME_END ({end}) < MAP_FRAME_START ({start})")

    step = int(MAP_STEP) if MAP_STEP is not None and MAP_STEP > 0 else 1

    frames_used = np.arange(start, end + 1, step)
    nuse = frames_used.size
    npeaks = len(seeds0)

    centers_trk = np.full((nuse, npeaks), np.nan)
    fwhm_trk    = np.full((nuse, npeaks), np.nan)
    height_trk  = np.full((nuse, npeaks), np.nan)
    area_trk    = np.full((nuse, npeaks), np.nan)

    iterator = range(nuse)
    if tqdm is not None:
        iterator = tqdm(iterator, desc="Fitting frames", ncols=80)

    for i_row in iterator:
        f = frames_used[i_row]
        y = I_full[f]
        res = fit_frame(x, y, seeds0, HALF_WINDOW)
        if not res["success"]:
            continue

        valid = np.isfinite(res["centers"]) & np.isfinite(res["height_fit"])
        if np.any(valid):
            c = res["centers"][valid]
            w = res["fwhm"][valid]
            h = res["height_fit"][valid]
            a = res.get("area_fit", np.full_like(h, np.nan))[valid]
        else:
            c = w = h = a = np.array([])

        if c.size:
            order = np.argsort(c)
            c, w, h, a = c[order], w[order], h[order], a[order]
        k = min(c.size, npeaks)
        centers_trk[i_row, :k] = c[:k]
        fwhm_trk[i_row, :k]    = w[:k]
        height_trk[i_row, :k]  = h[:k]
        area_trk[i_row, :k]    = a[:k]

    # Optional: save map arrays to HDF5
    if SAVE_MAP_TO_H5:
        base_dir = os.path.dirname(os.path.abspath(args.h5))
        base_name = os.path.splitext(os.path.basename(args.h5))[0]
        out_path = os.path.join(base_dir, base_name + "_peakmap.h5")
        with h5py.File(out_path, "w") as hf:
            hf.create_dataset("frame_index", data=frames_used, compression="gzip")
            if SEC_PER_FRAME is not None and SEC_PER_FRAME > 0:
                hf.create_dataset("time", data=frames_used * float(SEC_PER_FRAME), compression="gzip")
            hf.create_dataset("centers", data=centers_trk, compression="gzip")
            hf.create_dataset("fwhm",    data=fwhm_trk,    compression="gzip")
            hf.create_dataset("height",  data=height_trk,  compression="gzip")
            hf.create_dataset("area",    data=area_trk,    compression="gzip")
            hf.attrs["seeds0"] = seeds0
            hf.attrs["sec_per_frame"] = float(SEC_PER_FRAME) if SEC_PER_FRAME is not None else -1.0
        print(f"Saved peak map to {out_path}")

    # Visualization: centers vs time/frame, color = normalized area (0–100)
    apply_pub_style()
    plt.rcParams.update({"figure.figsize": (11.5, 4.6)})

    fig, ax = plt.subplots(); style_axes(ax, light_grid=True)

    if SEC_PER_FRAME is not None and SEC_PER_FRAME > 0:
        xvals = frames_used * float(SEC_PER_FRAME)
        xlabel = "Time (s)"
    else:
        xvals = frames_used
        xlabel = "Frame"

    area_norm = np.zeros_like(area_trk, float)
    if np.any(np.isfinite(area_trk)):
        global_max = np.nanmax(area_trk)
        if global_max > 0:
            area_norm = 100.0 * area_trk / global_max

    any_plotted = False
    for j in range(npeaks):
        mask = (
            np.isfinite(centers_trk[:, j]) &
            np.isfinite(area_norm[:, j]) &
            (height_trk[:, j] >= PEAK_HEIGHT_MIN)
        )
        if not np.any(mask):
            continue
        sc = ax.scatter(
            xvals[mask],
            centers_trk[mask, j],
            c=area_norm[mask, j],
            cmap="plasma",
            s=18,
            linewidths=0.0,
            edgecolors="none",
            vmin=0, vmax=100,
        )
        any_plotted = True

    lo = float(np.min(seeds0) - HALF_WINDOW)
    hi = float(np.max(seeds0) + HALF_WINDOW)
    ax.set_ylim(lo, hi)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("q (Å⁻¹)")
    ax.set_title("Peak centers over frames (color = normalized peak area)")

    if any_plotted:
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label("Normalized Peak Area (0–100)")

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()


