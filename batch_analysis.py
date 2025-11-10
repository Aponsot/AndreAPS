#!/usr/bin/env python3
# Core v1.6 — Gaussian multi-peak fitting (linear bkg, BIC-gated shoulder)
# - Single-frame and full-experiment tracking
# - Max number of peaks = number of user-provided seeds
# - Prune low-height components before shoulder attempt (frees slot if needed)
# - Shoulder = ONE residual-max candidate; admit only if BIC improves (ΔBIC<0)
# - If at cap, try replacing weakest component, keep only if BIC improves
# - Height floor is for reporting/plotting only (not admission)
# - Map == single-frame logic, using same seeds0 per frame (no cross-frame seeding)
# - Publishable plotting; plasma colormap for map; no CSV

import argparse, os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from lmfit.models import GaussianModel, LinearModel

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

# ------------------------------
# Tunables (minimal & clear)
# ------------------------------
HALF_WINDOW     = 0.14   # half-width of fit window around [min(seeds), max(seeds)]
MIN_POINTS      = 8      # min points in window to attempt a fit

PEAK_HEIGHT_MIN = 5.2    # floor for reporting/plotting (post-fit filtering)

SIGMA_MIN_FIT   = 0.0005  # global σ bounds (FWHM = 2.355 σ)
SIGMA_MAX_FIT   = 0.10

DRIFT_NEG       = 0.10   # allowed center drift around each seed per frame
DRIFT_POS       = 0.010

MIN_SEP         = 1e-8   # min separation between component centers (x-units)

SEC_PER_FRAME   = 0.004  # for titles; set None to show "Frame N" instead

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
    y_true = np.asarray(y_true, float)
    y_fit  = np.asarray(y_fit,  float)
    ss_res = np.sum((y_true - y_fit)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2) + 1e-12
    return 1.0 - ss_res/ss_tot

def load_q_and_I(h5_path):
    with h5py.File(h5_path, "r") as f:
        if "q" in f:     x = np.asarray(f["q"][:], float)
        elif "tth" in f: x = np.asarray(f["tth"][:], float)
        else: raise ValueError("HDF5 must contain 'q' or 'tth'.")
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

def _gaussian_y(x, amp, cen, sig):
    return amp * np.exp(-(x - cen)**2 / (2.0 * sig**2))

def _build_seed_model(xw, yw, seeds):
    # linear background in the window
    try:
        bkg_slope, bkg_intercept = np.polyfit(xw, yw, 1)
    except Exception:
        bkg_slope, bkg_intercept = 0.0, float(np.median(yw))

    model = LinearModel(prefix="bkg_")
    params = model.make_params(bkg_slope=bkg_slope, bkg_intercept=bkg_intercept)

    span = max(xw[-1] - xw[0], 1e-9)
    sigma0_base = max(span / (7.0 * len(seeds)), 1e-6)
    sigma0 = float(np.clip(sigma0_base, SIGMA_MIN_FIT, SIGMA_MAX_FIT))

    for i, c_seed in enumerate(seeds):
        g = GaussianModel(prefix=f"g{i}_")
        model = model + g
        idx = np.abs(xw - c_seed).argmin()
        y_at_seed = yw[idx]
        y_bkg = bkg_slope * xw[idx] + bkg_intercept
        height0 = max(y_at_seed - y_bkg, np.std(yw) * 0.5)
        amp0 = max(height0 * sigma0 * np.sqrt(2.0 * np.pi), 0.0)

        params.update(g.make_params(center=c_seed, sigma=sigma0, amplitude=amp0))
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
            heights[j] = amps[j] / (sigmas[j] * np.sqrt(2.0 * np.pi))
            fwhm[j] = sigma_to_fwhm(sigmas[j])
        peak_at_center[j] = (bkg_slope * centers[j] + bkg_intercept) + (heights[j] if np.isfinite(heights[j]) else 0.0)

    comps = []
    for a, c, s in zip(amps, centers, sigmas):
        if np.all(np.isfinite([a, c, s])) and s > 0:
            comps.append(_gaussian_y(xw, a, c, s))
        else:
            comps.append(np.full_like(xw, np.nan, float))
    return bkg_line, centers, sigmas, amps, heights, fwhm, peak_at_center, comps

def _rebuild_from_kept(xw, yw, result, keep_mask):
    """Rebuild model with only kept components; refit."""
    from lmfit import Parameters
    params_new = Parameters()
    model_new = LinearModel(prefix="bkg_")
    # background
    for nm in ["bkg_slope", "bkg_intercept"]:
        if nm in result.params:
            p = result.params[nm]
            params_new.add(nm, value=p.value, min=p.min, max=p.max, vary=p.vary)
        else:
            params_new.add(nm, value=0.0)

    next_idx = 0
    j = 0
    while f"g{j}_center" in result.params:
        if keep_mask[j]:
            gk = GaussianModel(prefix=f"g{next_idx}_")
            model_new = model_new + gk
            ccur = result.params[f"g{j}_center"].value
            scur = result.params[f"g{j}_sigma"].value
            acur = result.params[f"g{j}_amplitude"].value
            params_new.update(gk.make_params(center=ccur, sigma=scur, amplitude=acur))
            params_new[f"g{next_idx}_sigma"].set(min=SIGMA_MIN_FIT, max=SIGMA_MAX_FIT)
            params_new[f"g{next_idx}_amplitude"].set(min=0.0)
            params_new[f"g{next_idx}_center"].set(min=ccur - DRIFT_NEG, max=ccur + DRIFT_POS)
            next_idx += 1
        j += 1

    refit = model_new.fit(yw, params_new, x=xw, nan_policy="omit")
    return model_new, refit

def _build_params_from_result(res, drop_idx=None):
    """Copy fitted params to a new model; optionally drop component `drop_idx`."""
    from lmfit import Parameters
    params_new = Parameters()
    model_expr = LinearModel(prefix="bkg_")
    for nm in ["bkg_slope","bkg_intercept"]:
        if nm in res.params:
            p = res.params[nm]
            params_new.add(nm, value=p.value, min=p.min, max=p.max, vary=p.vary)
        else:
            params_new.add(nm, value=0.0)
    next_idx = 0
    j = 0
    while f"g{j}_center" in res.params:
        if drop_idx is not None and j == drop_idx:
            j += 1; continue
        g = GaussianModel(prefix=f"g{next_idx}_")
        model_expr = model_expr + g
        params_new.update(g.make_params(
            center=res.params[f"g{j}_center"].value,
            sigma =res.params[f"g{j}_sigma"].value,
            amplitude=res.params[f"g{j}_amplitude"].value))
        params_new[f"g{next_idx}_sigma"].set(min=SIGMA_MIN_FIT, max=SIGMA_MAX_FIT)
        params_new[f"g{next_idx}_amplitude"].set(min=0.0)
        ccur = res.params[f"g{j}_center"].value
        params_new[f"g{next_idx}_center"].set(min=ccur-DRIFT_NEG, max=ccur+DRIFT_POS)
        next_idx += 1; j += 1
    return model_expr, params_new, next_idx

def _try_residual_add_bic(xw, yw, result, max_n):
    """
    Single-pass shoulder: one candidate at max positive residual.
    Admit if BIC improves (ΔBIC < 0), subject to MIN_SEP and σ bounds.
    If at cap, try replacing the weakest component (by height); keep only if BIC improves.
    """
    # current component count
    n_now = 0
    while f"g{n_now}_center" in result.params:
        n_now += 1

    # positive residual and its maximum
    resid = yw - result.best_fit
    rpos = np.maximum(resid, 0.0)
    if not np.any(rpos > 0):
        return result, False
    idx = int(np.argmax(rpos))
    x0 = float(xw[idx])

    # min separation from existing centers
    centers_old = []
    j = 0
    while f"g{j}_center" in result.params:
        centers_old.append(result.params[f"g{j}_center"].value)
        j += 1
    centers_old = np.asarray(centers_old, float)
    if centers_old.size and np.min(np.abs(centers_old - x0)) < MIN_SEP:
        return result, False

    # seed σ from local residual spread in a small neighborhood
    # use ~10% of window span around x0 (clamped to data)
    span = max(xw[-1] - xw[0], 1e-9)
    rad  = 0.10 * span
    nbh  = (xw >= x0 - rad) & (xw <= x0 + rad)
    if not np.any(nbh):
        nbh = np.ones_like(xw, dtype=bool)
    w = rpos[nbh]
    xs = xw[nbh]
    if np.sum(w) <= 0:
        return result, False
    xc = np.sum(xs * w) / np.sum(w)
    var = max(np.sum(w * (xs - xc)**2) / np.sum(w), 1e-12)
    sigma_seed = float(np.clip(np.sqrt(var), SIGMA_MIN_FIT, SIGMA_MAX_FIT))

    # amplitude from area: area ≈ A * sqrt(2π) * σ  =>  A ≈ area / (sqrt(2π) σ)
    area = float(np.sum(w))
    amp_seed = max(area / (np.sqrt(2.0 * np.pi) * max(sigma_seed, 1e-12)), 1e-9)

    # Build trial: add if under cap, else replace weakest-by-height
    def _trial_with(add_at_end=True, drop_idx=None):
        model_expr, params_new, next_idx = _build_params_from_result(result, drop_idx=drop_idx)
        prefix = f"g{next_idx}_"
        gnew = GaussianModel(prefix=prefix)
        model_expr = model_expr + gnew
        params_new.update(gnew.make_params(center=x0, sigma=sigma_seed, amplitude=amp_seed))
        params_new[prefix + "sigma"].set(min=SIGMA_MIN_FIT, max=SIGMA_MAX_FIT)
        params_new[prefix + "amplitude"].set(min=0.0)
        params_new[prefix + "center"].set(min=xw[0], max=xw[-1])
        trial = model_expr.fit(yw, params_new, x=xw, nan_policy="omit")
        return trial

    base_bic = result.bic
    if n_now < max_n:
        trial = _trial_with()
        if trial.bic < base_bic +1:
            return trial, True
        return result, False

    # at cap: replace weakest-by-height
    _, _, _, _, h0, _, _, _ = _extract_metrics(result, xw)
    if h0.size == 0:
        return result, False
    weakest = int(np.nanargmin(h0))
    trial = _trial_with(drop_idx=weakest)
    if trial.bic < base_bic:
        return trial, True
    return result, False

def fit_frame(x, y, seeds, halfwidth):
    m = window_mask(x, seeds, halfwidth)
    if not np.any(m):
        return {"success": False}
    xw, yw = x[m], y[m]
    if xw.size < MIN_POINTS:
        return {"success": False}

    # 1) Seeded fit
    base_model, params = _build_seed_model(xw, yw, seeds)
    try:
        result = base_model.fit(yw, params, x=xw, nan_policy="omit")
    except Exception:
        return {"success": False}

    # 2) Prune sub-threshold heights BEFORE shoulder (frees slot under cap), then refit
    bkg_line, c_all, s_all, a_all, h_all, w_all, p_all, comps = _extract_metrics(result, xw)
    keep = np.isfinite(h_all) & (h_all >= PEAK_HEIGHT_MIN)
    if keep.size and not np.all(keep):
        _, result = _rebuild_from_kept(xw, yw, result, keep)
        bkg_line, c_all, s_all, a_all, h_all, w_all, p_all, comps = _extract_metrics(result, xw)

    # 3) ONE residual-max candidate; admit only if BIC improves (ΔBIC<0), cap respected
    max_allowed = len(seeds)
    result, _ = _try_residual_add_bic(xw, yw, result, max_allowed)
    bkg_line, c_all, s_all, a_all, h_all, w_all, p_all, comps = _extract_metrics(result, xw)

    # 4) Reporting mask (height floor)
    valid = np.isfinite(h_all) & (h_all >= PEAK_HEIGHT_MIN)
    centers_out = c_all.copy(); centers_out[~valid] = np.nan
    fwhm_out    = w_all.copy(); fwhm_out[~valid]    = np.nan
    height_out  = h_all.copy(); height_out[~valid]  = np.nan
    peakfit_out = p_all.copy(); peakfit_out[~valid] = np.nan

    comp_sum = bkg_line + (np.sum(np.vstack(comps), axis=0) if len(comps) else 0.0)
    r2 = r2_score(yw, result.best_fit)
    return {
        "success": True,
        "xw": xw, "yw": yw, "yfit": result.best_fit, "bkg": bkg_line,
        "centers": centers_out, "fwhm": fwhm_out, "height_fit": height_out,
        "peak_fit": peakfit_out, "components": comps, "comp_sum": comp_sum,
        "r2": r2, "result": result
    }

# ------------------------------
# Visual style
# ------------------------------
def apply_pub_style():
    plt.rcParams.update({
        "figure.figsize": (6.5, 4.8),
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "font.size": 12,
        "axes.labelsize": 14,
        "legend.fontsize": 12,
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

# ------------------------------
# Main
# ------------------------------
def main():
    ap = argparse.ArgumentParser(description="Gaussian multi-peak tracker (linear bkg, BIC-gated shoulder).")
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

        vis = np.isfinite(res["centers"])
        centers_v = res["centers"][vis]
        fwhm_v    = res["fwhm"][vis]
        hfit_v    = res["height_fit"][vis]
        pfit_v    = res["peak_fit"][vis]

        # ---- Plot (single frame)
        apply_pub_style()
        from matplotlib.gridspec import GridSpec
        fig = plt.figure()
        gs = GridSpec(2, 1, height_ratios=[3.2, 1.2], hspace=0.18)

        ax = fig.add_subplot(gs[0]); style_axes(ax, light_grid=True)
        ax.plot(res["xw"], res["yw"],  lw=1.2, label="Data")
        ax.plot(res["xw"], res["yfit"], lw=1.6, label="Total fit")
        ax.plot(res["xw"], res["bkg"],  "--", lw=1.0, label="Linear bkg")

        for comp in res["components"]:
            ax.plot(res["xw"], comp, ":", lw=1.0, alpha=0.75)

        ax.plot(res["xw"], res["comp_sum"], lw=1.0, color="k", alpha=0.55, label="Gaussians + bkg")

        for c in centers_v:
            ax.axvline(c, linestyle="--", alpha=0.6, lw=1.0, color="0.45")

        title = (f"{args.frame*float(SEC_PER_FRAME):.1f} s | " if SEC_PER_FRAME is not None else f"Frame {args.frame} | ")
        ax.set_title(title + f"R²={res['r2']:.4f}", pad=6)
        ax.set_xlabel("q (1/Å)")
        ax.set_ylabel("Intensity (a.u.)")
        ax.legend(loc="upper right", ncol=1)

        # Compact table
        ax_tbl = fig.add_subplot(gs[1]); style_axes(ax_tbl, light_grid=False); ax_tbl.axis("off")
        rows = [[f"peak{i}", f"{c:.6f}", f"{w:.6f}", f"{h:.6f}", f"{p:.6f}"]
                for i, (c, w, h, p) in enumerate(zip(centers_v, fwhm_v, hfit_v, pfit_v))]
        col_labels = ["Peak", "Center", "FWHM", "Height (fit)", "Peak@Center (fit)"]
        tbl = ax_tbl.table(cellText=rows, colLabels=col_labels, loc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(10)
        for (r, _c), cell in tbl.get_celld().items():
            cell.set_edgecolor("0.85"); cell.set_linewidth(0.6)
            cell.set_alpha(0.0 if r == 0 else 0.06)

        fig.tight_layout()
        plt.show()
        return

    # -------- Mapping (all frames) --------
    # Map uses EXACT same single-frame logic per frame with the ORIGINAL seeds0 (no per-frame seeding)
    nuse = nframes
    npeaks = len(seeds0)            # hard cap
    centers_trk = np.full((nuse, npeaks), np.nan)
    fwhm_trk    = np.full((nuse, npeaks), np.nan)
    height_trk  = np.full((nuse, npeaks), np.nan)

    iterator = range(nuse)
    if tqdm is not None:
        iterator = tqdm(iterator, desc="Fitting frames", ncols=80)

    for f in iterator:
        y = I_full[f]
        res = fit_frame(x, y, seeds0, HALF_WINDOW)
        if not res["success"]:
            continue

        # Drop anything below the height floor (belt-and-suspenders)
        valid = np.isfinite(res["centers"]) & np.isfinite(res["height_fit"])
        if np.any(valid):
            c = res["centers"][valid]
            w = res["fwhm"][valid]
            h = res["height_fit"][valid]
            hi = h >= PEAK_HEIGHT_MIN
            c, w, h = c[hi], w[hi], h[hi]
        else:
            c = w = h = np.array([])

        # Put up to npeaks into columns (sorted by center for stability)
        if c.size:
            order = np.argsort(c)
            c, w, h = c[order], w[order], h[order]
        k = min(c.size, npeaks)
        centers_trk[f, :k] = c[:k]
        fwhm_trk[f, :k]    = w[:k]
        height_trk[f, :k]  = h[:k]

    # ---- Map plot
    apply_pub_style()
    fig, ax = plt.subplots(); style_axes(ax, light_grid=True)

    frames = np.arange(nuse)
    handles, labels = [], []
    for j in range(npeaks):
        mask = (
            np.isfinite(centers_trk[:, j]) &
            np.isfinite(height_trk[:, j]) &
            (height_trk[:, j] >= PEAK_HEIGHT_MIN)
        )
        if not np.any(mask):
            continue
        sc = ax.scatter(
            frames[mask],
            centers_trk[mask, j],
            c=height_trk[mask, j],
            cmap="plasma",
            s=18,
            linewidths=0.0,
            edgecolors="none",
        )
        handles.append(sc); labels.append(f"Peak {j}")

    ax.set_xlabel("Frame")
    ax.set_ylabel("Center (q or 2θ)")
    ax.set_title("Peak centers over frames (color = fitted height)")

    if handles:
        cbar = fig.colorbar(handles[0], ax=ax, pad=0.02)
        cbar.set_label("Height (fit)")
        ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    fig.tight_layout(rect=[0.0, 0.0, 0.82, 1.0])
    plt.show()

if __name__ == "__main__":
    main()






