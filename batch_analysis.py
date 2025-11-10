#!/usr/bin/env python3
# Core v1 — Gaussian multi-peak fitting with linear background
# - Single-frame mode and full-experiment tracking
# - Fixed number of peaks from --centers (hard cap)
# - Shoulder robustness via sigma bounds, per-seed drift limits, residual SNR + AIC checks (cap respected)
# - Clean CSV and simple plots for iteration

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
# Tunables (minimal & focused)
# ------------------------------
HALF_WINDOW = 0.25        # window half-width around [min(seeds), max(seeds)]
MIN_POINTS  = 8           # min points in window to attempt a fit

# Peak reporting floor
PEAK_HEIGHT_MIN = 5.0     # fitted height above background to report/track

# Global sigma bounds (handles tall/narrow and small/wide)
SIGMA_MIN_FIT = 0.0002
SIGMA_MAX_FIT = 0.080

# Per-seed drift limits per frame (asymmetric helps avoid identity swaps)
# center_i ∈ [seed_i - DRIFT_NEG, seed_i + DRIFT_POS]
DRIFT_NEG = 0.10
DRIFT_POS = 0.010

# Residual-shoulder logic (never exceed seeded count)
ENABLE_RESIDUAL = True
RESIDUAL_SNR    = 0.25     # residual peak must be ≥ SNR * robust_noise
MIN_SEP         = 0.0007  # min separation from existing centers
AIC_IMPROVE     = 8.0     # require ΔAIC ≤ -AIC_IMPROVE to accept replacement

SEC_PER_FRAME   = 0.004   # for titles; set None to show "Frame N" instead

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
    # background init via linear fit
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

def _hit_bounds(result, eps=1e-10):
    hits = []
    for name, par in result.params.items():
        if par.vary and par.min is not None and par.max is not None:
            if abs(par.value - par.min) <= eps: hits.append(f"{name}==min")
            elif abs(par.value - par.max) <= eps: hits.append(f"{name}==max")
    return hits

def _try_residual_replacement(xw, yw, result, max_n):
    """Try to add a residual-based component, but NEVER exceed max_n.
       If we'd exceed, replace the weakest (height) component if the AIC improves."""
    # current components
    centers_old, sigmas_old, amps_old = [], [], []
    i = 0
    while f"g{i}_center" in result.params:
        centers_old.append(result.params[f"g{i}_center"].value)
        sigmas_old.append (result.params[f"g{i}_sigma"].value)
        amps_old.append   (result.params[f"g{i}_amplitude"].value)
        i += 1
    n_now = len(centers_old)

    resid = yw - result.best_fit
    noise = robust_sigma(resid)
    if noise <= 0: 
        return result, False

    idx = int(np.argmax(resid))
    if resid[idx] < RESIDUAL_SNR * noise:
        return result, False

    x0 = float(xw[idx])
    centers_old = np.asarray(centers_old, float)
    if centers_old.size and np.min(np.abs(centers_old - x0)) < MIN_SEP:
        return result, False

    span = max(xw[-1] - xw[0], 1e-9)
    mean_sigma = np.nanmean(np.asarray(sigmas_old)) if n_now else span/10.0
    sigma_seed = float(np.clip(min(mean_sigma, span/12.0), SIGMA_MIN_FIT, SIGMA_MAX_FIT))
    amp_seed = max(resid[idx] * sigma_seed * np.sqrt(2.0 * np.pi), 1e-9)

    # Build a trial model by cloning the old result
    from lmfit import Parameters
    params_new = Parameters()
    model_expr = LinearModel(prefix="bkg_")
    # copy background
    for nm in ["bkg_slope","bkg_intercept"]:
        if nm in result.params:
            p = result.params[nm]
            params_new.add(nm, value=p.value, min=p.min, max=p.max, vary=p.vary)
        else:
            params_new.add(nm, value=0.0)

    # copy existing components
    for j in range(n_now):
        g = GaussianModel(prefix=f"g{j}_")
        model_expr = model_expr + g
        for nm in [f"g{j}_center", f"g{j}_sigma", f"g{j}_amplitude"]:
            p = result.params[nm]
            params_new.add(nm, value=p.value, min=p.min, max=p.max, vary=p.vary)

    # add candidate as the next index
    cand_idx = n_now
    gnew = GaussianModel(prefix=f"g{cand_idx}_")
    model_expr = model_expr + gnew
    params_new.update(gnew.make_params(center=x0, sigma=sigma_seed, amplitude=amp_seed))
    params_new[f"g{cand_idx}_sigma"].set(min=SIGMA_MIN_FIT, max=SIGMA_MAX_FIT)
    params_new[f"g{cand_idx}_amplitude"].set(min=0.0)
    params_new[f"g{cand_idx}_center"].set(min=xw[0], max=xw[-1])

    trial = model_expr.fit(yw, params_new, x=xw, nan_policy="omit")
    dAIC = trial.aic - result.aic  # negative is better

    # If we are under the cap, accept if it clearly helps and the new peak is real enough.
    bkg_line, c_all, s_all, a_all, h_all, _, _, _ = _extract_metrics(trial, xw)
    if cand_idx < max_n:
        new_h = h_all[cand_idx] if cand_idx < h_all.size else -np.inf
        if (dAIC <= -AIC_IMPROVE) and np.isfinite(new_h) and new_h >= PEAK_HEIGHT_MIN:
            return trial, True
        return result, False

    # If we'd exceed the cap, consider replacing the weakest existing component.
    if n_now >= max_n:
        # find weakest (by fitted height) among existing
        heights_old = []
        for j in range(n_now):
            if s_all[j] > 0:
                heights_old.append(a_all[j] / (s_all[j]*np.sqrt(2.0*np.pi)))
            else:
                heights_old.append(-np.inf)
        weakest = int(np.argmin(heights_old))
        # Build a "replacement" parameterset: drop weakest, keep others + candidate
        from lmfit import Parameters
        params_rep = Parameters()
        model_rep = LinearModel(prefix="bkg_")
        for nm in ["bkg_slope","bkg_intercept"]:
            if nm in result.params:
                p = result.params[nm]
                params_rep.add(nm, value=p.value, min=p.min, max=p.max, vary=p.vary)
            else:
                params_rep.add(nm, value=0.0)

        next_idx = 0
        # keep all except weakest, but reindex to keep param names compact
        for j in range(n_now):
            if j == weakest: 
                continue
            gk = GaussianModel(prefix=f"g{next_idx}_")
            model_rep = model_rep + gk
            params_rep.update(gk.make_params(
                center=result.params[f"g{j}_center"].value,
                sigma =result.params[f"g{j}_sigma"].value,
                amplitude=result.params[f"g{j}_amplitude"].value))
            params_rep[f"g{next_idx}_sigma"].set(min=SIGMA_MIN_FIT, max=SIGMA_MAX_FIT)
            params_rep[f"g{next_idx}_amplitude"].set(min=0.0)
            # keep the original drift band roughly around current value
            ccur = result.params[f"g{j}_center"].value
            params_rep[f"g{next_idx}_center"].set(min=ccur-DRIFT_NEG, max=ccur+DRIFT_POS)
            next_idx += 1

        # add candidate as last
        gk = GaussianModel(prefix=f"g{next_idx}_")
        model_rep = model_rep + gk
        params_rep.update(gk.make_params(center=x0, sigma=sigma_seed, amplitude=amp_seed))
        params_rep[f"g{next_idx}_sigma"].set(min=SIGMA_MIN_FIT, max=SIGMA_MAX_FIT)
        params_rep[f"g{next_idx}_amplitude"].set(min=0.0)
        params_rep[f"g{next_idx}_center"].set(min=xw[0], max=xw[-1])

        rep = model_rep.fit(yw, params_rep, x=xw, nan_policy="omit")
        if rep.aic <= result.aic - AIC_IMPROVE:
            return rep, True
        return result, False

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

    max_allowed = len(seeds)
    if ENABLE_RESIDUAL and max_allowed > 0:
        result, _ = _try_residual_replacement(xw, yw, result, max_allowed)

    bkg_line, c_all, s_all, a_all, h_all, w_all, p_all, comps = _extract_metrics(result, xw)

    # apply reporting floor
    valid = (h_all >= PEAK_HEIGHT_MIN) & np.isfinite(h_all)
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

def apply_pub_style():
    plt.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 300,
        "font.size": 12, "axes.titlesize": 12, "axes.labelsize": 14,
        "axes.linewidth": 1.0, "xtick.labelsize": 12, "ytick.labelsize": 12,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.size": 4, "ytick.major.size": 4
    })

def style_axes(ax):
    for side in ("top","right","bottom","left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.0)
    ax.minorticks_on()
    ax.grid(True, which="major", alpha=0.12, linestyle="-", linewidth=0.6)

# ------------------------------
# Main
# ------------------------------
def main():
    ap = argparse.ArgumentParser(description="Gaussian multi-peak tracker (linear background, fixed peak count).")
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

        # numeric dump
        print(f"# Frame {args.frame}")
        for i, (c, w, h, p) in enumerate(zip(centers_v, fwhm_v, hfit_v, pfit_v)):
            print(f"peak{i}_center={c:.6f}, peak{i}_FWHM={w:.6f}, peak{i}_height_fit={h:.6f}, peak{i}_peak_fit={p:.6f}")

        # plot
        apply_pub_style()
        from matplotlib.gridspec import GridSpec
        fig = plt.figure(figsize=(6.6, 4.8))
        gs = GridSpec(2, 1, height_ratios=[3.0, 1.45], hspace=0.18)
        ax = fig.add_subplot(gs[0]); style_axes(ax)

        ax.plot(res["xw"], res["yw"],  lw=1.2, label="Data")
        ax.plot(res["xw"], res["yfit"], lw=1.4, label="Fit")
        ax.plot(res["xw"], res["bkg"],  "--", lw=1.0, label="Background")

        for comp in res["components"]:
            ax.plot(res["xw"], comp, ":", lw=1.0, alpha=0.7)

        ax.plot(res["xw"], res["comp_sum"], lw=0.9, color="k", alpha=0.6, label="Gaussian Sum + Background")

        for c in centers_v:
            ax.axvline(c, linestyle="--", alpha=0.45, lw=0.95, color="0.5")

        title = (f"{args.frame*float(SEC_PER_FRAME):.1f} sec | " if SEC_PER_FRAME is not None else f"Frame {args.frame} | ")
        ax.set_title(title + f"R²={res['r2']:.4f}", pad=6)
        ax.set_xlabel("q (1/Å)")
        ax.set_ylabel("Intensity (a.u.)")
        leg = ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
        fig.tight_layout(rect=[0.0, 0.0, 0.80, 1.0])

        # table
        ax_tbl = fig.add_subplot(gs[1]); style_axes(ax_tbl); ax_tbl.axis("off")
        rows = [[f"peak{i}", f"{c:.6f}", f"{w:.6f}", f"{h:.6f}", f"{p:.6f}"]
                for i, (c, w, h, p) in enumerate(zip(centers_v, fwhm_v, hfit_v, pfit_v))]
        col_labels = ["Peak", "Center", "FWHM", "Height (fit)", "Peak@Center (fit)"]
        tbl = ax_tbl.table(cellText=rows, colLabels=col_labels, loc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(9)
        for key, cell in tbl.get_celld().items():
            cell.set_edgecolor("0.8"); cell.set_linewidth(0.6)
            if key[0] != 0: cell.set_alpha(0.08)

        fig.tight_layout(rect=[0.0, 0.0, 0.80, 1.0])
        plt.show()
        return

    # -------- Mapping (all frames) --------
    nuse = nframes
    npeaks = len(seeds0)            # hard cap
    centers_trk = np.full((nuse, npeaks), np.nan)
    fwhm_trk    = np.full((nuse, npeaks), np.nan)
    height_trk  = np.full((nuse, npeaks), np.nan)

    seeds = seeds0.copy()
    iterator = range(nuse)
    if tqdm is not None:
        iterator = tqdm(iterator, desc="Fitting frames", ncols=80)

    for f in iterator:
        y = I_full[f]
        res = fit_frame(x, y, seeds, HALF_WINDOW)
        if res["success"]:
            vis = np.isfinite(res["centers"])
            cvis = res["centers"][vis]
            wvis = res["fwhm"][vis]
            hvis = res["height_fit"][vis]
            k = min(cvis.size, npeaks)
            centers_trk[f, :k] = cvis[:k]
            fwhm_trk[f, :k]    = wvis[:k]
            height_trk[f, :k]  = hvis[:k]
            if k == npeaks:
                seeds = cvis[:npeaks]   # per-frame seeding

    # Preview
    print("# preview:")
    for r in range(min(5, nuse)):
        row = [str(r)] + [f"{v:.6f}" if np.isfinite(v) else "nan" for v in centers_trk[r]] \
                    + [f"{v:.6f}" if np.isfinite(v) else "nan" for v in fwhm_trk[r]]
        print(",".join(row))

    # Map plot
    apply_pub_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.2)); style_axes(ax)
    frames = np.arange(nuse)
    for j in range(npeaks):
        mask = np.isfinite(centers_trk[:, j]) & np.isfinite(height_trk[:, j])
        if not np.any(mask): 
            continue
        sc = ax.scatter(frames[mask], centers_trk[mask, j], c=height_trk[mask, j], cmap="plasma", s=14, edgecolors="none", label=f"peak {j}")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Center (q or 2θ)")
    ax.set_title("Peak Centers over Frames (colored by fitted height)")
    if np.any(np.isfinite(height_trk)):
        cbar = fig.colorbar(ax.collections[0], ax=ax, pad=0.02)
        cbar.set_label("Height (fit)")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    fig.tight_layout(rect=[0.0, 0.0, 0.82, 1.0])
    plt.show()

if __name__ == "__main__":
    main()





