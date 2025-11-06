#!/usr/bin/env python3
# Multi-peak tracker: residual-added shoulder, global sigma bounds,
# fitted-height filtering, asymmetric seed-shift limits, polished plotting,
# component-sum check, tqdm progress, and a "map" scatter plot of centers vs frame
# colored by fitted height.

import argparse, os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from lmfit.models import GaussianModel, LinearModel

# tqdm (optional)
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

# ------------------------------
# Tunables
# ------------------------------
WINDOW = 0.50          # half-width around the seeded peak range (x-units, e.g., q)
MIN_POINTS = 8         # minimum points in window to attempt a fit
PEAK_HEIGHT_MIN = 5.0  # min *fitted* height above background to report/plot

# Global sigma bounds for ALL components (seeded and residual-added)
SIGMA_MIN_FIT = 0.0002  # raise to avoid needle spikes
SIGMA_MAX_FIT = 0.080   # raise to allow broad shoulders

# Asymmetric per-frame seed shift limits (relative to current seed)
# center_i ∈ [seed_i - CENTER_SHIFT_NEG, seed_i + CENTER_SHIFT_POS]
CENTER_SHIFT_NEG = 0.10
CENTER_SHIFT_POS = 0.010

# Residual-shoulder controls
ENABLE_RESIDUAL_SHOULDER = True
RESIDUAL_SNR = 0.5      # residual SNR threshold
MIN_SEP = 0.0010        # min separation from existing peaks (x units)
AIC_IMPROVE = 6.0       # require ΔAIC <= -AIC_IMPROVE to accept new peak

# Plotting preferences
PANEL_LABEL = ""            # set "" to hide
SHOW_SEEDS = True           # show input seeds as light-blue vlines
SEC_PER_FRAME = .004        # e.g., 0.01 -> title "t sec | R²=..."
PLOT_ONLY_VALID_COMPONENTS = False  # if True, only components that pass PEAK_HEIGHT_MIN are drawn
SHOW_COMPONENT_SUM = True   # draw thin black line of (background + sum(components))
LEGEND_COLS = 3

# ------------------------------
# Helpers
# ------------------------------
def parse_centers(s: str):
    vals = [float(v) for v in s.split(",") if v.strip()]
    if not vals:
        raise ValueError("No centers parsed from --centers.")
    return np.array(vals, float)

def sigma_to_fwhm(sigma): return 2.354820045 * sigma

def robust_sigma(y):
    y = np.asarray(y, float)
    med = np.median(y)
    return 1.4826 * np.median(np.abs(y - med)) + 1e-12

def r2_score(y_true, y_fit):
    y_true = np.asarray(y_true, float)
    y_fit  = np.asarray(y_fit, float)
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

def combined_window_mask(x, centers, halfwidth):
    lo = np.min(centers) - halfwidth
    hi = np.max(centers) + halfwidth
    return (x >= lo) & (x <= hi)

def initial_params_for_frame(xw, yw, seeds, halfwidth):
    """
    Build model (linear background + one Gaussian per seed) & params.
    Enforces asymmetric per-seed shift limits around 'seeds'.
    """
    # background initial guess
    try:
        bkg_slope, bkg_intercept = np.polyfit(xw, yw, 1)
    except Exception:
        bkg_slope, bkg_intercept = 0.0, float(np.median(yw))

    model = LinearModel(prefix="bkg_")
    params = model.make_params(bkg_slope=bkg_slope, bkg_intercept=bkg_intercept)

    span = max(xw[-1] - xw[0], 1e-9)
    sigma0_base = max(span / (7.0 * len(seeds)), 1e-6)
    sigma0 = float(np.clip(sigma0_base, SIGMA_MIN_FIT, SIGMA_MAX_FIT))

    # seeded peaks
    for i, c_seed in enumerate(seeds):
        g = GaussianModel(prefix=f"g{i}_")
        model = model + g
        idx = np.abs(xw - c_seed).argmin()
        y_at_seed = yw[idx]
        y_bkg = bkg_slope * xw[idx] + bkg_intercept
        height0 = max(y_at_seed - y_bkg, np.std(yw) * 0.5)
        amp0 = max(height0 * sigma0 * np.sqrt(2.0 * np.pi), 0.0)
        params.update(g.make_params(center=c_seed, sigma=sigma0, amplitude=amp0))
        # enforce global sigma bounds
        params[f"g{i}_sigma"].set(min=SIGMA_MIN_FIT, max=SIGMA_MAX_FIT)
        params[f"g{i}_amplitude"].set(min=0.0)
        # asymmetric seed-shift limits
        params[f"g{i}_center"].set(min=c_seed - CENTER_SHIFT_NEG,
                                   max=c_seed + CENTER_SHIFT_POS)

    return model, params

def _gaussian_y(x, amp, cen, sig):
    # pure Gaussian component (no background)
    return amp * np.exp(-(x - cen)**2 / (2.0 * sig**2))

def _compute_fitted_metrics(result, xw):
    # background line
    bkg_slope = result.params.get("bkg_slope").value if "bkg_slope" in result.params else 0.0
    bkg_intercept = result.params.get("bkg_intercept").value if "bkg_intercept" in result.params else 0.0
    bkg_line = bkg_slope * xw + bkg_intercept

    # collect gaussians
    i = 0; centers=[]; sigmas=[]; amps=[]
    while f"g{i}_center" in result.params:
        centers.append(result.params[f"g{i}_center"].value)
        sigmas.append(result.params[f"g{i}_sigma"].value)
        amps.append(result.params[f"g{i}_amplitude"].value)
        i += 1
    centers = np.asarray(centers, float)
    sigmas  = np.asarray(sigmas, float)
    amps    = np.asarray(amps, float)

    heights = np.full_like(centers, np.nan, float)
    fwhm    = np.full_like(centers, np.nan, float)
    peakfit = np.full_like(centers, np.nan, float)
    for j in range(centers.size):
        if np.isfinite(sigmas[j]) and sigmas[j] > 0:
            heights[j] = amps[j] / (sigmas[j] * np.sqrt(2.0 * np.pi))
            fwhm[j] = sigma_to_fwhm(sigmas[j])
        peakfit[j] = (bkg_slope * centers[j] + bkg_intercept) + (heights[j] if np.isfinite(heights[j]) else 0.0)

    # component curves (pure Gaussians) and their sum
    comps = []
    for a, c, s in zip(amps, centers, sigmas):
        if np.all(np.isfinite([a, c, s])):
            comps.append(_gaussian_y(xw, a, c, s))
        else:
            comps.append(np.full_like(xw, np.nan, float))
    comps = comps if len(comps) else []

    return bkg_line, centers, sigmas, amps, heights, fwhm, peakfit, comps

# --- MODIFIED: add a cap on number of components via max_allowed ---
def _try_add_shoulder(xw, yw, model, result, max_allowed):
    """
    Try adding a residual-based shoulder, but NEVER exceed max_allowed components.
    max_allowed should typically be len(seeds) from the CLI.
    """
    # count current components
    n_now = 0
    while f"g{n_now}_center" in result.params:
        n_now += 1
    if n_now >= max_allowed:
        # Already at or above the cap → refuse to add.
        return model, result, False

    resid = yw - result.best_fit
    noise = robust_sigma(resid)
    if noise <= 0:
        return model, result, False

    idx = int(np.argmax(resid))
    peak_resid = resid[idx]
    if peak_resid < RESIDUAL_SNR * noise:
        return model, result, False

    x0 = xw[idx]
    _, centers, sigmas, amps, heights, fwhm, peakfit, _ = _compute_fitted_metrics(result, xw)
    if centers.size and np.min(np.abs(centers - x0)) < MIN_SEP:
        return model, result, False

    span = max(xw[-1] - xw[0], 1e-9)
    mean_sigma = np.nanmean(sigmas) if sigmas.size else span/10.0
    sigma_seed = float(np.clip(min(mean_sigma, span/12.0), SIGMA_MIN_FIT, SIGMA_MAX_FIT))
    amp_seed = max(peak_resid * sigma_seed * np.sqrt(2.0 * np.pi), 1e-9)

    # next prefix
    new_idx = n_now  # append at the end
    prefix = f"g{new_idx}_"

    g = GaussianModel(prefix=prefix)
    new_model = model + g
    new_params = result.params.copy()
    new_params.update(g.make_params(center=float(x0), sigma=sigma_seed, amplitude=amp_seed))
    new_params[prefix + "sigma"].set(min=SIGMA_MIN_FIT, max=SIGMA_MAX_FIT)
    new_params[prefix + "amplitude"].set(min=0.0)
    new_params[prefix + "center"].set(min=xw[0], max=xw[-1])  # keep within window

    new_result = new_model.fit(yw, new_params, x=xw, nan_policy="omit")

    dAIC = new_result.aic - result.aic   # negative is better
    _, c_new, s_new, a_new, h_new, f_new, p_new, _ = _compute_fitted_metrics(new_result, xw)
    if c_new.size <= new_idx:
        return model, result, False
    new_height = h_new[new_idx]

    if (dAIC <= -AIC_IMPROVE) and np.isfinite(new_height) and (new_height >= PEAK_HEIGHT_MIN):
        return new_model, new_result, True
    return model, result, False

def _check_hit_bounds(result, eps=1e-10):
    """Return list of strings noting params that hit min/max bounds."""
    hits = []
    for name, par in result.params.items():
        if par.vary and par.min is not None and par.max is not None:
            if abs(par.value - par.min) <= eps:
                hits.append(f"{name}==min")
            elif abs(par.value - par.max) <= eps:
                hits.append(f"{name}==max")
    return hits

def fit_frame(x, y, seeds, halfwidth):
    m = combined_window_mask(x, seeds, halfwidth)
    if not np.any(m):
        return {"success": False}
    xw, yw = x[m], y[m]
    if xw.size < MIN_POINTS:
        return {"success": False}

    base_model, params = initial_params_for_frame(xw, yw, seeds, halfwidth)
    try:
        result = base_model.fit(yw, params, x=xw, nan_policy="omit")
    except Exception:
        return {"success": False}

    model_used = base_model
    if ENABLE_RESIDUAL_SHOULDER:
        # --- pass cap: at most as many components as user-supplied centers ---
        max_allowed = len(seeds)
        model_used, result, _ = _try_add_shoulder(xw, yw, model_used, result, max_allowed)

    # metrics
    bkg_line, c_all, s_all, a_all, h_all, w_all, p_all, comps_all = _compute_fitted_metrics(result, xw)

    # filter by fitted height (for reporting)
    valid = (h_all >= PEAK_HEIGHT_MIN) & np.isfinite(h_all)
    centers_out = c_all.copy(); centers_out[~valid] = np.nan
    fwhm_out    = w_all.copy(); fwhm_out[~valid]    = np.nan
    height_out  = h_all.copy(); height_out[~valid]  = np.nan
    peakfit_out = p_all.copy(); peakfit_out[~valid] = np.nan

    # choose which components to plot
    if PLOT_ONLY_VALID_COMPONENTS:
        comps_plot = [c for j,c in enumerate(comps_all) if valid[j]]
    else:
        comps_plot = comps_all

    # component sum (for sanity overlay)
    if len(comps_all):
        comp_sum = bkg_line + np.sum(np.vstack(comps_all), axis=0)
    else:
        comp_sum = bkg_line.copy()

    # R^2 on the window
    r2 = r2_score(yw, result.best_fit)

    # bounds diagnostics
    hit_bounds = _check_hit_bounds(result)

    return {
        "success": True,
        "xw": xw, "yw": yw, "yfit": result.best_fit, "bkg": bkg_line,
        "centers": centers_out, "fwhm": fwhm_out, "height_fit": height_out,
        "peak_fit": peakfit_out, "components": comps_plot, "comp_sum": comp_sum,
        "r2": r2, "hit_bounds": hit_bounds, "result": result
    }

def apply_nature_style():
    plt.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 300,
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 10,
        "axes.linewidth": 1.1, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.size": 4, "ytick.major.size": 4,
        "xtick.minor.size": 2, "ytick.minor.size": 2,
        "legend.frameon": True, "legend.edgecolor": "0.7"
    })

def style_axes_box(ax):
    # full outline + minor ticks + subtle grid
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.1)
    ax.minorticks_on()
    ax.grid(True, which="major", alpha=0.12, linestyle="-", linewidth=0.6)

# ------------------------------
# Main
# ------------------------------
def main():
    ap = argparse.ArgumentParser(description="Gaussian multi-peak tracker (linear background).")
    ap.add_argument("--h5", required=True, help="HDF5 with 'q' (or 'tth') and 'int'")
    ap.add_argument("--centers", required=True,
                    help="Comma-separated initial peak centers (e.g., 2.975,3.124)")
    ap.add_argument("--frame", type=int, default=None,
                    help="Fit a single frame index. Omit to track all frames.")
    args = ap.parse_args()

    seeds0 = parse_centers(args.centers)   # seeds for first use
    x, I_full = load_q_and_I(args.h5)
    nframes = I_full.shape[0]

    # ---------- Single-frame mode ----------
    if args.frame is not None:
        if not (0 <= args.frame < nframes):
            raise ValueError(f"--frame {args.frame} is out of range [0, {nframes-1}]")
        y = I_full[args.frame]
        res = fit_frame(x, y, seeds0, WINDOW/2.0)
        if not res["success"]:
            # one retry with wider window
            res = fit_frame(x, y, seeds0, WINDOW)
            if not res["success"]:
                print("Fit failed for the requested frame.")
                return

        # visible (non-NaN) peaks for table
        vis = np.isfinite(res["centers"])
        centers_v = res["centers"][vis]
        fwhm_v    = res["fwhm"][vis]
        hfit_v    = res["height_fit"][vis]
        pfit_v    = res["peak_fit"][vis]

        # numeric results + diagnostics
        print(f"# Frame {args.frame}")
        for i_vis, (c, w, h, p) in enumerate(zip(centers_v, fwhm_v, hfit_v, pfit_v)):
            print(f"peak{i_vis}_center={c:.6f}, peak{i_vis}_FWHM={w:.6f}, "
                  f"peak{i_vis}_height_fit={h:.6f}, peak{i_vis}_peak_fit={p:.6f}")
        if res["hit_bounds"]:
            print("HIT_BOUNDS:", "; ".join(res["hit_bounds"]))

        # ---- Plot ----
        apply_nature_style()
        from matplotlib.gridspec import GridSpec
        fig = plt.figure(figsize=(6.6, 4.8))
        gs = GridSpec(2, 1, height_ratios=[3.0, 1.45], hspace=0.18)
        ax = fig.add_subplot(gs[0])

        style_axes_box(ax)

        if PANEL_LABEL:
            ax.text(-0.12, 1.02, PANEL_LABEL, transform=ax.transAxes,
                    fontsize=12, fontweight="bold", va="bottom", ha="left")

        # data / fit / background
        ax.plot(res["xw"], res["yw"], lw=1.2, label="Data")
        ax.plot(res["xw"], res["yfit"], lw=1.4, label="Fit")
        ax.plot(res["xw"], res["bkg"],  "--", lw=1.0, label="Background", color="tab:green")

        # pure Gaussian components (no background) dotted
        for comp in res["components"]:
            ax.plot(res["xw"], comp, ":", lw=1.0, alpha=0.7, color="0.35")

        # optional check: background + sum(components) overlays the fit
        if SHOW_COMPONENT_SUM:
            ax.plot(res["xw"], res["comp_sum"], lw=0.9, color="k", alpha=0.6, label="Idividual Guassian Fit")

        # fitted centers (for visible peaks)
        for c in centers_v:
            ax.axvline(c, linestyle="--", alpha=0.5, lw=0.95, color="0.5")

        # title with R^2 (and time if provided)
        if SEC_PER_FRAME is not None:
            t = args.frame * float(SEC_PER_FRAME)
            title = f"{t:.1f} sec | R²={res['r2']:.4f}"
        else:
            title = f"Frame {args.frame} | R²={res['r2']:.4f}"
        ax.set_title(title, pad=6)

        ax.set_xlabel("q (1/Å)")
        ax.set_ylabel("Intensity (a.u.)")

        # Legend in a boxed panel to the RIGHT of the axes
        leg = ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                        borderaxespad=0.0, frameon=True, edgecolor="0.7",
                        facecolor="white", fontsize=8, ncol=LEGEND_COLS)
        # leave right margin for legend
        fig.tight_layout(rect=[0.0, 0.0, 0.80, 1.0])

        # table (fitted values)
        ax_tbl = fig.add_subplot(gs[1])
        style_axes_box(ax_tbl)
        ax_tbl.axis("off")
        table_data = [[f"peak{i}", f"{c:.6f}", f"{w:.6f}", f"{h:.6f}", f"{p:.6f}"]
                      for i, (c, w, h, p) in enumerate(zip(centers_v, fwhm_v, hfit_v, pfit_v))]
        col_labels = ["Peak", "Center", "FWHM", "Height (fit)", "Peak@Center (fit)"]
        tbl = ax_tbl.table(cellText=table_data, colLabels=col_labels, loc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(8)
        for key, cell in tbl.get_celld().items():
            cell.set_edgecolor("0.8"); cell.set_linewidth(0.6)
            cell.set_height(0.18); cell.set_alpha(0.0 if key[0] == 0 else 0.10)

        fig.tight_layout(rect=[0.0, 0.0, 0.80, 1.0])
        plt.show()
        return

    # ---------- Mapping mode (all frames, with tqdm) ----------
    nuse = nframes
    npeaks_seeded = len(seeds0)
    # allow one optional shoulder column (still allocated, but residual add is capped)
    ncols = npeaks_seeded + 1
    centers_trk = np.full((nuse, ncols), np.nan)
    fwhm_trk    = np.full((nuse, ncols), np.nan)
    height_trk  = np.full((nuse, ncols), np.nan)   # store fitted heights for coloring

    seeds = seeds0.copy()
    iterator = range(nuse)
    if tqdm is not None:
        iterator = tqdm(iterator, desc="Fitting frames", ncols=80)

    for f in iterator:
        y = I_full[f]
        res = fit_frame(x, y, seeds, WINDOW/2.0)
        if not res["success"]:
            res = fit_frame(x, y, seeds, WINDOW)
        if res["success"]:
            vis = np.isfinite(res["centers"])
            cvis = res["centers"][vis]
            wvis = res["fwhm"][vis]
            hvis = res["height_fit"][vis]
            k = min(cvis.size, ncols)
            centers_trk[f, :k] = cvis[:k]
            fwhm_trk[f, :k]    = wvis[:k]
            height_trk[f, :k]  = hvis[:k]
            if cvis.size >= npeaks_seeded:
                seeds = cvis[:npeaks_seeded]

    # Write CSV
    base = os.path.splitext(os.path.basename(args.h5))[0]
    csv_path = f"{base}_multi_peak_tracking.csv"
    header_cols = ["frame"] + [f"center_{i}" for i in range(ncols)] + \
                  [f"FWHM_{i}" for i in range(ncols)]
    arr = np.column_stack([np.arange(nuse), centers_trk, fwhm_trk])
    np.savetxt(csv_path, arr, delimiter=",", header=",".join(header_cols),
               comments="", fmt="%.10g")
    print(f"Wrote: {csv_path}")

    # Preview
    preview_rows = min(5, nuse)
    print("# preview:")
    for r in range(preview_rows):
        parts = [str(r)] + [f"{v:.6f}" if np.isfinite(v) else "nan" for v in centers_trk[r]] + \
                [f"{v:.6f}" if np.isfinite(v) else "nan" for v in fwhm_trk[r]]
        print(",".join(parts))

    # ----- Map scatter: frame vs center, color by fitted height -----
    apply_nature_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    style_axes_box(ax)

    frames = np.arange(nuse)
    # Plot each peak column as its own series (so legend keys make sense)
    scatters = []
    for j in range(ncols):
        mask = np.isfinite(centers_trk[:, j]) & np.isfinite(height_trk[:, j])
        if not np.any(mask):
            continue
        sc = ax.scatter(frames[mask], centers_trk[mask, j],
                        c=height_trk[mask, j], cmap="plasma", s=14,
                        edgecolors="none")
        scatters.append((sc, j))

    ax.set_xlabel("Frame")
    ax.set_ylabel("Center (q or 2θ)")
    ax.set_title("Peak Centers over Frames (colored by fitted height)")

    # Colorbar
    if scatters:
        cbar = fig.colorbar(scatters[0][0], ax=ax, pad=0.02)
        cbar.set_label("Height (fit)")

    # Legend (one entry per plotted peak column)
    labels = [f"peak col {j}" for _, j in scatters]
    if scatters:
        # place legend outside on right in a box
        leg_elements = [s for s,_ in scatters]
        leg = ax.legend(leg_elements, labels, loc="center left", bbox_to_anchor=(1.02, 0.5),
                        borderaxespad=0.0, frameon=True, edgecolor="0.7", facecolor="white",
                        fontsize=8)
        fig.tight_layout(rect=[0.0, 0.0, 0.82, 1.0])
    else:
        fig.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()
