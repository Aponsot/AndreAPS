import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from lmfit.models import GaussianModel, LinearModel
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Window (in q units) used around the mean of specified peak positions
WINDOW = 0.15

# Peak width constraints
MIN_SIGMA = 1e-6
MAX_SIGMA_FRAC = 0.22

# Physical bounds on peak center shifts relative to specified positions
# Adjust as needed for your system/instrument:
# Allow larger negative shift to accommodate melting/solidification (~0.07-0.1q)
SHIFT_NEG_MAX = 0.12
# Allow small positive shift (rare but possible)
SHIFT_POS_MAX = 0.03


def robust_sigma(y):
    med = np.median(y)
    return 1.4826 * np.median(np.abs(y - med)) + 1e-12


def compute_r2(y, yfit):
    ss_res = np.sum((y - yfit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2) + 1e-16
    return 1.0 - ss_res / ss_tot


def build_model(xw, yw, centers, baseline):
    """
    Build a model with a linear background and Gaussian peaks at given centers,
    with physically plausible bounds on center, sigma, and amplitude.
    """
    dx = np.mean(np.diff(xw)) if len(xw) > 1 else WINDOW
    min_sigma = max(dx / 3.0, MIN_SIGMA)
    max_sigma = MAX_SIGMA_FRAC * WINDOW

    bkg = LinearModel(prefix="bkg_")
    model = bkg
    params = bkg.make_params(slope=0.0, intercept=baseline)

    for i, cx in enumerate(centers):
        gi = GaussianModel(prefix=f"g{i}_")
        model += gi

        # Initial estimates
        p = np.argmin(np.abs(xw - cx))
        ypk = yw[p]
        height0 = max(ypk - baseline, robust_sigma(yw))
        sigma0 = np.clip(dx, min_sigma, max_sigma)  # use local sampling for initial sigma
        amp0 = height0 * sigma0 * np.sqrt(2 * np.pi)

        params.update(gi.make_params(center=cx, sigma=sigma0, amplitude=amp0))

        # Physically plausible bounds for center:
        cmin = max(xw[0], cx - SHIFT_NEG_MAX)
        cmax = min(xw[-1], cx + SHIFT_POS_MAX)
        params[f"g{i}_center"].set(min=cmin, max=cmax, value=cx)

        # Width constraints
        params[f"g{i}_sigma"].set(min=min_sigma, max=max_sigma, value=sigma0)

        # Non-negative amplitude
        params[f"g{i}_amplitude"].set(min=0.0, value=amp0)

    return model, params


def extract_peaks(result):
    peaks = []
    i = 0
    while f"g{i}_center" in result.params:
        ctr = result.params[f"g{i}_center"].value
        sig = result.params[f"g{i}_sigma"].value
        amp = result.params[f"g{i}_amplitude"].value
        hgt = amp / (sig * np.sqrt(2 * np.pi)) if sig > 0 else 0.0
        fwhm = 2.354820045 * sig
        peaks.append({
            "index": i, "center": ctr, "height": hgt,
            "fwhm": fwhm, "amplitude": amp
        })
        i += 1
    return peaks


def fit_peaks(h5_path, frame, peak_positions, plot=True):
    """
    Always fit the specified number of peaks at provided q positions, with
    physically constrained center bounds and non-negative amplitudes.
    """
    with h5py.File(h5_path, "r") as f:
        x = f["q"][:] if "q" in f else f["tth"][:]
        yfull = f["int"][frame, :]

    x = np.asarray(x, float)
    yfull = np.asarray(yfull, float)

    center = np.mean(peak_positions)
    half = WINDOW / 2.0
    m = (x >= center - half) & (x <= center + half)
    xw, yw = x[m], yfull[m]
    mfin = np.isfinite(xw) & np.isfinite(yw)
    xw, yw = xw[mfin], yw[mfin]

    if xw.size < 5:
        raise ValueError("Too few points in window.")

    baseline = np.median(yw)
    noise = robust_sigma(yw)

    # Always fit all specified peaks with physical bounds
    model, params = build_model(xw, yw, peak_positions, baseline)
    result = model.fit(yw, params, x=xw, calc_covar=False,
                       method="least_squares", max_nfev=600)

    peaks = extract_peaks(result)

    bkg_slope = result.params["bkg_slope"].value
    bkg_intercept = result.params["bkg_intercept"].value
    r2 = compute_r2(yw, result.best_fit)

    rows = [[p["index"], p["center"], p["height"], p["fwhm"], p["amplitude"]]
            for p in peaks]

    if plot:
        plt.rcParams.update({
            "figure.dpi": 160, "savefig.dpi": 300,
            "font.size": 16, "axes.labelsize": 18, "axes.titlesize": 20,
            "xtick.labelsize": 14, "ytick.labelsize": 14,
        })
        fig, (ax, ax_tbl) = plt.subplots(2, 1, figsize=(10, 6.8),
                                         gridspec_kw={"height_ratios": [3, 1]})

        ax.plot(xw, yw, lw=1.8, label="Data")
        ax.plot(xw, result.best_fit, lw=2.2, label="Fit")

        comps = result.eval_components(x=xw)
        if "bkg_" in comps:
            ax.plot(xw, comps["bkg_"], ls="--", label="Background")

        for i in range(len(peaks)):
            key = f"g{i}_"
            if key in comps:
                ax.plot(xw, comps[key], ls=":", alpha=0.7, label=f"Peak {i+1}")
            ax.axvline(result.params[f"g{i}_center"].value, alpha=0.25, ls="--")

        ax.set_xlabel("q (1/Å)")
        ax.set_ylabel("Intensity")
        ax.set_title(f"Frame {frame} | {len(peaks)} peaks | R²={r2:.4f}")
        ax.legend(loc="best")
        ax.grid(alpha=0.3)
        ax.set_xlim(center - half, center + half)

        ax_tbl.axis("off")
        cols = ["Peak #", "Center", "Height", "FWHM", "Amplitude"]
        table = ax_tbl.table(
            cellText=[[f"{r[0]}", f"{r[1]:.6g}", f"{r[2]:.6g}",
                      f"{r[3]:.6g}", f"{r[4]:.6g}"] for r in rows],
            colLabels=cols, loc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.25)

        plt.tight_layout()
        plt.show()

    return {
        "frame": frame,
        "window": (center - half, center + half),
        "background": {"slope": bkg_slope, "intercept": bkg_intercept},
        "r2": r2,
        "rows": rows,
        "result": result,
        "x": xw, "y": yw, "yfit": result.best_fit,
        "noise": noise,
        "n_peaks": len(peaks)
    }


def _fit_one(args):
    h5_path, frame, peak_positions = args
    try:
        out = fit_peaks(h5_path, frame, peak_positions, plot=False)
        # Return center and height for each peak
        peaks = [(r[1], r[2]) for r in out.get("rows", [])]
        return frame, peaks
    except Exception:
        return frame, []


def peak_map_parallel(h5_path, peak_positions, marker_size=10):
    with h5py.File(h5_path, "r") as f:
        nframes = f["int"].shape[0]

    xs, ys, cs = [], [], []

    with ProcessPoolExecutor() as ex:
        futures = [ex.submit(_fit_one, (h5_path, fr, peak_positions))
                   for fr in range(nframes)]
        for fut in tqdm(as_completed(futures), total=nframes,
                        desc="Building peak map", unit="frame"):
            frame, peaks = fut.result()
            for q, height in peaks:
                xs.append(q)
                ys.append(frame)
                cs.append(height)

    if not xs:
        print("No peaks found.")
        return

    plt.rcParams.update({
        "figure.dpi": 160, "savefig.dpi": 300,
        "font.size": 20, "axes.labelsize": 20, "axes.titlesize": 20,
        "xtick.labelsize": 16, "ytick.labelsize": 16,
    })

    plt.figure(figsize=(9, 5))
    sc = plt.scatter(ys, xs, c=cs, s=marker_size, cmap="plasma")
    # You can adjust vmin/vmax to your instrument and data range
    norm = Normalize(vmin=0.0, vmax=70, clip=True)
    cbar = plt.colorbar(sc, norm=norm)
    cbar.set_label("Peak height (a.u.)")
    plt.ylabel("q (1/Å)")
    plt.xlabel("Frame")
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Fit Gaussian peaks with linear background at specified q-positions"
    )
    parser.add_argument("h5", help="HDF5 file with 'q' and 'int' datasets")
    parser.add_argument("peaks", type=float, nargs='+',
                        help="Peak q-positions (e.g., 3.025 3.012)")
    parser.add_argument("--frame", type=int, help="Fit single frame and show plot")
    parser.add_argument("--map", action="store_true",
                        help="Generate peak map for all frames")

    args = parser.parse_args()

    peak_positions = sorted(args.peaks)
    print(f"Fitting {len(peak_positions)} peak(s) at q = {peak_positions}")
    print(f"Center bounds per peak: [-{SHIFT_NEG_MAX}, +{SHIFT_POS_MAX}] around each specified q")

    if args.frame is not None:
        fit_peaks(args.h5, args.frame, peak_positions, plot=True)
    elif args.map:
        peak_map_parallel(args.h5, peak_positions)
    else:
        print("Specify --frame N to fit a single frame, or --map for all frames")


if __name__ == "__main__":
    main()
