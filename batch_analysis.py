import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.ndimage import gaussian_filter1d, percentile_filter
from lmfit.models import PolynomialModel, GaussianModel
import warnings
import csv
from dataclasses import dataclass
from tqdm import tqdm  
import argparse


def _sigma_from_fwhm(fwhm_pts: float, dq: float) -> float:
    return (fwhm_pts * dq) / 2.355

def _poisson_weights(y):
    return 1.0 / np.sqrt(np.clip(y, 1.0, None))

def _robust_sigma(x):
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))

def detect_background_floor(y_win, pct=15, win_frac=0.05, smooth_sigma=2):
    n = len(y_win)
    win = max(7, int(win_frac * n) | 1)
    bg = percentile_filter(y_win, percentile=pct, size=win)
    if smooth_sigma and smooth_sigma > 0:
        bg = gaussian_filter1d(bg, sigma=smooth_sigma)
    return bg

def detect_background_adaptive(y_win, floor_pct=14, floor_win_frac=0.05,
                               floor_smooth=2, gauss_frac=0.03,
                               target_frac=0.55, max_iter=12):
    n = len(y_win)
    bg_floor = detect_background_floor(y_win, pct=floor_pct,
                                       win_frac=floor_win_frac,
                                       smooth_sigma=floor_smooth)
    sg = max(3, min(12, int(gauss_frac * n)))
    bg_gauss = gaussian_filter1d(y_win, sigma=sg)
    lo, hi = 0.0, 1.0
    for _ in range(max_iter):
        alpha = 0.5 * (lo + hi)
        bg = alpha * bg_gauss + (1 - alpha) * bg_floor
        frac_below = float(np.mean(y_win <= bg))
        if frac_below < target_frac:
            lo = alpha
        else:
            hi = alpha
    alpha = 0.5 * (lo + hi)
    bg = alpha * bg_gauss + (1 - alpha) * bg_floor
    return bg

def fit_metrics(result, x, y, weights=None):
    yhat = result.model.eval(params=result.params, x=x)
    resid = y - yhat
    n = len(y)
    k = sum(p.vary for p in result.params.values())
    if weights is not None:
        w = np.asarray(weights, float)
        sse = float(np.sum((w * resid) ** 2))
        ybar = float(np.sum((w ** 2) * y) / np.sum(w ** 2))
        sst = float(np.sum((w * (y - ybar)) ** 2))
    else:
        sse = float(np.sum(resid ** 2))
        ybar = float(np.mean(y))
        sst = float(np.sum((y - ybar) ** 2))
    r2 = 1.0 - sse / max(sst, 1e-18)
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / max(n - k - 1, 1)
    red_chisq = sse / max(n - k, 1)
    aic = result.aic
    bic = result.bic
    rmse = np.sqrt(sse / n)
    max_abs = float(np.max(np.abs(resid)))
    return dict(r2=r2, adj_r2=adj_r2, red_chisq=red_chisq, aic=aic, bic=bic,
                rmse=rmse, max_abs=max_abs)

def fit_multi_peaks(q, y, peaks_idx, props, bg_degree=1):
    if len(peaks_idx) == 0:
        return None
    dq = float(np.diff(q).mean())
    composite = PolynomialModel(degree=bg_degree, prefix="bg_")
    params = composite.make_params()
    params["bg_c0"].set(value=float(np.median(y)), min=0)
    if bg_degree >= 1:
        params["bg_c1"].set(value=0)
    params.add("qshift", value=0.0, min=-5e-3, max=5e-3)
    params.add("qscale", value=1.0, min=0.999, max=1.001)

    fwhm_pts = np.asarray(props.get("widths", np.full_like(peaks_idx, 3.0, dtype=float)), float)
    prominences = np.asarray(props.get("prominences", np.ones_like(peaks_idx, dtype=float)), float)

    for i, pidx in enumerate(peaks_idx):
        center0 = float(q[pidx])
        sigma0 = max(_sigma_from_fwhm(float(fwhm_pts[i]), dq), 1e-6)
        g = GaussianModel(prefix=f"g{i}_")
        composite += g
        height0 = float(prominences[i])
        amp0 = max(height0 * sigma0 * np.sqrt(2 * np.pi), 1e-9)
        params.update(g.make_params(center=center0, sigma=sigma0, amplitude=amp0))
        params.add(f"g{i}_c0", value=center0, vary=False)
        dwig = max(3 * dq, 0.0018)
        params.add(f"g{i}_dcenter", value=0.0, min=-dwig, max=dwig)
        params[f"g{i}_center"].set(expr=f"qscale*(g{i}_c0) + qshift + g{i}_dcenter")
        params[f"g{i}_sigma"].set(min=0.25 * sigma0, max=4.0 * sigma0)
        params[f"g{i}_amplitude"].set(min=0.2 * abs(amp0), max=10 * abs(amp0))

    w = _poisson_weights(y)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in sqrt")
        result = composite.fit(
            y, params, x=q, weights=w,
            method="least_squares", fit_kws={"loss": "soft_l1", "f_scale": 1.0}
        )
    return result

@dataclass
class FitSummary:
    frame: int
    ok: bool
    r2: float = np.nan
    adj_r2: float = np.nan
    red_chisq: float = np.nan
    aic: float = np.nan
    bic: float = np.nan
    rmse: float = np.nan
    max_abs: float = np.nan
    n_peaks: int = 0
    peaks: list = None
    err: str = ""
    params: object = None

def fit_one_for_sweep(q_full, int_row, peak_pos, window=0.1):
    q_min, q_max = peak_pos - window, peak_pos + window
    mask = (q_full >= q_min) & (q_full <= q_max)
    q = q_full[mask]
    y = int_row[mask]
    if q.size < 5:
        return None, "Fit window too small."
    bg = detect_background_adaptive(y)
    y_det = y - bg
    sig = _robust_sigma(y_det)
    prom_full = max(1.0, 3.6 * sig)
    w_guess_pts = 3.0
    wmin = max(1, int(0.6 * w_guess_pts))
    wmax = int(20 * w_guess_pts)
    min_dist_pts = int(max(1, round(0.6 * w_guess_pts)))
    peaks, props = signal.find_peaks(
        y_det, prominence=prom_full, width=(wmin, wmax),
        distance=min_dist_pts, rel_height=0.5
    )
    if len(peaks) == 0:
        return None, "No peaks detected."
    result = fit_multi_peaks(q, y, peaks, props, bg_degree=1)
    if result is None:
        return None, "Fit failed."
    m = fit_metrics(result, q, y, weights=_poisson_weights(y))
    peak_centers = [result.params[k].value for k in result.params if k.endswith("_center")]
    return FitSummary(
    frame=-1, ok=True, r2=float(m["r2"]), adj_r2=float(m["adj_r2"]),
    red_chisq=float(m["red_chisq"]), aic=float(m["aic"]), bic=float(m["bic"]),
    rmse=float(m["rmse"]), max_abs=float(m["max_abs"]),
    n_peaks=len(peak_centers), peaks=peak_centers,
    params=result.params      # <-- add this
), ""

def sweep_frames(h5_path, peak_pos, window=0.1, frames=None, csv_path=None, print_every=50):
    with h5py.File(h5_path, "r") as f:
        Int = f["int"][:]
        q = f["q"][:]
        nframes = Int.shape[0]
    if frames is None:
        frames = range(nframes)

    summaries = []
    for idx, fr in enumerate(tqdm(frames, desc="Fitting frames"), 1):
        int_row = Int[fr, :]
        s, err = fit_one_for_sweep(q, int_row, peak_pos, window=window)
        if s is None:
            summaries.append(FitSummary(frame=fr, ok=False, err=err))
        else:
            s.frame = fr
            summaries.append(s)
    

    # Plot R² vs frame
    xs = [s.frame for s in summaries]
    ys = [s.r2 for s in summaries]
    plt.figure(figsize=(10, 4))
    plt.plot(xs, ys, marker=".", linestyle="-")
    plt.xlabel("Frame")
    plt.ylabel("R²")
    plt.title(f"R² vs Frame @ q={peak_pos} ± {window}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Plot all fitted peak positions vs frame, colored by R²
    plt.figure(figsize=(12, 6))
    for s in summaries:
        if s.ok and s.peaks:
            amps = []
            for i in range(s.n_peaks):
                p = s.params.get(f"g{i}_amplitude")
                amps.append(p.value if p is not None else np.nan)

            plt.scatter(
                [s.frame] * len(s.peaks),
                s.peaks,
                c=amps,
                cmap="plasma",
                s=5
            )

    cbar = plt.colorbar(label="Peak Amplitude (area under Gaussian)")
    plt.xlabel("Frame Number")
    plt.ylabel("Peak Position (q)")
    plt.title(f"Peak Positions vs Frame Number @ q={peak_pos} ± {window} (colored by amplitude)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    bad = [s.frame for s in summaries if not s.ok]
    if bad:
        print(f"failed frames: {bad[:20]}{'...' if len(bad)>20 else ''}")
    else:
        print("all frames fit successfully")
    return summaries
# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Polar integration of diffraction experiments.")
    parser.add_argument("h5", type=str, help="Path to the folder containing TIFF images.")
    parser.add_argument("peak_pos", type=float, help="Location of Peak of study")
    args = parser.parse_args()
    sweep_frames(args.h5, args.peak_pos, window=0.1, frames=None, csv_path="peakfit_with_peaks.csv")

