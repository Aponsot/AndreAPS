#!/usr/bin/env python3
import argparse
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt

# ------------------------------
# Plot styling
# ------------------------------
def apply_pub_style():
    plt.rcParams.update({
        "figure.figsize": (8, 6),
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "font.size": 14,
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
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.15)
    ax.minorticks_on()
    if light_grid:
        ax.grid(True, which="major", alpha=0.12, linestyle="-", linewidth=0.6)

# ------------------------------
# Helpers
# ------------------------------
def parse_indices(s: str):
    """Parse comma-separated peak indices, e.g. '0,1'."""
    return [int(v) for v in s.split(",") if v.strip()]

# ------------------------------
# Main logic
# ------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Fit smooth curves to tracked peak centers and compare their derivatives."
    )
    ap.add_argument(
        "--map_h5", required=True,
        help="HDF5 file produced by the tracking script (with 'centers' and 'frame_index'/'time')."
    )
    ap.add_argument(
        "--peaks", default="0,1",
        help="Comma-separated peak indices to analyze (default: '0,1')."
    )
    ap.add_argument(
        "--deg", type=int, default=3,
        help="Polynomial degree for q(t) fit (default: 3)."
    )
    ap.add_argument(
        "--tmin", type=float, default=None,
        help="Optional minimum time/frame for fit."
    )
    ap.add_argument(
        "--tmax", type=float, default=None,
        help="Optional maximum time/frame for fit."
    )
    ap.add_argument(
        "--use_frame", action="store_true",
        help="Use frame index as x instead of time (ignore sec_per_frame)."
    )
    ap.add_argument(
        "--save_deriv_h5", default=None,
        help="Optional path to save fitted q(t) and dq/dt curves to HDF5."
    )
    args = ap.parse_args()

    peak_indices = parse_indices(args.peaks)
    deg = args.deg

    # --------------------------
    # Load map H5 from tracker
    # --------------------------
    map_path = os.path.abspath(os.path.expanduser(args.map_h5))
    if not os.path.exists(map_path):
        raise FileNotFoundError(f"Map H5 not found: {map_path}")

    with h5py.File(map_path, "r") as hf:
        centers = np.asarray(hf["centers"][:], float)  # (nuse, npeaks)
        frames = np.asarray(hf["frame_index"][:], int)

        if args.use_frame:
            xvals = frames.astype(float)
            x_label = "Frame"
        else:
            if "time" in hf:
                xvals = np.asarray(hf["time"][:], float)
                x_label = "Time (s)"
            else:
                sec_per = float(hf.attrs.get("sec_per_frame", -1.0))
                if sec_per > 0:
                    xvals = frames * sec_per
                    x_label = "Time (s)"
                else:
                    xvals = frames.astype(float)
                    x_label = "Frame"

    nframes, npeaks_total = centers.shape
    for idx in peak_indices:
        if idx < 0 or idx >= npeaks_total:
            raise ValueError(
                f"Requested peak index {idx} out of range [0, {npeaks_total-1}]."
            )

    # Optional x-window
    mask_range = np.ones_like(xvals, dtype=bool)
    if args.tmin is not None:
        mask_range &= (xvals >= args.tmin)
    if args.tmax is not None:
        mask_range &= (xvals <= args.tmax)

    # --------------------------
    # Fit each requested peak
    # --------------------------
    fit_results = {}  # peak_idx -> dict with t_fit, q_fit, dqdt_fit, raw_x, raw_q

    for pidx in peak_indices:
        q_raw = centers[:, pidx]

        # valid points: in range + finite
        valid = mask_range & np.isfinite(q_raw)
        x_use = xvals[valid]
        q_use = q_raw[valid]

        if x_use.size < (deg + 1):
            print(f"Peak {pidx}: not enough points for degree {deg} fit, skipping.")
            continue

        # Polynomial fit q(t)
        coeffs = np.polyfit(x_use, q_use, deg)
        poly_q = np.poly1d(coeffs)
        poly_dq = poly_q.deriv()  # derivative polynomial

        # Create a smooth grid for plotting
        x_fit = np.linspace(x_use.min(), x_use.max(), 400)
        q_fit = poly_q(x_fit)
        dqdt_fit = poly_dq(x_fit)

        fit_results[pidx] = {
            "raw_x": x_use,
            "raw_q": q_use,
            "x_fit": x_fit,
            "q_fit": q_fit,
            "dqdt_fit": dqdt_fit,
            "coeffs": coeffs,
        }

        # Simple scalar summary of "average slope" over range
        avg_slope = (q_use[-1] - q_use[0]) / (x_use[-1] - x_use[0])
        print(f"Peak {pidx}: average dq/dx over window ≈ {avg_slope:.4e}")

    if not fit_results:
        print("No peaks were successfully fitted. Nothing to plot.")
        return

    # --------------------------
    # Optional save to HDF5
    # --------------------------
    if args.save_deriv_h5 is not None:
        outp = os.path.abspath(os.path.expanduser(args.save_deriv_h5))
        out_dir = os.path.dirname(outp)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        with h5py.File(outp, "w") as hf_out:
            hf_out.attrs["poly_degree"] = deg
            hf_out.attrs["x_label"] = x_label
            for pidx, data in fit_results.items():
                grp = hf_out.create_group(f"peak_{pidx}")
                grp.create_dataset("raw_x", data=data["raw_x"])
                grp.create_dataset("raw_q", data=data["raw_q"])
                grp.create_dataset("x_fit", data=data["x_fit"])
                grp.create_dataset("q_fit", data=data["q_fit"])
                grp.create_dataset("dqdt_fit", data=data["dqdt_fit"])
                grp.create_dataset("coeffs", data=data["coeffs"])
        print(f"Saved derivative analysis to {outp}")

    # --------------------------
    # Plot q(t) and dq/dt
    # --------------------------
    apply_pub_style()
    from matplotlib.gridspec import GridSpec
    fig = plt.figure()
    gs = GridSpec(2, 1, height_ratios=[3.0, 2.0], hspace=0.35)

    # Top: positions
    ax_pos = fig.add_subplot(gs[0])
    style_axes(ax_pos, light_grid=True)

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for i, pidx in enumerate(peak_indices):
        if pidx not in fit_results:
            continue
        col = colors[i % len(colors)]
        data = fit_results[pidx]
        ax_pos.scatter(
            data["raw_x"], data["raw_q"],
            s=15, alpha=0.6, label=f"Peak {pidx} data", edgecolors="none"
        )
        ax_pos.plot(
            data["x_fit"], data["q_fit"],
            lw=1.8, color=col, label=f"Peak {pidx} fit"
        )

    ax_pos.set_xlabel(x_label)
    ax_pos.set_ylabel("q (Å⁻¹)")
    ax_pos.legend(loc="best")

    # Bottom: derivatives
    ax_der = fig.add_subplot(gs[1])
    style_axes(ax_der, light_grid=True)

    for i, pidx in enumerate(peak_indices):
        if pidx not in fit_results:
            continue
        col = colors[i % len(colors)]
        data = fit_results[pidx]
        ax_der.plot(
            data["x_fit"], data["dqdt_fit"],
            lw=1.8, color=col, label=f"dq/d{x_label} (peak {pidx})"
        )

    ax_der.set_xlabel(x_label)
    ax_der.set_ylabel("dq/d" + ("t" if "Time" in x_label else "d(frame)"))
    ax_der.legend(loc="best")

    fig.suptitle("Peak position and derivative comparison", y=0.99)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
