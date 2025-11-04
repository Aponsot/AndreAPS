import argparse
import os
import time
from typing import Tuple, List, Optional

import numpy as np
import h5py
import tifffile as tiff
import yaml
from scipy.ndimage import median_filter
import concurrent.futures
import gc

from hexrd import instrument
from hexrd.projections.polar import PolarView


ENERGY_KEV = 61.335
PLANCK_CONSTANT = 4.135667696e-18
SPEED_OF_LIGHT = 2.99792458e10
CM_TO_ANGSTROM = 1e8

FRAME_DURATION_MS = 4

TTH_MIN = 1.0
TTH_MAX = 14.0
ETA_MIN = -180.0
ETA_MAX = 180.0
NDIV = 3

SATURATION_VALUE = 2**32 - 1
BACKGROUND_VALUES = [0.0, 0.5]
BACKGROUND_TOLERANCE = 0.2
MEDIAN_FILTER_SIZE = 5
INTENSITY_THRESHOLD = 0.7
INFLATE_FACTOR = 20

DEFAULT_OUTPUT_DIR = "/home/beams/PONSOT/Data/h5"


def calculate_wavelength(energy_kev: float) -> float:
    wavelength_cm = (PLANCK_CONSTANT * SPEED_OF_LIGHT) / energy_kev
    return wavelength_cm * CM_TO_ANGSTROM


def load_tiff_files(tiff_folder: str) -> Tuple[List[str], str]:
    tiff_files = sorted([
        f for f in os.listdir(tiff_folder)
        if f.lower().endswith(('.tiff', '.tif'))
    ])

    if not tiff_files:
        raise ValueError(f"No TIFF files found in {tiff_folder}")

    experiment_name = os.path.commonprefix(tiff_files).rstrip('_-')
    if not experiment_name:
        experiment_name = "experiment"

    return tiff_files, experiment_name


def tth_to_q(tth_degrees: np.ndarray, wavelength_angstrom: float) -> np.ndarray:
    theta_radians = np.radians(tth_degrees / 2)
    return (4 * np.pi * np.sin(theta_radians)) / wavelength_angstrom


def process_frame_path(path: str,
                       pv: PolarView,
                       det_keys: List[str]) -> np.ndarray:
    # Read frame from disk
    img = tiff.imread(path)
    image = img.astype(np.float32, copy=False)

    # Boolean mask for bad pixels/background
    mask = (image == SATURATION_VALUE) | (image <= 0)
    for val in BACKGROUND_VALUES:
        mask |= (np.abs(image - val) <= BACKGROUND_TOLERANCE)

    # Background subtraction (creates a new array)
    background = median_filter(image, size=MEDIAN_FILTER_SIZE)

    # In-place subtract to avoid creating another large array
    np.subtract(image, background, out=image)
    background = None  # let GC reclaim
    # Apply mask
    image[mask] = 0.0

    # In-place thresholding/inflation to minimize temporaries
    # First zero everything <= threshold
    idx_low = image <= INTENSITY_THRESHOLD
    image[idx_low] = 0.0
    # Inflate the remaining positives
    np.multiply(image, INFLATE_FACTOR, out=image, where=image > 0)

    # Map to detector(s); adjust if you truly have multiple distinct detectors
    local_imsd = {det_key: image for det_key in det_keys}

    # Polar warp (original behavior with padding and interpolation)
    polar_image = pv.warp_image(local_imsd, pad_with_nans=True, do_interpolation=True)

    # Average over full azimuthal (eta) to get 1D intensity
    if np.ma.isMaskedArray(polar_image):
        integrated_intensity = np.array(np.ma.average(polar_image, axis=0), dtype=np.float32)
    else:
        integrated_intensity = np.nanmean(polar_image, axis=0).astype(np.float32)

    # Help GC
    del img, image, local_imsd, polar_image
    gc.collect()

    return integrated_intensity


def integrate_em(tiff_folder: str,
                 instr_file: str,
                 output_dir: str = DEFAULT_OUTPUT_DIR,
                 plot: bool = False,
                 max_workers: Optional[int] = None) -> str:
    start_time = time.time()

    print("=" * 70)
    print("POLAR INTEGRATION WORKFLOW")
    print("=" * 70)

    wavelength = calculate_wavelength(ENERGY_KEV)
    print(f"Energy: {ENERGY_KEV} keV | Wavelength: {wavelength:.6f} Å")

    with open(instr_file, 'r') as f:
        instr_cfg = yaml.safe_load(f)
    instr = instrument.HEDMInstrument(instr_cfg)
    det_keys = list(instr.detectors.keys())

    tiff_files, experiment_name = load_tiff_files(tiff_folder)
    nframes = len(tiff_files)
    print(f"Found {nframes} frames | Experiment: {experiment_name}")

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{experiment_name}.h5")

    # Pixel resolution (degrees)
    tth_stats, eta_stats = np.degrees(instrument.hedm_instrument.pixel_resolution(instr))

    pv = PolarView(
        np.r_[TTH_MIN, TTH_MAX],
        instr,
        ETA_MIN,
        ETA_MAX,
        pixel_size=(tth_stats[1] / NDIV, eta_stats[1] / NDIV),
        cache_coordinate_map=True
    )

    print(f"2θ: {TTH_MIN}-{TTH_MAX}° | η: {ETA_MIN}-{ETA_MAX}°")
    print(f"Polar image shape (eta, tth): {pv.shape}")

    # Axes
    tth = np.linspace(pv.tth_min, pv.tth_max, pv.shape[1]) * 180 / np.pi
    q_values = tth_to_q(tth, wavelength)
    time_axis = np.arange(nframes) * FRAME_DURATION_MS / 1000.0

    print(f"\nSaving to {output_file}...")
    with h5py.File(output_file, 'w') as h5_file:
        d_int = h5_file.create_dataset(
            "int",
            shape=(nframes, pv.shape[1]),
            dtype='f4',
            chunks=(min(256, nframes), pv.shape[1]),
            compression="gzip"
        )
        h5_file.create_dataset("q", data=q_values)
        h5_file.create_dataset("time", data=time_axis)

        h5_file.attrs["experiment_name"] = experiment_name
        h5_file.attrs["nframes"] = nframes
        h5_file.attrs["energy_kev"] = ENERGY_KEV
        h5_file.attrs["wavelength_angstrom"] = wavelength

        # Concurrency: modest default; override via CLI
        if max_workers is None:
            max_workers = min(16, os.cpu_count() or 1)
        print(f"Processing with {max_workers} workers...")

        paths = [os.path.join(tiff_folder, f) for f in tiff_files]
        processing_start = time.time()
        completed = 0

        # Limit the number of in-flight tasks tightly to avoid memory spikes
        max_pending = max_workers  # not 2x; keep ≤ workers alive
        future_to_idx = {}
        pending = set()

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for i, p in enumerate(paths):
                fut = executor.submit(process_frame_path, p, pv, det_keys)
                future_to_idx[fut] = i
                pending.add(fut)

                # Keep at most max_pending in-flight
                if len(pending) >= max_pending:
                    done, pending = concurrent.futures.wait(
                        pending, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    for f in done:
                        idx = future_to_idx.pop(f)
                        intensity = f.result()
                        d_int[idx, :] = intensity
                        completed += 1
                        if completed % 100 == 0 or completed == nframes:
                            elapsed = time.time() - processing_start
                            rate = completed / max(elapsed, 1e-6)
                            print(f"  {completed}/{nframes} frames ({rate:.1f} fps)")

            # Drain remaining futures
            for f in concurrent.futures.as_completed(pending):
                idx = future_to_idx.pop(f)
                intensity = f.result()
                d_int[idx, :] = intensity
                completed += 1
                if completed % 100 == 0 or completed == nframes:
                    elapsed = time.time() - processing_start
                    rate = completed / max(elapsed, 1e-6)
                    print(f"  {completed}/{nframes} frames ({rate:.1f} fps)")

    if plot:
        import matplotlib.pyplot as plt
        plt.rcParams.update({"figure.dpi": 140, "font.size": 14})
        with h5py.File(output_file, 'r') as h5_file:
            intensity_stack = h5_file["int"][...]
            q_values = h5_file["q"][...]
            time_axis = h5_file["time"][...]
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(intensity_stack, aspect='auto',
                       extent=[q_values.min(), q_values.max(), time_axis.max(), 0],
                       origin='upper', cmap='plasma')
        plt.colorbar(im, ax=ax, label='Intensity')
        ax.set_xlabel('q (Å⁻¹)')
        ax.set_ylabel('Time (s)')
        ax.set_title(f"{experiment_name}")
        plt.tight_layout()
        plt.show()

    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Polar integration of diffraction TIFF images")
    parser.add_argument("tiff_folder", help="TIFF folder path")
    parser.add_argument("--instr_file", required=True, help="Instrument YAML file")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--plot", action="store_true", help="Show plot")
    parser.add_argument("--max_workers", type=int, help="Parallel workers")

    args = parser.parse_args()

    integrate_em(args.tiff_folder, args.instr_file, args.output_dir, args.plot, args.max_workers)
