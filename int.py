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
CAKE_WIDTH = 10.0

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


def generate_cake_slices(eta_min: float, eta_max: float, cake_width: float) -> List[Tuple[float, float]]:
    n_cakes = int((eta_max - eta_min) / cake_width)
    return [(eta_min + i * cake_width, eta_min + (i + 1) * cake_width)
            for i in range(n_cakes)]


def tth_to_q(tth_degrees: np.ndarray, wavelength_angstrom: float) -> np.ndarray:
    theta_radians = np.radians(tth_degrees / 2)
    return (4 * np.pi * np.sin(theta_radians)) / wavelength_angstrom


def process_frame_path(path: str,
                       pv: PolarView,
                       det_keys: List[str],
                       cake_slices: Optional[List[Tuple[float, float]]] = None) -> Tuple[np.ndarray, Optional[List[np.ndarray]]]:
    # Read frame from disk
    img = tiff.imread(path)
    image = img.astype(np.float32, copy=False)

    # Boolean mask for bad pixels/background
    mask = (image == SATURATION_VALUE) | (image <= 0)
    for val in BACKGROUND_VALUES:
        mask |= (np.abs(image - val) <= BACKGROUND_TOLERANCE)

    # Background subtraction
    background = median_filter(image, size=MEDIAN_FILTER_SIZE)
    image_subtracted = image - background

    # Apply mask and threshold
    image_subtracted[mask] = 0.0
    image_processed = np.where(
        image_subtracted > INTENSITY_THRESHOLD,
        image_subtracted * INFLATE_FACTOR,
        0.0
    ).astype(np.float32, copy=False)

    # Map to detector(s); adjust if you truly have multiple, distinct detector images
    local_imsd = {det_key: image_processed for det_key in det_keys}

    # Polar warp (match original behavior with padding and interpolation)
    polar_image = pv.warp_image(local_imsd, pad_with_nans=True, do_interpolation=True).astype(np.float32, copy=False)

    # Average over eta to get 1D intensity
    integrated_intensity = np.array(np.ma.average(polar_image, axis=0)).astype(np.float32)

    cake_intensities = None
    if cake_slices is not None:
        cake_intensities = []
        eta_axis = np.linspace(pv.eta_min, pv.eta_max, polar_image.shape[0]) * 180 / np.pi

        for eta_start, eta_end in cake_slices:
            mask_eta = (eta_axis >= eta_start) & (eta_axis < eta_end)
            if np.any(mask_eta):
                cake_int = np.array(np.ma.average(polar_image[mask_eta, :], axis=0)).astype(np.float32)
            else:
                cake_int = np.zeros(polar_image.shape[1], dtype=np.float32)
            cake_intensities.append(cake_int)

    return integrated_intensity, cake_intensities


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

    cake_slices = generate_cake_slices(ETA_MIN, ETA_MAX, CAKE_WIDTH)
    print(f"2θ: {TTH_MIN}-{TTH_MAX}° | η: {ETA_MIN}-{ETA_MAX}° | Cakes: {len(cake_slices)}")
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

        d_cake = None
        if cake_slices:
            d_cake = h5_file.create_dataset(
                "cake_int",
                shape=(nframes, len(cake_slices), pv.shape[1]),
                dtype='f4',
                chunks=(min(64, nframes), 1, pv.shape[1]),
                compression="gzip"
            )
            h5_file.create_dataset("cake_eta_ranges", data=np.array(cake_slices, dtype='f4'))

        h5_file.attrs["experiment_name"] = experiment_name
        h5_file.attrs["nframes"] = nframes
        h5_file.attrs["energy_kev"] = ENERGY_KEV
        h5_file.attrs["wavelength_angstrom"] = wavelength

        # Concurrency: default to 16 workers (override via CLI)
        if max_workers is None:
            max_workers = min(16, os.cpu_count() or 1)
        print(f"Processing with {max_workers} workers...")

        processing_start = time.time()
        paths = [os.path.join(tiff_folder, f) for f in tiff_files]

        # Stream frames and write incrementally
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_frame_path, p, pv, det_keys, cake_slices): i for i, p in enumerate(paths)}
            for completed_idx, future in enumerate(concurrent.futures.as_completed(futures)):
                i = futures[future]
                intensity, cake_int = future.result()
                d_int[i, :] = intensity
                if d_cake is not None and cake_int is not None:
                    for k, arr in enumerate(cake_int):
                        d_cake[i, k, :] = arr
                if (completed_idx + 1) % 100 == 0 or (completed_idx + 1) == nframes:
                    elapsed = time.time() - processing_start
                    rate = (completed_idx + 1) / max(elapsed, 1e-6)
                    print(f"  {completed_idx + 1}/{nframes} frames ({rate:.1f} fps)")

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

