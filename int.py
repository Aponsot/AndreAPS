import argparse
import os
import time
from typing import Tuple, List, Optional

import numpy as np
import h5py
import yaml
from scipy.ndimage import median_filter
import concurrent.futures
import threading
import gc

from hexrd import instrument
from hexrd.projections.polar import PolarView

# Constants
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

# Preload chunk factor: frames per worker per chunk when preload="chunk"
DEFAULT_PRELOAD_FACTOR = 2

# Where the raw 2D image(s) lives in each per-frame h5
H5_IMAGE_DATASET = "exchange/data"


def calculate_wavelength(energy_kev: float) -> float:
    wavelength_cm = (PLANCK_CONSTANT * SPEED_OF_LIGHT) / energy_kev
    return wavelength_cm * CM_TO_ANGSTROM


def load_h5_frame_files(h5_folder: str) -> Tuple[List[str], str]:
    h5_files = sorted([
        f for f in os.listdir(h5_folder)
        if f.lower().endswith(('.h5', '.hdf5'))
    ])
    if not h5_files:
        raise ValueError(f"No H5/HDF5 files found in {h5_folder}")

    experiment_name = os.path.commonprefix(h5_files).rstrip('_-.') or "experiment"
    return h5_files, experiment_name


def tth_to_q(tth_degrees: np.ndarray, wavelength_angstrom: float) -> np.ndarray:
    theta_radians = np.radians(tth_degrees / 2)
    return (4 * np.pi * np.sin(theta_radians)) / wavelength_angstrom


def read_frame_image_from_h5(path: str, dataset_path: str = H5_IMAGE_DATASET) -> np.ndarray:
    with h5py.File(path, "r") as f:
        if dataset_path not in f:
            raise KeyError(f"Dataset '{dataset_path}' not found in file: {path}")
        img = f[dataset_path][...]
    return img


def process_and_write_from_image(idx: int,
                                 image: np.ndarray,
                                 pv: PolarView,
                                 det_keys: List[str],
                                 d_int,
                                 writer_lock: threading.Lock) -> None:
    # Ensure float32
    image = image.astype(np.float32, copy=False)

    # Boolean mask
    mask = (image == SATURATION_VALUE) | (image <= 0)
    for val in BACKGROUND_VALUES:
        mask |= (np.abs(image - val) <= BACKGROUND_TOLERANCE)

    # Background subtraction
    background = median_filter(image, size=MEDIAN_FILTER_SIZE)
    np.subtract(image, background, out=image)
    background = None
    image[mask] = 0.0

    # Threshold + inflation
    idx_low = image <= INTENSITY_THRESHOLD
    image[idx_low] = 0.0
    np.multiply(image, INFLATE_FACTOR, out=image, where=image > 0)

    # Polar warp (same image for all detectors)
    local_imsd = {det_key: image for det_key in det_keys}
    polar_image = pv.warp_image(local_imsd, pad_with_nans=True, do_interpolation=True)

    # Integrate over full azimuthal (eta)
    if np.ma.isMaskedArray(polar_image):
        integrated_intensity = np.array(np.ma.average(polar_image, axis=0), dtype=np.float32)
    else:
        integrated_intensity = np.nanmean(polar_image, axis=0).astype(np.float32)

    # Thread-safe write
    with writer_lock:
        d_int[idx, :] = integrated_intensity

    # Cleanup
    del local_imsd, polar_image, image, integrated_intensity
    gc.collect()


def process_and_write_from_multidet_images(idx: int,
                                          images_3d: np.ndarray,
                                          pv: PolarView,
                                          det_keys: List[str],
                                          d_int,
                                          writer_lock: threading.Lock) -> None:
    """
    images_3d: shape (n_det, ny, nx) where n_det == len(det_keys)
    Process each detector panel independently, then warp+integrate.
    """
    if images_3d.ndim != 3:
        raise ValueError(f"Expected 3D array (n_det, ny, nx), got shape {images_3d.shape}")

    if images_3d.shape[0] != len(det_keys):
        raise ValueError(
            f"First dimension of image stack ({images_3d.shape[0]}) "
            f"does not match number of detectors ({len(det_keys)}). "
            f"Image shape={images_3d.shape}, det_keys={det_keys}"
        )

    local_imsd = {}

    for det_i, det_key in enumerate(det_keys):
        image = images_3d[det_i].astype(np.float32, copy=False)

        mask = (image == SATURATION_VALUE) | (image <= 0)
        for val in BACKGROUND_VALUES:
            mask |= (np.abs(image - val) <= BACKGROUND_TOLERANCE)

        background = median_filter(image, size=MEDIAN_FILTER_SIZE)
        np.subtract(image, background, out=image)
        background = None
        image[mask] = 0.0

        idx_low = image <= INTENSITY_THRESHOLD
        image[idx_low] = 0.0
        np.multiply(image, INFLATE_FACTOR, out=image, where=image > 0)

        local_imsd[det_key] = image

    polar_image = pv.warp_image(local_imsd, pad_with_nans=True, do_interpolation=True)

    if np.ma.isMaskedArray(polar_image):
        integrated_intensity = np.array(np.ma.average(polar_image, axis=0), dtype=np.float32)
    else:
        integrated_intensity = np.nanmean(polar_image, axis=0).astype(np.float32)

    with writer_lock:
        d_int[idx, :] = integrated_intensity

    del local_imsd, polar_image, integrated_intensity
    gc.collect()


def process_and_write_from_path(idx: int,
                                path: str,
                                pv: PolarView,
                                det_keys: List[str],
                                d_int,
                                writer_lock: threading.Lock,
                                dataset_path: str = H5_IMAGE_DATASET) -> None:
    img = read_frame_image_from_h5(path, dataset_path=dataset_path)

    # Common cases:
    # - 2D (ny, nx): single assembled image
    # - 3D (n_det, ny, nx): per-detector panels (e.g., 11 panels)
    # - 3D (1, ny, nx): single image stored with leading singleton dim
    if img.ndim == 3 and img.shape[0] == 1:
        img = img[0]

    if img.ndim == 2:
        process_and_write_from_image(idx, img, pv, det_keys, d_int, writer_lock)
    elif img.ndim == 3:
        process_and_write_from_multidet_images(idx, img, pv, det_keys, d_int, writer_lock)
    else:
        raise ValueError(f"Unsupported image ndim={img.ndim} for file {path} with shape {img.shape}")


def integrate_em(h5_folder: str,
                 instr_file: str,
                 output_dir: str = DEFAULT_OUTPUT_DIR,
                 plot: bool = False,
                 max_workers: Optional[int] = None,
                 preload: str = "none",
                 preload_factor: int = DEFAULT_PRELOAD_FACTOR,
                 dataset_path: str = H5_IMAGE_DATASET) -> str:

    print("=" * 70)
    print("POLAR INTEGRATION WORKFLOW")
    print("=" * 70)

    wavelength = calculate_wavelength(ENERGY_KEV)
    print(f"Energy: {ENERGY_KEV} keV | Wavelength: {wavelength:.6f} Å")

    with open(instr_file, 'r') as f:
        instr_cfg = yaml.safe_load(f)
    instr = instrument.HEDMInstrument(instr_cfg)
    det_keys = list(instr.detectors.keys())

    h5_files, experiment_name = load_h5_frame_files(h5_folder)
    nframes = len(h5_files)
    print(f"Found {nframes} frames | Experiment: {experiment_name}")
    print(f"Reading per-frame images from dataset: '{dataset_path}'")

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
        h5_file.attrs["input_frame_dataset"] = dataset_path

        # Concurrency
        if max_workers is None:
            max_workers = min(16, os.cpu_count() or 1)
        print(f"Processing with {max_workers} workers...")

        writer_lock = threading.Lock()
        processing_start = time.time()
        completed = 0

        # Normalize preload mode
        preload = preload.lower()
        if preload not in ("none", "chunk", "all"):
            print(f"Unknown preload mode '{preload}', defaulting to 'none'")
            preload = "none"

        paths = [os.path.join(h5_folder, f) for f in h5_files]

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            if preload == "all":
                print("Preloading all frames into memory...")
                images = []
                for p in paths:
                    img = read_frame_image_from_h5(p, dataset_path=dataset_path)
                    if img.ndim == 3 and img.shape[0] == 1:
                        img = img[0]
                    images.append(img)

                print("Submitting all tasks...")
                futures = []
                for i in range(nframes):
                    if images[i].ndim == 2:
                        futures.append(executor.submit(
                            process_and_write_from_image, i, images[i], pv, det_keys, d_int, writer_lock
                        ))
                    elif images[i].ndim == 3:
                        futures.append(executor.submit(
                            process_and_write_from_multidet_images, i, images[i], pv, det_keys, d_int, writer_lock
                        ))
                    else:
                        raise ValueError(f"Unsupported preloaded image ndim={images[i].ndim} shape={images[i].shape}")

                for f in concurrent.futures.as_completed(futures):
                    f.result()
                    completed += 1
                    if completed % 100 == 0 or completed == nframes:
                        elapsed = time.time() - processing_start
                        rate = completed / max(elapsed, 1e-6)
                        print(f"  {completed}/{nframes} frames ({rate:.1f} fps)")

                del images
                gc.collect()

            elif preload == "chunk":
                chunk_size = max(1, preload_factor * max_workers)
                print(f"Chunked preloading with chunk_size={chunk_size} (factor={preload_factor})")
                for start in range(0, nframes, chunk_size):
                    end = min(start + chunk_size, nframes)
                    preload_buf = []
                    for i in range(start, end):
                        img = read_frame_image_from_h5(paths[i], dataset_path=dataset_path)
                        if img.ndim == 3 and img.shape[0] == 1:
                            img = img[0]
                        preload_buf.append((i, img))

                    futures = []
                    for (idx, arr) in preload_buf:
                        if arr.ndim == 2:
                            futures.append(executor.submit(
                                process_and_write_from_image, idx, arr, pv, det_keys, d_int, writer_lock
                            ))
                        elif arr.ndim == 3:
                            futures.append(executor.submit(
                                process_and_write_from_multidet_images, idx, arr, pv, det_keys, d_int, writer_lock
                            ))
                        else:
                            raise ValueError(f"Unsupported preloaded image ndim={arr.ndim} shape={arr.shape}")

                    for f in concurrent.futures.as_completed(futures):
                        f.result()
                        completed += 1
                        if completed % 100 == 0 or completed == nframes:
                            elapsed = time.time() - processing_start
                            rate = completed / max(elapsed, 1e-6)
                            print(f"  {completed}/{nframes} frames ({rate:.1f} fps)")

                    del preload_buf
                    gc.collect()

            else:
                next_i = 0
                in_flight = {}
                initial = min(max_workers, nframes)

                for _ in range(initial):
                    fut = executor.submit(
                        process_and_write_from_path,
                        next_i, paths[next_i], pv, det_keys, d_int, writer_lock, dataset_path
                    )
                    in_flight[fut] = next_i
                    next_i += 1

                while in_flight:
                    done, _ = concurrent.futures.wait(
                        in_flight.keys(),
                        return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    for f in done:
                        in_flight.pop(f)
                        f.result()
                        completed += 1
                        if completed % 100 == 0 or completed == nframes:
                            elapsed = time.time() - processing_start
                            rate = completed / max(elapsed, 1e-6)
                            print(f"  {completed}/{nframes} frames ({rate:.1f} fps)")

                        if next_i < nframes:
                            nfut = executor.submit(
                                process_and_write_from_path,
                                next_i, paths[next_i], pv, det_keys, d_int, writer_lock, dataset_path
                            )
                            in_flight[nfut] = next_i
                            next_i += 1

    if plot:
        import matplotlib.pyplot as plt
        plt.rcParams.update({"figure.dpi": 140, "font.size": 14})
        with h5py.File(output_file, 'r') as h5_file:
            intensity_stack = h5_file["int"][...]
            q_values = h5_file["q"][...]
            time_axis = h5_file["time"][...]
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(
            intensity_stack,
            aspect='auto',
            extent=[q_values.min(), q_values.max(), time_axis.min(), time_axis.max()],
            origin='lower',
            cmap='plasma'
        )
        plt.colorbar(im, ax=ax, label='Intensity')
        ax.set_xlabel('q (Å⁻¹)')
        ax.set_ylabel('Time (s)')
        ax.set_title(f"{experiment_name}")
        plt.tight_layout()
        plt.show()

    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Polar integration of diffraction frames stored as per-frame HDF5 files"
    )
    parser.add_argument("h5_folder", help="Folder containing per-frame .h5/.hdf5 files")
    parser.add_argument("--instr_file", required=True, help="Instrument YAML file")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--plot", action="store_true", help="Show plot")
    parser.add_argument("--max_workers", type=int, help="Parallel workers")
    parser.add_argument("--preload", choices=["none", "chunk", "all"], default="none",
                        help="Preloading mode: none (workers read), chunk (small batches), all (load all frames)")
    parser.add_argument("--preload_factor", type=int, default=DEFAULT_PRELOAD_FACTOR,
                        help="Frames per worker per chunk when preload='chunk'")
    parser.add_argument("--dataset_path", default=H5_IMAGE_DATASET,
                        help="Dataset path inside each per-frame H5 (default: exchange/data)")
    args = parser.parse_args()

    integrate_em(
        args.h5_folder,
        args.instr_file,
        args.output_dir,
        args.plot,
        args.max_workers,
        args.preload,
        args.preload_factor,
        args.dataset_path
    )
