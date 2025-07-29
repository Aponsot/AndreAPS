import argparse
import numpy as np
import os
import tifffile as tiff
import yaml
from hexrd import instrument
from hexrd.projections.polar import PolarView
import concurrent.futures
import h5py
import matplotlib.pyplot as plt


def integrate_em(Tiff_fold, output_file, instr, plot=False):
    """
    Perform polar integration of TIFF images and save results in HDF5 format.

    Parameters:
    - Tiff_fold: Path to the folder containing TIFF images.
    - output_file: Path to the output HDF5 file.
    - instr: HEXRD instrument configuration.
    - plot: Boolean flag to enable plotting of time-resolved data.
    """
    # Extract energy from the instrument configuration
    energy = instr.beam.energy  # Energy in keV
    h = 4.135667696e-18  # Planck's constant in keV·s
    c = 2.99792458e10  # Speed of light in cm/s
    wavelength = h * c / energy  # Wavelength in cm
    wavelength *= 1e8  # Convert to angstroms

    print(f"Beam energy: {energy} keV")
    print(f"Calculated wavelength: {wavelength:.6f} Å")

    # Load TIFF images
    tifs = sorted([f for f in os.listdir(Tiff_fold) if f.lower().endswith(('.tiff', '.tif'))])
    if not tifs:
        raise ValueError("No TIFF files found in the folder.")

    # Extract experiment name from TIFF files (common prefix)
    experiment_name = os.path.commonprefix(tifs).rstrip('_-')
    print(f"Experiment name: {experiment_name}")

    first_img = tiff.imread(os.path.join(Tiff_fold, tifs[0]))
    image_shape = first_img.shape
    image_stack = np.empty((len(tifs), *image_shape), dtype=first_img.dtype)
    image_stack[0] = first_img
    for i, fname in enumerate(tifs[1:], start=1):
        img = tiff.imread(os.path.join(Tiff_fold, fname))
        image_stack[i] = img
        print(f"Loaded {fname} with shape {img.shape}")
    images = image_stack
    nframes = images.shape[0]
    print(f"Number of frames: {nframes}")

    # Setup for polar remap
    tth_min = 1.0
    tth_max = 24.0
    eta_min = -180.0
    eta_max = 180.0
    ndiv = 1
    tth_stats, eta_stats = np.degrees(instrument.hedm_instrument.pixel_resolution(instr))
    det_keys = instr.detectors.keys()
    imsd = dict.fromkeys(det_keys)

    pv = PolarView(
        np.r_[tth_min, tth_max], instr,
        eta_min, eta_max,
        pixel_size=(tth_stats[1] / ndiv, eta_stats[1] / ndiv),
        cache_coordinate_map=True
    )

    def process_frame(image_1):
        image_1 = np.ma.masked_where(image_1 == (2**32 - 1), image_1)
        local_imsd = dict.fromkeys(det_keys)
        for det_key in det_keys:
            local_imsd[det_key] = image_1
        pimg = pv.warp_image(local_imsd, pad_with_nans=True, do_interpolation=True)
        Int = np.array(np.ma.average(pimg, axis=0))  # 1D array
        return Int

    all_int = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_frame, images[i]) for i in range(nframes)]
        for idx, future in enumerate(concurrent.futures.as_completed(futures)):
            all_int.append(future.result())
            if (idx + 1) % 100 == 0 or (idx + 1) == nframes:
                print(f"Processed {idx + 1} of {nframes} frames (parallel)")

    # Convert to 2D array: (Z = image/frame index, X = tth points)
    intensity_stack = np.array(all_int)

    # Compute tth and q values
    tth = np.linspace(pv.tth_min, pv.tth_max, pv.shape[1]) * 180 / np.pi
    q_values = (4 * np.pi * np.sin(np.radians(tth / 2))) / wavelength

    # Save data to HDF5 format
    with h5py.File(output_file, 'w') as h5_file:
        h5_file.create_dataset(f"{experiment_name}/intensities", data=intensity_stack)
        h5_file.create_dataset(f"{experiment_name}/tth", data=tth)
        h5_file.create_dataset(f"{experiment_name}/q_values", data=q_values)
    print(f"Polar integration completed. Results saved to {output_file}")

    # Plot time-resolved data if the plot flag is set
    if plot:
        plt.figure(figsize=(10, 6))
        plt.imshow(intensity_stack, aspect='auto', extent=[q_values.min(), q_values.max(), 0, nframes],
                   origin='lower', cmap='viridis')
        plt.colorbar(label='Intensity')
        plt.xlabel('q (1/Å)')
        plt.ylabel('Frame Index')
        plt.title(f"Time-Resolved Data: {experiment_name}")
        plt.show()


if __name__ == "__main__":
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Polar integration of diffraction experiments.")
    parser.add_argument("Tiff_fold", type=str, help="Path to the folder containing TIFF images.")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Path to the output HDF5 file. Defaults to current working directory.")
    parser.add_argument("--instr_file", type=str, required=True, help="Path to the instrument YAML file.")
    parser.add_argument("--plot", action="store_true", help="Enable plotting of time-resolved data.")

    args = parser.parse_args()

    # Set default output file if not provided
    if args.output_file is None:
        args.output_file = os.path.join(os.getcwd(), "polar_integration_results.h5")

    # Load instrument configuration
    with open(args.instr_file, 'r') as f:
        instr_cfg = yaml.safe_load(f)
    instr = instrument.HEDMInstrument(instr_cfg)

    # Run integration
    integrate_em(args.Tiff_fold, args.output_file, instr, plot=args.plot)