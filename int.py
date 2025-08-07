import argparse
import numpy as np
import os
import tifffile as tiff
import yaml
from hexrd import instrument
from hexrd.projections.polar import PolarView
import concurrent.futures
import h5py
import argparse
import matplotlib.pyplot as plt


def integrate_em(Tiff_fold, instr_file, plot=False):
    """
    Perform polar integration of TIFF images and save results in HDF5 format.

    Parameters:
    - Tiff_fold: Path to the folder containing TIFF images.
    - instr_file: Path to the instrument YAML file.
    - plot: Boolean flag to enable plotting of time-resolved data.
    """
    # Extract energy from the instrument configuration
    energy = 61.335  # Energy in keV
    h = 4.135667696e-18  # Planck's constant in keV·s
    c = 2.99792458e10  # Speed of light in cm/s
    wavelength = h * c / energy  # Wavelength in cm
    wavelength *= 1e8  # Convert to angstroms
    with open(instr_file, 'r') as f:
        instr_cfg = yaml.safe_load(f)
    instr = instrument.HEDMInstrument(instr_cfg)
    # Load TIFF images
    tifs = sorted([f for f in os.listdir(Tiff_fold) if f.lower().endswith(('.tiff', '.tif'))])
    if not tifs:
        raise ValueError("No TIFF files found in the folder.")

    # Extract experiment name from TIFF files (common prefix)
    experiment_name = os.path.commonprefix(tifs).rstrip('_-')
    output_file = os.path.join("/home/beams/PONSOT/Data/h5", f"{experiment_name}.h5") 

    first_img = tiff.imread(os.path.join(Tiff_fold, tifs[0]))
    image_shape = first_img.shape
    image_stack = np.empty((len(tifs), *image_shape), dtype=first_img.dtype)
    image_stack[0] = first_img
    for i, fname in enumerate(tifs[1:], start=1):
        img = tiff.imread(os.path.join(Tiff_fold, fname))
        image_stack[i] = img
    images = image_stack
    nframes = images.shape[0]
    print(f"Number of frames: {nframes}")

    # Setup for polar remap
    tth_min = 1.0
    tth_max = 14.0
    eta_min = -180.0
    eta_max = 180.0
    ndiv = 3
    cake_width = 10.0  # degrees, set your cake slice width here
    n_cakes = int((eta_max - eta_min) / cake_width)
    tth_stats, eta_stats = np.degrees(instrument.hedm_instrument.pixel_resolution(instr))
    det_keys = instr.detectors.keys()
    imsd = dict.fromkeys(det_keys)

    pv = PolarView(
        np.r_[tth_min, tth_max], instr,
        eta_min, eta_max,
        pixel_size=(tth_stats[1] / ndiv, eta_stats[1] / ndiv),
        cache_coordinate_map=True
    )

    from sklearn.cluster import DBSCAN

    def detect_spots(image, eps=5, min_samples=3):
        coords = np.column_stack(np.where(image > 0))
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
        spot_mask = np.zeros_like(image, dtype=bool)
        for cluster_label in np.unique(clustering.labels_):
            if cluster_label != -1:
                cluster_coords = coords[clustering.labels_ == cluster_label]
                spot_mask[cluster_coords[:, 0], cluster_coords[:, 1]] = True
        return spot_mask

    from scipy.ndimage import median_filter
    def mask_background(image, fluctuation_values=[0.0, .5], tolerance=0.2):
        background_mask = np.zeros_like(image, dtype=bool)
        for value in fluctuation_values:
            background_mask |= (np.abs(image - value) <= tolerance)
        masked_image = np.ma.masked_where(background_mask, image)
        return masked_image

    def process_frame(image_1, fluctuation_values=[0.0, 0.5], tolerance=0.2, cake_slices=None):
        image_1 = np.ma.masked_where((image_1 == (2**32 - 1)) | (image_1 <= 0), image_1)
        image_1 = mask_background(image_1, fluctuation_values, tolerance=tolerance)
        background = median_filter(image_1, size=5)
        image_1 = image_1 - background
        spot_mask = detect_spots(image_1)
        image_1_masked = np.ma.masked_where(~spot_mask, image_1)
        threshold = 0.7
        inflate_factor = 20
        image_1_masked = np.where(image_1_masked > threshold, image_1_masked * inflate_factor, image_1_masked)
        image_1_masked = np.where(image_1_masked <= threshold, image_1_masked * 0, image_1_masked)
        local_imsd = dict.fromkeys(det_keys)
        for det_key in det_keys:
            local_imsd[det_key] = image_1_masked
        pimg = pv.warp_image(local_imsd, pad_with_nans=True, do_interpolation=True)
        # Integrate over all eta (azimuth) as before
        Int = np.array(np.ma.average(pimg, axis=0))
        # If cake_slices is provided, bin by cake slices
        cake_intensities = None
        if cake_slices is not None:
            cake_intensities = []
            eta_axis = np.linspace(pv.eta_min, pv.eta_max, pimg.shape[0]) * 180 / np.pi
            for eta_start, eta_end in cake_slices:
                mask = (eta_axis >= eta_start) & (eta_axis < eta_end)
                # Average only over the eta slice
                cake_int = np.array(np.ma.average(pimg[mask, :], axis=0))
                cake_intensities.append(cake_int)
        return Int, cake_intensities

    # Prepare cake slices
    cake_slices = []
    for i in range(n_cakes):
        eta_start = eta_min + i * cake_width
        eta_end = eta_start + cake_width
        cake_slices.append((eta_start, eta_end))

    all_int = []
    all_cake_int = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_frame, images[i], cake_slices=cake_slices) for i in range(nframes)]
        for idx, future in enumerate(concurrent.futures.as_completed(futures)):
            Int, cake_intensities = future.result()
            all_int.append(Int)
            if cake_intensities is not None:
                all_cake_int.append(cake_intensities)
            if (idx + 1) % 100 == 0 or (idx + 1) == nframes:
                print(f"Processed {idx + 1} of {nframes} frames (parallel)")

    intensity_stack = np.array(all_int)
    # If cake integration was performed, stack cakes: shape (nframes, n_cakes, tth)
    if all_cake_int:
        cake_intensity_stack = np.array(all_cake_int)  # shape: (nframes, n_cakes, tth)
    else:
        cake_intensity_stack = None

    tth = np.linspace(pv.tth_min, pv.tth_max, pv.shape[1]) * 180 / np.pi
    q_values = (4 * np.pi * np.sin(np.radians(tth / 2))) / wavelength

    # Save data to HDF5 format
    with h5py.File(output_file, 'w') as h5_file:
        h5_file.create_dataset("int", data=intensity_stack)  # shape: (nframes, tth)
        h5_file.create_dataset("q", data=q_values)           # shape: (tth,)
        h5_file.attrs["nframes"] = nframes
        if cake_intensity_stack is not None:
            h5_file.create_dataset("cake_int", data=cake_intensity_stack)  # shape: (nframes, n_cakes, tth)
            h5_file.create_dataset("cake_eta_ranges", data=np.array(cake_slices))  # shape: (n_cakes, 2)
    print(f"Polar integration completed. Results saved to {output_file}")

   
     
    # Plot time-resolved data if the plot flag is set
    if plot:
        plt.figure(figsize=(10, 6))
        plt.imshow(intensity_stack, aspect='auto', extent=[q_values.min(), q_values.max(), 0, nframes],
                   origin='lower', cmap='jet')
        plt.colorbar(label='Intensity')
        plt.xlabel('q (1/Å)', fontsize = 16)
        plt.ylabel('Frame Index', fontsize = 16)
        plt.title(f"Time-Resolved Data: {experiment_name}")
        plt.show()
if __name__ == "__main__":
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Polar integration of diffraction experiments.")
    parser.add_argument("Tiff_fold", type=str, help="Path to the folder containing TIFF images.")
    parser.add_argument("--instr_file", type=str, required=True, help="Path to the instrument YAML file.")
    parser.add_argument("--plot", action="store_true", help="Enable plotting of time-resolved data.")
    args = parser.parse_args()
    
    # Run the integration workflow
    integrate_em(args.Tiff_fold, args.instr_file, plot=args.plot)
