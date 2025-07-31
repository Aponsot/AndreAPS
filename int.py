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
    tth_max = 15.0
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

    from sklearn.cluster import DBSCAN

    def detect_spots(image, eps=5, min_samples=3):
        """
        Detect spots in the image using DBSCAN clustering.
    
        Parameters:
        - image: Input image (2D array).
        - eps: Maximum distance between points to be considered a cluster.
        - min_samples: Minimum number of points to form a cluster.
    
        Returns:
        - spot_mask: Binary mask indicating detected spots.
        """
        # Extract coordinates of non-zero pixels
        coords = np.column_stack(np.where(image > 0))
    
        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    
        # Create a binary mask for detected spots
        spot_mask = np.zeros_like(image, dtype=bool)
        for cluster_label in np.unique(clustering.labels_):
            if cluster_label != -1:  # Ignore noise points
                cluster_coords = coords[clustering.labels_ == cluster_label]
                spot_mask[cluster_coords[:, 0], cluster_coords[:, 1]] = True
        return spot_mask

    def process_frame(image_1):
        """
        Process a single frame for polar integration with spot detection.
    
        Parameters:
        - image_1: Input image (2D array).
    
        Returns:
        - Int: Integrated intensity values for the frame.
        """
        # Mask invalid pixels
        image_1 = np.ma.masked_where((image_1 == (2**32 - 1)) | (image_1 <= 0), image_1)
    
        # Detect spots
        spot_mask = detect_spots(image_1)
    
        # Apply spot mask
        image_1_masked = np.ma.masked_where(~spot_mask, image_1)
    
        # Initialize local detector images
        local_imsd = dict.fromkeys(det_keys)
        for det_key in det_keys:
            local_imsd[det_key] = image_1_masked
    
        # Perform polar remapping
        pimg = pv.warp_image(local_imsd, pad_with_nans=True, do_interpolation=True)
        Int = np.array(np.ma.average(pimg, axis=0))  # Integrate only meaningful bins

        return Int  # Ensure the result is returned

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
        h5_file.create_dataset(f"int", data=intensity_stack)
        h5_file.attrs["q_values"] = q_values
        h5_file.attrs["nframes"] = nframes
   
    print(f"Polar integration completed. Results saved to {output_file}")

    int1 = intensity_stack  
    int_max = np.max(intensity_stack) 
    normalized = 15 * int1 / int_max	
     
    # Plot time-resolved data if the plot flag is set
    if plot:
        plt.figure(figsize=(10, 6))
        plt.imshow(normalized, aspect='auto', extent=[q_values.min(), q_values.max(), 0, nframes],
                   origin='lower', cmap='jet')
        plt.colorbar(label='Intensity')
        plt.xlabel('q (1/Å)')
        plt.ylabel('Frame Index')
        plt.title(f"Time-Resolved Data: {experiment_name}")
        plt.show()
