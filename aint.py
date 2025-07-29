import argparse

def integrate_em(tiff_folder, instr_file, plot=False):
    
    import os
    import h5py
    import yaml
    from hexrd import instrument
    from concurrent.futures import ProcessPoolExecutor
    import numpy as np
    import tifffile as tiff
    import pyFAI
    import matplotlib.pyplot as plt

    # Load instrument configuration
    with open(instr_file, 'r') as f:
        instr_cfg = yaml.safe_load(f)
    instr = instrument.HEDMInstrument(instr_cfg)

    # Extract instrument parameters
    detector_distance = instr_cfg["detectors"]["detector_1"]["transform"]["translation"][2]  # in meters
    pixel_size = instr_cfg["detectors"]["detector_1"]["pixels"]["size"]  # [pixel_size_x, pixel_size_y] in meters
    cols = instr_cfg["detectors"]["detector_1"]["pixels"]["columns"]
    rows = instr_cfg["detectors"]["detector_1"]["pixels"]["rows"] 
    beam_center = np.array([cols / 2, rows / 2])  
    energy = instr_cfg["beam"]["energy"]  # Energy in keV

    # Calculate wavelength in meters
    h = 4.135667696e-18  # Planck's constant in keV·s
    c = 2.99792458e10  # Speed of light in cm/s
    wavelength_m = h * c / energy  # Wavelength in cm
    wavelength_m *= 1e-2  # Convert to meters

    # Initialize pyFAI AzimuthalIntegrator
    ai = pyFAI.AzimuthalIntegrator()
    ai.set_pixel1(pixel_size[1])  # Pixel size in the vertical direction (meters)
    ai.set_pixel2(pixel_size[0])  # Pixel size in the horizontal direction (meters)
    ai.set_wavelength(wavelength_m)  # Wavelength in meters
    ai.dist = detector_distance  # Detector distance in meters
    ai.poni1 = beam_center[1] * pixel_size[1] 
    ai.poni2 = beam_center[0] * pixel_size[0] 
    



    # Load TIFF images
    tiffs = sorted([f for f in os.listdir(tiff_folder) if f.lower().endswith(('.tiff', '.tif'))])
    if not tiffs:
        raise ValueError("No TIFF files found in the folder.")

    # Extract experiment name from TIFF files (common prefix)
    experiment_name = os.path.commonprefix(tiffs).rstrip('_-')
    print(f"Experiment name: {experiment_name}")

    # Prepare output file
    output_file = os.path.join("/home/beams/PONSOT/Data/", f"{experiment_name}.h5")

    # Load images into a stack
    first_img = tiff.imread(os.path.join(tiff_folder, tiffs[0]))
    image_shape = first_img.shape
    image_stack = np.empty((len(tiffs), *image_shape), dtype=first_img.dtype)
    image_stack[0] = first_img
    for i, fname in enumerate(tiffs[1:], start=1):
        img = tiff.imread(os.path.join(tiff_folder, fname))
        image_stack[i] = img
        
    images = image_stack
    nframes = images.shape[0]
    print(f"Number of frames: {nframes}")

    # Define the frame processing function
    def process_frame(image):
        # Perform azimuthal integration
        npt = 1000  # Number of bins in the 1D profile
        result = ai.integrate1d(
            image,
            npt=npt,
            unit="q_nm^-1",  # q values in nm^-1
            radial_range=0,10,  # Default radial range
            azimuth_range=-none,  # Default azimuthal range
            mask=np.zeros_like(sam , dtype =bool) 
            mask{sam <threshold] = true  # Add a mask here if needed
        )

        q = result.radial  # q values (nm^-1)
        intensity = result.intensity  # Intensity values
        #two_theta = ai.q2twotheta(q) * (180 / np.pi)  # Convert q to 2θ in degrees
        
        return q, intensity
    sam = images[0] 
    print(ai)
    q, intensity = process_frame(sam) 
    print(f"sample int res") 
    print(f"q range: {q.min()} to {q.max()}")
    print(f"int range: {intensity.min()} tp {intensity.max()}") 
    
    
    # Process frames in parallel and save results to HDF5
    with h5py.File(output_file, "w") as h5file:
        # Create datasets for q, 2θ, intensity, and frame numbers
        n_bins = 1000  # Number of bins in the 1D profile
        h5file.create_dataset("q", (n_bins,), dtype="f", compression="gzip")  # q values (shared across frames)
        h5file.create_dataset("two_theta", (n_bins,), dtype="f", compression="gzip")  # 2θ values (shared across frames)
        h5file.create_dataset("intensity", (nframes, n_bins), dtype="f", compression="gzip")  # Intensity for each frame
        h5file.create_dataset("frame_numbers", (nframes,), dtype="i")  # Frame numbers

        # Process frames in parallel
        with ProcessPoolExecutor() as executor:
            results = executor.map(lambda img: process_frame(img), images)
            for i, (q, intensity) in enumerate(executor.map(results)):
                if i == 0:
                    # Save q and 2θ values (only need to save once, as they are the same for all frames)
                    h5file["q"][:] = q
                    h5file["two_theta"][:] = two_theta
                # Save intensity and frame number for the current frame
                h5file["intensity"][i, :] = intensity
                h5file["frame_numbers"][i] = i
    print(f"Integrated results saved to {output_file}")
    
    if plot:
    	with h5py.File(output_file, "r") as h5file: 
            q_val = h5file["q"][:] 
            int_val = h5file["intensity"][:] 
            frames = int_val.shape[0]
           
            plt.figure(figsize=(10, 6))
            plt.imshow(int_val, aspect='auto', extent=[q_val.min(), q_val.max(), 0, frames],
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
    parser.add_argument("--instr_file", type=str, required=True, help="Path to the instrument YAML file.")
    parser.add_argument("--plot", action="store_true", help="Enable plotting of time-resolved data.")

    args = parser.parse_args()

    # Run the integration workflow
    integrate_em(args.Tiff_fold, args.instr_file, plot=args.plot)
