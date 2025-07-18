import numpy as np  
import matplotlib as plt 
import os
import tifffile as tiff
#file importing
import tkinter as tk
from tkinter import filedialog
#Hexrd Imports 
import yaml
from hexrd import instrument 
from hexrd.projections.polar import PolarView
import concurrent.futures

"Define Integration Script"
Data = [] 
def integrate_em(Tiff_fold, output_folder, data_label, instr):  #Inputs of Tiff Folder, The Output Folder, And User Given Data Label, Load yaml Instrument file
        tifs = sorted([f for f in os.listdir(Tiff_fold) if f.lower().endswith(('.tiff','.tif'))])
        if not tifs:
            raise ValueError("No TIFF files found in the folder.")
        # Read the first image to get the shape
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
        tth_min = 1.         # HEXRD variables TTH is the theta range that it will create the polar mapp of. ETA is the range of ring that will be anaylized 0-360
        tth_max = 24.
        eta_min = -180.
        eta_max = 180.
        ndiv = 1  # angular resolution
        tth_stats, eta_stats = np.degrees(instrument.hedm_instrument.pixel_resolution(instr)) # Defines pixel size from instrument file 
        det_keys = instr.detectors.keys() 
        imsd = dict.fromkeys(det_keys)

        pv = PolarView(                  
            np.r_[tth_min, tth_max], instr,                     
            eta_min, eta_max,
            pixel_size=(tth_stats[1]/ndiv, eta_stats[1]/ndiv),
            cache_coordinate_map=True
        )
       
        all_int = []

        def process_frame(image_1):
            image_1 = np.ma.masked_where(image_1 == (2**32 - 1), image_1)
    # Assign the image to the correct detector key
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
        intensity_stack = np.array(all_int)  # shape = (nframes, len(tth))

        # Compute tth once (same for all images)
        tth = np.linspace(pv.tth_min, pv.tth_max, pv.shape[1]) * 180 / np.pi
        
        # Save both to .npz
        file_name = f'{data_label}'
        Folder = f'{output_folder}'
        full_path = os.path.join(file_name,Folder)
        np.savez(full_path, intensities=intensity_stack, tth=tth)
#User inputs 
Tiff_fold = filedialog.askdirectory(title="Select Folder Containing Tiff Images")
output_folder = filedialog.askdirectory(title="Select Output Folder")
data_label = input("Enter a label for the output files (e.g., sampleID_scanNumber): ").strip()
instr_file = filedialog.askopenfilename(title="Select Instrument YAML file", filetypes=[("YAML files", "*.yml")])

with open(instr_file, 'r') as f:
        instr_cfg = yaml.safe_load(f)
instr = instrument.HEDMInstrument(instr_cfg)
#Run Integration 
integrate_em(Tiff_fold, output_folder, data_label, instr)