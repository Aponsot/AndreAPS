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


"Define Integration Script"
Data = [] 
def integrate_em(Tiff_fold, output_folder, data_label, instr):  #Inputs of Tiff Folder, The Output Folder, And User Given Data Label, Load yaml Instrument file
    tifs = sorted([f for f in os.listdir(Tiff_fold) if f.lower().endswith(('.tiff','.tif'))]) # Sorts through Given Tiff Folder and finds Tiff and tif files 
    image_stack = [] # preallocate image_stack of tiff images 
    for fname in tiff: 
        img = tiff.imread(os.path.join(tifs, fname)) #pulls tiff files out of folder  
        image_stack.append(img)
        image_stack = np.array(image_stack) #Creates array of all tiff files (Z,X,Y) Z is frame number XY are the image pixel values of each diffraction immage
        images = image_stack() 
        nframes = np.shape(image_stack[0]) 

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
            np.r_[tth_min, tth_max], instr,                     # Using Defined HEXRD variables and given instrument file this block sets up the polor view
            eta_min, eta_max,
            pixel_size=(tth_stats[1]/ndiv, eta_stats[1]/ndiv),
            cache_coordinate_map=True
        )
       
        all_int = []

        for i in range(nframes):
            image_1 = images[i]
            image_1 = np.ma.masked_where(image_1 == (2**32 - 1), image_1)

            pimg = pv.warp_image(imsd, pad_with_nans=True, do_interpolation=True)

            Int = np.array(np.ma.average(pimg, axis=0))  # 1D array
            all_int.append(Int)

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
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    Tiff_fold = filedialog.askdirectory(title="Select Folder Containing Tiff Images", filetypes=[("tiff", "*.tif")])
   
    # Select output folder
    output_folder = filedialog.askdirectory(title="Select Output Folder")

    # Enter sample/scan label
    data_label = input("Enter a label for the output files (e.g., sampleID_scanNumber): ").strip()
    
    # Load instrument YAML
    instr_file = r'D:\APS_Rev.yml'

    with open(instr_file, 'r') as f:
        instr_cfg = yaml.safe_load(f)
    instr = instrument.HEDMInstrument(instr_cfg)
#Run Integration 
integrate_em(Tiff_fold, output_folder, data_label, instr)