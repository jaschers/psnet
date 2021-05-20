import numpy as np
import re
from io import StringIO
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import pandas as pd
from tqdm import tqdm

# properties = np.array([9, 0, 0, 0, 10, 100000, 2, 0, 20, 20, 3]) # a1, a2, dl1, dl2, dh1, dh2, m1, m2, n1, n2, f 
# pattern spectra properties
a = np.array([9, 0])
dl = np.array([0, 0])
dh = np.array([10, 100000])
m = np.array([2, 0])
n = np.array([20, 20])
f = 3

particle_type = "gamma"
image_type = "minimalistic"
run = np.array([107, 1012, 1024, 1034, 1037, 1054, 1057, 1069, 1073, 1086, 1098]) #1012, 1024, 1034, 1037, 1054, 1057, 1069, 1073, 1086, 1098

for r in range(len(run)):
    # read csv file
    print("Run", run[r])
    run_filename = f"{particle_type}_20deg_0deg_run{run[r]}___cta-prod5-paranal_desert-2147m-Paranal-dark_merged.DL1"
    csv_directory = f"dm-finder/data/{particle_type}/tables/" + run_filename + ".csv"

    table = pd.read_csv(csv_directory)

    # extract unique obs_id and all event_id
    obs_id_unique = np.unique(table["obs_id"])
    event_id = table["event_id"]

    # add image column to table to be filled in
    table["pattern spectrum"] = np.nan
    table["pattern spectrum"] = table["pattern spectrum"].astype(object)

    # for loop to create pattern spectra from cta images with pattern spectra code
    for i in tqdm(range(len(table))): #len(table)
        # create folder 
        path_mat = f"dm-finder/data/{particle_type}/pattern_spectra/{image_type}" + f"/a_{a[0]}_{a[1]}__dl_{dl[0]}_{dl[1]}__dh_{dh[0]}_{dh[1]}__m_{m[0]}_{m[1]}__n_{n[0]}_{n[1]}__f_{f}/" + run_filename + "/obs_id_" + f"{table['obs_id'][i]}/mat/"
        path_tif = f"dm-finder/data/{particle_type}/pattern_spectra/{image_type}" + f"/a_{a[0]}_{a[1]}__dl_{dl[0]}_{dl[1]}__dh_{dh[0]}_{dh[1]}__m_{m[0]}_{m[1]}__n_{n[0]}_{n[1]}__f_{f}/" + run_filename + "/obs_id_" + f"{table['obs_id'][i]}/tif/"
        filename = "obs_id_" + f"{table['obs_id'][i]}" + "__" "event_id_" + f"{table['event_id'][i]}"

        try:
            os.makedirs(path_mat)
        except OSError:
            pass #print("Directory could not be created")
        
        try:
            os.makedirs(path_tif)
        except OSError:
            pass #print("Directory could not be created")

        # command to create pattern spectra
        command = "./dm-finder/software/pattern_spectra/xmaxtree/xmaxtree" + f" dm-finder/data/{particle_type}/images/{image_type}/" + run_filename + "/obs_id_" + f"{table['obs_id'][i]}" + "/pgm/" + "obs_id_" + f"{table['obs_id'][i]}" + "__" "event_id_" + f"{table['event_id'][i]}.pgm" f" a {a[0]}, {a[1]} dl {dl[0]}, {dl[1]} dh {dh[0]}, {dh[1]} m {m[0]}, {m[1]} n {n[0]}, {n[1]} f {f} nogui e " + path_mat + filename + " &> /dev/null"

        # apply command in terminal
        os.system(command)

        # open pattern spectra file
        file = open(path_mat + filename + ".m", "r")
        # remove unnecessary "Granulometry(:,:)" from the string
        image = file.read()[18:]
        # convert string to numpy array with proper shape and remove unnecessary colomns and rows
        image = "0" + re.sub(" +", " ", image)
        image = np.genfromtxt(StringIO(image))[2:, 2:]
        # take log of the image and replace -inf values with 0
        image = np.log10(image)
        image[image == -np.inf] = 0
        # add image to table
        table["pattern spectrum"][i] = image

        if r == 0 and i < 50:
            plt.rcParams['figure.facecolor'] = 'black'
            plt.figure()
            plt.imshow(image, cmap = "Greys_r")
            plt.savefig(path_tif + filename + ".tif")
            plt.close()

    # save tabel as h5 file
    path_cnn_input = f"dm-finder/cnn/pattern_spectra/input/{particle_type}/{image_type}/" + f"/a_{a[0]}_{a[1]}__dl_{dl[0]}_{dl[1]}__dh_{dh[0]}_{dh[1]}__m_{m[0]}_{m[1]}__n_{n[0]}_{n[1]}__f_{f}/" 
    try:
        os.makedirs(path_cnn_input)
    except OSError:
        pass

    output_filename = path_cnn_input + run_filename + "_ps.h5"
    # print(output_filename)
    table.to_hdf(output_filename, key = 'events', mode = 'w', index = False)
