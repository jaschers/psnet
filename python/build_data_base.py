import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py
pd.options.mode.chained_assignment = None  # default='warn'

run = np.array([107, 1012, 1024, 1034, 1037, 1054, 1057, 1069, 1073, 1086, 1098, 3667])
for r in range(len(run)):
    # read csv file
    print("Run", run[r])
    run_filename = f"gamma_20deg_0deg_run{run[r]}___cta-prod5-paranal_desert-2147m-Paranal-dark_merged.DL1"
    csv_directory = "cta/data/csv/" + run_filename + ".csv"

    table = pd.read_csv(csv_directory)

    # extract unique obs_id and all event_id
    obs_id_unique = np.unique(table["obs_id"])
    event_id = table["event_id"]

    # add image column to table to be filled in
    table["image"] = np.nan
    table["image"] = table["image"].astype(object)

    # for loop to load all tif images and put it into the table
    for i in tqdm(range(len(table))): #len(table)
        # load tif images
        input_directory = "cta/data/images/" + run_filename + "/obs_id_" + f"{table['obs_id'][i]}" + "/tif/" + "obs_id_" + f"{table['obs_id'][i]}" + "__" "event_id_" + f"{table['event_id'][i]}.tif"

        image = cv2.imread(input_directory, 0)

        # put image into table (and remove blank edges of the tif image)
        table["image"][i] = image[15:-15, 77:-77]

    # save tabel as h5 file
    output_filename = "dm-finder/cnn/input/images/" + run_filename + "_images.h5"
    print(output_filename)
    table.to_hdf(output_filename, key = 'events', mode = 'w', index = False)
