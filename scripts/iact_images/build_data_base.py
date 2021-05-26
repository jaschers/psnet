import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py
pd.options.mode.chained_assignment = None  # default='warn'

particle_type = "gamma"
image_type = "minimalistic"
run = np.array([107, 1012, 1024, 1034, 1037, 1054, 1057, 1069, 1073, 1086, 1098])
for r in range(len(run)):
    # read csv file
    print("Run", run[r])
    run_filename = f"{particle_type}_20deg_0deg_run{run[r]}___cta-prod5-paranal_desert-2147m-Paranal-dark_merged.DL1"
    table_path = f"dm-finder/data/{particle_type}/tables/" + run_filename + ".csv"

    table = pd.read_csv(table_path)

    # extract unique obs_id and all event_id
    obs_id_unique = np.unique(table["obs_id"])
    event_id = table["event_id"]

    # add image column to table to be filled in
    table["image"] = np.nan
    table["image"] = table["image"].astype(object)

    # for loop to load all tif images and put it into the table
    for i in tqdm(range(len(table))): #len(table)
        # load tif images
        input_path = f"dm-finder/data/{particle_type}/images/{image_type}/" + run_filename + "/obs_id_" + f"{table['obs_id'][i]}" + "/tif/" + "obs_id_" + f"{table['obs_id'][i]}" + "__" "event_id_" + f"{table['event_id'][i]}.tif"

        image = cv2.imread(input_path, 0)

        # put image into table (and remove blank edges of the tif image)
        table["image"][i] = image

    # save tabel as h5 file
    output_filename = f"dm-finder/cnn/iact_images/input/{particle_type}/{image_type}/" + run_filename + "_images.h5"
    table.to_hdf(output_filename, key = 'events', mode = 'w', index = False)
