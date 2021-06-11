import numpy as np
import matplotlib.pyplot as plt
from ctapipe.io import EventSource, read_table
from ctapipe.instrument import SubarrayDescription 
from ctapipe.visualization import CameraDisplay
from utilities import GetEventImage
from astropy.table import Table, join, vstack
from astropy.io import ascii
import sys
import os
from tqdm import tqdm
np.set_printoptions(threshold=sys.maxsize)

# ----------------------------------------------------------------------
# load and prepare data set
# ----------------------------------------------------------------------

# load data
particle_type = "gamma" # gamma/proton
image_type = "default" # default/minimalistic
run = np.array([1024, 1034, 1037, 1054, 1057, 1069, 1073, 1086, 1098]) #gamma 107, 1012, 1024, 1034, 1037, 1054, 1057, 1069, 1073, 1086, 1098
for r in range(len(run)):
    print("Run", run[r])
    input_filename = f"gamma_20deg_0deg_run{run[r]}___cta-prod5-paranal_desert-2147m-Paranal-dark_merged.DL1"
    input_directory = f"dm-finder/data/{particle_type}/event_files/" + input_filename + ".h5"

    source = EventSource(input_directory)
    print(f"Total number of events: {len(source)}")

    # get telescope subarray description
    subarray = SubarrayDescription.from_hdf(input_directory)
    subarray.info()

    # # create figure of telescope subarray layout
    # plt.figure()
    # subarray.peek()
    # plt.savefig(f"cta/data/info/telescope_subarray_layout_run{run[r]}.png")
    # plt.close()

    # get tables for all SST telescopes that include images and corresponding obs_id + event_id for each event
    sst_tel_id = np.append(range(30, 100), range(131, 181))
    images_table= vstack([read_table(input_directory, f"/dl1/event/telescope/images/tel_{t:03}") for t in sst_tel_id])

    # get true energy of each event
    simulated_parameter_table = read_table(input_directory, "/simulation/event/subarray/shower")

    # combine both tables and remove unnecessary columns
    complete_table = join(left = images_table, right = simulated_parameter_table, keys=["obs_id", "event_id"])
    complete_table.keep_columns(["obs_id", "event_id", "tel_id", "image", "true_energy"])

    # convert energy from TeV to GeV
    complete_table["true_energy"] = complete_table["true_energy"].to("GeV")

    # define telescope geometry to the SST geometry (telescope id 30)
    sst_camera_geometry = source.subarray.tel[30].camera.geometry

    ############################### save individual telescope images ###############################
    # for i in range(30):
    #     GetEventImage(complete_table["image"][i], sst_camera_geometry, clean_image = False, savefig = f"cta/data/images/tests/obs_id_{complete_table['obs_id'][i]}__event_id_{complete_table['event_id'][i]}__tel_id_{complete_table['tel_id'][i]}.png", colorbar = True, cmap = "Greys")
    #     print("_______________________________")
    #     print("obs_id, event_id, tel_id:", complete_table["obs_id"][i], complete_table["event_id"][i], complete_table["tel_id"][i])
    #     print("true energy:", complete_table["true_energy"][i])
    #     print("min/max pixel:", np.min(complete_table["image"][i]), np.max(complete_table["image"][i]))
    ################################################################################################

    # group table by same obs_id and event_id
    complete_table_by_obs_id_event_id = complete_table.group_by(["obs_id", "event_id", "true_energy"])

    # create new table in which we will add a combined images of telescopes originiting from the same event
    complete_table_tel_combined = complete_table_by_obs_id_event_id.groups.keys

    # # save data into a .csv file
    # ascii.write(complete_table_tel_combined, f"dm-finder/data/{particle_type}/tables/{particle_type}_20deg_0deg_run{run[r]}___cta-prod5-paranal_desert-2147m-Paranal-dark_merged.DL1.csv", format = "csv", fast_writer = False, overwrite = True)

    # create empty image combined list to be filled in
    image_combined = [[]] * len(complete_table_tel_combined)

    # combine all images of telescopes originiting from the same event and put them into the list
    k = 0
    for key, group in zip(complete_table_by_obs_id_event_id.groups.keys, complete_table_by_obs_id_event_id.groups):
        image_combined[k] = group["image"].groups.aggregate(np.add)
        k += 1

    # add combined images list to the table
    complete_table_tel_combined["image combined"] = image_combined

    # # display and save energy distribution
    # plt.figure()
    # plt.grid(alpha = 0.2)
    # plt.hist(complete_table_tel_combined["true_energy"].to("TeV").value)
    # plt.xlabel("true energy [TeV]")
    # plt.ylabel("number of events")
    # plt.yscale("log")
    # plt.savefig(f"dm-finder/data/{particle_type}/info/energy_distribution/energy_distribution_run{run[r]}.png")
    # plt.close()

    # save combined telescope images
    for i in tqdm(range(len(complete_table_tel_combined))):
        # create directory in which the images will be saved
        path = f"dm-finder/data/{particle_type}/images/{image_type}/{input_filename}/obs_id_{complete_table_tel_combined['obs_id'][i]}/tif"
        try:
            os.makedirs(path)
        except OSError:
            pass

        # save image
        GetEventImage(complete_table_tel_combined["image combined"][i][0], sst_camera_geometry, clean_image = True, savefig = f"dm-finder/data/{particle_type}/images/{image_type}/{input_filename}/obs_id_{complete_table_tel_combined['obs_id'][i]}/tif/obs_id_{complete_table_tel_combined['obs_id'][i]}__event_id_{complete_table_tel_combined['event_id'][i]}_r.tif", colorbar = False, cmap = "Greys_r")

        # # print information
        # print("_______________________________")
        # print("obs_id, event_id:", complete_table_tel_combined["obs_id"][i], complete_table_tel_combined["event_id"][i])
        # print("true energy:", np.round(complete_table_tel_combined["true_energy"][i], 10))
        # print("min/max pixel:", np.min(complete_table_tel_combined["image combined"][i][0]), np.max(complete_table_tel_combined["image combined"][i][0]))

    # convert png images to pgm
    for i in tqdm(range(len(complete_table_tel_combined))):
        # create directory in which the images will be saved
        path = f"dm-finder/data/{particle_type}/images/{image_type}/{input_filename}/obs_id_{complete_table_tel_combined['obs_id'][i]}/pgm"
        try:
            os.makedirs(path)
        except OSError:
            pass

        os.system(f"convert dm-finder/data/{particle_type}/images/{image_type}/{input_filename}/obs_id_{complete_table_tel_combined['obs_id'][i]}/tif/obs_id_{complete_table_tel_combined['obs_id'][i]}__event_id_{complete_table_tel_combined['event_id'][i]}_r.tif dm-finder/data/{particle_type}/images/{image_type}/{input_filename}/obs_id_{complete_table_tel_combined['obs_id'][i]}/pgm/obs_id_{complete_table_tel_combined['obs_id'][i]}__event_id_{complete_table_tel_combined['event_id'][i]}_r.pgm")

    # close event file
    source.close()
