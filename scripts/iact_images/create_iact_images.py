import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ctapipe.io import EventSource, read_table
from ctapipe.instrument import SubarrayDescription 
from ctapipe.visualization import CameraDisplay
from utilities import GetEventImage, GetEventImageBasic, GetEventImageBasicSmall
from astropy.table import Table, join, vstack
from astropy.io import ascii
import sys
import os
from tqdm import tqdm
import argparse
import h5py

np.set_printoptions(threshold=sys.maxsize)
plt.rcParams.update({'font.size': 14})

######################################## argparse setup ########################################

script_version=0.1
script_descr="""
This script creates Cherenkov images of gamma/diffuse-gamma/proton events simulated for CTA. The images are stored in a HDF table.
"""

# Open argument parser
parser = argparse.ArgumentParser(description=script_descr)
parser.add_argument("-v", "--version", action="version", version=f"v{script_version}")

# Define expected arguments
parser.add_argument("-pt", "--particle_type", type = str, metavar = "", choices = ["gamma", "gamma_diffuse", "proton"], help = "particle type [gamma, gamma_diffuse, proton], default: gamma", default = "gamma")
parser.add_argument("-tm", "--telescope_mode", type = str, required = False, metavar = "", choices = ["mono", "stereo_sum_cta"], help = "telescope mode [mono, stereo_sum_cta], default: stereo_sum_cta", default = "stereo_sum_cta")
parser.add_argument("-r", "--run", type = int, metavar = "-", help = "input run(s) from which the CTA images will be extracted, default: csv list", action='append', nargs='+')
parser.add_argument("-er", "--energy_range", type = float, required = True, metavar = "-", help = "set energy range of events in TeV", nargs = 2)

args = parser.parse_args()
print(f"################### Input summary ################### \nParticle type: {args.particle_type} \nTelescope mode: {args.telescope_mode} \nEnergy range: {args.energy_range} TeV \n#####################################################")
##########################################################################################

# ----------------------------------------------------------------------
# load and prepare data set
# ----------------------------------------------------------------------

# load data
if args.telescope_mode == "mono" and args.particle_type == "gamma":
    filename_run_csv = f"dm-finder/scripts/run_lists/{args.particle_type}_run_list_mono_alpha.csv"
else: 
    filename_run_csv = f"dm-finder/scripts/run_lists/{args.particle_type}_run_list_alpha.csv"
run = pd.read_csv(filename_run_csv)
run = run.to_numpy().reshape(len(run))

if args.run != None:
    run = args.run[0]
    
print("Runs:", run)
print("Total number of Runs:", len(run))

# loop over all runs
for r in range(len(run)): #len(run)
    print("Run", run[r])

    if args.particle_type == "gamma_diffuse":
        input_filename = f"gamma_20deg_0deg_run{run[r]}___cta-prod5-paranal_desert-2147m-Paranal-dark_cone10_merged.DL1"
    else:
        input_filename = f"{args.particle_type}_20deg_0deg_run{run[r]}___cta-prod5-paranal_desert-2147m-Paranal-dark_merged.DL1"
    input_directory = f"dm-finder/data/{args.particle_type}/event_files/" + input_filename + ".h5"

    output_filename = f"{args.particle_type}_20deg_0deg_run{run[r]}___cta-prod5-paranal_desert-2147m-Paranal-dark_merged.DL1"

    source = EventSource(input_directory)
    print(f"Total number of events: {len(source)}")

    # get telescope subarray description
    subarray = SubarrayDescription.from_hdf(input_directory)
    subarray.info()

    # create figure of telescope subarray layout
    path_array_layout = f"dm-finder/data/{args.particle_type}/info/array_layout/"
    path_energy_distribution = f"dm-finder/data/{args.particle_type}/info/energy_distribution/"
    os.makedirs(path_array_layout, exist_ok = True)
    os.makedirs(path_energy_distribution, exist_ok = True)

    plt.figure()
    subarray.peek()
    plt.savefig(path_array_layout + f"telescope_subarray_layout_run{run[r]}.pdf", dpi = 500)
    plt.close()

    # get tables for all SST telescopes that include images and corresponding obs_id + event_id for each event
    # alpha configuration
    sst_tel_id = np.array([30, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 133, 59, 61, 66, 67, 68, 69, 70, 71, 72, 73, 143, 144, 145, 146])
    # full configuration
    # sst_tel_id = np.append(range(30, 100), range(131, 181))
    images_table = vstack([read_table(input_directory, f"/dl1/event/telescope/images/tel_{t:03}") for t in sst_tel_id])

    # get true energy of each event
    simulated_parameter_table = read_table(input_directory, "/simulation/event/subarray/shower")

    # combine both tables and remove unnecessary columns
    complete_table = join(left = images_table, right = simulated_parameter_table, keys=["obs_id", "event_id"])
    complete_table.keep_columns(["obs_id", "event_id", "tel_id", "image", "true_energy"])

    # convert energy from TeV to GeV
    complete_table["true_energy"] = complete_table["true_energy"].to("GeV")

    # remove events outside energy range
    mask = (complete_table["true_energy"] >= args.energy_range[0] * 1e3) & (complete_table["true_energy"] <= args.energy_range[1] * 1e3)
    complete_table = complete_table[mask]

    # define telescope geometry to the SST geometry (telescope id 30)
    sst_camera_geometry = source.subarray.tel[30].camera.geometry

    if args.telescope_mode == "mono":
        # group table by same obs_id and event_id
        complete_table_by_obs_id_event_id = complete_table.group_by(["obs_id", "event_id", "tel_id", "true_energy"])

        # create new table in which we will add a combined images of telescopes originiting from the same event
        table_csv = complete_table_by_obs_id_event_id.groups.keys

        # save data into a .csv file
        path_tables = f"dm-finder/data/{args.particle_type}/tables/"
        os.makedirs(path_tables, exist_ok = True)

        ascii.write(table_csv, f"dm-finder/data/{args.particle_type}/tables/{args.particle_type}_20deg_0deg_run{run[r]}___cta-prod5-paranal_desert-2147m-Paranal-dark_merged.DL1_mono_alpha.csv", format = "csv", fast_writer = False, overwrite = True)

        # open csv file and prepare table for filling in images 
        table_path = f"dm-finder/data/{args.particle_type}/tables/" + f"{args.particle_type}_20deg_0deg_run{run[r]}___cta-prod5-paranal_desert-2147m-Paranal-dark_merged.DL1_mono_alpha" + ".csv"

        table = pd.read_csv(table_path)

        # add run information to the table
        table["run"] = run[r]

        # rearange the order of the columns
        columns = list(table.columns.values)
        columns = columns[-1:] + columns[:-1]
        table = table[columns]

        # add particle type to table
        if (args.particle_type == "gamma") or (args.particle_type == "gamma_diffuse"):
            table["particle"] = 1
        elif args.particle_type == "proton":
            table["particle"] = 0

        # add image columns to table to be filled in
        table["image"] = np.nan
        table["image"] = table["image"].astype(object)

############################################ CLEAN ############################################

        for i in tqdm(range(len(table))): # len(table)
  
            image = complete_table["image"][i]
            
            image = np.append(image, np.ones(4 * 8 * 8) * np.min(image))
            image = np.split(image, 36)
            image = np.reshape(image, newshape = (-1, 8, 8)).astype('float32')
            image = np.rot90(image, k = 1, axes = (1, 2))
            mask = np.array([32, 22, 10, 4, 16, 33, 30, 20, 8, 2, 14, 26, 28, 18, 6, 0, 12, 24, 29, 19, 7, 1, 13, 25, 31, 21, 9, 3, 15, 27, 34, 23, 11, 5, 17, 35])
            image = image[mask]
            image = image.reshape(6,6,8,8).transpose(0,2,1,3).reshape(48,48)
            image = np.round(image, 1)

            # create directory in which the float images will be saved
            path_hdf = f"dm-finder/data/{args.particle_type}/images/{output_filename}/float/mono_alpha/obs_id_{complete_table['obs_id'][i]}/hdf/"
            os.makedirs(path_hdf, exist_ok = True)

            image_HDF = np.array([np.reshape(image, (48 * 48))])

            output_hdf_filename = path_hdf + f"obs_id_{complete_table['obs_id'][i]}__event_id_{complete_table['event_id'][i]}__tel_id_{complete_table['tel_id'][i]}.h5"
            HDFfile = h5py.File(output_hdf_filename, "w")
            HDFfile.create_dataset("image", data = image_HDF)

            if run[r] == 10 and i <= 125:
                path_tif = f"dm-finder/data/{args.particle_type}/images/{output_filename}/float/mono_alpha/obs_id_{complete_table['obs_id'][i]}/tif/"
                os.makedirs(path_tif, exist_ok = True)
                output_tif_filename = path_tif + f"obs_id_{complete_table['obs_id'][i]}__event_id_{complete_table['event_id'][i]}__tel_id_{complete_table['tel_id'][i]}.tif"
                GetEventImageBasic(image, cmap = "Greys_r", show_frame = False, colorbar = True, clean_image = False, savefig = output_tif_filename)

            # implement image into table
            table["image"][i] = image

        path_cnn_input = f"dm-finder/cnn/iact_images/input/{args.particle_type}/"
        try:
            os.makedirs(path_cnn_input)
        except OSError:
            pass
        
        run_filename = f"{args.particle_type}_20deg_0deg_run{run[r]}___cta-prod5-paranal_desert-2147m-Paranal-dark_merged.DL1"

        output_hdf_filename = f"dm-finder/cnn/iact_images/input/{args.particle_type}/" + run_filename + "_images_mono_alpha"

        table.to_hdf(output_hdf_filename + ".h5", key = 'events', mode = 'w', index = False)
        # close event file
        source.close()


    if args.telescope_mode == "stereo_sum_cta":
        # group table by same obs_id and event_id
        complete_table_by_obs_id_event_id = complete_table.group_by(["obs_id", "event_id", "true_energy"])

        # create new table in which we will add a combined images of telescopes originiting from the same event
        complete_table_tel_combined = complete_table_by_obs_id_event_id.groups.keys

        # create empty image combined list to be filled in
        image_combined = [[]] * len(complete_table_tel_combined)

        # combine all images of telescopes originiting from the same event and put them into the list
        k = 0
        for key, group in zip(complete_table_by_obs_id_event_id.groups.keys, complete_table_by_obs_id_event_id.groups):
            image_combined[k] = group["image"].groups.aggregate(np.add)
            k += 1
            
        # # compute sum of all 'photon' counts in each combined image 
        # sum_image_combined = np.sum(image_combined, axis = 2)

        # # add combined sum of all 'photon' counts list to the table
        # complete_table_tel_combined["sum_image"] = sum_image_combined

        # save data into a .csv file
        path_tables = f"dm-finder/data/{args.particle_type}/tables/"
        try:
            os.makedirs(path_tables)
        except OSError:
            pass
        ascii.write(complete_table_tel_combined, f"dm-finder/data/{args.particle_type}/tables/{args.particle_type}_20deg_0deg_run{run[r]}___cta-prod5-paranal_desert-2147m-Paranal-dark_merged.DL1_alpha.csv", format = "csv", fast_writer = False, overwrite = True)

        # add combined images list to the table
        complete_table_tel_combined["image combined"] = image_combined

        # display and save energy distribution
        plt.figure()
        plt.grid(alpha = 0.2)
        plt.hist(complete_table_tel_combined["true_energy"].to("TeV").value)
        plt.xlabel("True energy [TeV]")
        plt.ylabel("Number of events")
        plt.yscale("log")
        plt.savefig(f"dm-finder/data/{args.particle_type}/info/energy_distribution/energy_distribution_run{run[r]}_alpha.png")
        plt.close()

        # open csv file and prepare table for filling in images 
        run_filename = f"{args.particle_type}_20deg_0deg_run{run[r]}___cta-prod5-paranal_desert-2147m-Paranal-dark_merged.DL1"
        # table_path = f"dm-finder/data/{args.particle_type}/tables/" + run_filename + ".csv"
        table_path = f"dm-finder/data/{args.particle_type}/tables/" + f"{args.particle_type}_20deg_0deg_run{run[r]}___cta-prod5-paranal_desert-2147m-Paranal-dark_merged.DL1_alpha" + ".csv"

        table = pd.read_csv(table_path)

        # extract unique obs_id and all event_id
        obs_id_unique = np.unique(table["obs_id"])
        event_id = table["event_id"]

        # add run information to the table
        table["run"] = run[r]

        # rearange the order of the columns
        columns = list(table.columns.values)
        columns = columns[-1:] + columns[:-1]
        table = table[columns]

        # add image columns to table to be filled in
        if (args.particle_type == "gamma") or (args.particle_type == "gamma_diffuse"):
            table["particle"] = 1
            print("gamma")
        elif args.particle_type == "proton":
            table["particle"] = 0
            print("proton")
        table["image"] = np.nan
        table["image"] = table["image"].astype(object)


        # save combined telescope images
        for i in tqdm(range(len(complete_table_tel_combined))): # len(complete_table_tel_combined)
            # create directory in which the images will be saved
            # path = f"dm-finder/data/{args.particle_type}/images/{output_filename}/obs_id_{complete_table_tel_combined['obs_id'][i]}/tif"
            # try:
            #     os.makedirs(path)
            # except OSError:
            #     pass

            # print("obs_id:", complete_table_tel_combined["obs_id"][i])
            # print("event_id:", complete_table_tel_combined["event_id"][i])
            # print(complete_table_tel_combined["image combined"][i][0])

            image = complete_table_tel_combined["image combined"][i][0]
            
            image = np.append(image, np.ones(4 * 8 * 8) * np.min(image))
            image = np.split(image, 36)
            image = np.reshape(image, newshape = (-1, 8, 8)).astype('float32')
            image = np.rot90(image, k = 1, axes = (1, 2))
            # mask = np.array([15, 21, 9, 27, 3, 33, 14, 20, 8, 26, 2, 32, 16, 22, 10, 28, 4, 34, 13, 19, 7, 25, 1, 31, 17, 23, 11, 29, 18, 6, 24, 0, 5, 30, 35])
            mask = np.array([32, 22, 10, 4, 16, 33, 30, 20, 8, 2, 14, 26, 28, 18, 6, 0, 12, 24, 29, 19, 7, 1, 13, 25, 31, 21, 9, 3, 15, 27, 34, 23, 11, 5, 17, 35])
            image = image[mask]
            image = image.reshape(6,6,8,8).transpose(0,2,1,3).reshape(48,48)
            image = np.round(image, 1)

            image_filename_tif = f"dm-finder/data/{args.particle_type}/images/{output_filename}/float_alpha/obs_id_{complete_table_tel_combined['obs_id'][i]}/tif/obs_id_{complete_table_tel_combined['obs_id'][i]}__event_id_{complete_table_tel_combined['event_id'][i]}"

            image_filename_pgm = f"dm-finder/data/{args.particle_type}/images/{output_filename}/float_alpha/obs_id_{complete_table_tel_combined['obs_id'][i]}/pgm/obs_id_{complete_table_tel_combined['obs_id'][i]}__event_id_{complete_table_tel_combined['event_id'][i]}"

            # create directory in which the float images will be saved
            path = f"dm-finder/data/{args.particle_type}/images/{output_filename}/float_alpha/obs_id_{complete_table_tel_combined['obs_id'][i]}/"
            os.makedirs(path, exist_ok = True)

            image_HDF = np.array([np.reshape(image, (48 * 48))])

            output_hdf_filename = path + f"obs_id_{complete_table_tel_combined['obs_id'][i]}__event_id_{complete_table_tel_combined['event_id'][i]}.h5"
            HDFfile = h5py.File(output_hdf_filename, "w")
            HDFfile.create_dataset("image", data = image_HDF)


            if run[r] == 10 and i <= 50:
                path_tif = f"dm-finder/data/{args.particle_type}/images/{output_filename}/float_alpha/obs_id_{complete_table['obs_id'][i]}/tif/"
                os.makedirs(path_tif, exist_ok = True)
                output_tif_filename = path_tif + f"obs_id_{complete_table['obs_id'][i]}__event_id_{complete_table['event_id'][i]}.tif"
                GetEventImageBasic(image, cmap = "Greys_r", show_frame = False, colorbar = True, clean_image = False, savefig = output_tif_filename)

            # implement image into table
            table["image"][i] = image

        path_cnn_input = f"dm-finder/cnn/iact_images/input/{args.particle_type}/"
        try:
            os.makedirs(path_cnn_input)
        except OSError:
            pass
        
        output_hdf_filename = f"dm-finder/cnn/iact_images/input/{args.particle_type}/" + run_filename + "_images_alpha"

        print(output_hdf_filename)

        table.to_hdf(output_hdf_filename + ".h5", key = 'events', mode = 'w', index = False)

        # close event file
        source.close()
