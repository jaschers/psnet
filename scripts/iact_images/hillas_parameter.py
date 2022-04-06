import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ctapipe.io import EventSource, read_table
from ctapipe.instrument import SubarrayDescription 
from ctapipe.visualization import CameraDisplay
from utilities import GetEventImage, GetEventImageBasic, GetEventImageBasicSmall, PlotHillasParametersDistribution
from astropy.table import Table, join, vstack
from astropy.io import ascii
import sys
import os
from tqdm import tqdm
import argparse
import h5py
import seaborn as sns

np.set_printoptions(threshold=sys.maxsize)
plt.rcParams.update({'font.size': 14})

######################################## argparse setup ########################################

script_version=0.1
script_descr="""
This script creates Cherenkov images of gamma/diffuse-gamma/proton events simulated for CTA. The images are saved in tif and pgm format and stored in a HDF table. One can choose between int8 and float64 images.
"""

# Open argument parser
parser = argparse.ArgumentParser(description=script_descr)
parser.add_argument("-v", "--version", action="version", version=f"v{script_version}")

# Define expected arguments
parser.add_argument("-pt", "--particle_type", type = str, metavar = "", choices = ["gamma", "gamma_diffuse", "proton"], help = "particle type [gamma, gamma_diffuse, proton], default: gamma", default = "gamma")
parser.add_argument("-er", "--energy_range", type = float, required = False, metavar = "-", help = "set energy range of events in TeV, default: 0.5 100", default = [0.5, 100], nargs = 2)
# parser.add_argument("-tm", "--telescope_mode", type = str, required = False, metavar = "", choices = ["mono", "stereo_sum_cta"], help = "telescope mode [mono, stereo_sum_cta], default: stereo_sum_cta", default = "stereo_sum_cta")
# parser.add_argument("-dt", "--data_type", type = str, required = False, metavar = "", choices = ["int8", "float64"], help = "data type of the output images [int8, float64], default: float64", default = "float64")
parser.add_argument("-r", "--run", type = int, metavar = "-", help = "input run(s) from which the CTA images will be extracted, default: csv list", action='append', nargs='+')

# parser.add_argument("-r", "--run_list", type = str, required = True, metavar = "", help = "path to the csv file that contains the run numbers")

args = parser.parse_args()
print(f"################### Input summary ################### \nParticle type: {args.particle_type}")
##########################################################################################

# ----------------------------------------------------------------------
# load and prepare data set
# ----------------------------------------------------------------------

# load data
filename_run_csv = f"dm-finder/scripts/run_lists/{args.particle_type}_run_list.csv"
run = pd.read_csv(filename_run_csv)
run = run.to_numpy().reshape(len(run))

if args.run != None:
    run = args.run[0]
    
print("Runs:", run)

tables = []
for r in range(len(run)): #len(run)
    print("Run", run[r])

    if args.particle_type == "gamma_diffuse":
        input_filename = f"gamma_20deg_0deg_run{run[r]}___cta-prod5-paranal_desert-2147m-Paranal-dark_cone10_merged.DL1"
    else:
        input_filename = f"{args.particle_type}_20deg_0deg_run{run[r]}___cta-prod5-paranal_desert-2147m-Paranal-dark_merged.DL1"
    input_directory = f"dm-finder/data/{args.particle_type}/event_files/" + input_filename + ".h5"

    input_filename = f"{args.particle_type}_20deg_0deg_run{run[r]}___cta-prod5-paranal_desert-2147m-Paranal-dark_merged.DL1"

    source = EventSource(input_directory)
    print(f"Total number of events: {len(source)}")

    # get telescope subarray description
    subarray = SubarrayDescription.from_hdf(input_directory)
    subarray.info()

    # get tables for all SST telescopes that include images and corresponding obs_id + event_id for each event
    sst_tel_id = np.append(range(30, 100), range(131, 181))
    images_table = vstack([read_table(input_directory, f"/dl1/event/telescope/images/tel_{t:03}") for t in sst_tel_id])
    parameters_table = vstack([read_table(input_directory, f"/dl1/event/telescope/parameters/tel_{t:03}") for t in sst_tel_id])

    # get true energy of each event
    simulated_parameter_table = read_table(input_directory, "/simulation/event/subarray/shower")

    # combine both tables and remove unnecessary columns
    combined_table = join(left = parameters_table, right = simulated_parameter_table, keys=["obs_id", "event_id"])
    # print(combined_table.info())
    combined_table.keep_columns(["obs_id", "event_id", "tel_id", "hillas_intensity", "hillas_length", "hillas_width", "hillas_skewness", "true_energy"])

    combined_table = combined_table.to_pandas()
    tables.append(combined_table)

    # (parameters_table.info("stats"))

full_table = pd.concat(tables)
full_table.reset_index(inplace = True)
full_table["hillas_ellipticity"] = full_table["hillas_length"] / full_table["hillas_width"]

full_table.replace([np.inf, -np.inf], np.nan, inplace=True)
# full_table = full_table.dropna().reset_index()

full_table.drop(full_table.loc[full_table["true_energy"] <= args.energy_range[0]].index, inplace=True)
full_table.drop(full_table.loc[full_table["true_energy"] >= args.energy_range[1]].index, inplace=True)
full_table.reset_index(inplace = True)

subset = full_table[["hillas_intensity", "hillas_length", "hillas_width", "hillas_skewness", "hillas_ellipticity", "true_energy"]] 

subset = subset.sort_values(by = "true_energy").reset_index()

# print(np.max(subset["hillas_ellipticity"]))

PlotHillasParametersDistribution(subset, "hillas_intensity", args.energy_range, 9, args.particle_type)

PlotHillasParametersDistribution(subset, "hillas_skewness", args.energy_range, 9, args.particle_type)

PlotHillasParametersDistribution(subset, "hillas_ellipticity", args.energy_range, 9, args.particle_type)