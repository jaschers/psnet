import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from ctapipe.io import EventSource, read_table
from ctapipe.instrument import SubarrayDescription 
from ctapipe.visualization import CameraDisplay
from astropy.table import Table, join, vstack
import argparse
from tqdm import tqdm

######################################## argparse setup ########################################

script_version=0.1
script_descr="""
This script selects CTA images based on selection criteria on the Hillas intensity, the leakage2 or the multiplicity parameter. The table including the obs_id, event_id and tel_id is saved in cnn/selection_cuts/
"""

# Open argument parser
parser = argparse.ArgumentParser(description=script_descr)
parser.add_argument("-v", "--version", action="version", version=f"v{script_version}")

# Define expected arguments
parser.add_argument("-pt", "--particle_type", type = str, metavar = "", choices = ["gamma", "gamma_diffuse", "proton"], help = "particle type [gamma, gamma_diffuse, proton], default: gamma", default = "gamma")
parser.add_argument("-r", "--run", type = int, metavar = "-", help = "input run(s) from which the CTA images will be extracted, default: csv list", action='append', nargs='+')
parser.add_argument("-tm", "--telescope_mode", type = str, required = False, metavar = "", choices = ["mono", "stereo_sum_cta"], help = "telescope mode [mono, stereo_sum_cta], default: stereo_sum_cta", default = "stereo_sum_cta")
parser.add_argument("-er", "--energy_range", type = float, required = True, metavar = "-", help = "set energy range of events in TeV", nargs = 2)
parser.add_argument("-hi", "--hillas_intensity", type = int, required = False, metavar = "", help = "Events with Hillas intensities less than X will be rejected", default = 50)
parser.add_argument("-l2", "--leakage2", type = float, required = False, metavar = "", help = "Events with leakage2 more than X will be rejected", default = 0.2)
parser.add_argument("-mu", "--multiplicity", type = int, required = False, metavar = "", help = "Events with multiplicity less than X will be rejected", default = 4)

args = parser.parse_args()
print(f"################### Input summary ################### \nParticle type: {args.particle_type} \nTelescope mode: {args.telescope_mode}\nEnergy range: {args.energy_range}\nHillas intensity cut: {args.hillas_intensity}\nLeakage2 cut: {args.leakage2}")
##########################################################################################

# ----------------------------------------------------------------------
# load and prepare data set
# ----------------------------------------------------------------------

# load data
if args.telescope_mode == "mono" and args.particle_type == "gamma":
    filename_run_csv = f"main/run_lists/{args.particle_type}_run_list_mono_alpha.csv"
else: 
    filename_run_csv = f"main/run_lists/{args.particle_type}_run_list_alpha.csv"
run = pd.read_csv(filename_run_csv)
run = run.to_numpy().reshape(len(run))

if args.run != None:
    run = args.run[0]
    
print("Runs:", run)
print("Total number of Runs:", len(run))

sst_tel_id = np.array([30, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 133, 59, 61, 66, 67, 68, 69, 70, 71, 72, 73, 143, 144, 145, 146])

for r in tqdm(range(len(run))): #len(run)

    if args.particle_type == "gamma_diffuse":
        input_filename = f"gamma_20deg_0deg_run{run[r]}___cta-prod5-paranal_desert-2147m-Paranal-dark_cone10_merged.DL1"
    else:
        input_filename = f"{args.particle_type}_20deg_0deg_run{run[r]}___cta-prod5-paranal_desert-2147m-Paranal-dark_merged.DL1"
    input_directory = f"data/{args.particle_type}/event_files/" + input_filename + ".h5"

    source = EventSource(input_directory)

    images_table = vstack([read_table(input_directory, f"/dl1/event/telescope/images/tel_{t:03}") for t in sst_tel_id])

    parameters_table = vstack([read_table(input_directory, f"/dl1/event/telescope/parameters/tel_{t:03d}") for t in sst_tel_id])

    simulated_parameter_table = read_table(input_directory, "/simulation/event/subarray/shower")

    parameters_table.sort(["obs_id", "event_id"])
    merged_table = join(left = parameters_table, right = simulated_parameter_table, keys=["obs_id", "event_id"])
    merged_table.keep_columns(["obs_id", "event_id", "tel_id", "hillas_intensity", "leakage_intensity_width_2", "true_energy"])
    mask = (merged_table["true_energy"] >= args.energy_range[0]) & (merged_table["true_energy"] <= args.energy_range[1])
    merged_table = merged_table[mask]
    
    if args.telescope_mode == "stereo_sum_cta":
        merged_table.keep_columns(["obs_id", "event_id", "tel_id"])
        merged_table_uniques = np.unique(merged_table["obs_id", "event_id"], return_counts = True)

        table = pd.DataFrame(data = merged_table_uniques[0], columns=["obs_id", "event_id"]) #.astype({"obs_id": int, })
        table["multiplicity"] = merged_table_uniques[1]

        table_cut = table[table["multiplicity"] >= args.multiplicity]
        table_cut = table_cut.reset_index(drop = True)
        table_cut = table_cut[["obs_id", "event_id"]] 
    
        os.makedirs(f"cnn/selection_cuts/{args.particle_type}/mult{args.multiplicity}/", exist_ok = True)
        table_cut.to_csv(f"cnn/selection_cuts/{args.particle_type}/mult{args.multiplicity}/run{run[r]}.csv", index = False)
        # table_cut.close()

        source.close()
    
    if args.telescope_mode == "mono":
        mask = (parameters_table["leakage_intensity_width_2"] < args.leakage2) & (parameters_table["hillas_intensity"] > args.hillas_intensity)
        table_cut = parameters_table[mask][["obs_id", "event_id", "tel_id"]]
        table_cut = pd.DataFrame(np.array(table_cut))

        os.makedirs(f"cnn/selection_cuts/{args.particle_type}/l2{args.leakage2}_hi{args.hillas_intensity}/", exist_ok = True)
        table_cut.to_csv(f"cnn/selection_cuts/{args.particle_type}/l2{args.leakage2}_hi{args.hillas_intensity}/run{run[r]}.csv", index = False)

        # table_cut.close()

        source.close()
