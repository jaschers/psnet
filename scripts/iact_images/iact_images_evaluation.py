import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
import time
import argparse
from utilities import *

print("Packages successfully loaded")

######################################## argparse setup ########################################
script_version=0.1
script_descr="""
Evaluates the iact images pixel distribution for different energies
"""

# Open argument parser
parser = argparse.ArgumentParser(description=script_descr)
parser.add_argument("-v", "--version", action="version", version=f"v{script_version}")

# Define expected arguments
parser.add_argument("-pt", "--particle_type", type = str, metavar = "", choices = ["gamma", "gamma_diffuse", "proton"], help = "particle type [gamma, gamma_diffuse, proton], default: gamma", default = "gamma")
parser.add_argument("-dt", "--data_type", type = str, required = False, metavar = "", choices = ["int8", "float64"], help = "data type of the output images [int8, float64], default: float64", default = "float64")
parser.add_argument("-r", "--run", type = int, metavar = "-", help = "input run(s) for CNN, default: csv list", action="append", nargs="+")
parser.add_argument("-er", "--energy_range", type = float, required = False, metavar = "-", help = "set energy range of events in GeV, default: 0.02 300", default = [0.02, 300], nargs = 2)

# parser.add_argument("-r", "--run_list", type = str, required = True, metavar = "", help = "path to the csv file that contains the run numbers")

args = parser.parse_args()
print(f"################### Input summary ################### \nParticle type: {args.particle_type} \nEnergy range: {args.energy_range} TeV \n")
##########################################################################################

######################################## Load and prepare dataset ########################################
# import data
filename_run = f"dm-finder/scripts/run_lists/{args.particle_type}_run_list_old.csv"
run = pd.read_csv(filename_run)
run = run.to_numpy().reshape(len(run))

if args.run != None:
    run = args.run[0]

table = pd.DataFrame()
for r in range(len(run)): # len(run)
    run_filename = f"gamma_20deg_0deg_run{run[r]}___cta-prod5-paranal_desert-2147m-Paranal-dark_merged.DL1"
    input_filename = f"dm-finder/cnn/iact_images/input/{args.particle_type}/" + run_filename + "_images.h5"

    table_individual_run = pd.read_hdf(input_filename)
    print(f"Number of events in Run {run[r]}:", len(table_individual_run))
    table = table.append(table_individual_run, ignore_index = True)

print("Total number of events:", len(table))

table.drop(table.loc[table["true_energy"] <= args.energy_range[0] * 1e3].index, inplace=True)
table.drop(table.loc[table["true_energy"] >= args.energy_range[1] * 1e3].index, inplace=True)
table.reset_index(inplace = True)

print("Total number of events after energy cut:", len(table))

# sort table by energy
table = table.sort_values(by = ["true_energy"], ignore_index = True)

# input features
images = [[]] * len(table)
for i in range(len(table)):
    images[i] = table["image"][i]
images = np.asarray(images) 

# output label: true energy / TeV
energy_true = np.asarray(table["true_energy"]) * 1e-3

# prepare energy binning
number_energy_ranges = 9 # number of energy ranges the whole energy range will be splitted
sst_energy_min = args.energy_range[0] # TeV
sst_energy_max = args.energy_range[1] # TeV
bins = np.logspace(np.log10(np.min(sst_energy_min)), np.log10(np.max(sst_energy_max)), number_energy_ranges + 1) 
indices = np.array([], dtype = int)
for b in range(len(bins) - 2):
    index = np.max(np.where(energy_true < bins[b+1])) + 1
    indices = np.append(indices, index)

energy_true_binned = np.split(energy_true, indices)
images_binned = np.split(images, indices)

path = f"dm-finder/data/{args.particle_type}/info/iact_images_distribution/"
os.makedirs(path, exist_ok = True)

PlotImages(number_energy_ranges, np.shape(images)[1:], bins, images_binned, path)