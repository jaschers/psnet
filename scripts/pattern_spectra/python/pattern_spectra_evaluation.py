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
Evaluates the pattern spectra pixel distribution for different energies
"""

# Open argument parser
parser = argparse.ArgumentParser(description=script_descr)
parser.add_argument("-v", "--version", action="version", version=f"v{script_version}")

# Define expected arguments
parser.add_argument("-pt", "--particle_type", type = str, metavar = "-", choices = ["gamma", "gamma_diffuse", "proton"], help = "particle type [gamma, gamma_diffuse, proton], default: gamma", default = "gamma")
parser.add_argument("-r", "--run", type = int, metavar = "-", help = "input run(s) for CNN, default: csv list", action="append", nargs="+")
parser.add_argument("-er", "--energy_range", type = float, required = False, metavar = "-", help = "set energy range of events in GeV, default: 0.02 300", default = [0.02, 300], nargs = 2)
parser.add_argument("-a", "--attribute", type = int, metavar = "-", choices = np.arange(1, 19, dtype = int), help = "attribute [0, 1 ... 18] (two required), default: 9 0", default = [9, 0], nargs = 2)
parser.add_argument("-dl", "--domain_lower", type = int, metavar = "-", help = "Granulometry: domain - start at <value> <value>, default: 0 0", default = [0, 0], nargs = 2)
parser.add_argument("-dh", "--domain_higher", type = int, metavar = "-", help = "Granulometry: domain - end at <value> <value>, default: 10 100000", default = [10, 100000], nargs = 2)
parser.add_argument("-m", "--mapper", type = int, metavar = "-", help = "Granulometry: use lambdamappers <mapper1> <mapper2>, default: 2 0", default = [2, 0], nargs = 2)
parser.add_argument("-n", "--size", type = int, metavar = "-", help = "Granulometry: size <n1>x<n2>, default: 20 20", default = [20, 20], nargs = 2)
parser.add_argument("-f", "--filter", type = int, metavar = "-", help = "Use decision <filter>, default: 3", default = 3, nargs = 1)

args = parser.parse_args()
##########################################################################################


######################################## Define some strings based on the input of the user ########################################
print(f"################### Input summary ################### \nParticle type: {args.particle_type} \nEnergy range: {args.energy_range} TeV \nAttribute: {args.attribute} \nDomain lower: {args.domain_lower} \nDomain higher: {args.domain_higher} \nMapper: {args.mapper} \nSize: {args.size} \nFilter: {args.filter} \n")
string_ps_input = f"a_{args.attribute[0]}_{args.attribute[1]}__dl_{args.domain_lower[0]}_{args.domain_lower[1]}__dh_{args.domain_higher[0]}_{args.domain_higher[1]}__m_{args.mapper[0]}_{args.mapper[1]}__n_{args.size[0]}_{args.size[1]}__f_{args.filter}/"
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
    input_filename = f"dm-finder/cnn/pattern_spectra/input/{args.particle_type}/" + string_ps_input + run_filename + "_ps.h5"

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
pattern_spectra = [[]] * len(table)
for i in range(len(table)):
    pattern_spectra[i] = table["pattern spectrum"][i]
pattern_spectra = np.asarray(pattern_spectra) 

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
pattern_spectra_binned = np.split(pattern_spectra, indices)

path = f"dm-finder/data/{args.particle_type}/info/pattern_spectra_distribution/" + f"/a_{args.attribute[0]}_{args.attribute[1]}__dl_{args.domain_lower[0]}_{args.domain_lower[1]}__dh_{args.domain_higher[0]}_{args.domain_higher[1]}__m_{args.mapper[0]}_{args.mapper[1]}__n_{args.size[0]}_{args.size[1]}__f_{args.filter}/" 
os.makedirs(path, exist_ok = True)

PlotPatternSpectra(number_energy_ranges, args.size, bins, pattern_spectra_binned, path)