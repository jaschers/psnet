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
Evaluates the pattern spectra pixel distribution for different energies and primary particles
"""

# Open argument parser
parser = argparse.ArgumentParser(description=script_descr)
parser.add_argument("-v", "--version", action="version", version=f"v{script_version}")

# Define expected arguments
parser.add_argument("-m", "--mode", type = str, required = True, metavar = "-", choices = ["energy", "separation"], help = "CNN mode - energy or gamma/proton separation [energy, separation]")
parser.add_argument("-pt", "--particle_type", type = str, metavar = "-", choices = ["gamma", "gamma_diffuse", "proton"], help = "particle type [gamma, gamma_diffuse, proton], default: gamma", default = "gamma")
parser.add_argument("-r", "--run", type = int, metavar = "-", help = "input run(s) for CNN, default: csv list", action="append", nargs="+")
parser.add_argument("-er", "--energy_range", type = float, required = False, metavar = "-", help = "set energy range of events in GeV, default: 0.02 300", default = [0.02, 300], nargs = 2)
parser.add_argument("-a", "--attribute", type = int, metavar = "-", choices = np.arange(0, 19, dtype = int), help = "attribute [0, 1 ... 18] (two required), default: 9 0", default = [9, 0], nargs = 2)
parser.add_argument("-dl", "--domain_lower", type = int, metavar = "-", help = "Granulometry: domain - start at <value> <value>, default: 0 0", default = [0, 0], nargs = 2)
parser.add_argument("-dh", "--domain_higher", type = int, metavar = "-", help = "Granulometry: domain - end at <value> <value>, default: 10 100000", default = [10, 100000], nargs = 2)
parser.add_argument("-ma", "--mapper", type = int, metavar = "-", help = "Granulometry: use lambdamappers <mapper1> <mapper2>, default: 2 0", default = [2, 0], nargs = 2)
parser.add_argument("-n", "--size", type = int, metavar = "-", help = "Granulometry: size <n1>x<n2>, default: 20 20", default = [20, 20], nargs = 2)
parser.add_argument("-f", "--filter", type = int, metavar = "-", help = "Use decision <filter>, default: 3", default = 3, nargs = 1)
parser.add_argument("-t", "--test", type = str, metavar = "-", help = "If yes, csv test list is used [y/n]", default = "n")


args = parser.parse_args()
##########################################################################################


######################################## Define some strings based on the input of the user ########################################
print(f"################### Input summary ################### \nMode: {args.mode} \nParticle type: {args.particle_type} \nEnergy range: {args.energy_range} TeV \nAttribute: {args.attribute} \nDomain lower: {args.domain_lower} \nDomain higher: {args.domain_higher} \nMapper: {args.mapper} \nSize: {args.size} \nFilter: {args.filter} \n")
string_ps_input = f"a_{args.attribute[0]}_{args.attribute[1]}__dl_{args.domain_lower[0]}_{args.domain_lower[1]}__dh_{args.domain_higher[0]}_{args.domain_higher[1]}__m_{args.mapper[0]}_{args.mapper[1]}__n_{args.size[0]}_{args.size[1]}__f_{args.filter}/"
##########################################################################################


######################################## Load and prepare dataset ########################################
# import data
if args.mode == "energy":
    filename_run = f"dm-finder/scripts/run_lists/{args.particle_type}_run_list.csv"
    if args.test == "y":
            filename_run = f"dm-finder/scripts/run_lists/{args.particle_type}_run_list_test.csv"
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

elif args.mode == "separation":
    particle_type = np.array(["gamma_diffuse", "proton"])

    table = pd.DataFrame()
    events_count = np.array([0, 0])
    for p in range(len(particle_type)):
        filename_run = f"dm-finder/scripts/run_lists/{particle_type[p]}_run_list.csv"
        if args.test == "y":
            filename_run = f"dm-finder/scripts/run_lists/{particle_type[p]}_run_list_test.csv"
        run = pd.read_csv(filename_run)
        run = run.to_numpy().reshape(len(run))

        for r in range(len(run)):
            run_filename = f"{particle_type[p]}_20deg_0deg_run{run[r]}___cta-prod5-paranal_desert-2147m-Paranal-dark_merged.DL1"
            input_filename = f"dm-finder/cnn/pattern_spectra/input/{particle_type[p]}/" + string_ps_input + run_filename + "_ps.h5"
            table_individual_run = pd.read_hdf(input_filename)
            print(f"Number of events in {particle_type[p]} Run {run[r]}:", len(table_individual_run))
            if (particle_type[p] == "gamma_diffuse") or particle_type[p] == "gamma":
                events_count[0] += len(table_individual_run)
            if particle_type[p] == "proton":
                events_count[1] += len(table_individual_run)
            table = table.append(table_individual_run, ignore_index = True)

    print("______________________________________________")
    print("Total number of gamma events:", events_count[0])
    print("Total number of proton events:", events_count[1])
    print("Total number of events:", len(table))

table.drop(table.loc[table["true_energy"] <= args.energy_range[0] * 1e3].index, inplace=True)
table.drop(table.loc[table["true_energy"] >= args.energy_range[1] * 1e3].index, inplace=True)
table.reset_index(inplace = True)

print("______________________________________________")
if args.mode == "separation":
    # shuffle data set
    table = table.sample(frac=1).reset_index(drop=True)
    print("Total number of gamma events after energy cut:", len(table.loc[table["particle"] == 1]))
    print("Total number of proton events after energy cut:", len(table.loc[table["particle"] == 0]))
print("Total number of events after energy cut:", len(table))

# sort table by energy
table = table.sort_values(by = ["true_energy"], ignore_index = True)

if args.mode == "energy":
    # collect pattern spectra
    pattern_spectra = [[]] * len(table)
    for i in range(len(table)):
        pattern_spectra[i] = table["pattern spectrum"][i]
    pattern_spectra = np.asarray(pattern_spectra) 

    # collect true energy
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

    # extract the normed sum of all pattern spectra in each specific energy bin
    pattern_spectra_mean = ExtractPatternSpectraMean(number_energy_ranges, args.size, pattern_spectra_binned)
    pattern_spectra_mean_min, pattern_spectra_mean_max = ExtractPatternSpectraMinMax(number_energy_ranges, pattern_spectra_mean)

    # subtract normed pattern spectrum of the first energy bin from all other normed pattern spectra of each other energy bin 
    pattern_spectra_mean_difference = ExtractPatternSpectraDifference(number_energy_ranges, args.size, pattern_spectra_mean)
    # extract the min and max value of pattern_spectra_mean_difference to set the colourbar equally 
    pattern_spectra_mean_difference_min, pattern_spectra_mean_difference_max = ExtractPatternSpectraMinMax(number_energy_ranges, pattern_spectra_mean_difference)
    # plot normed and difference pattern spectra
    PlotPatternSpectra(number_energy_ranges, pattern_spectra_mean, pattern_spectra_mean_min, pattern_spectra_mean_max, pattern_spectra_mean_difference, pattern_spectra_mean_difference_min, pattern_spectra_mean_difference_max, bins, args.particle_type, path)


elif args.mode == "separation":
    # for energy dependend comparison
    pattern_spectra_mean_gamma_proton = [[]] * len(particle_type)
    pattern_spectra_mean_min = [[]] * len(particle_type)
    pattern_spectra_mean_max = [[]] * len(particle_type)
    # for energy independend comparison
    pattern_spectra_total_mean_gamma_proton = np.zeros(shape = (len(particle_type), args.size[0], args.size[1]))
    for p in range(len(particle_type)):
        table_individual = table.loc[table["particle"] == p]
        table_individual = table_individual.reset_index(drop = True)
        pattern_spectra = [[]] * len(table_individual)
        for i in range(len(table_individual)):
            pattern_spectra[i] = table_individual["pattern spectrum"][i]
            pattern_spectra_total_mean_gamma_proton[p] = np.add(pattern_spectra_total_mean_gamma_proton[p], pattern_spectra[i])
        pattern_spectra = np.asarray(pattern_spectra)

        # normalise it to 1
        pattern_spectra_total_mean_gamma_proton[p] = pattern_spectra_total_mean_gamma_proton[p] / len(pattern_spectra)

        # collect true energy
        energy_true = np.asarray(table_individual["true_energy"]) * 1e-3

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

        path = f"dm-finder/data/{particle_type[p]}/info/pattern_spectra_distribution/" + f"/a_{args.attribute[0]}_{args.attribute[1]}__dl_{args.domain_lower[0]}_{args.domain_lower[1]}__dh_{args.domain_higher[0]}_{args.domain_higher[1]}__m_{args.mapper[0]}_{args.mapper[1]}__n_{args.size[0]}_{args.size[1]}__f_{args.filter}/" 
        os.makedirs(path, exist_ok = True)

        # extract the normed sum of all pattern spectra in each specific energy bin
        pattern_spectra_mean = ExtractPatternSpectraMean(number_energy_ranges, args.size, pattern_spectra_binned)
        pattern_spectra_mean_min[p], pattern_spectra_mean_max[p] = ExtractPatternSpectraMinMax(number_energy_ranges, pattern_spectra_mean)
        pattern_spectra_mean_gamma_proton[p] = pattern_spectra_mean
        # subtract normed pattern spectrum of the first energy bin from all other normed pattern spectra of each other energy bin 
        pattern_spectra_mean_difference = ExtractPatternSpectraDifference(number_energy_ranges, args.size, pattern_spectra_mean)
        # extract the min and max value of pattern_spectra_mean_difference to set the colourbar equally 
        pattern_spectra_mean_difference_min, pattern_spectra_mean_difference_max = ExtractPatternSpectraMinMax(number_energy_ranges, pattern_spectra_mean_difference)
        # plot normed and difference pattern spectra
        PlotPatternSpectra(number_energy_ranges, pattern_spectra_mean, pattern_spectra_mean_min[p], pattern_spectra_mean_max[p], pattern_spectra_mean_difference, pattern_spectra_mean_difference_min, pattern_spectra_mean_difference_max, bins, particle_type[p], path)

    PlotPatternSpectraComparisonTotal(pattern_spectra_total_mean_gamma_proton, particle_type, args.attribute, path)
    PlotPatternSpectraComparison(number_energy_ranges, pattern_spectra_mean_gamma_proton, bins, particle_type, path)