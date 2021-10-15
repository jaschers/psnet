import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import pandas as pd
import os
import sys
import argparse
import warnings
from utilities import *
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import CSVLogger
from keras.models import Model
import logging

plt.rcParams.update({'font.size': 14})

######################################## argparse setup ########################################
script_version=0.1
script_descr="""
This script loads the output csv files from the CNN and plots the results.
"""

# Open argument parser
parser = argparse.ArgumentParser(description=script_descr)
parser.add_argument("-v", "--version", action="version", version=f"v{script_version}")

# Define expected arguments
parser.add_argument("-i", "--input", type = str, required = True, metavar = "-", choices = ["cta", "cta_int8", "ps"], help = "input for the CNN [cta, cta_int8, ps]", action='append', nargs='+')
parser.add_argument("-l", "--label", type = str, required = False, metavar = "-", help = "plotting label for the individual experiments", action='append', nargs='+')
parser.add_argument("-pt", "--particle_type", type = str, metavar = "-", choices = ["gamma", "gamma_diffuse", "proton"], help = "particle type [gamma, gamma_diffuse, proton], default: gamma", default = "gamma")
parser.add_argument("-er", "--energy_range", type = float, required = False, metavar = "-", help = "set energy range of events in GeV, default: 0.02 300", default = [0.02, 300], nargs = 2)
parser.add_argument("-na", "--name", type = str, required = False, metavar = "-", help = "Name of this particular experiment(s)", action='append', nargs='+')
parser.add_argument("-a", "--attribute", type = int, metavar = "-", choices = np.arange(1, 19, dtype = int), help = "attribute [0, 1 ... 18] (two required), default: 9 0", default = [9, 0], nargs = 2)
parser.add_argument("-dl", "--domain_lower", type = int, metavar = "-", help = "Granulometry: domain - start at <value> <value>, default: 0 0", default = [0, 0], nargs = 2)
parser.add_argument("-dh", "--domain_higher", type = int, metavar = "-", help = "Granulometry: domain - end at <value> <value>, default: 10 100000", default = [10, 100000], nargs = 2)
parser.add_argument("-m", "--mapper", type = int, metavar = "-", help = "Granulometry: use lambdamappers <mapper1> <mapper2>, default: 2 0", default = [2, 0], nargs = 2)
parser.add_argument("-n", "--size", type = int, metavar = "-", help = "Granulometry: size <n1>x<n2>, default: 20 20", default = [20, 20], nargs = 2)
parser.add_argument("-f", "--filter", type = int, metavar = "-", help = "Use decision <filter>, default: 3", default = 3, nargs = 1)

# parser.add_argument("-r", "--run_list", type = str, required = True, metavar = "-", help = "path to the csv file that contains the run numbers")

args = parser.parse_args()
##########################################################################################


######################################## Error messages and warnings ########################################
# if "ps" in args.input[0] and args.data_type == "int8":
#     raise ValueError("-i ps -dt int8 -> using pattern spectra from int8 CTA images is not supported")
##########################################################################################

######################################## Define some strings based on the input of the user ########################################
string_name = np.array([])
string_data_type = np.array([])
string_input = np.array([])
string_input_short = np.array([])
string_ps_input = np.array([])
string_table_column = np.array([])
string_summary = f"################### Input summary ################### \n"

for i in range(len(args.input[0])):
    if args.name != None and args.name[0][i] != "None":
        string_name = np.append(string_name, f"_{args.name[0][i]}")
    else:
        string_name = np.append(string_name, "")

    if args.input[0][i] == "cta":
        string_summary += f"\nInput: CTA \nParticle type: {args.particle_type} \nEnergy range: {args.energy_range} \n"
        string_input = np.append(string_input, "iact_images")
        string_input_short = np.append(string_input_short, "_images")
        string_ps_input = np.append(string_ps_input, "")
        string_table_column = np.append(string_table_column, "image")
        string_data_type = np.append(string_data_type, "")
    if args.input[0][i] == "cta_int8":
        string_summary += f"\nInput: CTA 8-bit \nParticle type: {args.particle_type} \nEnergy range: {args.energy_range} \n"
        string_input = np.append(string_input, "iact_images")
        string_input_short = np.append(string_input_short, "_images")
        string_ps_input = np.append(string_ps_input, "")
        string_table_column = np.append(string_table_column, "image")
        string_data_type = np.append(string_data_type, "_int8")
    if args.input[0][i] == "ps":
        string_summary += f"\nInput: pattern spectra \nParticle type: {args.particle_type} \nEnergy range: {args.energy_range} \nAttribute: {args.attribute} \nDomain lower: {args.domain_lower} \nDomain higher: {args.domain_higher} \nMapper: {args.mapper} \nSize: {args.size} \nFilter: {args.filter}\n"
        string_input = np.append(string_input, "pattern_spectra")
        string_input_short = np.append(string_input_short, "_ps")
        string_ps_input = np.append(string_ps_input, f"a_{args.attribute[0]}_{args.attribute[1]}__dl_{args.domain_lower[0]}_{args.domain_lower[1]}__dh_{args.domain_higher[0]}_{args.domain_higher[1]}__m_{args.mapper[0]}_{args.mapper[1]}__n_{args.size[0]}_{args.size[1]}__f_{args.filter}/")
        string_table_column = np.append(string_table_column, "pattern spectrum")
        string_data_type = np.append(string_data_type, "")

if args.label == None:
    label = args.input
else:
    label = args.label

print(string_summary)
##########################################################################################

median_all = [[]] * len(args.input[0])
sigma_all = [[]] * len(args.input[0])

for i in range(len(args.input[0])):
    # create folder
    os.makedirs(f"dm-finder/cnn/{string_input[i]}/results/" + string_ps_input[i] + f"{string_name[i][1:]}/", exist_ok = True)

    # load loss history file
    history_path = f"dm-finder/cnn/{string_input[i]}/history/" + string_ps_input[i] + "history" + string_data_type[i] + string_name[i] + ".csv"
    table_history = pd.read_csv(history_path)

    # plot loss history
    PlotLoss(table_history["epoch"], table_history["loss"], table_history["val_loss"], f"dm-finder/cnn/{string_input[i]}/results/" + string_ps_input[i] + f"{string_name[i][1:]}/" + "loss" + string_data_type[i] + string_name[i] + ".png")

    ####################################### Filters and feature maps ########################################
    # define exaple run and load example data
    run = 107
    X, Y = LoadExampleData(run, string_input[i], args.particle_type, string_ps_input[i], string_input_short[i], string_data_type[i], string_table_column[i])

    # load model
    model_path = f"dm-finder/cnn/{string_input[i]}/model/" + string_ps_input[i] + "model" + string_data_type[i] + string_name[i] + ".h5"
    model = keras.models.load_model(model_path)
    
    # plot filters
    PlotFilters(model, f"dm-finder/cnn/{string_input[i]}/results/" + string_ps_input[i] + f"{string_name[i][1:]}/" + "filters" + string_data_type[i] + string_name[i])

    # plot feature maps for an example image
    index_example = 39
    PlotFeatureMaps(X, model, index_example, f"dm-finder/cnn/{string_input[i]}/results/" + string_ps_input[i] + f"{string_name[i][1:]}/" + "feature_maps" + string_data_type[i] + string_name[i] )
    ##########################################################################################

    # load data file that contains E_true and E_rec from the test set
    filename_output = f"dm-finder/cnn/{string_input[i]}/output/" + string_ps_input[i] + "evaluation" + string_data_type[i] + string_name[i] + ".csv"

    table_output = pd.read_csv(filename_output)
    table_output = table_output.sort_values(by = ["log10(E_true / GeV)"])

    # convert energy to TeV
    energy_true = np.asarray((10**table_output["log10(E_true / GeV)"] * 1e-3))
    energy_rec = np.asarray((10**table_output["log10(E_rec / GeV)"] * 1e-3))

    # create 2D energy scattering plot
    PlotEnergyScattering2D(table_output["log10(E_true / GeV)"], table_output["log10(E_rec / GeV)"], f"dm-finder/cnn/{string_input[i]}/results/" + string_ps_input[i] + f"{string_name[i][1:]}/" + "energy_scattering_2D" + string_data_type[i] + string_name[i] + ".png")

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
    energy_rec_binned = np.split(energy_rec, indices)

    # calculate relative energy error (E_rec - E_true) / E_true and corresponding median and sigma
    relative_energy_error_toal = (energy_rec - energy_true) / energy_true #10**(energy_rec - energy_true) - 1
    median_total = np.median(relative_energy_error_toal)
    sigma_total = np.std(relative_energy_error_toal)
    
    # save relative energy error histogram (total)
    PlotRelativeEnergyError(relative_energy_error_toal, median_total, sigma_total, f"dm-finder/cnn/{string_input[i]}/results/" + string_ps_input[i] + f"{string_name[i][1:]}/" + "relative_energy_error" + string_data_type[i] + string_name[i] + ".png")

    # save relative energy error histogram (binned)
    PlotRelativeEnergyErrorBinned(energy_true_binned, energy_rec_binned, bins, f"dm-finder/cnn/{string_input[i]}/results/" + string_ps_input[i] + f"{string_name[i][1:]}/" + "energy_binned_histogram" + string_data_type[i] + string_name[i] + ".png")

    # save corrected relative energy error histogram (binned)
    PlotRelativeEnergyErrorBinnedCorrected(energy_true_binned, energy_rec_binned, bins, f"dm-finder/cnn/{string_input[i]}/results/" + string_ps_input[i] + f"{string_name[i][1:]}/" + "energy_binned_histogram_corrected" + string_data_type[i] + string_name[i] + ".png")

    # get median and sigma68 values (binned)
    median, sigma = MedianSigma68(energy_true_binned, energy_rec_binned, bins)

    # collect median and sigma values from each experiment
    median_all[i] = median
    sigma_all[i] = sigma

    # plot energy accuracy
    PlotEnergyAccuracy(median, bins, f"dm-finder/cnn/{string_input[i]}/results/" + string_ps_input[i] + f"{string_name[i][1:]}/" + "energy_accuracy" + string_data_type[i] + string_name[i] + ".png")
   
    # plot energy resolution
    PlotEnergyResolution(sigma, bins, f"dm-finder/cnn/{string_input[i]}/results/" + string_ps_input[i] + f"{string_name[i][1:]}/" + "energy_resolution" + string_data_type[i] + string_name[i] + ".png")

# if more than two inputs are given -> compare the results
if len(args.input[0]) > 1:
    os.makedirs(f"dm-finder/cnn/comparison/", exist_ok = True)

    string_comparison = ""
    for i in range(len(args.input[0])):
        string_comparison += args.input[0][i] + string_name[i]

    for i in range(len(args.input[0])):
        if args.input[0][i] == "ps":
            string_comparison += "_" + string_ps_input[i][:-1]
            break

    # plot energy accuracy comparison
    PlotEnergyAccuracyComparison(median_all, bins, label[0], f"dm-finder/cnn/comparison/" + "energy_accuracy_" + string_comparison + ".png")

    # plot energy resolution comparison
    PlotEnergyResolutionComparison(sigma_all, bins, label[0], f"dm-finder/cnn/comparison/" + "energy_resolution_" + string_comparison + ".png")

