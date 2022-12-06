import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
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
import time

plt.rcParams.update({'font.size': 8}) # 8 (paper), 10 (poster)
# plt.rcParams.update({'font.family':'serif'}) #serif
# plt.rcParams["mathtext.fontset"] = 'dejavuserif' #dejavuserif
# pd.options.mode.chained_assignment = None 

plt.rc('text', usetex=True )
plt.rc('font', family='Times New Roman')#, weight='normal', size=14)
plt.rcParams['mathtext.fontset'] = 'cm'

######################################## argparse setup ########################################
script_version=0.1
script_descr="""
This script loads the output csv files from the CNN and plots the results.
"""

# Open argument parser
parser = argparse.ArgumentParser(description=script_descr)
parser.add_argument("-v", "--version", action="version", version=f"v{script_version}")

# Define expected arguments
parser.add_argument("-m", "--mode", type = str, required = True, metavar = "-", choices = ["energy", "separation"], help = "CNN mode - energy reconstruction or gamma/proton separation [energy, separation]")
parser.add_argument("-i", "--input", type = str, required = True, metavar = "-", choices = ["cta", "cta_int8", "ps"], help = "input for the CNN [cta, cta_int8, ps]", action='append', nargs='+')
parser.add_argument("-na", "--name", type = str, required = False, metavar = "-", help = "Name of this particular experiment(s)", action='append', nargs='+')
parser.add_argument("-l", "--label", type = str, required = False, metavar = "-", help = "plotting label for the individual experiments", action='append', nargs='+')
parser.add_argument("-pt", "--particle_type", type = str, metavar = "-", choices = ["gamma", "gamma_diffuse", "proton"], help = "particle type [gamma, gamma_diffuse, proton], default: gamma", default = "gamma")
parser.add_argument("-er", "--energy_range", type = float, required = False, metavar = "-", help = "set energy range of events in TeV, default: 0.5 100", default = [0.5, 100], nargs = 2)
parser.add_argument("-gl", "--gammaness_limit", type = float, required = False, metavar = "-", help = "separation: set min / max limit for reconstructed gammaness to investigate wrongly classified gamma/proton events [g_min (gamma), g_max (gamma), g_min (proton), g_max (proton)], default: 0.0 0.0 0.0 0.0", default = [0.0, 0.0, 0.0, 0.0], nargs = 4)
parser.add_argument("-s", "--suffix", type = str, required = False, metavar = "-", help = "suffix for the output filenames")
parser.add_argument("-a", "--attribute", type = int, metavar = "-", choices = np.arange(0, 19, dtype = int), help = "attribute [0, 1 ... 18] (two required), default: 9 0", default = [9, 0], nargs = 2)
parser.add_argument("-dl", "--domain_lower", type = float, metavar = "-", help = "Granulometry: domain - start at <value> <value>, default: 0.8 0.8", default = [0.8, 0.8], nargs = 2)
parser.add_argument("-dh", "--domain_higher", type = float, metavar = "-", help = "Granulometry: domain - end at <value> <value>, default: 7 3000", default = [7., 3000.], nargs = 2)
parser.add_argument("-ma", "--mapper", type = int, metavar = "-", help = "Granulometry: use lambdamappers <mapper1> <mapper2>, default: 4 4", default = [4, 4], nargs = 2)
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
        string_input_short = np.append(string_input_short, "_ps_float_alpha")
        string_ps_input = np.append(string_ps_input, f"a_{args.attribute[0]}_{args.attribute[1]}__dl_{args.domain_lower[0]}_{args.domain_lower[1]}__dh_{args.domain_higher[0]}_{args.domain_higher[1]}__m_{args.mapper[0]}_{args.mapper[1]}__n_{args.size[0]}_{args.size[1]}__f_{args.filter}/")
        string_table_column = np.append(string_table_column, "pattern spectrum")
        string_data_type = np.append(string_data_type, "")

if args.label == None:
    label = args.input
else:
    label = args.label

print(string_summary)
##########################################################################################

median_all, sigma_all = [[]] * len(args.input[0]), [[]] * len(args.input[0])

epochs_all, loss_train_all, loss_val_all, true_positive_rate_all, false_positive_rate_all, true_negative_rate_all, false_negative_rate_all, area_under_ROC_curve_all, accuracy_gammaness_all, precision_gammaness_all, thresholds_all, threshold_cut_all, area_under_ROC_curve_energy_all, accuracy_energy_all, true_positive_rate_energy_all, false_positive_rate_energy_all = [[]] * len(args.input[0]), [[]] * len(args.input[0]), [[]] * len(args.input[0]), [[]] * len(args.input[0]), [[]] * len(args.input[0]), [[]] * len(args.input[0]), [[]] * len(args.input[0]), [[]] * len(args.input[0]), [[]] * len(args.input[0]), [[]] * len(args.input[0]), [[]] * len(args.input[0]), [[]] * len(args.input[0]), [[]] * len(args.input[0]), [[]] * len(args.input[0]), [[]] * len(args.input[0]), [[]] * len(args.input[0])

for i in range(len(args.input[0])):
    print(f"Processing input {i+1}...")
    # create folder
    os.makedirs(f"dm-finder/cnn/{string_input[i]}/{args.mode}/results/" + string_ps_input[i] + f"{string_name[i][1:]}/", exist_ok = True)

    # load loss history file
    history_path = f"dm-finder/cnn/{string_input[i]}/{args.mode}/history/" + string_ps_input[i] + "history" + string_data_type[i] + string_name[i] + ".csv"
    table_history = pd.read_csv(history_path)

    # plot loss history
    PlotLoss(table_history["epoch"], table_history["loss"], table_history["val_loss"], f"dm-finder/cnn/{string_input[i]}/{args.mode}/results/" + string_ps_input[i] + f"{string_name[i][1:]}/" + "loss.pdf")

    epochs_all[i] = table_history["epoch"] + 1
    loss_train_all[i] = table_history["loss"]
    loss_val_all[i] = table_history["val_loss"]

    ####################################### Filters and feature maps ########################################
    # define example run and load example data
    # run = 107
    # X, Y = LoadExampleData(run, string_input[i], args.particle_type, string_ps_input[i], string_input_short[i], string_data_type[i], string_table_column[i])

    # load model
    model_path = f"dm-finder/cnn/{string_input[i]}/{args.mode}/model/" + string_ps_input[i] + "model" + string_data_type[i] + string_name[i] + ".h5"
    model = keras.models.load_model(model_path)
    
    # plot filters
    # PlotFilters(model, f"dm-finder/cnn/{string_input[i]}/{args.mode}/results/" + string_ps_input[i] + f"{string_name[i][1:]}/" + "filters")

    # # plot feature maps for an example image
    # index_example = 39
    # PlotFeatureMaps(X, model, index_example, f"dm-finder/cnn/{string_input[i]}/{args.mode}/results/" + string_ps_input[i] + f"{string_name[i][1:]}/" + "feature_maps")
    ##########################################################################################

    # load data file that contains E_true and E_rec from the test set
    filename_output = f"dm-finder/cnn/{string_input[i]}/{args.mode}/output/" + string_ps_input[i] + "evaluation" + string_data_type[i] + string_name[i] + ".csv"

    table_output = pd.read_csv(filename_output)

    if args.mode == "energy":
        table_output = table_output.sort_values(by = ["log10(E_true / GeV)"])

        # convert energy to TeV
        energy_true = np.asarray((10**table_output["log10(E_true / GeV)"] * 1e-3))
        energy_rec = np.asarray((10**table_output["log10(E_rec / GeV)"] * 1e-3))

        # create 2D energy scattering plot
        PlotEnergyScattering2D(np.log10(energy_true), np.log10(energy_rec), f"dm-finder/cnn/{string_input[i]}/{args.mode}/results/" + string_ps_input[i] + f"{string_name[i][1:]}/" + "energy_scattering_2D.pdf")

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
        PlotRelativeEnergyError(relative_energy_error_toal, median_total, sigma_total, f"dm-finder/cnn/{string_input[i]}/{args.mode}/results/" + string_ps_input[i] + f"{string_name[i][1:]}/" + "relative_energy_error.pdf")

        # save relative energy error histogram (binned)
        PlotRelativeEnergyErrorBinned(energy_true_binned, energy_rec_binned, bins, f"dm-finder/cnn/{string_input[i]}/{args.mode}/results/" + string_ps_input[i] + f"{string_name[i][1:]}/" + "energy_binned_histogram.pdf")

        # save corrected relative energy error histogram (binned)
        PlotRelativeEnergyErrorBinnedCorrected(energy_true_binned, energy_rec_binned, bins, f"dm-finder/cnn/{string_input[i]}/{args.mode}/results/" + string_ps_input[i] + f"{string_name[i][1:]}/" + "energy_binned_histogram_corrected.pdf")

        # get median and sigma68 values (binned)
        median, sigma = MedianSigma68(energy_true_binned, energy_rec_binned, bins)

        # collect median and sigma values from each experiment
        median_all[i] = median
        sigma_all[i] = sigma

        # plot energy accuracy
        PlotEnergyAccuracy(median, bins, f"dm-finder/cnn/{string_input[i]}/{args.mode}/results/" + string_ps_input[i] + f"{string_name[i][1:]}/" + "energy_accuracy.pdf")
    
        # plot energy resolution
        PlotEnergyResolution(sigma, bins, f"dm-finder/cnn/{string_input[i]}/{args.mode}/results/" + string_ps_input[i] + f"{string_name[i][1:]}/" + "energy_resolution.pdf")

        # save energy accuracy & resolution in csv files
        SaveCSV(median, bins, "accuracy", f"dm-finder/cnn/{string_input[i]}/{args.mode}/results/" + string_ps_input[i] + f"{string_name[i][1:]}/" + "energy_accuracy.csv")
        SaveCSV(sigma, bins, "resolution", f"dm-finder/cnn/{string_input[i]}/{args.mode}/results/" + string_ps_input[i] + f"{string_name[i][1:]}/" + "energy_resolution.csv")
        
    
    elif args.mode == "separation":
        particle_type = np.array(["gamma_diffuse", "proton"])
        gammaness_true = np.asarray(table_output["true gammaness"])
        gammaness_rec = np.asarray(table_output["reconstructed gammaness"])
        energy_true = np.asarray(table_output["E_true / GeV"])

        PlotGammaness(gammaness_true, gammaness_rec, f"dm-finder/cnn/{string_input[i]}/separation/results/{string_ps_input[i]}/{string_name[i][1:]}/gammaness.pdf")

        # perform an energy dependend analysis of accuracy, AUC and gammaness
        bins, bins_central, area_under_ROC_curve_energy, accuracy_energy, true_positive_rate_energy, false_positive_rate_energy = PlotGammanessEnergyBinned(table_output, args.energy_range, f"dm-finder/cnn/{string_input[i]}/separation/results/{string_ps_input[i]}/{string_name[i][1:]}/")

        true_positive_rate, false_positive_rate, true_negative_rate, false_negative_rate, rejection_power, area_under_ROC_curve = ROC(gammaness_true, gammaness_rec)

        PlotROC(true_positive_rate, false_positive_rate, area_under_ROC_curve, f"dm-finder/cnn/{string_input[i]}/separation/results/{string_ps_input[i]}/{string_name[i][1:]}/roc.pdf")

        SaveROC(true_positive_rate, false_positive_rate, area_under_ROC_curve, f"dm-finder/cnn/{string_input[i]}/separation/results/{string_ps_input[i]}/{string_name[i][1:]}/roc.csv")

        # prepare accuracy vs gammaness threshold plot
        accuracy_gammaness, thresholds = AccuracyGammaness(gammaness_true, gammaness_rec)

        # plot accuracy vs gammaness threshold plot
        PlotAccuracyGammaness(accuracy_gammaness, thresholds, f"dm-finder/cnn/{string_input[i]}/separation/results/{string_ps_input[i]}/{string_name[i][1:]}/accuracy_gammaness.pdf")

        # prepare precision vs gammaness treshold plot
        precision_gammaness, threshold_cut = PrecisionGammaness(gammaness_true, gammaness_rec)

        # plot precision vs gammaness treshold plot
        PlotPrecisionGammaness(precision_gammaness, thresholds[:threshold_cut], f"dm-finder/cnn/{string_input[i]}/separation/results/{string_ps_input[i]}/{string_name[i][1:]}/precision_gammaness.pdf")

        PlotPurityGammaness(true_positive_rate, false_positive_rate, rejection_power, thresholds, f"dm-finder/cnn/{string_input[i]}/separation/results/{string_ps_input[i]}/{string_name[i][1:]}/purity_gammaness.pdf")

        true_positive_rate_all[i] = true_positive_rate
        false_positive_rate_all[i] = false_positive_rate
        true_negative_rate_all[i] = true_negative_rate
        false_negative_rate_all[i] = false_negative_rate
        area_under_ROC_curve_all[i] = area_under_ROC_curve
        accuracy_gammaness_all[i] = accuracy_gammaness
        precision_gammaness_all[i] = precision_gammaness
        threshold_cut_all[i] = thresholds[:threshold_cut]
        thresholds_all[i] = thresholds
        area_under_ROC_curve_energy_all[i] = area_under_ROC_curve_energy
        accuracy_energy_all[i] = accuracy_energy 
        true_positive_rate_energy_all[i] = true_positive_rate_energy
        false_positive_rate_energy_all[i] = false_positive_rate_energy

        # Plot wrongly classified CTA images / pattern spectra
        if args.gammaness_limit != [0.0, 0.0, 0.0, 0.0]:
            # load CTA images or pattern spectra into the table_output 
            table_output = ExtendTable(table_output, string_table_column[i], string_input[i], string_ps_input[i], string_input_short[i], string_data_type[i])

            # Plot 15 examples of wrongly classified events
            PlotWronglyClassifiedEvents(table_output, particle_type, string_table_column[i], args.gammaness_limit, f"dm-finder/cnn/{string_input[i]}/separation/results/{string_ps_input[i]}/{string_name[i][1:]}/misclassified_examples")

            # get normed sum pattern spectra
            pattern_spectra_mean_gammaness_limit = ExtractPatternSpectraMean(table_output, particle_type, args.size, args.gammaness_limit, string_table_column[i])
            # get correctly and wrongly classified gamma events
            pattern_spectra_mean_gammaness_limit_gamma = ExtractPatternSpectraMeanGamma(table_output, args.size, [0.9, 1.0, 0.0, 0.1], string_table_column[i])
            # get correctly and wrongly classified proton events
            pattern_spectra_mean_gammaness_limit_proton = ExtractPatternSpectraMeanProton(table_output, args.size, [0.0, 0.1, 0.9, 1.0], string_table_column[i])
            # create own colour map
            N = 256
            vals = np.ones((N, 4))
            vals[:, 0] = np.linspace(cstm_RdBu(6)[0], cstm_PuBu(12)[0], N)
            vals[:, 1] = np.linspace(cstm_RdBu(6)[1], cstm_PuBu(12)[1], N)
            vals[:, 2] = np.linspace(cstm_RdBu(6)[2], cstm_PuBu(12)[2], N)
            newcmp = ListedColormap(vals)
            # plot normed sum of pattern spectra of missclassified events
            PlotPatternSpectraMean(pattern_spectra_mean_gammaness_limit, particle_type, args.attribute, args.gammaness_limit, newcmp, f"dm-finder/cnn/{string_input[i]}/separation/results/{string_ps_input[i]}/{string_name[i][1:]}/misclassified_sum")

            # plot pattern spectra difference (gamma - proton) of missclassified events
            PlotPatternSpectraDifference(pattern_spectra_mean_gammaness_limit, particle_type, args.attribute, args.gammaness_limit, f"dm-finder/cnn/{string_input[i]}/separation/results/{string_ps_input[i]}/{string_name[i][1:]}/misclassified_difference")
            # plot pattern spectra difference of correctly and wrongly classified gamma events
            PlotPatternSpectraDifference(pattern_spectra_mean_gammaness_limit_gamma, ["gamma_diffuse", "gamma_diffuse"], args.attribute, [0.9, 1.0, 0.0, 0.1], f"dm-finder/cnn/{string_input[i]}/separation/results/{string_ps_input[i]}/{string_name[i][1:]}/correct_wrong_gamma_difference")
            # plot pattern spectra difference of correctly and wrongly classified proton events
            PlotPatternSpectraDifference(pattern_spectra_mean_gammaness_limit_proton, ["proton", "proton"], args.attribute, [0.0, 0.1, 0.9, 1.0], f"dm-finder/cnn/{string_input[i]}/separation/results/{string_ps_input[i]}/{string_name[i][1:]}/correct_wrong_proton_difference")


if args.mode == "energy":
    # if more than two inputs are given -> compare the results
    if len(args.input[0]) > 1:
        os.makedirs(f"dm-finder/cnn/comparison/energy", exist_ok = True)

        if args.suffix == None:
            string_comparison = ""
            for i in range(len(args.input[0])):
                string_comparison += args.input[0][i] + string_name[i] + "_"

            for i in range(len(args.input[0])):
                if args.input[0][i] == "ps":
                    string_comparison += "_" + string_ps_input[i][:-1]
                    break
            
            if len(string_comparison) > 200:
                string_comparison = string_comparison[:200]
        else:
            string_comparison = args.suffix
            print(string_comparison)

        
        if len(string_comparison) > 200:
            string_comparison = string_comparison[:200]
        
        # plot Loss comparison
        PlotLossComparison(epochs_all, loss_train_all, loss_val_all, args.input[0], f"dm-finder/cnn/comparison/energy/" + "loss_comparison_" + string_comparison + ".pdf")

        # plot energy accuracy comparison
        PlotEnergyAccuracyComparison(median_all, bins, label[0], f"dm-finder/cnn/comparison/energy/" + "energy_accuracy_" + string_comparison + ".pdf")

        PlotEnergyAccuracyComparisonMean(median_all, bins, label[0], args.input[0], f"dm-finder/cnn/comparison/energy/" + "energy_accuracy_mean_" + string_comparison + ".pdf")

        # plot energy resolution comparison
        PlotEnergyResolutionComparison(sigma_all, bins, label[0], f"dm-finder/cnn/comparison/energy/" + "energy_resolution_" + string_comparison + ".pdf")

        # plot mean energy resolution
        PlotEnergyResolutionComparisonMean(args.input[0], sigma_all, bins, label[0], f"dm-finder/cnn/comparison/energy/" + "energy_resolution_mean_" + string_comparison + ".pdf")


if (args.mode == "separation") and (len(args.input[0]) > 1):
    os.makedirs(f"dm-finder/cnn/comparison/separation", exist_ok = True)

    string_comparison = ""
    for i in range(len(args.input[0])):
        string_comparison += args.input[0][i] + string_name[i] + "_"

    for i in range(len(args.input[0])):
        if args.input[0][i] == "ps":
            string_comparison += "_" + string_ps_input[i][:-1]
            break
    
    if len(string_comparison) > 200:
        string_comparison = string_comparison[:200]

    PlotLossComparison(epochs_all, loss_train_all, loss_val_all, args.input[0], f"dm-finder/cnn/comparison/separation/" + "loss_comparison_" + string_comparison + ".pdf")

    if len(args.input[0]) > 3:
        PlotROCComparison(true_positive_rate_all, false_positive_rate_all, area_under_ROC_curve_all, args.input[0], f"dm-finder/cnn/comparison/separation/" + "ROC_comparison_" + string_comparison + ".pdf")

    PlotAccuracyGammanessComparison(accuracy_gammaness_all, thresholds_all, args.input[0], f"dm-finder/cnn/comparison/separation/" + "accuracy_gammaness_comparison_" + string_comparison + ".pdf")

    PlotPrecisionGammanessComparison(precision_gammaness_all, threshold_cut_all, args.input[0], f"dm-finder/cnn/comparison/separation/" + "precision_gammaness_comparison_" + string_comparison + ".pdf")

    PlotPurityGammanessComparison(thresholds_all, true_positive_rate_all, false_positive_rate_all, args.input[0], f"dm-finder/cnn/comparison/separation/" + "purity_gammaness_comparison_" + string_comparison + ".pdf")

    PlotAUCEnergyComparison(bins, bins_central, area_under_ROC_curve_energy_all, args.input[0], f"dm-finder/cnn/comparison/separation/" + "AUC_energy_comparison_" + string_comparison + ".pdf")

    PlotAccuracyEnergyComparison(bins, bins_central, accuracy_energy_all, args.input[0], f"dm-finder/cnn/comparison/separation/" + "accuracy_energy_comparison_" + string_comparison + ".pdf")

    PlotEfficiencyEnergyComparison(bins, bins_central, true_positive_rate_energy_all, false_positive_rate_energy_all, args.input[0], f"dm-finder/cnn/comparison/separation/" + "efficiency_energy_comparison_" + string_comparison + ".pdf")

    MeanStdAUC(area_under_ROC_curve_all, args.input[0])


print("CNN evaluation completed!")