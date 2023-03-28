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
from ctapipe.io import EventSource, read_table
from astropy.table import Table, join, vstack
from pyirf.simulations import SimulatedEventsInfo
import astropy.units as u

# do not print tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

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
Evaluation of the CNN performance on the test data
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
parser.add_argument("-erg", "--energy_range_gamma", type = float, required = False, metavar = "-", help = "set energy range of events in TeV, default: 1.5 100", default = [1.5, 100], nargs = 2)
parser.add_argument("-erp", "--energy_range_proton", type = float, required = False, metavar = "-", help = "set energy range of events in TeV, default: 1.5 100", default = [1.5, 100], nargs = 2)
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
        string_summary += f"\nInput: CTA \nParticle type: {args.particle_type} \nEnergy range (energy reco): {args.energy_range} \nEnergy range gamma (sig-bkg sep): {args.energy_range_gamma} \nEnergy range proton (sig-bkg sep): {args.energy_range_proton} \n"
        string_input = np.append(string_input, "iact_images")
        string_input_short = np.append(string_input_short, "_images")
        string_ps_input = np.append(string_ps_input, "")
        string_table_column = np.append(string_table_column, "image")
        string_data_type = np.append(string_data_type, "")
    if args.input[0][i] == "cta_int8":
        string_summary += f"\nInput: CTA 8-bit \nParticle type: {args.particle_type} \nEnergy range (energy reco): {args.energy_range} \nEnergy range gamma (sig-bkg sep): {args.energy_range_gamma} \nEnergy range proton (sig-bkg sep): {args.energy_range_proton} \n"
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

epochs_all, loss_train_all, loss_val_all, true_positive_rate_all, false_positive_rate_all, true_negative_rate_all, false_negative_rate_all,  area_under_ROC_curve_all, accuracy_gammaness_all, precision_gammaness_all, thresholds_all, threshold_cut_all, area_under_ROC_curve_energy_all, accuracy_energy_all, true_positive_rate_energy_all, false_positive_rate_energy_all, true_positive_rate_fixed_eta_all, false_positive_rate_fixed_eta_all = [[]] * len(args.input[0]), [[]] * len(args.input[0]), [[]] * len(args.input[0]), [[]] * len(args.input[0]), [[]] * len(args.input[0]), [[]] * len(args.input[0]), [[]] * len(args.input[0]), [[]] * len(args.input[0]), [[]] * len(args.input[0]), [[]] * len(args.input[0]), [[]] * len(args.input[0]), [[]] * len(args.input[0]), [[]] * len(args.input[0]), [[]] * len(args.input[0]), [[]] * len(args.input[0]), [[]] * len(args.input[0]), [[]] * len(args.input[0]), [[]] * len(args.input[0])

tpr_fixed_gammaness_cta_all, fpr_fixed_gammaness_cta_all = [], []
tpr_ps_proton_efficiency_fixed_to_cta_all, fpr_ps_proton_efficiency_fixed_to_cta_all = [], []

if args.mode == "separation":
    dl0_gamma = Table()
    dl0_proton = Table()
    # load simulation info which is necesarry to calculate signal and bkg efficiencies
    filename_run_gamma_diffuse = f"dm-finder/scripts/run_lists/gamma_diffuse_run_list_alpha.csv"
    filename_run_proton = f"dm-finder/scripts/run_lists/proton_run_list_alpha.csv"

    run_gamma = pd.read_csv(filename_run_gamma_diffuse)
    run_gamma = run_gamma.to_numpy().reshape(len(run_gamma))
    run_proton = pd.read_csv(filename_run_proton)
    run_proton = run_proton.to_numpy().reshape(len(run_proton))

    print("loading gamma simulation information...")
    for r in tqdm(range(len(run_gamma))):
        dl1_filename_gamma_diffuse = f"gamma_20deg_0deg_run{run_gamma[r]}___cta-prod5-paranal_desert-2147m-Paranal-dark_cone10_merged.DL1"
        dl1_directory_gamma_diffuse = f"dm-finder/data/gamma_diffuse/event_files/" + dl1_filename_gamma_diffuse + ".h5"
        dl0_gamma_temp = read_table(dl1_directory_gamma_diffuse, "/configuration/simulation/run")
        dl0_gamma = vstack([dl0_gamma, dl0_gamma_temp])
        # break
        

    print("loading proton simulation information...")
    for r in tqdm(range(len(run_proton))):
        dl1_filename_proton = f"proton_20deg_0deg_run{run_proton[r]}___cta-prod5-paranal_desert-2147m-Paranal-dark_merged.DL1"
        dl1_directory_proton = f"dm-finder/data/proton/event_files/" + dl1_filename_proton + ".h5"
        dl0_proton_temp = read_table(dl1_directory_proton, "/configuration/simulation/run")
        dl0_proton = vstack([dl0_proton, dl0_proton_temp])
        # break
        

    if len(np.unique(dl0_gamma["energy_range_min"])) > 1:
        print("WARNING: min energy range is not the same for all obs_ids (gamma_diffuse)")
    if len(np.unique(dl0_gamma["energy_range_max"])) > 1:
        print("WARNING: max energy range is not the same for all obs_ids (gamma_diffuse)")
    if len(np.unique(dl0_gamma["spectral_index"])) > 1:
        print("WARNING: spectral index is not the same for all obs_ids (gamma_diffuse)")
    if len(np.unique(dl0_gamma["max_scatter_range"])) > 1:
        print("WARNING: max scatter range is not the same for all obs_ids (gamma_diffuse)")
    if len(np.unique(dl0_gamma["max_viewcone_radius"])) > 1:
        print("WARNING: max viewcone radius is not the same for all obs_ids (gamma_diffuse)")

    if len(np.unique(dl0_gamma["energy_range_min"])) > 1:
        print("WARNING: min energy range is not the same for all obs_ids (proton)")
    if len(np.unique(dl0_gamma["energy_range_max"])) > 1:
        print("WARNING: max energy range is not the same for all obs_ids (proton)")
    if len(np.unique(dl0_gamma["spectral_index"])) > 1:
        print("WARNING: spectral index is not the same for all obs_ids (proton)")
    if len(np.unique(dl0_gamma["max_scatter_range"])) > 1:
        print("WARNING: max scatter range is not the same for all obs_ids (proton)")
    if len(np.unique(dl0_gamma["max_viewcone_radius"])) > 1:
        print("WARNING: max viewcone radius is not the same for all obs_ids (proton)")

    percentage_test_data = 0.1
    n_showers_total_gamma = np.sum(dl0_gamma["num_showers"] * dl0_gamma["shower_reuse"]) * percentage_test_data
    n_showers_total_proton = np.sum(dl0_proton["num_showers"] * dl0_proton["shower_reuse"]) * percentage_test_data

    print(n_showers_total_gamma)
    print(n_showers_total_proton)

    # use 
    simulation_info_gamma = SimulatedEventsInfo(
    energy_min=dl0_gamma["energy_range_min"][0] * dl0_gamma["energy_range_min"].unit, # energy_range_min
    energy_max=dl0_gamma["energy_range_max"][0] * dl0_gamma["energy_range_max"].unit, # energy_range_max
    spectral_index=dl0_gamma["spectral_index"][0], # spectral_index
    n_showers=n_showers_total_gamma,
    max_impact= dl0_gamma["max_scatter_range"][0] * dl0_gamma["max_scatter_range"].unit, # max_scatter_range
    viewcone=dl0_gamma["max_viewcone_radius"][0] * dl0_gamma["max_viewcone_radius"].unit, # max_viewcone_radius
    )

    simulation_info_proton = SimulatedEventsInfo(
    energy_min=dl0_proton["energy_range_min"][0] * dl0_proton["energy_range_min"].unit, # energy_range_min
    energy_max=dl0_proton["energy_range_max"][0] * dl0_proton["energy_range_max"].unit, # energy_range_max
    spectral_index=dl0_proton["spectral_index"][0], # spectral_index
    n_showers=n_showers_total_proton,
    max_impact= dl0_proton["max_scatter_range"][0] * dl0_proton["max_scatter_range"].unit, # max_scatter_range
    viewcone=dl0_proton["max_viewcone_radius"][0] * dl0_proton["max_viewcone_radius"].unit, # max_viewcone_radius
    )

    bins = np.logspace(np.log10(args.energy_range_gamma[0]), np.log10(args.energy_range_gamma[1]), 10) * u.TeV
    bins_width = (bins[1:] - bins[:-1])
    bins_central =  bins[:-1] + bins_width / 2

    dl0_gamma_hist = simulation_info_gamma.calculate_n_showers_per_energy(bins) #* 500 # temp solution
    dl0_proton_hist = simulation_info_proton.calculate_n_showers_per_energy(bins) #* 1000 # temp solution

    # plt.figure()
    # plt.bar(bins_central, dl0_gamma_hist, width = bins_width, label = "gamma", alpha = 0.5)
    # plt.bar(bins_central, dl0_proton_hist, width = bins_width, label = "proton", alpha = 0.5)
    # plt.xscale("log")
    # plt.legend()
    # plt.show()


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
        # bins, bins_central, area_under_ROC_curve_energy, accuracy_energy, true_positive_rate_energy, false_positive_rate_energy = PlotGammanessEnergyBinned(table_output, args.energy_range, f"dm-finder/cnn/{string_input[i]}/separation/results/{string_ps_input[i]}/{string_name[i][1:]}/")
        PlotGammanessEnergyBinned(table_output, bins, bins_central, f"dm-finder/cnn/{string_input[i]}/separation/results/{string_ps_input[i]}/{string_name[i][1:]}/")

        PlotDL0DL1Hist(table_output, bins, bins_central, bins_width, dl0_gamma_hist, dl0_proton_hist, f"dm-finder/cnn/{string_input[i]}/separation/results/{string_ps_input[i]}/{string_name[i][1:]}/")

        # perform an gamma & proton energy dependend analysis of accuracy, AUC and gammaness
        true_positive_rate_fixed_eta, false_positive_raten_fixed_eta = GetEfficienciesEnergyBinnedFixedBackground(table_output, bins, bins_central, dl0_gamma_hist, dl0_proton_hist, 0.001)

        # get AUC energy binned based on proton energy
        area_under_ROC_curve_energy = GetAUCEnergyBinned(table_output, args.energy_range_proton)

        # if args.input[0][i] == "cta":
        #     tpr_fixed_gammaness_cta, fpr_fixed_gammaness_cta = GetEfficienciesEnergyBinnedFixedGammaness(table_output, args.energy_range_gamma, args.energy_range_proton, 0.8) #0.5075
        #     PlotEfficienciesEnergyBinned(bins_gamma, bins_proton, bins_central_gamma, bins_central_proton, tpr_fixed_gammaness_cta, fpr_fixed_gammaness_cta, f"dm-finder/cnn/{string_input[i]}/separation/results/{string_ps_input[i]}/{string_name[i][1:]}/efficiencies_fixed_gammaness_energy.pdf")
        # if (args.input[0][i] == "ps") and ("cta" in args.input[0]):
        #     tpr_ps_proton_efficiency_fixed_to_cta, fpr_ps_proton_efficiency_fixed_to_cta = GetEfficienciesEnergyBinnedFixedProtonEfficiency(table_output, args.energy_range_gamma, args.energy_range_proton, fpr_fixed_gammaness_cta_mean)
        #     PlotEfficienciesEnergyBinned(bins_gamma, bins_proton, bins_central_gamma, bins_central_proton, tpr_ps_proton_efficiency_fixed_to_cta, fpr_ps_proton_efficiency_fixed_to_cta, f"dm-finder/cnn/{string_input[i]}/separation/results/{string_ps_input[i]}/{string_name[i][1:]}/efficiencies_proton_efficiency_fixed_to_cta_energy.pdf")

        PlotEfficienciesEnergyBinned(bins, bins_central, true_positive_rate_fixed_eta, false_positive_raten_fixed_eta, f"dm-finder/cnn/{string_input[i]}/separation/results/{string_ps_input[i]}/{string_name[i][1:]}/efficiencies_fixed_eta_p.pdf")

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

        PlotEfficiencyGammaness(true_positive_rate, false_positive_rate, rejection_power, thresholds, f"dm-finder/cnn/{string_input[i]}/separation/results/{string_ps_input[i]}/{string_name[i][1:]}/efficiency_gammaness.pdf")

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
        # accuracy_energy_all[i] = accuracy_energy 
        # true_positive_rate_energy_all[i] = true_positive_rate_energy
        # false_positive_rate_energy_all[i] = false_positive_rate_energy
        true_positive_rate_fixed_eta_all[i] = true_positive_rate_fixed_eta
        false_positive_rate_fixed_eta_all[i] = false_positive_raten_fixed_eta
        
        # if args.input[0][i] == "cta":
        #     tpr_fixed_gammaness_cta_all.append(tpr_fixed_gammaness_cta)
        #     fpr_fixed_gammaness_cta_all.append(fpr_fixed_gammaness_cta)
        #     # print("fpr_fixed_gammaness_cta_all", fpr_fixed_gammaness_cta_all)
        #     fpr_fixed_gammaness_cta_mean = np.mean(fpr_fixed_gammaness_cta_all, axis = 0)
        #     # print("fpr_fixed_gammaness_cta_mean", fpr_fixed_gammaness_cta_mean)
        # if (args.input[0][i] == "ps") and ("cta" in args.input[0]):
        #     tpr_ps_proton_efficiency_fixed_to_cta_all.append(tpr_ps_proton_efficiency_fixed_to_cta)
        #     fpr_ps_proton_efficiency_fixed_to_cta_all.append(fpr_ps_proton_efficiency_fixed_to_cta)


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

        PlotEfficiencyGammanessComparison(thresholds_all, true_positive_rate_all, false_positive_rate_all, args.input[0], f"dm-finder/cnn/comparison/separation/" + "efficiency_gammaness_comparison_" + string_comparison + ".pdf")

    PlotAccuracyGammanessComparison(accuracy_gammaness_all, thresholds_all, args.input[0], f"dm-finder/cnn/comparison/separation/" + "accuracy_gammaness_comparison_" + string_comparison + ".pdf")

    PlotPrecisionGammanessComparison(precision_gammaness_all, threshold_cut_all, args.input[0], f"dm-finder/cnn/comparison/separation/" + "precision_gammaness_comparison_" + string_comparison + ".pdf")

    PlotAUCEnergyComparison(bins.value, bins_central.value, area_under_ROC_curve_energy_all, args.input[0], f"dm-finder/cnn/comparison/separation/" + "AUC_energy_comparison_" + string_comparison + ".pdf")

    # PlotAccuracyEnergyComparison(bins, bins_central, accuracy_energy_all, args.input[0], f"dm-finder/cnn/comparison/separation/" + "accuracy_energy_comparison_" + string_comparison + ".pdf")

    # PlotEfficiencyEnergyComparison(bins, bins_central, true_positive_rate_energy_all, false_positive_rate_energy_all, args.input[0], f"dm-finder/cnn/comparison/separation/" + "efficiency_energy_comparison_" + string_comparison + ".pdf")

    PlotEfficiencyEnergyComparison(bins.value, bins_central.value, true_positive_rate_fixed_eta_all, false_positive_rate_fixed_eta_all, args.input[0], f"dm-finder/cnn/comparison/separation/" + "efficiency_energy_comparison_" + string_comparison + ".pdf")

    # PlotGammaEfficiencyEnergyComparison(bins_gamma, bins_proton, bins_central_gamma, bins_central_proton, true_positive_rate_fixed_eta_all, false_positive_rate_fixed_eta_all, args.input[0], f"dm-finder/cnn/comparison/separation/" + "gamma_efficiency_energy_comparison_" + string_comparison + ".pdf")

    # if ("ps" in args.input[0]) and ("cta" in args.input[0]):
    #     # print("tpr_fixed_gammaness_cta_all")
    #     # print(tpr_fixed_gammaness_cta_all)
    #     # print("tpr_ps_proton_efficiency_fixed_to_cta_all")
    #     # print(tpr_ps_proton_efficiency_fixed_to_cta_all)
    #     # print("fpr_fixed_gammaness_cta_all")
    #     # print(fpr_fixed_gammaness_cta_all)
    #     # print("fpr_ps_proton_efficiency_fixed_to_cta_all")
    #     # print(fpr_ps_proton_efficiency_fixed_to_cta_all)

    #     tpr_proton_efficiency_fixed_to_cta = np.concatenate((tpr_fixed_gammaness_cta_all, tpr_ps_proton_efficiency_fixed_to_cta_all), axis = 0)
    #     fpr_proton_efficiency_fixed_to_cta = np.concatenate((fpr_fixed_gammaness_cta_all, fpr_ps_proton_efficiency_fixed_to_cta_all), axis = 0)

    #     # print("tpr_proton_efficiency_fixed_to_cta")
    #     # print(tpr_proton_efficiency_fixed_to_cta)
    #     # print("fpr_proton_efficiency_fixed_to_cta")
    #     # print(fpr_proton_efficiency_fixed_to_cta)

    #     PlotEfficiencyEnergyComparison(bins_gamma, bins_proton, bins_central_gamma, bins_central_proton, tpr_proton_efficiency_fixed_to_cta, fpr_proton_efficiency_fixed_to_cta, args.input[0], f"dm-finder/cnn/comparison/separation/" + "efficiency_energy_fixed_peff_comparison_" + string_comparison + ".pdf")

    #     PlotGammaEfficiencyEnergyComparison(bins_gamma, bins_proton, bins_central_gamma, bins_central_proton, tpr_proton_efficiency_fixed_to_cta, fpr_proton_efficiency_fixed_to_cta, args.input[0], f"dm-finder/cnn/comparison/separation/" + "gamma_efficiency_energy_fixed_peff_comparison_" + string_comparison + ".pdf")

    MeanStdAUC(area_under_ROC_curve_all, args.input[0])


print("CNN evaluation completed!")