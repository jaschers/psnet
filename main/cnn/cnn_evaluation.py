import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import os
import argparse
from utilities import *
from tensorflow import keras
from keras.models import Model
from ctapipe.io import read_table
from astropy.table import Table, vstack
from pyirf.simulations import SimulatedEventsInfo
import astropy.units as u

# do not print tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

plt.rcParams.update({'font.size': 8}) # 8 (paper), 10 (poster)

plt.rc('text', usetex=True)
plt.rc('font', family='Times New Roman')#, weight='normal', size=14)
plt.rcParams['mathtext.fontset'] = 'cm'

######################################## argparse setup ########################################
script_version=1.0
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


args = parser.parse_args()
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

# define a dictonary with lists to be filled in
dict = {
    "median_all": [],
    "sigma_all": [],
    "epochs_all": [],
    "loss_train_all": [],
    "loss_val_all": [],
    "true_positive_rate_all": [],
    "false_positive_rate_all": [],
    "true_negative_rate_all": [],
    "false_negative_rate_all": [],
    "area_under_ROC_curve_all": [],
    "accuracy_gammaness_all": [],
    "precision_gammaness_all": [],
    "thresholds_all": [],
    "threshold_cut_all": [],
    "area_under_ROC_curve_energy_all": [],
    "true_positive_rate_fixed_eta_all": [],
    "false_positive_rate_fixed_eta_all": [],
    "area_eff_all": [],
    "tpr_fixed_gammaness_cta_all": [],
    "fpr_fixed_gammaness_cta_all": []
}

if args.mode == "separation":
    dl0_gamma = Table()
    dl0_proton = Table()
    # load simulation info which is necesarry to calculate signal and bkg efficiencies
    filename_run_gamma_diffuse = f"main/run_lists/gamma_diffuse_run_list_alpha.csv"
    filename_run_proton = f"scripts/run_lists/proton_run_list_alpha.csv"

    run_gamma = pd.read_csv(filename_run_gamma_diffuse)
    run_gamma = run_gamma.to_numpy().reshape(len(run_gamma))
    run_proton = pd.read_csv(filename_run_proton)
    run_proton = run_proton.to_numpy().reshape(len(run_proton))

    print("loading gamma simulation information...")
    for r in tqdm(range(len(run_gamma))):
        dl1_filename_gamma_diffuse = f"gamma_20deg_0deg_run{run_gamma[r]}___cta-prod5-paranal_desert-2147m-Paranal-dark_cone10_merged.DL1"
        dl1_directory_gamma_diffuse = f"data/gamma_diffuse/event_files/" + dl1_filename_gamma_diffuse + ".h5"
        dl0_gamma_temp = read_table(dl1_directory_gamma_diffuse, "/configuration/simulation/run")
        dl0_gamma = vstack([dl0_gamma, dl0_gamma_temp])        

    print("loading proton simulation information...")
    for r in tqdm(range(len(run_proton))):
        dl1_filename_proton = f"proton_20deg_0deg_run{run_proton[r]}___cta-prod5-paranal_desert-2147m-Paranal-dark_merged.DL1"
        dl1_directory_proton = f"data/proton/event_files/" + dl1_filename_proton + ".h5"
        dl0_proton_temp = read_table(dl1_directory_proton, "/configuration/simulation/run")
        dl0_proton = vstack([dl0_proton, dl0_proton_temp])        

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

    # use 
    simulation_info_gamma = SimulatedEventsInfo(
    energy_min = dl0_gamma["energy_range_min"][0] * dl0_gamma["energy_range_min"].unit, # energy_range_min
    energy_max = dl0_gamma["energy_range_max"][0] * dl0_gamma["energy_range_max"].unit, # energy_range_max
    spectral_index = dl0_gamma["spectral_index"][0], # spectral_index
    n_showers = n_showers_total_gamma,
    max_impact = dl0_gamma["max_scatter_range"][0] * dl0_gamma["max_scatter_range"].unit, # max_scatter_range
    viewcone = (dl0_gamma["max_viewcone_radius"][0] - dl0_gamma["min_viewcone_radius"][0]) * dl0_gamma["max_viewcone_radius"].unit, # max_viewcone_radius
    )

    simulation_info_proton = SimulatedEventsInfo(
    energy_min = dl0_proton["energy_range_min"][0] * dl0_proton["energy_range_min"].unit, # energy_range_min
    energy_max = dl0_proton["energy_range_max"][0] * dl0_proton["energy_range_max"].unit, # energy_range_max
    spectral_index = dl0_proton["spectral_index"][0], # spectral_index
    n_showers = n_showers_total_proton,
    max_impact = dl0_proton["max_scatter_range"][0] * dl0_proton["max_scatter_range"].unit, # max_scatter_range
    viewcone = (dl0_proton["max_viewcone_radius"][0] - dl0_proton["min_viewcone_radius"][0]) * dl0_proton["max_viewcone_radius"].unit, # max_viewcone_radius
    )

    bins = np.logspace(np.log10(args.energy_range_gamma[0]), np.log10(args.energy_range_gamma[1]), 8) * u.TeV
    bins_width = (bins[1:] - bins[:-1])
    bins_central =  bins[:-1] + bins_width / 2

    dl0_gamma_hist = simulation_info_gamma.calculate_n_showers_per_energy(bins) 
    dl0_proton_hist = simulation_info_proton.calculate_n_showers_per_energy(bins) 

    # calculate geometric area from simulation info
    area = np.pi * simulation_info_gamma.max_impact ** 2


for i in range(len(args.input[0])):
    print(f"Processing input {i+1}...")
    # create folder
    os.makedirs(f"cnn/{string_input[i]}/{args.mode}/results/" + string_ps_input[i] + f"{string_name[i][1:]}/", exist_ok = True)

    # load loss history file
    history_path = f"cnn/{string_input[i]}/{args.mode}/history/" + string_ps_input[i] + "history" + string_data_type[i] + string_name[i] + ".csv"
    table_history = pd.read_csv(history_path)

    # plot loss history
    PlotLoss(table_history["epoch"], table_history["loss"], table_history["val_loss"], f"cnn/{string_input[i]}/{args.mode}/results/" + string_ps_input[i] + f"{string_name[i][1:]}/" + "loss.pdf")

    dict["epochs_all"].append(table_history["epoch"] + 1)
    dict["loss_train_all"].append(table_history["loss"])
    dict["loss_val_all"].append(table_history["val_loss"])

    # load model
    model_path = f"cnn/{string_input[i]}/{args.mode}/model/" + string_ps_input[i] + "model" + string_data_type[i] + string_name[i] + ".h5"
    model = keras.models.load_model(model_path)

    # load data file that contains E_true and E_rec from the test set
    filename_output = f"cnn/{string_input[i]}/{args.mode}/output/" + string_ps_input[i] + "evaluation" + string_data_type[i] + string_name[i] + ".csv"

    table_output = pd.read_csv(filename_output)

    if args.mode == "energy":
        table_output = table_output.sort_values(by = ["log10(E_true / GeV)"])

        # convert energy to TeV
        energy_true = np.asarray((10**table_output["log10(E_true / GeV)"] * 1e-3))
        energy_rec = np.asarray((10**table_output["log10(E_rec / GeV)"] * 1e-3))

        # create 2D energy scattering plot
        PlotEnergyScattering2D(np.log10(energy_true), np.log10(energy_rec), f"cnn/{string_input[i]}/{args.mode}/results/" + string_ps_input[i] + f"{string_name[i][1:]}/" + "energy_scattering_2D.pdf")

        # prepare energy binning
        number_energy_ranges = 7 # number of energy ranges the whole energy range will be splitted
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
        PlotRelativeEnergyError(relative_energy_error_toal, median_total, sigma_total, f"cnn/{string_input[i]}/{args.mode}/results/" + string_ps_input[i] + f"{string_name[i][1:]}/" + "relative_energy_error.pdf")

        # save relative energy error histogram (binned)
        PlotRelativeEnergyErrorBinned(energy_true_binned, energy_rec_binned, bins, f"cnn/{string_input[i]}/{args.mode}/results/" + string_ps_input[i] + f"{string_name[i][1:]}/" + "energy_binned_histogram.pdf")

        # save corrected relative energy error histogram (binned)
        PlotRelativeEnergyErrorBinnedCorrected(energy_true_binned, energy_rec_binned, bins, f"cnn/{string_input[i]}/{args.mode}/results/" + string_ps_input[i] + f"{string_name[i][1:]}/" + "energy_binned_histogram_corrected.pdf")

        # get median and sigma68 values (binned)
        median, sigma = MedianSigma68(energy_true_binned, energy_rec_binned, bins)

        # plot energy accuracy
        PlotEnergyAccuracy(median, bins, f"cnn/{string_input[i]}/{args.mode}/results/" + string_ps_input[i] + f"{string_name[i][1:]}/" + "energy_accuracy.pdf")
    
        # plot energy resolution
        PlotEnergyResolution(sigma, bins, f"cnn/{string_input[i]}/{args.mode}/results/" + string_ps_input[i] + f"{string_name[i][1:]}/" + "energy_resolution.pdf")

        # save energy accuracy & resolution in csv files
        SaveCSV(median, bins, "accuracy", f"cnn/{string_input[i]}/{args.mode}/results/" + string_ps_input[i] + f"{string_name[i][1:]}/" + "energy_accuracy.csv")
        SaveCSV(sigma, bins, "resolution", f"cnn/{string_input[i]}/{args.mode}/results/" + string_ps_input[i] + f"{string_name[i][1:]}/" + "energy_resolution.csv")

        # collect median and sigma values from each experiment
        dict["median_all"].append(median)
        dict["sigma_all"].append(sigma)
        
    
    elif args.mode == "separation":
        particle_type = np.array(["gamma_diffuse", "proton"])
        gammaness_true = np.asarray(table_output["true gammaness"])
        gammaness_rec = np.asarray(table_output["reconstructed gammaness"])
        energy_true = np.asarray(table_output["E_true / GeV"])

        PlotGammaness(gammaness_true, gammaness_rec, f"cnn/{string_input[i]}/separation/results/{string_ps_input[i]}/{string_name[i][1:]}/gammaness.pdf")

        # perform an energy dependend analysis of accuracy, AUC and gammaness
        PlotGammanessEnergyBinned(table_output, bins, bins_central, f"cnn/{string_input[i]}/separation/results/{string_ps_input[i]}/{string_name[i][1:]}/")

        PlotDL0DL1Hist(table_output, bins, bins_central, bins_width, dl0_gamma_hist, dl0_proton_hist, f"cnn/{string_input[i]}/separation/results/{string_ps_input[i]}/{string_name[i][1:]}/")

        # perform an gamma & proton energy dependend analysis of accuracy, AUC and gammaness
        true_positive_rate_fixed_eta, false_positive_raten_fixed_eta = GetEfficienciesEnergyBinnedFixedBackground(table_output, bins, bins_central, dl0_gamma_hist, dl0_proton_hist, 0.001)

        # get effective area
        area_eff = GetEffectiveArea(true_positive_rate_fixed_eta, area)

        # get AUC energy binned based on proton energy
        area_under_ROC_curve_energy = GetAUCEnergyBinned(table_output, bins)

        PlotEfficienciesEnergyBinned(bins.value, bins_central.value, true_positive_rate_fixed_eta, false_positive_raten_fixed_eta, f"cnn/{string_input[i]}/separation/results/{string_ps_input[i]}/{string_name[i][1:]}/efficiencies_fixed_eta_p.pdf")

        PlotEffectiveAreaEnergyBinned(bins.value, bins_central.value, area_eff.value, f"cnn/{string_input[i]}/separation/results/{string_ps_input[i]}/{string_name[i][1:]}/area_eff.pdf")

        true_positive_rate, false_positive_rate, true_negative_rate, false_negative_rate, area_under_ROC_curve = ROC(gammaness_true, gammaness_rec)

        PlotROC(true_positive_rate, false_positive_rate, area_under_ROC_curve, f"cnn/{string_input[i]}/separation/results/{string_ps_input[i]}/{string_name[i][1:]}/roc.pdf")

        SaveROC(true_positive_rate, false_positive_rate, area_under_ROC_curve, f"cnn/{string_input[i]}/separation/results/{string_ps_input[i]}/{string_name[i][1:]}/roc.csv")

        # prepare accuracy vs gammaness threshold plot
        accuracy_gammaness, thresholds = AccuracyGammaness(gammaness_true, gammaness_rec)

        # plot accuracy vs gammaness threshold plot
        PlotAccuracyGammaness(accuracy_gammaness, thresholds, f"cnn/{string_input[i]}/separation/results/{string_ps_input[i]}/{string_name[i][1:]}/accuracy_gammaness.pdf")

        # prepare precision vs gammaness treshold plot
        precision_gammaness, threshold_cut = PrecisionGammaness(gammaness_true, gammaness_rec)

        # plot precision vs gammaness treshold plot
        PlotPrecisionGammaness(precision_gammaness, thresholds[:threshold_cut], f"cnn/{string_input[i]}/separation/results/{string_ps_input[i]}/{string_name[i][1:]}/precision_gammaness.pdf")

        PlotEfficiencyGammaness(true_positive_rate, false_positive_rate, thresholds, f"cnn/{string_input[i]}/separation/results/{string_ps_input[i]}/{string_name[i][1:]}/efficiency_gammaness.pdf")

        dict["true_positive_rate_all"].append(true_positive_rate)
        dict["false_positive_rate_all"].append(false_positive_rate)
        dict["true_negative_rate_all"].append(true_negative_rate)
        dict["false_negative_rate_all"].append(false_negative_rate)
        dict["area_under_ROC_curve_all"].append(area_under_ROC_curve)
        dict["accuracy_gammaness_all"].append(accuracy_gammaness)
        dict["precision_gammaness_all"].append(precision_gammaness)
        dict["threshold_cut_all"].append(thresholds[:threshold_cut])
        dict["thresholds_all"].append(thresholds)
        dict["area_under_ROC_curve_energy_all"].append(area_under_ROC_curve_energy)
        dict["true_positive_rate_fixed_eta_all"].append(true_positive_rate_fixed_eta)
        dict["false_positive_rate_fixed_eta_all"].append(false_positive_raten_fixed_eta)
        dict["area_eff_all"].append(area_eff)


        # Plot wrongly classified CTA images / pattern spectra
        if args.gammaness_limit != [0.0, 0.0, 0.0, 0.0]:
            # load CTA images or pattern spectra into the table_output 
            table_output = ExtendTable(table_output, string_table_column[i], string_input[i], string_ps_input[i], string_input_short[i], string_data_type[i])

            # Plot 15 examples of wrongly classified events
            PlotWronglyClassifiedEvents(table_output, particle_type, string_table_column[i], args.gammaness_limit, f"cnn/{string_input[i]}/separation/results/{string_ps_input[i]}/{string_name[i][1:]}/misclassified_examples")

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
            PlotPatternSpectraMean(pattern_spectra_mean_gammaness_limit, particle_type, args.attribute, args.gammaness_limit, newcmp, f"cnn/{string_input[i]}/separation/results/{string_ps_input[i]}/{string_name[i][1:]}/misclassified_sum")

            # plot pattern spectra difference (gamma - proton) of missclassified events
            PlotPatternSpectraDifference(pattern_spectra_mean_gammaness_limit, particle_type, args.attribute, args.gammaness_limit, f"cnn/{string_input[i]}/separation/results/{string_ps_input[i]}/{string_name[i][1:]}/misclassified_difference")

            # plot pattern spectra difference of correctly and wrongly classified gamma events
            PlotPatternSpectraDifference(pattern_spectra_mean_gammaness_limit_gamma, ["gamma_diffuse", "gamma_diffuse"], args.attribute, [0.9, 1.0, 0.0, 0.1], f"cnn/{string_input[i]}/separation/results/{string_ps_input[i]}/{string_name[i][1:]}/correct_wrong_gamma_difference")

            # plot pattern spectra difference of correctly and wrongly classified proton events
            PlotPatternSpectraDifference(pattern_spectra_mean_gammaness_limit_proton, ["proton", "proton"], args.attribute, [0.0, 0.1, 0.9, 1.0], f"cnn/{string_input[i]}/separation/results/{string_ps_input[i]}/{string_name[i][1:]}/correct_wrong_proton_difference")


if args.mode == "energy":
    # if more than two inputs are given -> compare the results
    if len(args.input[0]) > 1:
        os.makedirs(f"cnn/comparison/energy", exist_ok = True)

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
        PlotLossComparison(dict["epochs_all"], dict["loss_train_all"], dict["loss_val_all"], args.input[0], f"cnn/comparison/energy/" + "loss_comparison_" + string_comparison + ".pdf")

        # plot energy accuracy comparison
        PlotEnergyAccuracyComparison(dict["median_all"], bins, label[0], f"cnn/comparison/energy/" + "energy_accuracy_" + string_comparison + ".pdf")

        PlotEnergyAccuracyComparisonMean(dict["median_all"], bins, label[0], args.input[0], f"cnn/comparison/energy/" + "energy_accuracy_mean_" + string_comparison + ".pdf")

        # plot energy resolution comparison
        PlotEnergyResolutionComparison(dict["sigma_all"], bins, label[0], f"cnn/comparison/energy/" + "energy_resolution_" + string_comparison + ".pdf")

        # plot mean energy resolution
        PlotEnergyResolutionComparisonMean(args.input[0], dict["sigma_all"], bins, label[0], f"cnn/comparison/energy/" + "energy_resolution_mean_" + string_comparison + ".pdf")


if (args.mode == "separation") and (len(args.input[0]) > 1):
    os.makedirs(f"cnn/comparison/separation", exist_ok = True)

    string_comparison = ""
    for i in range(len(args.input[0])):
        string_comparison += args.input[0][i] + string_name[i] + "_"

    for i in range(len(args.input[0])):
        if args.input[0][i] == "ps":
            string_comparison += "_" + string_ps_input[i][:-1]
            break
    
    if len(string_comparison) > 200:
        string_comparison = string_comparison[:200]

    PlotLossComparison(dict["epochs_all"], dict["loss_train_all"], dict["loss_val_all"], args.input[0], f"cnn/comparison/separation/" + "loss_comparison_" + string_comparison + ".pdf")

    if len(args.input[0]) > 3:
        PlotROCComparison(dict["true_positive_rate_all"], dict["false_positive_rate_all"], dict["area_under_ROC_curve_all"], args.input[0], f"cnn/comparison/separation/" + "ROC_comparison_" + string_comparison + ".pdf")

        PlotEfficiencyGammanessComparison(dict["thresholds_all"], dict["true_positive_rate_all"], dict["false_positive_rate_all"], args.input[0], f"cnn/comparison/separation/" + "efficiency_gammaness_comparison_" + string_comparison + ".pdf")

    PlotAccuracyGammanessComparison(dict["accuracy_gammaness_all"], dict["thresholds_all"], args.input[0], f"cnn/comparison/separation/" + "accuracy_gammaness_comparison_" + string_comparison + ".pdf")

    PlotPrecisionGammanessComparison(dict["precision_gammaness_all"], dict["threshold_cut_all"], args.input[0], f"cnn/comparison/separation/" + "precision_gammaness_comparison_" + string_comparison + ".pdf")

    PlotAUCEnergyComparison(bins.value, bins_central.value, dict["area_under_ROC_curve_energy_all"], args.input[0], f"cnn/comparison/separation/" + "AUC_energy_comparison_" + string_comparison + ".pdf")

    PlotEfficiencyEnergyComparison(bins.value, bins_central.value, dict["true_positive_rate_fixed_eta_all"], dict["false_positive_rate_fixed_eta_all"], args.input[0], f"cnn/comparison/separation/" + "efficiency_energy_comparison_" + string_comparison + ".pdf")
    
    PlotAeffEnergyComparison(bins.value, bins_central.value, dict["area_eff_all"], args.input[0], f"cnn/comparison/separation/" + "area_eff_comparison_" + string_comparison + ".pdf")

print("CNN evaluation completed!")