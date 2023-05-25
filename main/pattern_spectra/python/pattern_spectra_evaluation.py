import pandas as pd
import numpy as np
import os 
import argparse
from matplotlib.colors import ListedColormap
from utilities import *
import random

np.seterr(divide='ignore', invalid='ignore')

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
parser.add_argument("-tm", "--telescope_mode", type = str, required = False, metavar = "", choices = ["mono", "stereo_sum_cta", "stereo_sum_ps"], help = "telescope mode [mono, stereo_sum_cta, stereo_sum_ps], default: stereo_sum_cta", default = "stereo_sum_cta")
parser.add_argument("-r", "--run", type = int, metavar = "-", help = "input run(s) for CNN, default: csv list", action="append", nargs="+")
parser.add_argument("-ere", "--energy_range_en", type = float, required = False, metavar = "-", help = "set energy range of events in GeV for energy reconstruction mode, default: 0.5 100", default = [0.5, 100], nargs = 2)
parser.add_argument("-ers", "--energy_range_sep", type = float, required = False, metavar = "-", help = "set energy range of events in GeV for separation mode, default: 1.5 100", default = [1.5, 100], nargs = 2)
parser.add_argument("-a", "--attribute", type = int, metavar = "", choices = np.arange(0, 19, dtype = int), help = "attribute [0, 1 ... 18] (two required), default: 9 0", default = [9, 0], nargs = 2)
parser.add_argument("-dl", "--domain_lower", type = float, metavar = "", help = "Granulometry: domain - start at <value> <value>, default: 0.8 0.8", default = [0.8, 0.8], nargs = 2)
parser.add_argument("-dh", "--domain_higher", type = float, metavar = "", help = "Granulometry: domain - end at <value> <value>, default: 7 3000", default = [7., 3000.], nargs = 2)
parser.add_argument("-ma", "--mapper", type = int, metavar = "", help = "Granulometry: use lambdamappers <mapper1> <mapper2>, default: 4 4", default = [4, 4], nargs = 2)
parser.add_argument("-n", "--size", type = int, metavar = "", help = "Granulometry: size <n1>x<n2>, default: 20 20", default = [20, 20], nargs = 2)
parser.add_argument("-f", "--filter", type = int, metavar = "", help = "Use decision <filter>, default: 3", default = 3, nargs = 1)
parser.add_argument("-t", "--test", type = str, metavar = "-", help = "If yes, csv test list is used [y/n]", default = "n")


args = parser.parse_args()
##########################################################################################


######################################## Define some strings based on the input of the user ########################################
print(f"################### Input summary ################### \nMode: {args.mode} \nTelescope mode: {args.telescope_mode} \nEnergy range (E reco): {args.energy_range_en} TeV\nEnergy range (sig/bkg sep): {args.energy_range_sep} TeV \nAttribute: {args.attribute} \nDomain lower: {args.domain_lower} \nDomain higher: {args.domain_higher} \nMapper: {args.mapper} \nSize: {args.size} \nFilter: {args.filter} \n#######################################################")
string_ps_input = f"a_{args.attribute[0]}_{args.attribute[1]}__dl_{args.domain_lower[0]}_{args.domain_lower[1]}__dh_{args.domain_higher[0]}_{args.domain_higher[1]}__m_{args.mapper[0]}_{args.mapper[1]}__n_{args.size[0]}_{args.size[1]}__f_{args.filter}/"
if args.telescope_mode == "stereo_sum_cta":
    string_telescope_mode = "_ps_float_alpha"
elif args.telescope_mode == "mono":
    string_telescope_mode = "_ps_float_mono_alpha"
elif args.telescope_mode == "stereo_sum_ps":
    string_telescope_mode = "_ps_float_stereo_sum_alpha"
##########################################################################################

# import data
# energy mode
if args.mode == "energy":
    if args.test == "y":
        filename_run_csv = f"main/run_lists/gamma_run_list_alpha_test.csv"
    elif args.telescope_mode == "mono" or args.telescope_mode == "stereo_sum_ps":
        filename_run_csv = f"main/run_lists/gamma_run_list_mono_alpha.csv"
    else: 
        filename_run_csv = f"main/run_lists/gamma_run_list_alpha.csv"
    run = pd.read_csv(filename_run_csv)
    run = run.to_numpy().reshape(len(run))

    if args.run != None:
        run = args.run[0]

    table = pd.DataFrame()
    for r in range(len(run)): 
        run_filename = f"gamma_20deg_0deg_run{run[r]}___cta-prod5-paranal_desert-2147m-Paranal-dark_merged.DL1"
        input_filename = f"cnn/pattern_spectra/input/gamma/" + string_ps_input + run_filename + string_telescope_mode + ".h5"
        table_individual_run = pd.read_hdf(input_filename)
        print(f"Number of events in Run {run[r]}:", len(table_individual_run))
        table = table.append(table_individual_run, ignore_index = True)
    
    # apply energy cut
    table.drop(table.loc[table["true_energy"] <= args.energy_range_en[0] * 1e3].index, inplace=True)
    table.drop(table.loc[table["true_energy"] >= args.energy_range_en[1] * 1e3].index, inplace=True)
    table.reset_index(inplace = True)
    table = table.sort_values(by = ["true_energy"], ignore_index = True)

    print("Total number of events:", len(table))

# signal separation mode
elif args.mode == "separation":
    particle_type = np.array(["gamma_diffuse", "proton"])
    gammaness = np.array([1, 0])

    table = pd.DataFrame()
    events_count = np.array([0, 0])
    for p in range(len(particle_type)):
        if args.test == "y":
            filename_run_csv = f"main/run_lists/{particle_type[p]}_run_list_test.csv"
        elif args.telescope_mode == "mono":
            filename_run_csv = f"main/run_lists/{particle_type[p]}_run_list_mono.csv"
        else: 
            filename_run_csv = f"main/run_lists/{particle_type[p]}_run_list.csv"
        run = pd.read_csv(filename_run_csv)
        run = run.to_numpy().reshape(len(run))

        for r in range(len(run)):
            run_filename = f"{particle_type[p]}_20deg_0deg_run{run[r]}___cta-prod5-paranal_desert-2147m-Paranal-dark_merged.DL1"
            input_filename = f"cnn/pattern_spectra/input/{particle_type[p]}/" + string_ps_input + run_filename + string_telescope_mode + ".h5"
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

    # apply energy cut
    table.drop(table.loc[table["true_energy"] <= args.energy_range_sep[0] * 1e3].index, inplace=True)
    table.drop(table.loc[table["true_energy"] >= args.energy_range_sep[1] * 1e3].index, inplace=True)
    table.reset_index(inplace = True)
    table = table.sort_values(by = ["true_energy"], ignore_index = True)

print("______________________________________________")
if args.mode == "separation":
    print("Total number of gamma events after energy cut:", len(table.loc[table["particle"] == 1]))
    print("Total number of proton events after energy cut:", len(table.loc[table["particle"] == 0]))
    print("Total number of events after energy cut:", len(table))
    print("______________________________________________")

    # plot original energy distribution of data set
    path_energy_distribution = f"data/{particle_type[0]}/info/energy_distribution/{args.telescope_mode}/"
    os.makedirs(path_energy_distribution, exist_ok = True)
    PlotEnergyDistribution(table, args.energy_range_sep, path_energy_distribution + "energy_distribution_original.pdf")

    path_energy_distribution = f"data/{particle_type[1]}/info/energy_distribution/{args.telescope_mode}/"
    os.makedirs(path_energy_distribution, exist_ok = True)
    PlotEnergyDistribution(table, args.energy_range_sep, path_energy_distribution + "energy_distribution_original.pdf")

    # create same (similar) energy distribution for gammas and protons
    # extract gamma and proton tables
    table_gamma = table[table["particle"] == 1]
    table_proton = table[table["particle"] == 0]
    
    # remove all entries from table (will be filled in later again)
    table.drop(table.index[range(0, len(table))], inplace = True)
    table.reset_index(drop = True, inplace = True)

    # define proper energy ranges
    number_energy_ranges_redistribution = 200 # number of energy ranges the whole energy range will be splitted
    bins_redistribution = np.logspace(np.log10(args.energy_range_sep[0] * 1e3), np.log10(args.energy_range_sep[1] * 1e3), number_energy_ranges_redistribution + 1) 

    # count the number of removed events
    difference_total = 0
    # for loop over energy ranges
    for n in range(number_energy_ranges_redistribution):
        # select gamma events in the corresponding energy range
        table_gamma_split = table_gamma.loc[(table_gamma["true_energy"] >= bins_redistribution[n])]
        table_gamma_split = table_gamma_split.loc[(table_gamma_split["true_energy"] <= bins_redistribution[n+1])]
        table_gamma_split.reset_index(inplace = True, drop = True)

        # select proton events in the corresponding energy range
        table_proton_split = table_proton.loc[(table_proton["true_energy"] >= bins_redistribution[n])]
        table_proton_split = table_proton_split.loc[(table_proton_split["true_energy"] <= bins_redistribution[n+1])]
        table_proton_split.reset_index(inplace = True, drop = True)

        # determine the number of different events between gammas and protons in the corresponding energy range
        difference = np.abs(len(table_gamma_split) - len(table_proton_split))
        # keep track of the total difference
        difference_total += difference

        # if number gamma events > number proton events -> remove gamma events
        if len(table_gamma_split) > len(table_proton_split):
            random_indices = random.sample(range(0, len(table_gamma_split)), difference)
            table_gamma_split.drop(table_gamma_split.index[random_indices], inplace = True)
            table_gamma_split.reset_index(drop = True, inplace = True)
        # if number gamma events < number proton events -> remove proton events
        elif len(table_gamma_split) < len(table_proton_split):
            random_indices = random.sample(range(0, len(table_proton_split)), difference)
            table_proton_split.drop(table_proton_split.index[random_indices], inplace = True)
            table_proton_split.reset_index(drop = True, inplace = True)

        # append the updated gamma and proton tables to the total table
        table = table.append(table_gamma_split)
        table = table.append(table_proton_split)
        table.reset_index(drop = True, inplace = True)

    print("Total number of gamma events after energy redistribution:", len(table.loc[table["particle"] == 1]))
    print("Total number of proton events after energy redistribution:", len(table.loc[table["particle"] == 0]))
    print("Total number of events after energy redistribution:", len(table))
    print("______________________________________________")


# sort table by energy
table = table.sort_values(by = ["true_energy"], ignore_index = True)

if args.mode == "energy":
    print("Total number of events after energy cut:", len(table))
    print("______________________________________________")

    # collect pattern spectra
    pattern_spectra = []
    for i in range(len(table)):
        pattern_spectra.append(table["pattern spectrum"][i].astype(np.float32))
    pattern_spectra = np.asarray(pattern_spectra) 

    # collect true energy
    energy_true = np.asarray(table["true_energy"]) * 1e-3

    # prepare energy binning
    number_energy_ranges = 9 # number of energy ranges the whole energy range will be splitted
    sst_energy_min = args.energy_range_en[0] # TeV
    sst_energy_max = args.energy_range_en[1] # TeV
    bins = np.logspace(np.log10(np.min(sst_energy_min)), np.log10(np.max(sst_energy_max)), number_energy_ranges + 1) 
    indices = np.array([], dtype = int)
    for b in range(len(bins) - 2):
        index = np.max(np.where(energy_true < bins[b+1])) + 1
        indices = np.append(indices, index)

    energy_true_binned = np.split(energy_true, indices)
    ps_binned = np.split(pattern_spectra, indices)

    path = f"data/gamma/info/pattern_spectra_distribution" + f"/a_{args.attribute[0]}_{args.attribute[1]}__dl_{args.domain_lower[0]}_{args.domain_lower[1]}__dh_{args.domain_higher[0]}_{args.domain_higher[1]}__m_{args.mapper[0]}_{args.mapper[1]}__n_{args.size[0]}_{args.size[1]}__f_{args.filter}/" + f"{args.telescope_mode}/"
    os.makedirs(path, exist_ok = True)

    # extract the normed sum of all pattern spectra in each specific energy bin
    ps_mean = ExtractPatternSpectraMean(number_energy_ranges, args.size, ps_binned)
    ps_mean_min, ps_mean_max = ExtractPatternSpectraMinMax(number_energy_ranges, ps_mean)

    # subtract normed pattern spectrum of the first energy bin from all other normed pattern spectra of each other energy bin 
    ps_mean_difference = ExtractPatternSpectraDifference(number_energy_ranges, args.size, ps_mean)
    # extract the min and max value of ps_mean_difference to set the colourbar equally 
    ps_mean_difference_min, ps_mean_difference_max = ExtractPatternSpectraMinMax(number_energy_ranges, ps_mean_difference)

    # plot normed and difference pattern spectra
    PlotPatternSpectraMean(number_energy_ranges, ps_mean, ps_mean_min, ps_mean_max, ps_mean_difference, ps_mean_difference_min, ps_mean_difference_max, bins, args.particle_type, args.attribute, path)


elif args.mode == "separation":
    # for energy dependend comparison
    ps_mean_gamma_proton = []
    ps_median_gamma_proton = []
    ps_variance_gamma_proton = []
    ps_mean_min = []
    ps_mean_max = []
    ps_median_min = []
    ps_median_max = []
    ps_variance_min = []
    ps_variance_max = []
    
    # for energy independend comparison
    ps_total_mean_gamma_proton = np.zeros(shape = (len(particle_type), args.size[0], args.size[1]))
    ps_total_median_gamma_proton = np.zeros(shape = (len(particle_type), args.size[0], args.size[1]))
    ps_total_variance_gamma_proton = np.zeros(shape = (len(particle_type), args.size[0], args.size[1]))
    for p in range(len(particle_type)):
        # create proper path
        path_energy_distribution = f"data/{particle_type[p]}/info/energy_distribution/{args.telescope_mode}/"
        os.makedirs(path_energy_distribution, exist_ok = True)
        # plot energy distribution of data set
        PlotEnergyDistribution(table, args.energy_range_sep, path_energy_distribution + "energy_distribution_redistributed.pdf")

        print(f"Starting analysis for {particle_type[p]} pattern spectra")
        table_individual = table.loc[table["particle"] == gammaness[p]]
        table_individual = table_individual.reset_index(drop = True)
        pattern_spectra = []
        for i in range(len(table_individual)):
            pattern_spectra.append(table_individual["pattern spectrum"][i])
        pattern_spectra = np.asarray(pattern_spectra, dtype = np.float32)

        # calculate total mean
        ps_total_mean_gamma_proton[p] = np.mean(pattern_spectra, axis = 0)
        # # calculate total median
        ps_total_median_gamma_proton[p] = np.median(pattern_spectra, axis = 0)
        # # calculate total variance
        ps_total_variance_gamma_proton[p] = np.var(pattern_spectra, axis = 0)

        # create own colour map
        N = 256
        vals = np.ones((N, 4))
        vals[:, 0] = np.linspace(cstm_RdBu(6)[0], cstm_PuBu(12)[0], N)
        vals[:, 1] = np.linspace(cstm_RdBu(6)[1], cstm_PuBu(12)[1], N)
        vals[:, 2] = np.linspace(cstm_RdBu(6)[2], cstm_PuBu(12)[2], N)
        newcmp = ListedColormap(vals)

        # create proper path
        path = f"data/{particle_type[p]}/info/pattern_spectra_distribution/" + f"/a_{args.attribute[0]}_{args.attribute[1]}__dl_{args.domain_lower[0]}_{args.domain_lower[1]}__dh_{args.domain_higher[0]}_{args.domain_higher[1]}__m_{args.mapper[0]}_{args.mapper[1]}__n_{args.size[0]}_{args.size[1]}__f_{args.filter}/" + f"{args.telescope_mode}/"
        os.makedirs(path, exist_ok = True)

        # plot total median, mean and variance
        PlotPatternSpectraTotal(ps_total_mean_gamma_proton[p], ps_total_median_gamma_proton[p], ps_total_variance_gamma_proton[p], particle_type[p], args.attribute, path)

        # collect true energy
        energy_true = np.asarray(table_individual["true_energy"]) * 1e-3

        # prepare energy binning
        number_energy_ranges = 9 # number of energy ranges the whole energy range will be splitted
        sst_energy_min = args.energy_range_sep[0] # TeV
        sst_energy_max = args.energy_range_sep[1] # TeV
        bins = np.logspace(np.log10(np.min(sst_energy_min)), np.log10(np.max(sst_energy_max)), number_energy_ranges + 1) 
        indices = np.array([], dtype = int)
        for b in range(len(bins) - 2):
            index = np.max(np.where(energy_true < bins[b+1])) + 1
            indices = np.append(indices, index)

        energy_true_binned = np.split(energy_true, indices)
        ps_binned = np.split(pattern_spectra, indices)

        # extract the mean of all pattern spectra in each specific energy bin
        ps_mean = ExtractPatternSpectraMean(number_energy_ranges, args.size, ps_binned)
        # extract the median of all pattern spectra in each specific energy bin
        ps_median = ExtractPatternSpectraMedian(number_energy_ranges, args.size, ps_binned)
        # extract the variance of all pattern spectra in each specific energy bin
        ps_variance = ExtractPatternSpectraVariance(number_energy_ranges, args.size, ps_binned)

        # determine the min and max mean pattern spectra value over all energy bins
        ps_mean_min_, ps_mean_max_ = ExtractPatternSpectraMinMax(number_energy_ranges, ps_mean)
        ps_mean_min.append(ps_mean_min_)
        ps_mean_max.append(ps_mean_max_)

        # determine the min and max median pattern spectra value over all energy bins
        ps_median_min_, ps_median_max_ = ExtractPatternSpectraMinMax(number_energy_ranges, ps_median)
        ps_median_min.append(ps_median_min_)
        ps_median_max.append(ps_median_max_)

        # determine the min and max variance pattern spectra value over all energy bins
        ps_variance_min_, ps_variance_max_ = ExtractPatternSpectraMinMax(number_energy_ranges, ps_variance)
        ps_variance_min.append(ps_variance_min_)
        ps_variance_max.append(ps_variance_max_)

        # add ps_mean of gammas/protons to empty array for a comparison later
        ps_mean_gamma_proton.append(ps_mean)
        # add ps_median of gammas/protons to empty array for a comparison later
        ps_median_gamma_proton.append(ps_median)
        # add ps_variance of gammas/protons to empty array for a comparison later
        ps_variance_gamma_proton.append(ps_variance)

        # subtract mean pattern spectrum of the first energy bin from all other mean pattern spectra of each other energy bin to get mean difference pattern spectra 
        ps_mean_difference = ExtractPatternSpectraDifference(number_energy_ranges, args.size, ps_mean)
        # subtract median pattern spectrum of the first energy bin from all other mean pattern spectra of each other energy bin to get median difference pattern spectra 
        ps_median_difference = ExtractPatternSpectraDifference(number_energy_ranges, args.size, ps_median)
        # subtract variance pattern spectrum of the first energy bin from all other mean pattern spectra of each other energy bin to get variance difference pattern spectra 
        ps_variance_difference = ExtractPatternSpectraDifference(number_energy_ranges, args.size, ps_variance)

        # extract the min and max value of ps_mean_difference to set the color bar equally 
        ps_mean_difference_min, ps_mean_difference_max = ExtractPatternSpectraMinMax(number_energy_ranges, ps_mean_difference)
        # extract the min and max value of ps_median_difference to set the color bar equally 
        ps_median_difference_min, ps_median_difference_max = ExtractPatternSpectraMinMax(number_energy_ranges, ps_median_difference)
        # extract the min and max value of ps_variance_difference to set the color bar equally 
        ps_variance_difference_min, ps_variance_difference_max = ExtractPatternSpectraMinMax(number_energy_ranges, ps_variance_difference)

        # plot indivdual pixel distribution
        PlotPatternSpectraPixelDistribution(pattern_spectra, ps_binned, number_energy_ranges, bins, particle_type[p], path)

        # plot mean and mean difference pattern spectra
        PlotPatternSpectraMean(number_energy_ranges, ps_mean, ps_mean_min[p], ps_mean_max[p], ps_mean_difference, ps_mean_difference_min, ps_mean_difference_max, bins, particle_type[p], args.attribute, path)
        # plot median and median difference pattern spectra
        PlotPatternSpectraMedian(number_energy_ranges, ps_median, ps_median_min[p], ps_median_max[p], ps_median_difference, ps_median_difference_min, ps_median_difference_max, bins, particle_type[p], args.attribute, path)
        # plot variance and variance difference pattern spectra
        PlotPatternSpectraVariance(number_energy_ranges, ps_variance, ps_variance_min[p], ps_variance_max[p], ps_variance_difference, ps_variance_difference_min, ps_variance_difference_max, bins, particle_type[p], args.attribute, path)

    # Plot mean pattern spectra (gamma - proton) for the total energy range
    PlotPatternSpectraMeanComparisonTotal(ps_total_mean_gamma_proton, particle_type, args.attribute, path)
    # Plot median pattern spectra (gamma - proton) for the total energy range
    PlotPatternSpectraMedianComparisonTotal(ps_total_median_gamma_proton, particle_type, args.attribute, path)

    # Plot mean pattern spectra (gamma - proton) with binned energy
    PlotPatternSpectraMeanComparison(number_energy_ranges, ps_mean_gamma_proton, bins, particle_type, args.attribute, path)
    # Plot median pattern spectra (gamma - proton) with binned energy
    PlotPatternSpectraMedianComparison(number_energy_ranges, ps_median_gamma_proton, bins, particle_type, args.attribute, path)
