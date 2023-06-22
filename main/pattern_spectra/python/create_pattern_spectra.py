import numpy as np
import re
from io import StringIO
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm
import argparse
from utilities import PlotPatternSpectrum

plt.rcParams.update({'font.size': 16})

######################################## argparse setup ########################################
script_version=1.0
script_descr="""
This script creates pattern spectra from the CTA images of gamma/gamma_diffuse/proton events.
attr =  0 - Area (default) 
        1 - Area of the minimum enclosing rectangle 
        2 - Length of the diagonal of the minimum encl. rect. 
        3 - Area (Peri) 
        4 - Perimeter (Peri) 
        5 - Complexity (Peri) 
        6 - Simplicity (Peri) 
        7 - Compactness (Peri) 
        8 - Moment Of Inertia 
        9 - (Moment Of Inertia) / (Area*Area) 
        10 - Compactnes                          (Jagged) 
        11 - (Moment Of Inertia) / (Area*Area)   (Jagged) 
        12 - Jaggedness                          (Jagged)
        13 - Entropy 
        14 - Lambda-Max (not idempotent -> not a filter) 
        15 - Max. Pos. X 
        16 - Max. Pos. Y 
        17 - Grey level 
        18 - Sum grey levels 
filter = 0 - "Min" decision 
        1 - "Direct" decision (default) 
        2 - "Max" decision 
        3 - Wilkinson decision 
mapper = 0 - Area mapper 
        1 - Linear mapper 
        2 - Sqrt mapper 
        3 - Log2 mapper 
        4 - Log10 mapper 
"""

# Open argument parser
parser = argparse.ArgumentParser(description=script_descr, formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-v", "--version", action="version", version=f"v{script_version}")

# Define expected arguments
parser.add_argument("-pt", "--particle_type", type = str, metavar = "", choices = ["gamma", "gamma_diffuse", "proton"], help = "particle type [gamma, gamma_diffuse, proton], default: gamma", default = "gamma")
parser.add_argument("-tm", "--telescope_mode", type = str, required = False, metavar = "", choices = ["mono", "stereo_sum_cta", "stereo_sum_ps"], help = "telescope mode [mono, stereo_sum_cta, stereo_sum_ps], default: stereo_sum_cta", default = "stereo_sum_cta")
parser.add_argument("-r", "--run", type = int, metavar = "-", help = "input run(s) from which the pattern spectra will be extracted, default: csv list", action='append', nargs='+')
parser.add_argument("-er", "--energy_range", type = float, required = True, metavar = "-", help = "set energy range of events in TeV", nargs = 2)
parser.add_argument("-a", "--attribute", type = int, metavar = "", choices = np.arange(0, 19, dtype = int), help = "attribute [0, 1 ... 18] (two required), default: 9 0", default = [9, 0], nargs = 2)
parser.add_argument("-dl", "--domain_lower", type = float, metavar = "", help = "Granulometry: domain - start at <value> <value>, default: 0.8 0.8", default = [0.8, 0.8], nargs = 2)
parser.add_argument("-dh", "--domain_higher", type = float, metavar = "", help = "Granulometry: domain - end at <value> <value>, default: 7 3000", default = [7., 3000.], nargs = 2)
parser.add_argument("-m", "--mapper", type = int, metavar = "", help = "Granulometry: use lambdamappers <mapper1> <mapper2>, default: 4 4", default = [4, 4], nargs = 2)
parser.add_argument("-n", "--size", type = int, metavar = "", help = "Granulometry: size <n1>x<n2>, default: 20 20", default = [20, 20], nargs = 2)
parser.add_argument("-f", "--filter", type = int, metavar = "", help = "Use decision <filter>, default: 3", default = 3, nargs = 1)
parser.add_argument("-t", "--test", type = str, metavar = "-", help = "If yes, csv test list is used [y/n]", default = "n")

args = parser.parse_args()
print(f"################### Input summary ################### \nParticle type: {args.particle_type} \nTelescope mode: {args.telescope_mode} \nAttribute: {args.attribute} \nDomain lower: {args.domain_lower} \nDomain higher: {args.domain_higher} \nMapper: {args.mapper} \nSize: {args.size} \nFilter: {args.filter}\n#######################################################")
##########################################################################################

if args.test == "y":
    filename_run_csv = f"main/run_lists/{args.particle_type}_run_list_alpha_test.csv"
elif (args.telescope_mode == "mono" and args.particle_type == "gamma") or (args.telescope_mode == "stereo_sum_ps" and args.particle_type == "gamma"):
    filename_run_csv = f"main/run_lists/{args.particle_type}_run_list_mono_alpha.csv"
else: 
    filename_run_csv = f"main/run_lists/{args.particle_type}_run_list_alpha.csv"
run = pd.read_csv(filename_run_csv)
run = run.to_numpy().reshape(len(run))

if args.run != None:
    run = args.run[0]

print(f"Total number of runs: {len(run)}")
print(f"Run IDs: {run}")

for r in range(len(run)): #len(run)
    # read csv file
    print("Run", run[r])

    run_filename = f"{args.particle_type}_20deg_0deg_run{run[r]}___cta-prod5-paranal_desert-2147m-Paranal-dark_merged.DL1"

    if args.telescope_mode == "stereo_sum_cta":
        csv_directory = f"data/{args.particle_type}/tables/" + f"{args.particle_type}_20deg_0deg_run{run[r]}___cta-prod5-paranal_desert-2147m-Paranal-dark_merged.DL1_alpha" + ".csv"
    elif args.telescope_mode == "mono" or args.telescope_mode == "stereo_sum_ps":
        csv_directory = f"data/{args.particle_type}/tables/" + f"{args.particle_type}_20deg_0deg_run{run[r]}___cta-prod5-paranal_desert-2147m-Paranal-dark_merged.DL1_mono_alpha" + ".csv"

    table = pd.read_csv(csv_directory)

    # extract unique obs_id and all event_id
    obs_id_unique = np.unique(table["obs_id"])
    event_id = table["event_id"]

    # add run information to the table
    table["run"] = run[r]

    # rearange the order of the columns
    columns = list(table.columns.values)
    columns = columns[-1:] + columns[:-1]
    table = table[columns]

    # add particle type information to the table
    if (args.particle_type == "gamma") or (args.particle_type == "gamma_diffuse"):
        table["particle"] = 1
    elif args.particle_type == "proton":
        table["particle"] = 0

    # add pattern spectrum column to table to be filled in
    table["pattern spectrum"] = np.nan
    table["pattern spectrum"] = table["pattern spectrum"].astype(object)

    # apply energy cut 
    table.drop(table.loc[table["true_energy"] <= args.energy_range[0] * 1e3].index, inplace=True)
    table.drop(table.loc[table["true_energy"] >= args.energy_range[1] * 1e3].index, inplace=True)
    table.reset_index(inplace = True)

    # for loop to create pattern spectra from cta images with pattern spectra code
    for i in tqdm(range(len(table))): #len(table)
        
        if args.telescope_mode == "stereo_sum_cta":
            # create folder 
            path_mat = f"data/{args.particle_type}/pattern_spectra" + f"/a_{args.attribute[0]}_{args.attribute[1]}__dl_{args.domain_lower[0]}_{args.domain_lower[1]}__dh_{args.domain_higher[0]}_{args.domain_higher[1]}__m_{args.mapper[0]}_{args.mapper[1]}__n_{args.size[0]}_{args.size[1]}__f_{args.filter}/" + run_filename + "/float_alpha" + "/obs_id_" + f"{table['obs_id'][i]}/mat/"
            path_tif = f"data/{args.particle_type}/pattern_spectra" + f"/a_{args.attribute[0]}_{args.attribute[1]}__dl_{args.domain_lower[0]}_{args.domain_lower[1]}__dh_{args.domain_higher[0]}_{args.domain_higher[1]}__m_{args.mapper[0]}_{args.mapper[1]}__n_{args.size[0]}_{args.size[1]}__f_{args.filter}/" + run_filename + "/float_alpha" + "/obs_id_" + f"{table['obs_id'][i]}/tif/"

            os.makedirs(path_mat, exist_ok = True)
            os.makedirs(path_tif, exist_ok = True)

            filename = "obs_id_" + f"{table['obs_id'][i]}" + "__" "event_id_" + f"{table['event_id'][i]}"

            # command to create pattern spectra
            command = "./main/pattern_spectra/xmaxtree_HDF_float_single_MW/xmaxtree" + f" data/{args.particle_type}/images/" + run_filename + "/float_alpha" + "/obs_id_" + f"{table['obs_id'][i]}/" + "obs_id_" + f"{table['obs_id'][i]}" + "__" "event_id_" + f"{table['event_id'][i]}.h5" f" a {args.attribute[0]}, {args.attribute[1]} dl {args.domain_lower[0]}, {args.domain_lower[1]} dh {args.domain_higher[0]}, {args.domain_higher[1]} m {args.mapper[0]}, {args.mapper[1]} n {args.size[0]}, {args.size[1]} f {args.filter} nogui e " + path_mat + filename + " &> /dev/null"
        elif args.telescope_mode == "mono":
            # create folder 
            path_mat = f"data/{args.particle_type}/pattern_spectra" + f"/a_{args.attribute[0]}_{args.attribute[1]}__dl_{args.domain_lower[0]}_{args.domain_lower[1]}__dh_{args.domain_higher[0]}_{args.domain_higher[1]}__m_{args.mapper[0]}_{args.mapper[1]}__n_{args.size[0]}_{args.size[1]}__f_{args.filter}/" + run_filename + "/float_alpha" + "/mono" + "/obs_id_" + f"{table['obs_id'][i]}/mat/"
            path_tif = f"data/{args.particle_type}/pattern_spectra" + f"/a_{args.attribute[0]}_{args.attribute[1]}__dl_{args.domain_lower[0]}_{args.domain_lower[1]}__dh_{args.domain_higher[0]}_{args.domain_higher[1]}__m_{args.mapper[0]}_{args.mapper[1]}__n_{args.size[0]}_{args.size[1]}__f_{args.filter}/" + run_filename + "/float_alpha" + "/mono" + "/obs_id_" + f"{table['obs_id'][i]}/tif/"

            os.makedirs(path_mat, exist_ok = True)
            os.makedirs(path_tif, exist_ok = True)

            filename = "obs_id_" + f"{table['obs_id'][i]}" + "__event_id_" + f"{table['event_id'][i]}" + "__tel_id_" + f"{table['tel_id'][i]}"

            # command to create pattern spectra
            command = "./main/pattern_spectra/xmaxtree_HDF_float_single_MW/xmaxtree" + f" data/{args.particle_type}/images/" + run_filename + "/float" + "/mono_alpha" + "/obs_id_" + f"{table['obs_id'][i]}/" + "hdf/" + "obs_id_" + f"{table['obs_id'][i]}" + "__event_id_" + f"{table['event_id'][i]}" + "__tel_id_" + f"{table['tel_id'][i]}.h5" f" a {args.attribute[0]}, {args.attribute[1]} dl {args.domain_lower[0]}, {args.domain_lower[1]} dh {args.domain_higher[0]}, {args.domain_higher[1]} m {args.mapper[0]}, {args.mapper[1]} n {args.size[0]}, {args.size[1]} f {args.filter} nogui e " + path_mat + filename + " &> /dev/null"

        elif args.telescope_mode == "stereo_sum_ps":
            path_mat = f"data/{args.particle_type}/pattern_spectra" + f"/a_{args.attribute[0]}_{args.attribute[1]}__dl_{args.domain_lower[0]}_{args.domain_lower[1]}__dh_{args.domain_higher[0]}_{args.domain_higher[1]}__m_{args.mapper[0]}_{args.mapper[1]}__n_{args.size[0]}_{args.size[1]}__f_{args.filter}/" + run_filename + "/float_alpha" + "/mono" + "/obs_id_" + f"{table['obs_id'][i]}/mat/"

            filename = "obs_id_" + f"{table['obs_id'][i]}" + "__event_id_" + f"{table['event_id'][i]}" + "__tel_id_" + f"{table['tel_id'][i]}"


        if args.telescope_mode != "stereo_sum_ps":
            # apply command in terminal
            os.system(command)

        # open pattern spectra file
        file = open(path_mat + filename + ".m", "r")
        # remove unnecessary "Granulometry(:,:)" from the string
        pattern_spectrum = file.read()[18:]
        # convert string to numpy array with proper shape and remove unnecessary colomns and rows
        pattern_spectrum = "0" + re.sub(" +", " ", pattern_spectrum)
        pattern_spectrum = np.genfromtxt(StringIO(pattern_spectrum))[2:, 2:]

        # take log of the pattern_spectrum and replace -inf values with 0
        pattern_spectrum = np.log10(pattern_spectrum)
        pattern_spectrum[pattern_spectrum == -np.inf] = 0
        pattern_spectrum = pattern_spectrum.astype(np.float64)
        # add pattern_spectrum to table
        table["pattern spectrum"][i] = pattern_spectrum

        if args.telescope_mode != "stereo_sum_ps":
            if run[r] == 10 and i <= 150:
                PlotPatternSpectrum(pattern_spectrum, args.attribute, path_tif + filename + ".tif")

    if args.telescope_mode == "stereo_sum_ps":
        # sort the table by run, obs_id and event_id
        table.sort_values(by=["run", "obs_id", "event_id"])
        # sum up all pattern spectra of each event and update the table
        table = table.groupby(["run", "obs_id", "event_id", "true_energy", "particle"])["pattern spectrum"].apply(np.sum).reset_index()

        if run[r] == 10:
            for i in range(50):
                path_tif = f"data/{args.particle_type}/pattern_spectra" + f"/a_{args.attribute[0]}_{args.attribute[1]}__dl_{args.domain_lower[0]}_{args.domain_lower[1]}__dh_{args.domain_higher[0]}_{args.domain_higher[1]}__m_{args.mapper[0]}_{args.mapper[1]}__n_{args.size[0]}_{args.size[1]}__f_{args.filter}/" + run_filename + "/float_alpha" + "/stereo_sum_ps" + "/obs_id_" + f"{table['obs_id'][i]}/tif/"

                os.makedirs(path_tif, exist_ok = True)

                filename = "obs_id_" + f"{table['obs_id'][i]}" + "__event_id_" + f"{table['event_id'][i]}"

                PlotPatternSpectrum(table["pattern spectrum"][i], args.attribute, path_tif + filename + ".tif")
                

    # save tabel as h5 file
    path_cnn_input = f"dm-finder/cnn/pattern_spectra/input/{args.particle_type}/" + f"/a_{args.attribute[0]}_{args.attribute[1]}__dl_{args.domain_lower[0]}_{args.domain_lower[1]}__dh_{args.domain_higher[0]}_{args.domain_higher[1]}__m_{args.mapper[0]}_{args.mapper[1]}__n_{args.size[0]}_{args.size[1]}__f_{args.filter}/" 
    
    os.makedirs(path_cnn_input, exist_ok = True)

    if args.telescope_mode == "stereo_sum_cta":
        output_filename = path_cnn_input + run_filename + "_ps_float_alpha.h5"
    elif args.telescope_mode == "mono":
        output_filename = path_cnn_input + run_filename + "_ps_float_mono_alpha.h5"
    elif args.telescope_mode == "stereo_sum_ps":
        output_filename = path_cnn_input + run_filename + "_ps_float_stereo_sum_alpha.h5"

    print(output_filename)
    table.to_hdf(output_filename, key = 'events', mode = 'w', index = False)

