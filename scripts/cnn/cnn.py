import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import CSVLogger
from keras.models import Model
import time
import argparse
import warnings
from utilities import ExamplesEnergy, ExamplesSeparation, EnergyDistributionEnergy, EnergyDistributionSeparation, ResBlock

print("Packages successfully loaded")

######################################## argparse setup ########################################
script_version=0.2
script_descr="""
This script loads data (CTA images or pattern spectra), defines a CNN for energy reconstruction or gamma/proton separation, trains the CNN and puts the results in a csv table.
"""

# Open argument parser
parser = argparse.ArgumentParser(description=script_descr)
parser.add_argument("-v", "--version", action="version", version=f"v{script_version}")

# Define expected arguments
parser.add_argument("-m", "--mode", type = str, required = True, metavar = "-", choices = ["energy", "separation"], help = "CNN mode - energy reconstruction or gamma/proton separation [energy, separation]")
parser.add_argument("-i", "--input", type = str, required = True, metavar = "-", choices = ["cta", "ps"], help = "input for the CNN [cta, ps]")
parser.add_argument("-tm", "--telescope_mode", type = str, required = False, metavar = "", choices = ["mono", "stereo_sum_cta", "stereo_sum_ps"], help = "telescope mode [mono, stereo_sum_cta, stereo_sum_ps], default: stereo_sum_cta", default = "stereo_sum_cta")
parser.add_argument("-r", "--run", type = int, metavar = "-", help = "input run(s) for CNN, default: csv list", action="append", nargs="+")
parser.add_argument("-pt", "--particle_type", type = str, metavar = "-", choices = ["gamma", "gamma_diffuse", "proton"], help = "particle type [gamma, gamma_diffuse, proton], default: gamma", default = "gamma")
parser.add_argument("-dt", "--data_type", type = str, required = False, metavar = "-", choices = ["int8", "float64"], help = "data type of the output images [int8, float64], default: float64", default = "float64")
parser.add_argument("-er", "--energy_range", type = float, required = False, metavar = "-", help = "set energy range of events in TeV, default: 0.5 100", default = [0.5, 100], nargs = 2)
parser.add_argument("-na", "--name", type = str, required = False, metavar = "-", help = "Name of this particular experiment")
parser.add_argument("-a", "--attribute", type = int, metavar = "-", choices = np.arange(0, 19, dtype = int), help = "attribute [0, 1 ... 18] (two required), default: 9 0", default = [9, 0], nargs = 2)
parser.add_argument("-dl", "--domain_lower", type = int, metavar = "-", help = "Granulometry: domain - start at <value> <value>, default: 0 0", default = [0, 0], nargs = 2)
parser.add_argument("-dh", "--domain_higher", type = int, metavar = "-", help = "Granulometry: domain - end at <value> <value>, default: 10 100000", default = [10, 100000], nargs = 2)
parser.add_argument("-ma", "--mapper", type = int, metavar = "-", help = "Granulometry: use lambdamappers <mapper1> <mapper2>, default: 2 0", default = [2, 0], nargs = 2)
parser.add_argument("-n", "--size", type = int, metavar = "-", help = "Granulometry: size <n1>x<n2>, default: 20 20", default = [20, 20], nargs = 2)
parser.add_argument("-f", "--filter", type = int, metavar = "-", help = "Use decision <filter>, default: 3", default = 3, nargs = 1)
parser.add_argument("-e", "--epochs", type = int, metavar = "-", help = "Number of epochs for CNN training, default: 50", default = 50)
parser.add_argument("-t", "--test", type = str, metavar = "-", help = "If yes, csv test list is used [y/n]", default = "n")

# parser.add_argument("-r", "--run_list", type = str, required = True, metavar = "-", help = "path to the csv file that contains the run numbers")

args = parser.parse_args()
##########################################################################################


######################################## Error messages and warnings ########################################
if args.input == "ps" and args.data_type == "int8":
    raise ValueError("-i ps -dt int8 -> using pattern spectra from int8 CTA images is not supported")
##########################################################################################

######################################## Define some strings based on the input of the user ########################################
if args.name != None:
    string_name = f"_{args.name}"
else:
    string_name = ""

if args.data_type == "int8":
    string_data_type = "_int8"
else:
    string_data_type = ""

# if args.energy_range != [0.02, 300]:
#     string_energy_range = f"_{args.energy_range[0]}-{args.energy_range[1]}_TeV"
# else:
#     string_energy_range = ""

if args.input == "cta":
    print(f"################### Input summary ################### \nMode: {args.mode} \nInput: CTA images \nData type: {args.data_type} \nEnergy range: {args.energy_range} TeV \nEpochs: {args.epochs} \nTest run: {args.test}")
    string_input = "iact_images"
    if args.telescope_mode == "stereo_sum_cta":
        string_input_short = "_images"
    elif args.telescope_mode == "mono":
        string_input_short = "_images_mono"
    string_ps_input = ""
    string_table_column = "image"
elif args.input == "ps":
    print(f"################### Input summary ################### \nMode: {args.mode} \nInput: pattern spectra \nEnergy range: {args.energy_range} TeV \nAttribute: {args.attribute} \nDomain lower: {args.domain_lower} \nDomain higher: {args.domain_higher} \nMapper: {args.mapper} \nSize: {args.size} \nFilter: {args.filter} \nEpochs: {args.epochs} \nTest run: {args.test}")
    string_input = "pattern_spectra"
    if args.telescope_mode == "stereo_sum_cta":
        string_input_short = "_ps_float"
    elif args.telescope_mode == "mono":
        string_input_short = "_ps_float_mono"
    elif args.telescope_mode == "stereo_sum_ps":
        string_input_short = "_ps_float_stereo_sum"
    string_ps_input = f"a_{args.attribute[0]}_{args.attribute[1]}__dl_{args.domain_lower[0]}_{args.domain_lower[1]}__dh_{args.domain_higher[0]}_{args.domain_higher[1]}__m_{args.mapper[0]}_{args.mapper[1]}__n_{args.size[0]}_{args.size[1]}__f_{args.filter}/"
    string_table_column = "pattern spectrum"
##########################################################################################

######################################## Make some folders ########################################
path_results = f"dm-finder/cnn/{string_input}/{args.mode}/results/" + string_ps_input + f"{string_name[1:]}/"
path_history = f"dm-finder/cnn/{string_input}/{args.mode}/history/" + string_ps_input
path_model = f"dm-finder/cnn/{string_input}/{args.mode}/model/" + string_ps_input
path_output = f"dm-finder/cnn/{string_input}/{args.mode}/output/" + string_ps_input

os.makedirs(path_results, exist_ok = True)
os.makedirs(path_history, exist_ok = True)
os.makedirs(path_model, exist_ok = True)
os.makedirs(path_output, exist_ok = True)
##########################################################################################

######################################## Load and prepare dataset ########################################
# import data
if args.mode == "energy":
    if args.test == "y":
        filename_run_csv = f"dm-finder/scripts/run_lists/gamma_run_list_test.csv"
    elif args.telescope_mode == "mono":
        filename_run_csv = f"dm-finder/scripts/run_lists/gamma_run_list_mono.csv"
    else: 
        filename_run_csv = f"dm-finder/scripts/run_lists/gamma_run_list.csv"
    run = pd.read_csv(filename_run_csv)
    run = run.to_numpy().reshape(len(run))

    if args.run != None:
        run = args.run[0]

    print(filename_run_csv)

    table = pd.DataFrame()
    for r in range(len(run)): # len(run)
        run_filename = f"gamma_20deg_0deg_run{run[r]}___cta-prod5-paranal_desert-2147m-Paranal-dark_merged.DL1"
        input_filename = f"dm-finder/cnn/{string_input}/input/gamma/" + string_ps_input + run_filename + string_input_short + string_data_type + ".h5"

        table_individual_run = pd.read_hdf(input_filename)
        print(f"Number of events in Run {run[r]}:", len(table_individual_run))
        table = table.append(table_individual_run, ignore_index = True)
    
    print("Total number of events:", len(table))

elif args.mode == "separation":
    particle_type = np.array(["gamma_diffuse", "proton"])

    table = pd.DataFrame()
    events_count = np.array([0, 0])
    for p in range(len(particle_type)):
        if args.test == "y":
            filename_run_csv = f"dm-finder/scripts/run_lists/{particle_type[p]}_run_list_test.csv"
        elif args.telescope_mode == "mono":
            filename_run_csv = f"dm-finder/scripts/run_lists/{particle_type[p]}_run_list_mono.csv"
        else: 
            filename_run_csv = f"dm-finder/scripts/run_lists/{particle_type[p]}_run_list.csv"
        run = pd.read_csv(filename_run_csv)
        run = run.to_numpy().reshape(len(run))

        print(filename_run_csv)

        for r in range(len(run)):
            run_filename = f"{particle_type[p]}_20deg_0deg_run{run[r]}___cta-prod5-paranal_desert-2147m-Paranal-dark_merged.DL1"
            input_filename = f"dm-finder/cnn/{string_input}/input/{particle_type[p]}/" + string_ps_input + run_filename + string_input_short + string_data_type + ".h5"

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

    EnergyDistributionSeparation(table, f"dm-finder/cnn/{string_input}/{args.mode}/results/" + string_ps_input + f"{string_name[1:]}/" + "total_energy_distribution" + string_data_type + string_name + ".png")

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

# input features
X = [[]] * len(table)
for i in range(len(table)):
    X[i] = table[string_table_column][i]
X = np.asarray(X)

test_split_percentage = 1 / 10

if args.mode == "energy":
    # output label: log10(true energy)
    Y = np.asarray(table["true_energy"])
    Y = np.log10(np.asarray(table["true_energy"]))

    # plot a few examples
    ExamplesEnergy(X, Y, f"dm-finder/cnn/{string_input}/{args.mode}/results/{string_ps_input}/{string_name[1:]}/input_examples" + string_data_type + string_name + ".png")
    # display total energy distribution of data set
    EnergyDistributionEnergy(Y, f"dm-finder/cnn/{string_input}/{args.mode}/results/" + string_ps_input + f"{string_name[1:]}/" + "total_energy_distribution" + string_data_type + string_name + ".png")
elif args.mode == "separation":
    # output label: particle or gammaness (1 = gamma, 0 = proton)
    Y = np.asarray(table["particle"])
    Y = keras.utils.to_categorical(Y, 2)
    energy_true_test = np.asarray(table["true_energy"])[int(-len(table) * test_split_percentage):]

    # plot a few examples
    ExamplesSeparation(X, Y, f"dm-finder/cnn/{string_input}/{args.mode}/results/{string_ps_input}/{string_name[1:]}/input_examples" + string_data_type + string_name + ".png")

# reshape X data
X_shape = np.shape(X)
X = X.reshape(-1, X_shape[1], X_shape[2], 1)

# # hold out 10 percent as test data and extract the corresponding run, obs_id, event_id (and tel_id)
X_train, X_test = np.split(X, [int(-len(table) * test_split_percentage)])
Y_train, Y_test = np.split(Y, [int(-len(table) * test_split_percentage)])

print("Number training events: ", len(X_train))
print("Number test events: ", len(X_test))

if args.telescope_mode != "mono":
    run_test, obs_id_test, event_id_test = table.tail(int(len(table) * test_split_percentage))["run"], table.tail(int(len(table) * test_split_percentage))["obs_id"], table.tail(int(len(table) * test_split_percentage))["event_id"]
    run_test.reset_index(drop = True, inplace = True)
    obs_id_test.reset_index(drop = True, inplace = True)
    event_id_test.reset_index(drop = True, inplace = True)
    run_test, obs_id_test, event_id_test = np.asarray(run_test), np.asarray(obs_id_test), np.asarray(event_id_test)
elif args.telescope_mode == "mono":
    run_test, obs_id_test, event_id_test, tel_id_test = table.tail(int(len(table) * test_split_percentage))["run"], table.tail(int(len(table) * test_split_percentage))["obs_id"], table.tail(int(len(table) * test_split_percentage))["event_id"], table.tail(int(len(table) * test_split_percentage))["tel_id"]
    run_test.reset_index(drop = True, inplace = True)
    obs_id_test.reset_index(drop = True, inplace = True)
    event_id_test.reset_index(drop = True, inplace = True)
    tel_id_test.reset_index(drop = True, inplace = True)
    run_test, obs_id_test, event_id_test, tel_id_test = np.asarray(run_test), np.asarray(obs_id_test), np.asarray(event_id_test), np.asarray(tel_id_test)
# # remove strange outlier
# index = np.argmin(Y)
# Y = np.delete(Y, index)

##########################################################################################


######################################## Define a model ########################################
X_shape = np.shape(X_train)
Y_shape = np.shape(Y_train)
input = keras.layers.Input(shape = X_shape[1:])

if args.mode == "energy":
    # # ## cnn architecture number 1 (cnn1) ###
    # # define a suitable network 
    # z = keras.layers.Conv2D(4, # number of filters, the dimensionality of the output space
    #     kernel_size = (3,3), # size of filters 3x3
    #     activation = "relu")(input)
    # zl = [z]

    # for i in range(5):
    #     z = keras.layers.Conv2D(16, 
    #         kernel_size = (3,3), 
    #         padding = "same", # padding, "same" = on, "valid" = off
    #         activation = "relu")(z) 
    #     zl.append(z)
    #     z = keras.layers.concatenate(zl[:], axis=-1)

    # z = keras.layers.GlobalAveragePooling2D()(z)
    # z = keras.layers.Dense(8, activation = "relu")(z)

    # output = keras.layers.Dense(1, name = "energy")(z)

    # # ## cnn architecture number 2 (cnn2), based on VGG13 (Pietro Grespan) ###
    # z = keras.layers.BatchNormalization()(input)
    # z = keras.layers.Conv2D(64, kernel_size = (3,3), strides = 1, padding = "same", activation = "relu")(z)
    # z = keras.layers.Conv2D(64, kernel_size = (3,3), strides = 1, padding = "same", activation = "relu")(z)
    # z = keras.layers.MaxPooling2D(pool_size=(2, 2), strides = 2, padding="same")(z)
    # z = keras.layers.Conv2D(128, kernel_size = (3,3), strides = 1, padding = "same", activation = "relu")(z)
    # z = keras.layers.Conv2D(128, kernel_size = (3,3), strides = 1, padding = "same", activation = "relu")(z)
    # z = keras.layers.MaxPooling2D(pool_size=(2, 2), strides = 2, padding="same")(z)
    # z = keras.layers.Conv2D(256, kernel_size = (3,3), strides = 1, padding = "same", activation = "relu")(z)
    # z = keras.layers.Conv2D(256, kernel_size = (3,3), strides = 1, padding = "same", activation = "relu")(z)
    # z = keras.layers.Conv2D(256, kernel_size = (3,3), strides = 1, padding = "same", activation = "relu")(z)
    # z = keras.layers.MaxPooling2D(pool_size=(2, 2), strides = 2, padding="same")(z)
    # z = keras.layers.Conv2D(512, kernel_size = (3,3), strides = 1, padding = "same", activation = "relu")(z)
    # z = keras.layers.Conv2D(512, kernel_size = (3,3), strides = 1, padding = "same", activation = "relu")(z)
    # z = keras.layers.Conv2D(512, kernel_size = (3,3), strides = 1, padding = "same", activation = "relu")(z)
    # z = keras.layers.MaxPooling2D(pool_size=(2, 2), strides = 2, padding="same")(z)
    # z = keras.layers.Flatten()(z)
    # z = keras.layers.Dense(128, activation = "relu")(z)
    # output = keras.layers.Dense(1, name = "energy")(z)

    # # define the loss function
    # loss = "mse"

    # ## cnn architecture number 3 (cnn3) ###
    # # define a suitable network 
    # z = keras.layers.Conv2D(16, # number of filters, the dimensionality of the output space
    #     kernel_size = (3,3), # size of filters 3x3
    #     activation = "relu", padding = "same")(input)
    # zl = [z]

    # for i in range(5):
    #     z = keras.layers.Conv2D(32, 
    #         kernel_size = (3,3), 
    #         padding = "same", # padding, "same" = on, "valid" = off
    #         activation = "relu")(z) 
    #     zl.append(z)
    #     z = keras.layers.concatenate(zl[:], axis=-1)

    # z = keras.layers.GlobalAveragePooling2D()(z)
    # z = keras.layers.Dense(16, activation = "relu")(z)
    # z = keras.layers.Dense(8, activation = "relu")(z)

    # output = keras.layers.Dense(1, name = "energy")(z)

    #######################################

    # # cnn architecture number 9 (cnn9) ###
    # # define a suitable network 
    # z = keras.layers.Conv2D(16, kernel_size = (3,3), strides = 1, padding = "same", activation = "relu")(input)
    # z = keras.layers.Conv2D(32, kernel_size = (3,3), strides = 1, activation = "relu")(z)
    # z = keras.layers.Conv2D(32, kernel_size = (3,3), strides = 1, activation = "relu")(z)
    # z = keras.layers.AveragePooling2D(pool_size=(2, 2), strides = 1)(z)
    # z = keras.layers.Conv2D(64, kernel_size = (3,3), strides = 1, activation = "relu")(z)
    # z = keras.layers.Conv2D(64, kernel_size = (3,3), strides = 1, activation = "relu")(z)
    # z = keras.layers.AveragePooling2D(pool_size=(2, 2), strides = 1)(z)
    # z = keras.layers.Conv2D(128, kernel_size = (3,3), strides = 1, activation = "relu")(z)
    # z = keras.layers.Conv2D(128, kernel_size = (3,3), strides = 1, activation = "relu")(z)
    # z = keras.layers.AveragePooling2D(pool_size=(2, 2), strides = 1)(z)
    
    # z = keras.layers.GlobalAveragePooling2D()(z)
    # z = keras.layers.Dense(32, activation = "relu")(z)
    # z = keras.layers.Dropout(0.1)(z)
    # z = keras.layers.Dense(16, activation = "relu")(z)
    # z = keras.layers.Dropout(0.1)(z)

    # output = keras.layers.Dense(1, name = "energy")(z)

    # ########################################
    # # ## thin resnet architecture number 1 (trn1) ###
    # z = keras.layers.Conv2D(64, kernel_size = (7, 7), activation = "relu", padding = "same")(input)
    # z = keras.layers.MaxPooling2D(pool_size=(2, 2), strides = 2, padding="same")(z)
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [48, 96], increase_dim = True)
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [48, 96])
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [96, 128], increase_dim = True)
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [96, 128])
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [96, 128])
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [128, 256], increase_dim = True)
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [128, 256])
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [128, 256])
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [256, 512], increase_dim = True)
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [256, 512])
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [256, 512])
    # # z = MaxPooling2D(pool_size=(2, 2), strides = 2, padding="same")(z)
    # z = keras.layers.GlobalMaxPooling2D()(z)
    # # z = Flatten()(z)
    # z = keras.layers.Dense(64, activation = "relu")(z)
    # z = keras.layers.Dropout(0.1)(z)
    # z = keras.layers.Dense(32, activation = "relu")(z)
    # z = keras.layers.Dropout(0.1)(z)
    # z = keras.layers.Dense(16, activation = "relu")(z)
    # z = keras.layers.Dropout(0.1)(z)
    # output = keras.layers.Dense(1, activation="relu", name = "energy")(z)

    ########################################

    # ########################################
    # # ## thin resnet architecture number 2 (trn2) ###
    # z = keras.layers.Conv2D(16, kernel_size = (3, 3), activation = "relu", padding = "same")(input)
    # # z = keras.layers.MaxPooling2D(pool_size=(2, 2), strides = 2, padding="same")(z)
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [32, 64], increase_dim = True)
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [32, 64])
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [64, 128], increase_dim = True)
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [64, 128])
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [64, 128])
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [128, 256], increase_dim = True)
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [128, 256])
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [128, 256])
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [256, 512], increase_dim = True)
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [256, 512])
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [256, 512])
    # # z = MaxPooling2D(pool_size=(2, 2), strides = 2, padding="same")(z)
    # z = keras.layers.GlobalMaxPooling2D()(z)
    # # z = Flatten()(z)
    # # z = keras.layers.Dense(64, activation = "relu")(z)
    # # z = keras.layers.Dropout(0.1)(z)
    # z = keras.layers.Dense(32, activation = "relu")(z)
    # z = keras.layers.Dropout(0.1)(z)
    # z = keras.layers.Dense(16, activation = "relu")(z)
    # z = keras.layers.Dropout(0.1)(z)
    # z = keras.layers.Dense(8, activation = "relu")(z)
    # z = keras.layers.Dropout(0.1)(z)
    # output = keras.layers.Dense(1, activation="relu", name = "energy")(z)

    # ########################################

    # ########################################
    # ## thin resnet architecture number 3 (trn3) ###
    # z = keras.layers.Conv2D(16, kernel_size = (3, 3), activation = "relu", padding = "same")(input)
    # # z = keras.layers.MaxPooling2D(pool_size=(2, 2), strides = 2, padding="same")(z)
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [32, 64], increase_dim = True)
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [32, 64])
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [64, 128], increase_dim = True)
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [64, 128])
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [64, 128])
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [128, 256], increase_dim = True)
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [128, 256])
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [128, 256])
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [256, 512], increase_dim = True)
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [256, 512])
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [256, 512])
    # # z = MaxPooling2D(pool_size=(2, 2), strides = 2, padding="same")(z)
    # z = keras.layers.GlobalAveragePooling2D()(z)
    # # z = Flatten()(z)
    # # z = keras.layers.Dense(64, activation = "relu")(z)
    # # z = keras.layers.Dropout(0.1)(z)
    # # z = keras.layers.Dense(32, activation = "relu")(z)
    # # z = keras.layers.Dropout(0.1)(z)
    # z = keras.layers.Dense(16, activation = "relu")(z)
    # # z = keras.layers.Dropout(0.1)(z)
    # z = keras.layers.Dense(8, activation = "relu")(z)
    # # z = keras.layers.Dropout(0.1)(z)
    # output = keras.layers.Dense(1, name = "gammaness")(z)

    # ########################################

    # ########################################
    # ## thin resnet architecture number 4 (trn4) ###
    # z = keras.layers.Conv2D(16, kernel_size = (3, 3), activation = "relu", padding = "same")(input)
    # # z = keras.layers.MaxPooling2D(pool_size=(2, 2), strides = 2, padding="same")(z)
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [32, 64], increase_dim = True)
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [32, 64])
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [64, 128], increase_dim = True)
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [64, 128])
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [64, 128])
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [128, 256], increase_dim = True)
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [128, 256])
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [128, 256])
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [256, 512], increase_dim = True)
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [256, 512])
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [256, 512])
    # # z = MaxPooling2D(pool_size=(2, 2), strides = 2, padding="same")(z)
    # z = keras.layers.GlobalAveragePooling2D()(z)
    # # z = Flatten()(z)
    # z = keras.layers.Dense(128, activation = "relu")(z)
    # z = keras.layers.Dropout(0.1)(z)
    # z = keras.layers.Dense(64, activation = "relu")(z)
    # z = keras.layers.Dropout(0.1)(z)
    # z = keras.layers.Dense(32, activation = "relu")(z)
    # z = keras.layers.Dropout(0.1)(z)
    # # z = keras.layers.Dense(16, activation = "relu")(z)
    # # z = keras.layers.Dropout(0.1)(z)
    # # z = keras.layers.Dense(8, activation = "relu")(z)
    # # z = keras.layers.Dropout(0.1)(z)
    # output = keras.layers.Dense(1, name = "gammaness")(z)

    # ########################################

    # #######################################
    # ## thin resnet architecture number 5 (trn5) ###
    # z = keras.layers.Conv2D(16, kernel_size = (3, 3), activation = "relu", padding = "same")(input)
    # # z = keras.layers.MaxPooling2D(pool_size=(2, 2), strides = 2, padding="same")(z)
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [32, 64], increase_dim = True)
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [32, 64])
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [64, 128], increase_dim = True)
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [64, 128])
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [64, 128])
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [128, 256], increase_dim = True)
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [128, 256])
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [128, 256])
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [256, 512], increase_dim = True)
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [256, 512])
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [256, 512])
    # # z = MaxPooling2D(pool_size=(2, 2), strides = 2, padding="same")(z)
    # z = keras.layers.GlobalAveragePooling2D()(z)
    # # z = Flatten()(z)
    # z = keras.layers.Dense(64, activation = "relu")(z)
    # # z = keras.layers.Dropout(0.1)(z)
    # z = keras.layers.Dense(32, activation = "relu")(z)
    # # z = keras.layers.Dropout(0.1)(z)
    # # z = keras.layers.Dense(16, activation = "relu")(z)
    # # z = keras.layers.Dropout(0.1)(z)
    # # z = keras.layers.Dense(8, activation = "relu")(z)
    # # z = keras.layers.Dropout(0.1)(z)
    # output = keras.layers.Dense(1, name = "energy")(z)

    # ########################################

    # #######################################
    # ## thin resnet architecture number 6 (trn6) ###
    # z = keras.layers.Conv2D(16, kernel_size = (3, 3), activation = "relu", padding = "same")(input)
    # # z = keras.layers.MaxPooling2D(pool_size=(2, 2), strides = 2, padding="same")(z)
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [32, 64], increase_dim = True)
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [32, 64])
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [64, 128], increase_dim = True)
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [64, 128])
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [64, 128])
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [128, 256], increase_dim = True)
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [128, 256])
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [128, 256])
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [256, 512], increase_dim = True)
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [256, 512])
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [256, 512])
    # # z = MaxPooling2D(pool_size=(2, 2), strides = 2, padding="same")(z)
    # z = keras.layers.GlobalAveragePooling2D()(z)
    # # z = Flatten()(z)
    # # z = keras.layers.Dense(64, activation = "relu")(z)
    # # z = keras.layers.Dropout(0.1)(z)
    # # z = keras.layers.Dense(32, activation = "relu")(z)
    # # z = keras.layers.Dropout(0.1)(z)
    # z = keras.layers.Dense(16, activation = "relu")(z)
    # # z = keras.layers.Dropout(0.1)(z)
    # z = keras.layers.Dense(8, activation = "relu")(z)
    # # z = keras.layers.Dropout(0.1)(z)
    # output = keras.layers.Dense(1, name = "energy")(z)

    # ########################################

    # #######################################
    # ## thin resnet architecture number 7 (trn7) ###
    # z = keras.layers.Conv2D(16, kernel_size = (3, 3), activation = "relu", padding = "same")(input)
    # # z = keras.layers.MaxPooling2D(pool_size=(2, 2), strides = 2, padding="same")(z)
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [32, 64], increase_dim = True)
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [32, 64])
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [64, 128], increase_dim = True)
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [64, 128])
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [64, 128])
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [128, 256], increase_dim = True)
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [128, 256])
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [128, 256])
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [256, 512], increase_dim = True)
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [256, 512])
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [256, 512])
    # # z = MaxPooling2D(pool_size=(2, 2), strides = 2, padding="same")(z)
    # z = keras.layers.GlobalAveragePooling2D()(z)
    # # z = Flatten()(z)
    # z = keras.layers.Dense(64, activation = "relu")(z)
    # z = keras.layers.Dropout(0.1)(z)
    # z = keras.layers.Dense(32, activation = "relu")(z)
    # z = keras.layers.Dropout(0.1)(z)
    # # z = keras.layers.Dense(16, activation = "relu")(z)
    # # # z = keras.layers.Dropout(0.1)(z)
    # # z = keras.layers.Dense(8, activation = "relu")(z)
    # # # z = keras.layers.Dropout(0.1)(z)
    # output = keras.layers.Dense(1, name = "energy")(z)

    # #######################################

    # #######################################
    # ## thin resnet architecture number 8 (trn8) ###
    # z = keras.layers.Conv2D(16, kernel_size = (3, 3), activation = "relu", padding = "same")(input)
    # # z = keras.layers.MaxPooling2D(pool_size=(2, 2), strides = 2, padding="same")(z)
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [32, 64], increase_dim = True)
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [32, 64])
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [64, 128], increase_dim = True)
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [64, 128])
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [64, 128])
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [128, 256], increase_dim = True)
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [128, 256])
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [128, 256])
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [256, 512], increase_dim = True)
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [256, 512])
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [256, 512])
    # # z = MaxPooling2D(pool_size=(2, 2), strides = 2, padding="same")(z)
    # z = keras.layers.GlobalAveragePooling2D()(z)
    # # z = Flatten()(z)
    # z = keras.layers.Dense(64, activation = "relu")(z)
    # # # z = keras.layers.Dropout(0.1)(z)
    # z = keras.layers.Dense(32, activation = "relu")(z)
    # # z = keras.layers.Dropout(0.1)(z)
    # # z = keras.layers.Dense(16, activation = "relu")(z)
    # # # z = keras.layers.Dropout(0.1)(z)
    # # z = keras.layers.Dense(8, activation = "relu")(z)
    # # # z = keras.layers.Dropout(0.1)(z)
    # output = keras.layers.Dense(1, name = "energy")(z)

    # #######################################

    # #######################################
    # ## thin resnet architecture number 9 (trn9) ###
    # z = keras.layers.Conv2D(16, kernel_size = (3, 3), activation = "relu", padding = "same")(input)
    # # z = keras.layers.MaxPooling2D(pool_size=(2, 2), strides = 2, padding="same")(z)
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [32, 64], increase_dim = True)
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [32, 64])
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [64, 128], increase_dim = True)
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [64, 128])
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [64, 128])
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [128, 256], increase_dim = True)
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [128, 256])
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [128, 256])
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [256, 512], increase_dim = True)
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [256, 512])
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [256, 512])
    # # z = MaxPooling2D(pool_size=(2, 2), strides = 2, padding="same")(z)
    # z = keras.layers.GlobalAveragePooling2D()(z)
    # # z = Flatten()(z)
    # z = keras.layers.Dense(64, activation = "relu")(z)
    # # # z = keras.layers.Dropout(0.1)(z)
    # z = keras.layers.Dense(32, activation = "relu")(z)
    # # z = keras.layers.Dropout(0.1)(z)
    # # z = keras.layers.Dense(16, activation = "relu")(z)
    # # # z = keras.layers.Dropout(0.1)(z)
    # # z = keras.layers.Dense(8, activation = "relu")(z)
    # # # z = keras.layers.Dropout(0.1)(z)
    # output = keras.layers.Dense(1, name = "energy")(z)

    # #######################################

    #######################################
    ## thin resnet architecture number 10 (trn10) ###
    z = keras.layers.Conv2D(32, kernel_size = (3, 3), activation = "relu", padding = "same")(input)
    # z = keras.layers.MaxPooling2D(pool_size=(2, 2), strides = 2, padding="same")(z)
    z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [64, 128], increase_dim = True)
    z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [64, 128])
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [128, 256], increase_dim = True)
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [128, 256])
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [64, 128], increase_dim = True)
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [64, 128])
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [64, 128])
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [64, 128])
    #z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [128, 256], increase_dim = True)
    #z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [128, 256])
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [128, 256])
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [256, 512], increase_dim = True)
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [256, 512])
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [256, 512])
    # z = MaxPooling2D(pool_size=(2, 2), strides = 2, padding="same")(z)
    z = keras.layers.GlobalAveragePooling2D()(z)
    # z = Flatten()(z)
    z = keras.layers.Dense(64, activation = "relu")(z)
    # # z = keras.layers.Dropout(0.1)(z)
    z = keras.layers.Dense(32, activation = "relu")(z)
    # z = keras.layers.Dropout(0.1)(z)
    # z = keras.layers.Dense(16, activation = "relu")(z)
    # # z = keras.layers.Dropout(0.1)(z)
    # z = keras.layers.Dense(8, activation = "relu")(z)
    # # z = keras.layers.Dropout(0.1)(z)
    output = keras.layers.Dense(1, name = "energy")(z)

    #######################################

    # #######################################
    # ## thin resnet architecture number 11 (trn11) ###
    # z = keras.layers.Conv2D(16, kernel_size = (3, 3), activation = "relu", padding = "same")(input)
    # # z = keras.layers.MaxPooling2D(pool_size=(2, 2), strides = 2, padding="same")(z)
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [32, 64], increase_dim = True)
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [32, 64])
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [32, 64])
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [32, 64])
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [32, 64])
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [128, 256], increase_dim = True)
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [128, 256])
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [64, 128], increase_dim = True)
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [64, 128])
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [64, 128])
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [64, 128])
    # #z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [128, 256], increase_dim = True)
    # #z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [128, 256])
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [128, 256])
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [256, 512], increase_dim = True)
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [256, 512])
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [256, 512])
    # # z = MaxPooling2D(pool_size=(2, 2), strides = 2, padding="same")(z)
    # z = keras.layers.GlobalAveragePooling2D()(z)
    # # z = Flatten()(z)
    # z = keras.layers.Dense(64, activation = "relu")(z)
    # # # z = keras.layers.Dropout(0.1)(z)
    # z = keras.layers.Dense(32, activation = "relu")(z)
    # # z = keras.layers.Dropout(0.1)(z)
    # # z = keras.layers.Dense(16, activation = "relu")(z)
    # # # z = keras.layers.Dropout(0.1)(z)
    # # z = keras.layers.Dense(8, activation = "relu")(z)
    # # # z = keras.layers.Dropout(0.1)(z)
    # output = keras.layers.Dense(1, name = "energy")(z)

    # #######################################

    # define the loss function
    loss = "mse"

elif args.mode == "separation":

    # ## cnn architecture number 1 (cnn1) ###
    # # define a suitable network 
    # z = keras.layers.Conv2D(4, # number of filters, the dimensionality of the output space
    #     kernel_size = (3,3), # size of filters 3x3
    #     padding = "same", 
    #     activation = "relu")(input)
    # zl = [z]

    # for i in range(1):
    #     z = keras.layers.Conv2D(16, 
    #         kernel_size = (3,3), 
    #         padding = "same", # padding, "same" = on, "valid" = off
    #         activation = "relu")(z) 
    #     zl.append(z)
    #     z = keras.layers.concatenate(zl[:], axis = -1)

    # z = keras.layers.GlobalAveragePooling2D()(z)

    # z = keras.layers.Dense(8, activation = "relu")(z)

    # output = keras.layers.Dense(2, activation = "softmax", name = "gammaness")(z)
    # #######################################

    # # cnn architecture number 2 (cnn2) ###
    # z = keras.layers.Conv2D(4, kernel_size = (3,3), activation = "relu", padding = "same")(input)
    # z = keras.layers.Conv2D(16, kernel_size = (3,3), activation = "relu")(z)
    # z = keras.layers.Conv2D(64, kernel_size = (3,3), activation = "relu")(z)
    # z = keras.layers.GlobalAveragePooling2D()(z)

    # z = keras.layers.Dense(64, activation = "relu")(z)
    # z = keras.layers.Dense(32, activation = "relu")(z)
    # z = keras.layers.Dense(16, activation = "relu")(z)

    # output = keras.layers.Dense(2, activation="softmax", name = "gammaness")(z)
    ########################################

    #######################################

    # # cnn architecture number 3 (cnn3) ###
    # z = keras.layers.Conv2D(4, kernel_size = (3,3), activation = "relu", padding = "same")(input)
    # z = keras.layers.Conv2D(16, kernel_size = (3,3), activation = "relu")(z)
    # z = keras.layers.Conv2D(64, kernel_size = (3,3), activation = "relu")(z)
    # z = keras.layers.Flatten()(z)

    # z = keras.layers.Dense(256, activation = "relu")(z)
    # z = keras.layers.Dropout(0.1)(z)
    # z = keras.layers.Dense(128, activation = "relu")(z)
    # z = keras.layers.Dropout(0.1)(z)
    # z = keras.layers.Dense(64, activation = "relu")(z)
    # z = keras.layers.Dropout(0.1)(z)
    # z = keras.layers.Dense(32, activation = "relu")(z)
    # z = keras.layers.Dropout(0.1)(z)
    # z = keras.layers.Dense(16, activation = "relu")(z)

    # output = keras.layers.Dense(2, activation="softmax", name = "gammaness")(z)
    ####################################### 

    ######################################## 
    # # cnn architecture number 2 (cnn4) ###
    # z = keras.layers.Conv2D(16, kernel_size = (3,3), activation = "relu", padding = "same")(input)
    # zl = [z]
    # # z = keras.layers.Conv2D(16, kernel_size = (3,3), activation = "relu", padding = "same")(z)
    # # z = keras.layers.MaxPooling2D(pool_size = (2, 2), strides = 2, padding = "same")(z)
    # z = keras.layers.Conv2D(32, kernel_size = (3,3), activation = "relu", padding = "same")(z)
    # zl.append(z)  
    # # z = keras.layers.Conv2D(32, kernel_size = (3,3), activation = "relu", padding = "same")(z)
    # # z = keras.layers.MaxPooling2D(pool_size = (2, 2), strides = 2, padding = "same")(z)
    # z = keras.layers.Conv2D(64, kernel_size = (3,3), activation = "relu", padding = "same")(z)
    # zl.append(z)  
    # # z = keras.layers.Conv2D(64, kernel_size = (3,3), activation = "relu", padding = "same")(z)
    # # z = keras.layers.Conv2D(64, kernel_size = (3,3), activation = "relu", padding = "same")(z)
    # # z = keras.layers.MaxPooling2D(pool_size = (2, 2), strides = 2, padding = "same")(z)
    # z = keras.layers.concatenate(zl[:], axis = -1)
    # # z = keras.layers.MaxPooling2D(pool_size = (2, 2), strides = 2, padding = "same")(z)

    # z = keras.layers.Flatten()(z)
    # # z = keras.layers.Dense(128, activation = "relu")(z)
    # z = keras.layers.Dense(64, activation = "relu")(z)
    # z = keras.layers.Dense(32, activation = "relu")(z)
    # z = keras.layers.Dense(16, activation = "relu")(z)

    # output = keras.layers.Dense(2, activation="softmax", name = "gammaness")(z)   
    ########################################

    ########################################
    # ### fnn architecture number 1 (fnn1) ###
    # z = keras.layers.Flatten()(input)
    # z = keras.layers.Dense(564, activation = "relu")(z)
    # z = keras.layers.Dense(564, activation = "relu")(z)
    # z = keras.layers.Dense(256, activation = "relu")(z)
    # z = keras.layers.Dense(128, activation = "relu")(z)
    # # z = keras.layers.AveragePooling1D()(z)

    # z = keras.layers.Dense(64, activation = "relu")(z)
    # z = keras.layers.Dense(32, activation = "relu")(z)
    # z = keras.layers.Dense(16, activation = "relu")(z)
    # # z = keras.layers.AveragePooling1D()(z)

    # output = keras.layers.Dense(2, activation="softmax", name = "gammaness")(z)
    # ########################################

    ##########################################
    # ### cnn architecture number 5 (cnn5) ###
    # z = keras.layers.Conv2D(4, # number of filters, the dimensionality of the output space
    #     kernel_size = (3,3), # size of filters 3x3
    #     activation = "relu")(input)
    # zl = [z]

    # for i in range(5):
    #     z = keras.layers.Conv2D(16, 
    #         kernel_size = (3,3), 
    #         padding = "same", # padding, "same" = on, "valid" = off
    #         activation = "relu")(z) 
    #     zl.append(z)
    #     z = keras.layers.concatenate(zl[:], axis=-1)

    # z = keras.layers.GlobalAveragePooling2D()(z)
    # z = keras.layers.Dense(8, activation = "relu")(z)

    # output = keras.layers.Dense(2, activation="softmax", name = "gammaness")(z)
    #########################################

    # #######################################

    # # cnn architecture number 6 (cnn6) ###
    # z = keras.layers.BatchNormalization()
    # z = keras.layers.Conv2D(64, kernel_size = (3,3), activation = "relu", padding = "same")(input)
    # z = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid")(z)
    # z = keras.layers.Conv2D(128, kernel_size = (3,3), activation = "relu", padding = "same")(z)
    # z = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid")(z)
    # z = keras.layers.Conv2D(256, kernel_size = (3,3), activation = "relu", padding = "same")(z)
    # z = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid")(z)
    # # z = keras.layers.Conv2D(512, kernel_size = (3,3), activation = "relu", padding = "same")(z)
    # # z = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid")(z)

    # z = keras.layers.Flatten()(z)
    # z = keras.layers.Dense(512, activation = "relu")(z)

    # output = keras.layers.Dense(2, activation="softmax", name = "gammaness")(z)
    ########################################

    # ########################################
    # # ## cnn architecture number 2 (cnn7), based on VGG13 (Pietro Grespan) ###
    # z = keras.layers.BatchNormalization()(input)
    # z = keras.layers.Conv2D(64, kernel_size = (3,3), strides = 1, padding = "same", activation = "relu")(z)
    # z = keras.layers.Conv2D(64, kernel_size = (3,3), strides = 1, padding = "same", activation = "relu")(z)
    # z = keras.layers.MaxPooling2D(pool_size=(2, 2), strides = 2, padding="same")(z)
    # z = keras.layers.Conv2D(128, kernel_size = (3,3), strides = 1, padding = "same", activation = "relu")(z)
    # z = keras.layers.Conv2D(128, kernel_size = (3,3), strides = 1, padding = "same", activation = "relu")(z)
    # z = keras.layers.MaxPooling2D(pool_size=(2, 2), strides = 2, padding="same")(z)
    # z = keras.layers.Conv2D(256, kernel_size = (3,3), strides = 1, padding = "same", activation = "relu")(z)
    # z = keras.layers.Conv2D(256, kernel_size = (3,3), strides = 1, padding = "same", activation = "relu")(z)
    # z = keras.layers.Conv2D(256, kernel_size = (3,3), strides = 1, padding = "same", activation = "relu")(z)
    # z = keras.layers.MaxPooling2D(pool_size=(2, 2), strides = 2, padding="same")(z)
    # z = keras.layers.Conv2D(512, kernel_size = (3,3), strides = 1, padding = "same", activation = "relu")(z)
    # z = keras.layers.Conv2D(512, kernel_size = (3,3), strides = 1, padding = "same", activation = "relu")(z)
    # z = keras.layers.Conv2D(512, kernel_size = (3,3), strides = 1, padding = "same", activation = "relu")(z)
    # z = keras.layers.MaxPooling2D(pool_size=(2, 2), strides = 2, padding="same")(z)
    # z = keras.layers.Flatten()(z)
    # z = keras.layers.Dense(128, activation = "relu")(z)

    # output = keras.layers.Dense(2, activation="softmax", name = "gammaness")(z)

    # ########################################

    ######################################
    # ## cnn architecture number 8 (cnn8) ###
    # # # define a suitable network 
    # z = keras.layers.Conv2D(16, # number of filters, the dimensionality of the output space
    #     kernel_size = (3,3), # size of filters 3x3
    #     activation = "relu", padding = "same")(input)
    # zl = [z]

    # for i in range(5):
    #     z = keras.layers.Conv2D(32, 
    #         kernel_size = (3,3), 
    #         padding = "same", # padding, "same" = on, "valid" = off
    #         activation = "relu")(z) 
    #     zl.append(z)
    #     z = keras.layers.concatenate(zl[:], axis=-1)

    # z = keras.layers.GlobalAveragePooling2D()(z)
    # z = keras.layers.Dense(16, activation = "relu")(z)
    # z = keras.layers.Dense(8, activation = "relu")(z)

    # output = keras.layers.Dense(2, activation="softmax", name = "gammaness")(z)

    # ########################################
    # # cnn architecture number 9 (cnn9) ###
    # # define a suitable network 
    # z = keras.layers.Conv2D(16, kernel_size = (3,3), strides = 1, padding = "same", activation = "relu")(input)
    # z = keras.layers.Conv2D(32, kernel_size = (3,3), strides = 1, activation = "relu")(z)
    # z = keras.layers.Conv2D(32, kernel_size = (3,3), strides = 1, activation = "relu")(z)
    # z = keras.layers.AveragePooling2D(pool_size=(2, 2), strides = 1)(z)
    # z = keras.layers.Conv2D(64, kernel_size = (3,3), strides = 1, activation = "relu")(z)
    # z = keras.layers.Conv2D(64, kernel_size = (3,3), strides = 1, activation = "relu")(z)
    # z = keras.layers.AveragePooling2D(pool_size=(2, 2), strides = 1)(z)
    # z = keras.layers.Conv2D(128, kernel_size = (3,3), strides = 1, activation = "relu")(z)
    # z = keras.layers.Conv2D(128, kernel_size = (3,3), strides = 1, activation = "relu")(z)
    # z = keras.layers.AveragePooling2D(pool_size=(2, 2), strides = 1)(z)
    
    # z = keras.layers.GlobalAveragePooling2D()(z)
    # z = keras.layers.Dense(32, activation = "relu")(z)
    # z = keras.layers.Dropout(0.1)(z)
    # z = keras.layers.Dense(16, activation = "relu")(z)
    # z = keras.layers.Dropout(0.1)(z)

    # output = keras.layers.Dense(2, activation="softmax", name = "gammaness")(z)

    # ########################################

    ########################################
    # # ## thin resnet architecture number 1 (trn1) ###
    # z = keras.layers.Conv2D(64, kernel_size = (7, 7), activation = "relu", padding = "same")(input)
    # z = keras.layers.MaxPooling2D(pool_size=(2, 2), strides = 2, padding="same")(z)
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [48, 96], increase_dim = True)
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [48, 96])
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [96, 128], increase_dim = True)
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [96, 128])
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [96, 128])
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [128, 256], increase_dim = True)
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [128, 256])
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [128, 256])
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [256, 512], increase_dim = True)
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [256, 512])
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [256, 512])
    # # z = MaxPooling2D(pool_size=(2, 2), strides = 2, padding="same")(z)
    # z = keras.layers.GlobalMaxPooling2D()(z)
    # # z = Flatten()(z)
    # z = keras.layers.Dense(64, activation = "relu")(z)
    # z = keras.layers.Dropout(0.1)(z)
    # z = keras.layers.Dense(32, activation = "relu")(z)
    # z = keras.layers.Dropout(0.1)(z)
    # z = keras.layers.Dense(16, activation = "relu")(z)
    # z = keras.layers.Dropout(0.1)(z)
    # output = keras.layers.Dense(2, activation="softmax", name = "gammaness")(z)

    ########################################

    # ########################################
    # # ## thin resnet architecture number 2 (trn2) ###
    # z = keras.layers.Conv2D(16, kernel_size = (3, 3), activation = "relu", padding = "same")(input)
    # # z = keras.layers.MaxPooling2D(pool_size=(2, 2), strides = 2, padding="same")(z)
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [32, 64], increase_dim = True)
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [32, 64])
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [64, 128], increase_dim = True)
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [64, 128])
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [64, 128])
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [128, 256], increase_dim = True)
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [128, 256])
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [128, 256])
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [256, 512], increase_dim = True)
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [256, 512])
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [256, 512])
    # # z = MaxPooling2D(pool_size=(2, 2), strides = 2, padding="same")(z)
    # z = keras.layers.GlobalMaxPooling2D()(z)
    # # z = Flatten()(z)
    # # z = keras.layers.Dense(64, activation = "relu")(z)
    # # z = keras.layers.Dropout(0.1)(z)
    # z = keras.layers.Dense(32, activation = "relu")(z)
    # z = keras.layers.Dropout(0.1)(z)
    # z = keras.layers.Dense(16, activation = "relu")(z)
    # z = keras.layers.Dropout(0.1)(z)
    # z = keras.layers.Dense(8, activation = "relu")(z)
    # z = keras.layers.Dropout(0.1)(z)
    # output = keras.layers.Dense(2, activation="softmax", name = "gammaness")(z)

    # ########################################

    # #######################################
    # ## thin resnet architecture number 3 (trn3) ###
    # z = keras.layers.Conv2D(16, kernel_size = (3, 3), activation = "relu", padding = "same")(input)
    # # z = keras.layers.MaxPooling2D(pool_size=(2, 2), strides = 2, padding="same")(z)
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [32, 64], increase_dim = True)
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [32, 64])
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [64, 128], increase_dim = True)
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [64, 128])
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [64, 128])
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [128, 256], increase_dim = True)
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [128, 256])
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [128, 256])
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [256, 512], increase_dim = True)
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [256, 512])
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [256, 512])
    # # z = MaxPooling2D(pool_size=(2, 2), strides = 2, padding="same")(z)
    # z = keras.layers.GlobalAveragePooling2D()(z)
    # # z = Flatten()(z)
    # # z = keras.layers.Dense(64, activation = "relu")(z)
    # # z = keras.layers.Dropout(0.1)(z)
    # # z = keras.layers.Dense(32, activation = "relu")(z)
    # # z = keras.layers.Dropout(0.1)(z)
    # z = keras.layers.Dense(16, activation = "relu")(z)
    # # z = keras.layers.Dropout(0.1)(z)
    # z = keras.layers.Dense(8, activation = "relu")(z)
    # # z = keras.layers.Dropout(0.1)(z)
    # output = keras.layers.Dense(2, activation="softmax", name = "gammaness")(z)

    # ########################################

    # ########################################
    # ## thin resnet architecture number 4 (trn4) ###
    # z = keras.layers.Conv2D(16, kernel_size = (3, 3), activation = "relu", padding = "same")(input)
    # # z = keras.layers.MaxPooling2D(pool_size=(2, 2), strides = 2, padding="same")(z)
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [32, 64], increase_dim = True)
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [32, 64])
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [64, 128], increase_dim = True)
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [64, 128])
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [64, 128])
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [128, 256], increase_dim = True)
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [128, 256])
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [128, 256])
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [256, 512], increase_dim = True)
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [256, 512])
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [256, 512])
    # # z = MaxPooling2D(pool_size=(2, 2), strides = 2, padding="same")(z)
    # z = keras.layers.GlobalAveragePooling2D()(z)
    # # z = Flatten()(z)
    # z = keras.layers.Dense(128, activation = "relu")(z)
    # z = keras.layers.Dropout(0.1)(z)
    # z = keras.layers.Dense(64, activation = "relu")(z)
    # z = keras.layers.Dropout(0.1)(z)
    # z = keras.layers.Dense(32, activation = "relu")(z)
    # z = keras.layers.Dropout(0.1)(z)
    # # z = keras.layers.Dense(16, activation = "relu")(z)
    # # z = keras.layers.Dropout(0.1)(z)
    # # z = keras.layers.Dense(8, activation = "relu")(z)
    # # z = keras.layers.Dropout(0.1)(z)
    # output = keras.layers.Dense(2, activation="softmax", name = "gammaness")(z)

    # ########################################

    # #######################################
    # ## thin resnet architecture number 5 (trn5) ###
    # z = keras.layers.Conv2D(16, kernel_size = (3, 3), activation = "relu", padding = "same")(input)
    # # z = keras.layers.MaxPooling2D(pool_size=(2, 2), strides = 2, padding="same")(z)
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [32, 64], increase_dim = True)
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [32, 64])
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [64, 128], increase_dim = True)
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [64, 128])
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [64, 128])
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [128, 256], increase_dim = True)
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [128, 256])
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [128, 256])
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [256, 512], increase_dim = True)
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [256, 512])
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [256, 512])
    # # z = MaxPooling2D(pool_size=(2, 2), strides = 2, padding="same")(z)
    # z = keras.layers.GlobalAveragePooling2D()(z)
    # # z = Flatten()(z)
    # z = keras.layers.Dense(64, activation = "relu")(z)
    # # z = keras.layers.Dropout(0.1)(z)
    # z = keras.layers.Dense(32, activation = "relu")(z)
    # # z = keras.layers.Dropout(0.1)(z)
    # # z = keras.layers.Dense(16, activation = "relu")(z)
    # # z = keras.layers.Dropout(0.1)(z)
    # # z = keras.layers.Dense(8, activation = "relu")(z)
    # # z = keras.layers.Dropout(0.1)(z)
    # output = keras.layers.Dense(2, activation="softmax", name = "gammaness")(z)

    # ########################################

    # #######################################
    # ## thin resnet architecture number 6 (trn6) ###
    # z = keras.layers.Conv2D(16, kernel_size = (3, 3), activation = "relu", padding = "same")(input)
    # # z = keras.layers.MaxPooling2D(pool_size=(2, 2), strides = 2, padding="same")(z)
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [32, 64], increase_dim = True)
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [32, 64])
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [64, 128], increase_dim = True)
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [64, 128])
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [64, 128])
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [128, 256], increase_dim = True)
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [128, 256])
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [128, 256])
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [256, 512], increase_dim = True)
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [256, 512])
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [256, 512])
    # # z = MaxPooling2D(pool_size=(2, 2), strides = 2, padding="same")(z)
    # z = keras.layers.GlobalAveragePooling2D()(z)
    # # z = Flatten()(z)
    # # z = keras.layers.Dense(64, activation = "relu")(z)
    # # z = keras.layers.Dropout(0.1)(z)
    # # z = keras.layers.Dense(32, activation = "relu")(z)
    # # z = keras.layers.Dropout(0.1)(z)
    # z = keras.layers.Dense(16, activation = "relu")(z)
    # # z = keras.layers.Dropout(0.1)(z)
    # z = keras.layers.Dense(8, activation = "relu")(z)
    # # z = keras.layers.Dropout(0.1)(z)
    # output = keras.layers.Dense(2, activation="softmax", name = "gammaness")(z)

    # ########################################

    # #######################################
    # ## thin resnet architecture number 7 (trn7) ###
    # z = keras.layers.Conv2D(16, kernel_size = (3, 3), activation = "relu", padding = "same")(input)
    # # z = keras.layers.MaxPooling2D(pool_size=(2, 2), strides = 2, padding="same")(z)
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [32, 64], increase_dim = True)
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [32, 64])
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [64, 128], increase_dim = True)
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [64, 128])
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [64, 128])
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [128, 256], increase_dim = True)
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [128, 256])
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [128, 256])
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [256, 512], increase_dim = True)
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [256, 512])
    # # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [256, 512])
    # # z = MaxPooling2D(pool_size=(2, 2), strides = 2, padding="same")(z)
    # z = keras.layers.GlobalAveragePooling2D()(z)
    # # z = keras.layers.Flatten()(z)
    # z = keras.layers.Dense(64, activation = "relu")(z)
    # z = keras.layers.Dropout(0.1)(z)
    # z = keras.layers.Dense(32, activation = "relu")(z)
    # z = keras.layers.Dropout(0.1)(z)
    # # z = keras.layers.Dense(16, activation = "relu")(z)
    # # # z = keras.layers.Dropout(0.1)(z)
    # # z = keras.layers.Dense(8, activation = "relu")(z)
    # # # z = keras.layers.Dropout(0.1)(z)
    # output = keras.layers.Dense(2, activation="softmax", name = "gammaness")(z)

    # ########################################

    ########################################
    ## thin resnet architecture number 8 (trn8) ###
    z = keras.layers.Conv2D(16, kernel_size = (3, 3), activation = "relu", padding = "same")(input)
    # z = keras.layers.MaxPooling2D(pool_size=(2, 2), strides = 2, padding="same")(z)
    z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [32, 64], increase_dim = True)
    z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [32, 64])
    z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [64, 128], increase_dim = True)
    z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [64, 128])
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [64, 128])
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [128, 256], increase_dim = True)
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [128, 256])
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [128, 256])
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [256, 512], increase_dim = True)
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [256, 512])
    # z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [256, 512])
    # z = MaxPooling2D(pool_size=(2, 2), strides = 2, padding="same")(z)
    z = keras.layers.GlobalAveragePooling2D()(z)
    # z = Flatten()(z)
    z = keras.layers.Dense(64, activation = "relu")(z)
    # # z = keras.layers.Dropout(0.1)(z)
    z = keras.layers.Dense(32, activation = "relu")(z)
    # z = keras.layers.Dropout(0.1)(z)
    # z = keras.layers.Dense(16, activation = "relu")(z)
    # # z = keras.layers.Dropout(0.1)(z)
    # z = keras.layers.Dense(8, activation = "relu")(z)
    # # z = keras.layers.Dropout(0.1)(z)
    output = keras.layers.Dense(2, activation="softmax", name = "gammaness")(z)

    #######################################

    # # define the loss function
    loss = "categorical_crossentropy"
##########################################################################################


######################################## Train the model ########################################
model = keras.models.Model(inputs=input, outputs=output)

print(model.summary())

#weight estimations, relative values for all attributes:
weights = 1

model.compile(
    loss = loss,
    loss_weights = weights,  
    optimizer = keras.optimizers.Adam(learning_rate = 1E-3))

history_path = f"dm-finder/cnn/{string_input}/{args.mode}/history/" + string_ps_input + "history" + string_data_type + string_name + ".csv"

# start timer
start_time = time.time()

fit = model.fit(X_train,
    Y_train,
    batch_size = 32,
    epochs = args.epochs,
    verbose = 2,
    validation_split = 0.1,
    callbacks = [CSVLogger(history_path)])

# end timer and print training time
print("Time spend for training the CNN: ", np.round(time.time() - start_time, 1), "s")

model_path = f"dm-finder/cnn/{string_input}/{args.mode}/model/" + string_ps_input + "model" + string_data_type + string_name + ".h5"

model.save(model_path)
#########################################################################################

######################################## Results ########################################
model_path = f"dm-finder/cnn/{string_input}/{args.mode}/model/" + string_ps_input + "model" + string_data_type + string_name + ".h5"
model = keras.models.load_model(model_path)

losses = model.evaluate(X_test, Y_test, batch_size=128, verbose=0)
# print("Test loss with", title_names[i])
print("loss: %.5e" % losses)

# predict output for test set and undo feature scaling
yp = model.predict(X_test, batch_size=128)

if args.mode == "energy":
    if args.telescope_mode != "mono":
        yp = yp[:, 0]  # remove unnecessary last axis
        header = ["run", "obs_id", "event_id", "log10(E_true / GeV)", "log10(E_rec / GeV)"]
    elif args.telescope_mode == "mono":
        yp = yp[:, 0]  # remove unnecessary last axis
        header = ["run", "obs_id", "event_id", "tel_id", "log10(E_true / GeV)", "log10(E_rec / GeV)"]
elif args.mode == "separation":
    if args.telescope_mode != "mono":
        yp = yp[:, 1]  # get gammaness (or photon score)
        Y_test = np.argmax(Y_test, axis = 1)
        header = ["run", "obs_id", "event_id", "true gammaness", "reconstructed gammaness", "E_true / GeV"]
    elif args.telescope_mode == "mono":
        yp = yp[:, 1]  # get gammaness (or photon score)
        Y_test = np.argmax(Y_test, axis = 1)
        header = ["run", "obs_id", "event_id", "tel_id", "true gammaness", "reconstructed gammaness", "E_true / GeV"]

# create csv output file
Y_test = np.reshape(Y_test, (len(Y_test), 1))
yp = np.reshape(yp, (len(yp), 1))
run_test = np.reshape(run_test, (len(run_test), 1)).astype(int)
obs_id_test = np.reshape(obs_id_test, (len(obs_id_test), 1)).astype(int)
event_id_test = np.reshape(event_id_test, (len(event_id_test), 1)).astype(int)

if args.telescope_mode == "mono":
    tel_id_test = np.reshape(tel_id_test, (len(tel_id_test), 1)).astype(int)

if args.mode == "energy":
    if args.telescope_mode != "mono":
        table_output = np.hstack((run_test, obs_id_test, event_id_test, Y_test, yp))
    elif args.telescope_mode == "mono":
        table_output = np.hstack((run_test, obs_id_test, event_id_test, tel_id_test, Y_test, yp))
elif args.mode == "separation":
    if args.telescope_mode != "mono":
        energy_true_test = np.reshape(energy_true_test, (len(energy_true_test), 1))
        table_output = np.hstack((run_test, obs_id_test, event_id_test, Y_test, yp, energy_true_test))
    elif args.telescope_mode == "mono":
        energy_true_test = np.reshape(energy_true_test, (len(energy_true_test), 1))
        table_output = np.hstack((run_test, obs_id_test, event_id_test, tel_id_test, Y_test, yp, energy_true_test))

pd.DataFrame(table_output).to_csv(f"dm-finder/cnn/{string_input}/{args.mode}/output/" + string_ps_input + "/evaluation" + string_data_type + string_name + ".csv", index = None, header = header)
##########################################################################################
