import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
from tensorflow import keras
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from keras.models import Model
import time
import argparse
from utilities_cnn import *

print("Packages successfully loaded")

######################################## argparse setup ########################################
script_version=1.0
script_descr="""
This script loads data (CTA images or pattern spectra), defines a CNN for energy reconstruction or gamma/proton separation, trains the CNN and puts its predictions in a csv table.
"""

# Open argument parser
parser = argparse.ArgumentParser(description=script_descr)
parser.add_argument("-v", "--version", action="version", version=f"v{script_version}")

# Define expected arguments
parser.add_argument("-m", "--mode", type = str, required = True, metavar = "-", choices = ["energy", "separation"], help = "CNN mode - energy reconstruction or gamma/proton separation [energy, separation]")
parser.add_argument("-i", "--input", type = str, required = True, metavar = "-", choices = ["cta", "ps"], help = "input for the CNN [cta, ps]")
parser.add_argument("-tm", "--telescope_mode", type = str, required = False, metavar = "", choices = ["mono", "stereo_sum_cta", "stereo_sum_ps"], help = "telescope mode [mono, stereo_sum_cta, stereo_sum_ps], default: stereo_sum_cta", default = "stereo_sum_cta")
parser.add_argument("-erg", "--energy_range_gamma", type = float, required = False, metavar = "-", help = "set energy range of gamma events in TeV (energy reco. or sig/bkg sep.), default: 0.5 100", default = [0.5, 100], nargs = 2)
parser.add_argument("-erp", "--energy_range_proton", type = float, required = False, metavar = "-", help = "set energy range of proton events in TeV (sig/bkg sep. only), default: 1.5 100", default = [1.5, 100], nargs = 2)
parser.add_argument("-sctr", "--selection_cuts_train", type = str, required = False, metavar = "-", help = "Name of the selection cuts extracted from the pre_selection_cuty.py script for training")
parser.add_argument("-scte", "--selection_cuts_test", type = str, required = False, metavar = "-", help = "Name of the selection cuts extracted from the pre_selection_cuty.py script for testing")
parser.add_argument("-na", "--name", type = str, required = False, metavar = "-", help = "Name of this particular experiment")
parser.add_argument("-a", "--attribute", type = int, metavar = "-", choices = np.arange(0, 19, dtype = int), help = "Pattern spectra: attribute [0, 1 ... 18] (two required), default: 9 0", default = [9, 0], nargs = 2)
parser.add_argument("-dl", "--domain_lower", type = float, metavar = "-", help = "Pattern spectra: granulometry: domain - start at <value> <value>, default: 0.8 0.8", default = [0.8, 0.8], nargs = 2)
parser.add_argument("-dh", "--domain_higher", type = float, metavar = "-", help = "Pattern spectra: granulometry: domain - end at <value> <value>, default: 7 3000", default = [7., 3000.], nargs = 2)
parser.add_argument("-ma", "--mapper", type = int, metavar = "-", help = "Pattern spectra: granulometry: use lambdamappers <mapper1> <mapper2>, default: 4 4", default = [4, 4], nargs = 2)
parser.add_argument("-n", "--size", type = int, metavar = "-", help = "Pattern spectra: granulometry: size <n1>x<n2>, default: 20 20", default = [20, 20], nargs = 2)
parser.add_argument("-f", "--filter", type = int, metavar = "-", help = "Pattern spectra: Use decision <filter>, default: 3", default = 3, nargs = 1)
parser.add_argument("-e", "--epochs", type = int, metavar = "-", help = "Number of epochs for CNN training, default: 1000", default = 1000)
parser.add_argument("-sp", "--split_percentage", type = float, metavar = "-", help = "Percentage of data that is used for testing, default: 0.1", default = 0.1)
parser.add_argument("-sd", "--small_dataset", type = str, metavar = "-", help = "If yes, csv test list is used [y/n], default: n", default = "n")

args = parser.parse_args()
##########################################################################################


######################################## Define some strings based on the input of the user ########################################
string_name, string_input, string_ps_input, string_input_short, string_table_column = GetUserInputStr(args.name, args.input, args.mode, args.telescope_mode, args.energy_range_gamma, args.energy_range_proton, args.selection_cuts_train, args.selection_cuts_test, args.epochs, args.small_dataset, args.attribute, args.domain_lower, args.domain_higher, args.mapper, args.size, args.filter)
##########################################################################################

######################################## Create some folders ########################################
path_results = f"cnn/{string_input}/{args.mode}/results/" + string_ps_input + f"{args.name}/"
path_history = f"cnn/{string_input}/{args.mode}/history/" + string_ps_input
path_model = f"cnn/{string_input}/{args.mode}/model/" + string_ps_input
path_output = f"cnn/{string_input}/{args.mode}/output/" + string_ps_input

os.makedirs(path_results, exist_ok = True)
os.makedirs(path_history, exist_ok = True)
os.makedirs(path_model, exist_ok = True)
os.makedirs(path_output, exist_ok = True)
##########################################################################################

######################################## Load and prepare dataset ########################################

# import data for energy reconstruction
if args.mode == "energy":
    table_train, table_test = GetDataEnergyReconstruction(args.small_dataset, string_input, string_ps_input, string_input_short, args.energy_range_gamma, args.selection_cuts_train, args.selection_cuts_test, args.telescope_mode, args.split_percentage)

elif args.mode == "separation":
    table_train, table_test = GetDataSeparation(args.small_dataset, args.telescope_mode, string_input, string_ps_input, string_input_short, args.energy_range_gamma, args.energy_range_proton, args.selection_cuts_train, args.selection_cuts_test, args.split_percentage)

print("______________________________________________")

# input features
X_train = np.array(table_train[string_table_column].to_list())
X_test = np.array(table_test[string_table_column].to_list())


if args.mode == "energy":
    # output label: log10(true energy)
    Y_train = np.log10(np.asarray(table_train["true_energy"]))
    Y_test = np.log10(np.asarray(table_test["true_energy"]))

    # plot a few examples
    PlotExamplesEnergy(X_train, Y_train, f"cnn/{string_input}/{args.mode}/results/{string_ps_input}/{string_name[1:]}/input_examples" + string_name + ".pdf")

    # display total energy distribution of data set
    PlotEnergyDistributionEnergy(Y_train, f"cnn/{string_input}/{args.mode}/results/" + string_ps_input + f"{string_name[1:]}/" + "total_energy_distribution" + string_name + ".pdf")

########################################## CLEAN ################################################

elif args.mode == "separation":
    # output label: particle or gammaness (1 = gamma, 0 = proton)
    Y_train = np.asarray(table_train["particle"])
    Y_test = np.asarray(table_test["particle"])
    Y_train, Y_test = keras.utils.to_categorical(Y_train, 2), keras.utils.to_categorical(Y_test, 2)
    energy_true_test = np.asarray(table_test["true_energy"])

    PlotExamplesSeparation(X_train, Y_train, string_input, string_ps_input, string_name, args.mode)

    PlotEnergyDistributionSeparation(table_train, string_input, string_ps_input, string_name, args.mode)

# reshape X data
X_shape = np.shape(X_train)
X_train, X_test = X_train.reshape(-1, X_shape[1], X_shape[2], 1), X_test.reshape(-1, X_shape[1], X_shape[2], 1)

if args.telescope_mode != "mono":
    run_test, obs_id_test, event_id_test = table_test["run"], table_test["obs_id"], table_test["event_id"]
    run_test.reset_index(drop = True, inplace = True)
    obs_id_test.reset_index(drop = True, inplace = True)
    event_id_test.reset_index(drop = True, inplace = True)
    run_test, obs_id_test, event_id_test = np.asarray(run_test), np.asarray(obs_id_test), np.asarray(event_id_test)
else:
    run_test, obs_id_test, event_id_test, tel_id_test = table_test["run"], table_test["obs_id"], table_test["event_id"], table_test["tel_id"]
    run_test.reset_index(drop = True, inplace = True)
    obs_id_test.reset_index(drop = True, inplace = True)
    event_id_test.reset_index(drop = True, inplace = True)
    tel_id_test.reset_index(drop = True, inplace = True)
    run_test, obs_id_test, event_id_test, tel_id_test = np.asarray(run_test), np.asarray(obs_id_test), np.asarray(event_id_test), np.asarray(tel_id_test)

##########################################################################################


######################################## Define a model ########################################
X_shape = np.shape(X_train)
Y_shape = np.shape(Y_train)
input = keras.layers.Input(shape = X_shape[1:])

if args.mode == "energy":
    #######################################
    # thin resnet architecture number 
    z = keras.layers.Conv2D(16, kernel_size = (3, 3), activation = "relu",padding = "same")(input)
    z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [32, 64], increase_dim = True)
    z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [32, 64])
    z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [64, 128], increase_dim = True)
    z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [64, 128])
    z = keras.layers.GlobalAveragePooling2D()(z)
    z = keras.layers.Dense(64, activation = "relu")(z)
    z = keras.layers.Dense(32, activation = "relu")(z)
    output = keras.layers.Dense(1, name = "energy")(z)
    #######################################
    # define the loss function
    loss = "mse"

elif args.mode == "separation":
    ########################################
    # thin resnet architecture 
    z = keras.layers.Conv2D(16, kernel_size = (3, 3), activation = "relu", padding = "same")(input)
    z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [32, 64], increase_dim = True)
    z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [32, 64])
    z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [64, 128], increase_dim = True)
    z = ResBlock(z, kernelsizes = [(1, 1), (3, 3)], filters = [64, 128])
    z = keras.layers.GlobalAveragePooling2D()(z)
    z = keras.layers.Dense(64, activation = "relu")(z)
    z = keras.layers.Dense(32, activation = "relu")(z)
    output = keras.layers.Dense(2, activation="softmax", name = "gammaness")(z)
    #######################################

    # # define the loss function
    loss = "categorical_crossentropy"
##########################################################################################


######################################## Train the model ########################################
model = keras.models.Model(inputs=input, outputs=output)

print(model.summary())

model.compile(loss = loss, optimizer = keras.optimizers.Adam(learning_rate = 1E-3))

model_path = f"cnn/{string_input}/{args.mode}/model/" + string_ps_input + "model" + string_name + ".h5"
checkpointer = ModelCheckpoint(filepath = model_path, verbose = 2, save_best_only = True)

history_path = f"cnn/{string_input}/{args.mode}/history/" + string_ps_input + "history" + string_name + ".csv"

# start timer
start_time = time.time()

model.fit(X_train, Y_train, epochs = args.epochs, batch_size = 32, validation_split = 0.1, callbacks = [checkpointer, CSVLogger(history_path), EarlyStopping(monitor = "val_loss", patience = 20, min_delta = 0)], verbose = 2)

# end timer and print training time
print("Time spend for training the CNN: ", np.round(time.time() - start_time, 1), "s")

#########################################################################################

######################################## Results ########################################
test_loss = model.evaluate(X_test, Y_test, batch_size=32, verbose=0)
print("Test loss: %.5e" % test_loss)

# predict output for test set and undo feature scaling
yp = model.predict(X_test, batch_size=32)

if args.mode == "energy":
    if args.telescope_mode != "mono":
        yp = yp[:, 0]  # remove unnecessary last axis
        header = ["run", "obs_id", "event_id", "log10(E_true / GeV)", "log10(E_rec / GeV)"]
    else:
        yp = yp[:, 0]  # remove unnecessary last axis
        header = ["run", "obs_id", "event_id", "tel_id", "log10(E_true / GeV)", "log10(E_rec / GeV)"]
elif args.mode == "separation":
    if args.telescope_mode != "mono":
        yp = yp[:, 1]  # get gammaness (or photon score)
        Y_test = np.argmax(Y_test, axis = 1)
        header = ["run", "obs_id", "event_id", "true gammaness", "reconstructed gammaness", "E_true / GeV"]
    else:
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
    else:
        table_output = np.hstack((run_test, obs_id_test, event_id_test, tel_id_test, Y_test, yp))
elif args.mode == "separation":
    if args.telescope_mode != "mono":
        energy_true_test = np.reshape(energy_true_test, (len(energy_true_test), 1))
        table_output = np.hstack((run_test, obs_id_test, event_id_test, Y_test, yp, energy_true_test))
    else:
        energy_true_test = np.reshape(energy_true_test, (len(energy_true_test), 1))
        table_output = np.hstack((run_test, obs_id_test, event_id_test, tel_id_test, Y_test, yp, energy_true_test))

pd.DataFrame(table_output).to_csv(f"cnn/{string_input}/{args.mode}/output/" + string_ps_input + "/evaluation" + string_name + ".csv", index = None, header = header)
##########################################################################################
