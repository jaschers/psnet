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
from utilities import ExamplesEnergy, ExamplesSeparation, EnergyDistributionEnergy, EnergyDistributionSeparation

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
parser.add_argument("-r", "--run", type = int, metavar = "-", help = "input run(s) for CNN, default: csv list", action='append', nargs='+')
parser.add_argument("-pt", "--particle_type", type = str, metavar = "-", choices = ["gamma", "gamma_diffuse", "proton"], help = "particle type [gamma, gamma_diffuse, proton], default: gamma", default = "gamma")
parser.add_argument("-dt", "--data_type", type = str, required = False, metavar = "-", choices = ["int8", "float64"], help = "data type of the output images [int8, float64], default: float64", default = "float64")
parser.add_argument("-er", "--energy_range", type = float, required = False, metavar = "-", help = "set energy range of events in TeV, default: 0.02 300", default = [0.02, 300], nargs = 2)
parser.add_argument("-na", "--name", type = str, required = False, metavar = "-", help = "Name of this particular experiment")
parser.add_argument("-a", "--attribute", type = int, metavar = "-", choices = np.arange(1, 19, dtype = int), help = "attribute [0, 1 ... 18] (two required), default: 9 0", default = [9, 0], nargs = 2)
parser.add_argument("-dl", "--domain_lower", type = int, metavar = "-", help = "Granulometry: domain - start at <value> <value>, default: 0 0", default = [0, 0], nargs = 2)
parser.add_argument("-dh", "--domain_higher", type = int, metavar = "-", help = "Granulometry: domain - end at <value> <value>, default: 10 100000", default = [10, 100000], nargs = 2)
parser.add_argument("-ma", "--mapper", type = int, metavar = "-", help = "Granulometry: use lambdamappers <mapper1> <mapper2>, default: 2 0", default = [2, 0], nargs = 2)
parser.add_argument("-n", "--size", type = int, metavar = "-", help = "Granulometry: size <n1>x<n2>, default: 20 20", default = [20, 20], nargs = 2)
parser.add_argument("-f", "--filter", type = int, metavar = "-", help = "Use decision <filter>, default: 3", default = 3, nargs = 1)
parser.add_argument("-e", "--epochs", type = int, metavar = "-", help = "Number of epochs for CNN training, default: 50", default = 50)

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
    print(f"################### Input summary ################### \nMode: {args.mode} \nInput: CTA images \nParticle type: {args.particle_type} \nData type: {args.data_type} \nEnergy range: {args.energy_range} TeV \nEpochs: {args.epochs}")
    string_input = "iact_images"
    string_input_short = "_images"
    string_ps_input = ""
    string_table_column = "image"
elif args.input == "ps":
    print(f"################### Input summary ################### \nMode: {args.mode} \nInput: pattern spectra \nParticle type: {args.particle_type} \nEnergy range: {args.energy_range} TeV \nAttribute: {args.attribute} \nDomain lower: {args.domain_lower} \nDomain higher: {args.domain_higher} \nMapper: {args.mapper} \nSize: {args.size} \nFilter: {args.filter} \nEpochs: {args.epochs}")
    string_input = "pattern_spectra"
    string_input_short = "_ps"
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
    filename_run = f"dm-finder/scripts/run_lists/{args.particle_type}_run_list.csv"
    run = pd.read_csv(filename_run)
    run = run.to_numpy().reshape(len(run))

    if args.run != None:
        run = args.run[0]

    table = pd.DataFrame()
    for r in range(len(run)): # len(run)
        run_filename = f"gamma_20deg_0deg_run{run[r]}___cta-prod5-paranal_desert-2147m-Paranal-dark_merged.DL1"
        input_filename = f"dm-finder/cnn/{string_input}/input/{args.particle_type}/" + string_ps_input + run_filename + string_input_short + string_data_type + ".h5"

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
        run = pd.read_csv(filename_run)
        run = run.to_numpy().reshape(len(run))
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

    # shuffle data set
    table = table.sample(frac=1).reset_index(drop=True)


table.drop(table.loc[table["true_energy"] <= args.energy_range[0] * 1e3].index, inplace=True)
table.drop(table.loc[table["true_energy"] >= args.energy_range[1] * 1e3].index, inplace=True)
table.reset_index(inplace = True)

print("Total number of events after energy cut:", len(table))


# input features
X = [[]] * len(table)
for i in range(len(table)):
    X[i] = table[string_table_column][i]
X = np.asarray(X) 

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

    # plot a few examples
    ExamplesSeparation(X, Y, f"dm-finder/cnn/{string_input}/{args.mode}/results/{string_ps_input}/{string_name[1:]}/input_examples" + string_data_type + string_name + ".png")

# reshape X data
X_shape = np.shape(X)
X = X.reshape(-1, X_shape[1], X_shape[2], 1)

# # hold out 10 percent as test data
X_train, X_test = np.split(X, [int(-len(table) / 10)])
Y_train, Y_test = np.split(Y, [int(-len(table) / 10)])

# # remove strange outlier
# index = np.argmin(Y)
# Y = np.delete(Y, index)

##########################################################################################


######################################## Define a model ########################################
X_shape = np.shape(X_train)
Y_shape = np.shape(Y_train)
input1 = keras.layers.Input(shape = X_shape[1:])

if args.mode == "energy":
    # define a suitable network 
    z = keras.layers.Conv2D(4, # number of filters, the dimensionality of the output space
        kernel_size = (3,3), # size of filters 3x3
        activation = "relu")(input1)
    zl = [z]

    for i in range(5):
        z = keras.layers.Conv2D(16, 
            kernel_size = (3,3), 
            padding = "same", # padding, "same" = on, "valid" = off
            activation = "relu")(z) 
        zl.append(z)
        z = keras.layers.concatenate(zl[:], axis=-1)

    z = keras.layers.GlobalAveragePooling2D()(z)
    z = keras.layers.Dense(8, activation = "relu")(z)

    output = keras.layers.Dense(1, name = "energy")(z)

    # define the loss function
    loss = "mse"

elif args.mode == "separation":

    ### cnn architecture number 1 (cnn1) ###
    # # define a suitable network 
    # z = keras.layers.Conv2D(4, # number of filters, the dimensionality of the output space
    #     kernel_size = (3,3), # size of filters 3x3
    #     padding = "same", 
    #     activation = "relu")(input1)
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

    # output = keras.layers.Dense(2, activation = 'softmax', name = "gammaness")(z)
    ########################################

    ### cnn architecture number 2 (cnn2) ###
    z = keras.layers.Conv2D(4, kernel_size = (3,3), activation = "relu", padding = "same")(input1)
    z = keras.layers.Conv2D(16, kernel_size = (3,3), activation = "relu", padding = "same")(z)
    z = keras.layers.Conv2D(64, kernel_size = (3,3), activation = "relu", padding = "same")(z)
    z = keras.layers.GlobalAveragePooling2D()(z)

    z = keras.layers.Dense(64, activation = "relu")(z)
    z = keras.layers.Dense(32, activation = "relu")(z)
    z = keras.layers.Dense(16, activation = "relu")(z)

    output = keras.layers.Dense(2, activation='softmax', name = "gammaness")(z)
    ########################################

    # define the loss function
    loss = "categorical_crossentropy"
##########################################################################################


######################################## Train the model ########################################
model = keras.models.Model(inputs=input1, outputs=output)

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
    yp = yp[:, 0]  # remove unnecessary last axis
    header = ["log10(E_true / GeV)", "log10(E_rec / GeV)"]
elif args.mode == "separation":
    yp = yp[:, 1]  # get gammaness (or photon score)
    Y_test = np.argmax(Y_test, axis = 1)
    header = ["true gammaness", "reconstructed gammaness"]

# create csv output file
Y_test = np.reshape(Y_test, (len(Y_test), 1))
yp = np.reshape(yp, (len(yp), 1))
table_output = np.hstack((Y_test, yp))

pd.DataFrame(table_output).to_csv(f"dm-finder/cnn/{string_input}/{args.mode}/output/" + string_ps_input + "/evaluation" + string_data_type + string_name + ".csv", index = None, header = header)
##########################################################################################
