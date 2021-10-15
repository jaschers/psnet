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

print("Packages successfully loaded")

######################################## argparse setup ########################################
script_version=0.1
script_descr="""
This script loads data (CTA images or pattern spectra), defines a CNN, trains the CNN and puts the results in a csv table.
"""

# Open argument parser
parser = argparse.ArgumentParser(description=script_descr)
parser.add_argument("-v", "--version", action="version", version=f"v{script_version}")

# Define expected arguments
parser.add_argument("-i", "--input", type = str, required = True, metavar = "-", choices = ["cta", "ps"], help = "input for the CNN [cta, ps]")
parser.add_argument("-r", "--run", type = int, metavar = "-", help = "input run(s) for CNN, default: csv list", action='append', nargs='+')
parser.add_argument("-pt", "--particle_type", type = str, metavar = "-", choices = ["gamma", "gamma_diffuse", "proton"], help = "particle type [gamma, gamma_diffuse, proton], default: gamma", default = "gamma")
parser.add_argument("-dt", "--data_type", type = str, required = False, metavar = "-", choices = ["int8", "float64"], help = "data type of the output images [int8, float64], default: float64", default = "float64")
parser.add_argument("-er", "--energy_range", type = float, required = False, metavar = "-", help = "set energy range of events in GeV, default: 0.02 300", default = [0.02, 300], nargs = 2)
parser.add_argument("-na", "--name", type = str, required = False, metavar = "-", help = "Name of this particular experiment")
parser.add_argument("-a", "--attribute", type = int, metavar = "-", choices = np.arange(1, 19, dtype = int), help = "attribute [0, 1 ... 18] (two required), default: 9 0", default = [9, 0], nargs = 2)
parser.add_argument("-dl", "--domain_lower", type = int, metavar = "-", help = "Granulometry: domain - start at <value> <value>, default: 0 0", default = [0, 0], nargs = 2)
parser.add_argument("-dh", "--domain_higher", type = int, metavar = "-", help = "Granulometry: domain - end at <value> <value>, default: 10 100000", default = [10, 100000], nargs = 2)
parser.add_argument("-m", "--mapper", type = int, metavar = "-", help = "Granulometry: use lambdamappers <mapper1> <mapper2>, default: 2 0", default = [2, 0], nargs = 2)
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

if args.energy_range != [0.02, 300]:
    string_energy_range = f"_{args.energy_range[0]}-{args.energy_range[1]}_TeV"
else:
    string_energy_range = ""

if args.input == "cta":
    print(f"################### Input summary ################### \nInput: CTA \nParticle type: {args.particle_type} \nData type: {args.data_type} \nEpochs: {args.epochs}")
    string_input = "iact_images"
    string_input_short = "_images"
    string_ps_input = ""
    string_table_column = "image"
elif args.input == "ps":
    print(f"################### Input summary ################### \nInput: pattern spectra \nParticle type: {args.particle_type} \nAttribute: {args.attribute} \nDomain lower: {args.domain_lower} \nDomain higher: {args.domain_higher} \nMapper: {args.mapper} \nSize: {args.size} \nFilter: {args.filter} \nEpochs: {args.epochs}")
    string_input = "pattern_spectra"
    string_input_short = "_ps"
    string_ps_input = f"a_{args.attribute[0]}_{args.attribute[1]}__dl_{args.domain_lower[0]}_{args.domain_lower[1]}__dh_{args.domain_higher[0]}_{args.domain_higher[1]}__m_{args.mapper[0]}_{args.mapper[1]}__n_{args.size[0]}_{args.size[1]}__f_{args.filter}/"
    string_table_column = "pattern spectrum"
##########################################################################################


######################################## Load and prepare dataset ########################################
# import data
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

table.drop(table.loc[table["true_energy"] <= args.energy_range[0] * 1e3].index, inplace=True)
table.drop(table.loc[table["true_energy"] >= args.energy_range[1] * 1e3].index, inplace=True)
table.reset_index(inplace = True)

print("Total number of events after energy cut:", len(table))

# input features
X = [[]] * len(table)
for i in range(len(table)):
    X[i] = table[string_table_column][i]
X = np.asarray(X) 

# output label: log10(true energy)
Y = np.asarray(table["true_energy"])
Y = np.log10(np.asarray(table["true_energy"]))


# make some folders
path_results = f"dm-finder/cnn/{string_input}/results/" + string_ps_input + f"{string_name[1:]}/"
path_history = f"dm-finder/cnn/{string_input}/history/" + string_ps_input
path_model = f"dm-finder/cnn/{string_input}/model/" + string_ps_input

os.makedirs(path_results, exist_ok = True)
os.makedirs(path_history, exist_ok = True)
os.makedirs(path_model, exist_ok = True)

# plot a few examples
fig, ax = plt.subplots(3, 3)
ax = ax.ravel()
for i in range(9):
    ax[i].imshow(X[i], cmap = "Greys_r")
    ax[i].title.set_text(f"{int(np.round(10**Y[i]))} GeV")
    ax[i].axis("off")
plt.savefig(f"dm-finder/cnn/{string_input}/results/{string_ps_input}/{string_name[1:]}/input_examples" + string_data_type + string_name + string_energy_range + ".png", dpi = 500)

# reshape X data
X_shape = np.shape(X)
X = X.reshape(-1, X_shape[1], X_shape[2], 1)

# # hold out 10 percent as test data
X_train, X_test = np.split(X, [int(-len(table) / 10)])
Y_train, Y_test = np.split(Y, [int(-len(table) / 10)])


# # remove strange outlier
# index = np.argmin(Y)
# Y = np.delete(Y, index)

# display total energy distribution of data set
plt.figure()
plt.hist(10**Y, bins=np.logspace(np.log10(np.min(10**Y)),np.log10(np.max(10**Y)), 50))
plt.xlabel("True energy [GeV]")
plt.ylabel("Number of events")
plt.xscale("log")
plt.yscale("log")
plt.savefig(f"dm-finder/cnn/{string_input}/results/" + string_ps_input + f"{string_name[1:]}/" + "total_energy_distribution" + string_data_type + string_name + string_energy_range + ".png", dpi = 250)
plt.close()
##########################################################################################


######################################## Define a model ########################################
# define initializers (starting parameters)

X_shape = np.shape(X_train)
Y_shape = np.shape(Y_train)
input1 = keras.layers.Input(shape = X_shape[1:])

# define a suitable network 
z = keras.layers.Conv2D(4, # number of filters, the dimensionality of the output space
    kernel_size = (3,3), # size of filters 3x3
    activation = "relu", kernel_initializer = "zeros", bias_initializer = "zeros")(input1)
zl = [z]

z = keras.layers.GlobalAveragePooling2D()(z)

output = keras.layers.Dense(1, name = "energy", kernel_initializer = "zeros", bias_initializer = "zeros")(z)
##########################################################################################


######################################## Train the model ########################################
model = keras.models.Model(inputs=input1, outputs=output)

print(model.summary())

#weight estimations, relative values for all attributes:
weight_energy = 1

model.compile(
    loss="mse",
    loss_weights=weight_energy,  
    optimizer=keras.optimizers.Adam(learning_rate = 1E-3))

history_path = f"dm-finder/cnn/{string_input}/history/" + string_ps_input + "history" + string_data_type + string_name + string_energy_range + ".csv"

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

model_path = f"dm-finder/cnn/{string_input}/model/" + string_ps_input + "model" + string_data_type + string_name + string_energy_range + ".h5"

model.save(model_path)
#########################################################################################

######################################## Results ########################################
model_path = f"dm-finder/cnn/{string_input}/model/" + string_ps_input + "model" + string_data_type + string_name + string_energy_range + ".h5"
model = keras.models.load_model(model_path)

losses = model.evaluate(X_test, Y_test, batch_size=128, verbose=0)
# print("Test loss with", title_names[i])
print("%.5e (energy)" % losses)

# predict output for test set and undo feature scaling
yp = model.predict(X_test, batch_size=128)

yp = yp[:, 0]  # remove unnecessary last axis

# create csv output file
Y_test = np.reshape(Y_test, (len(Y_test), 1))
yp = np.reshape(yp, (len(yp), 1))
table_output = np.hstack((Y_test, yp))

path_output = f"dm-finder/cnn/{string_input}/output/" + string_ps_input
os.makedirs(path_output, exist_ok = True)

pd.DataFrame(table_output).to_csv(f"dm-finder/cnn/{string_input}/output/" + string_ps_input + "/evaluation" + string_data_type + string_name + string_energy_range + ".csv", index = None, header = ["log10(E_true / GeV)", "log10(E_rec / GeV)"])
##########################################################################################
