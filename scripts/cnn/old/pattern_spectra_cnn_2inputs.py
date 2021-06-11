import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import *
import os 
from keras.callbacks import CSVLogger
import tensorflow as tf
import time
layers = keras.layers

print("Packages successfully loaded")

# ---------------------------------------------------
# Load and prepare dataset
# ---------------------------------------------------

# pattern spectra properties
a = np.array([9, 0])
dl = np.array([0, 0])
dh = np.array([10, 100000])
m = np.array([2, 0])
n = np.array([20, 20])
f = 3

particle_type = "gamma"
image_type = "minimalistic"
run = np.array([1012, 1024, 1034, 1037, 1054, 1057, 1069, 1073, 107, 1086, 1098, 1108, 1117, 1119, 1121, 1146, 1196, 1213, 1232, 1234, 1257, 1258, 1275, 1305, 1308, 1330, 134, 1364, 1368, 1369, 1373, 1394, 1413, 1467, 1475, 1477, 1489, 148, 1514, 1517, 1521]) #, 1024, 1034, 1037, 1054, 1057, 1069, 1073, 107, 1086, 1098, 1108, 1117, 1119, 1121, 1146, 1196, 1213, 1232, 1234, 1257, 1258, 1275, 1305, 1308, 1330, 134, 1364, 1368, 1369, 1373, 1394, 1413, 1467, 1475, 1477, 1489, 148, 1514, 1517, 1521

table = pd.DataFrame()
for r in range(len(run)): # len(run)
    run_filename = f"gamma_20deg_0deg_run{run[r]}___cta-prod5-paranal_desert-2147m-Paranal-dark_merged.DL1"
    input_filename = f"dm-finder/cnn/pattern_spectra/input/{particle_type}/{image_type}/" + f"a_{a[0]}_{a[1]}__dl_{dl[0]}_{dl[1]}__dh_{dh[0]}_{dh[1]}__m_{m[0]}_{m[1]}__n_{n[0]}_{n[1]}__f_{f}/" + run_filename + "_ps.h5"

    table_individual_run = pd.read_hdf(input_filename)
    print(f"Number of events in Run {run[r]}:", len(table_individual_run))
    table = table.append(table_individual_run, ignore_index = True)

print("Total number of events:", len(table))

# first input feature: pattern spectra
X1 = [[]] * len(table)
for i in range(len(table)):
    X1[i] = table["pattern spectrum"][i]
X1 = np.asarray(X1) # / 255
X1_shape = np.shape(X1)
X1 = X1.reshape(-1, X1_shape[1], X1_shape[2], 1)

# scond input feature: sum of combined CTA image
X2 = table["sum_image"]
X2 = X1.reshape(-1, 1)

# output label: log10(true energy)
Y = np.log10(np.asarray(table["true_energy"]))

# # hold out the last 10 % events as test data
X1_train, X1_test = np.split(X1, [int(-len(table) / 10)])
X2_train, X2_test = np.split(X2, [int(-len(table) / 10)])
Y_train, Y_test = np.split(Y, [int(-len(table) / 10)])

path = f"dm-finder/cnn/pattern_spectra/results/{image_type}/" + f"a_{a[0]}_{a[1]}__dl_{dl[0]}_{dl[1]}__dh_{dh[0]}_{dh[1]}__m_{m[0]}_{m[1]}__n_{n[0]}_{n[1]}__f_{f}/"

try:
    os.makedirs(path)
except OSError:
    pass #print("Directory could not be created")

# display total energy distribution of data set
plt.figure()
plt.hist(table["true_energy"], bins=np.logspace(np.log10(np.min(table["true_energy"])),np.log10(np.max(table["true_energy"])), 50))
plt.xlabel("true energy [GeV]")
plt.ylabel("number of events")
plt.xscale("log")
plt.yscale("log")
plt.savefig(path + "total_energy_distribution.png")
plt.close()

# ----------------------------------------------------------------------
# Define the model
# ----------------------------------------------------------------------
X1_shape = np.shape(X1_train)
X2_shape = np.shape(X2_train)
Y_shape = np.shape(Y_train)
input1 = layers.Input(shape = X1_shape[1:])
input2 = layers.Input(shape = X2_shape[1:])

# first branch of CNN
z1 = Conv2D(4, # number of filters, the dimensionality of the output space
    kernel_size = (3,3), # size of filters 3x3
    activation = "relu")(input1)
zl = [z1]

for i in range(5):
    z1 = Conv2D(16, 
        kernel_size = (3,3), 
        padding="same", # padding, "same" = on, "valid" = off
        activation="relu")(z1) 
    zl.append(z1)
    z1 = concatenate(zl[:], axis=-1)

z1 = GlobalAveragePooling2D()(z1)
z1 = Dense(8,activation="relu")(z1)
z1 = keras.models.Model(inputs=input1, outputs=z1)

# second branch of CNN
z2 = Dense(64, activation="relu")(input2)
z2 = Dense(32, activation="relu")(z2)
z2 = keras.models.Model(inputs=input2, outputs=z2)

# combine the output of the two branches
combined = concatenate([z1.output, z2.output])

# apply a FC layer and then a regression prediction on the
# combined outputs
z = Dense(16, activation="relu")(combined)
z = Dense(8, activation="relu")(z)
output = Dense(1, name = "energy")(z)


# ----------------------------------------------------------------------
# Train the model
# ----------------------------------------------------------------------

# our model will accept the inputs of the two branches and then output a single value
model = keras.models.Model(inputs=[z1.input, z2.input], outputs=output)
# model = keras.models.Model(inputs=input1, outputs=output)

print(model.summary())

#weight estimations, relative values for all attributes:
weight_energy = 1

model.compile(
    loss="mse",
    loss_weights=weight_energy,  
    optimizer=keras.optimizers.Adam(lr=1E-3))

history_path = f"dm-finder/cnn/pattern_spectra/history/{image_type}/" + f"history_2inputs_a_{a[0]}_{a[1]}__dl_{dl[0]}_{dl[1]}__dh_{dh[0]}_{dh[1]}__m_{m[0]}_{m[1]}__n_{n[0]}_{n[1]}__f_{f}.csv"

# start timer
start_time = time.time()

fit = model.fit([X1_train, X2_train],
    Y_train,
    batch_size=32,
    epochs=50,
    verbose=2,
    validation_split=0.1,
    callbacks=[CSVLogger(history_path)])

# end timer and print training time
print("Time spend for training the CNN: ", time.time() - start_time)

model_path = f"dm-finder/cnn/pattern_spectra/model/{image_type}/" + f"model_2inputs_a_{a[0]}_{a[1]}__dl_{dl[0]}_{dl[1]}__dh_{dh[0]}_{dh[1]}__m_{m[0]}_{m[1]}__n_{n[0]}_{n[1]}__f_{f}.h5"

model.save(model_path)