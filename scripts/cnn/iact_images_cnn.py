import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import *
import os 
from keras.callbacks import CSVLogger
import tensorflow as tf
layers = keras.layers

print("Packages successfully loaded")

# ---------------------------------------------------
# Load and prepare dataset
# ---------------------------------------------------

# import data
particle_type = "gamma"
image_type = "minimalistic"
run = np.array([107, 1012, 1024, 1034, 1037, 1054, 1057, 1069, 1073, 1086, 1098]) # 107, 1012, 1024, 1034, 1037, 1054, 1057, 1069, 1073, 1086, 1098, 
table = pd.DataFrame()
for r in range(len(run)):
    run_filename = f"gamma_20deg_0deg_run{run[r]}___cta-prod5-paranal_desert-2147m-Paranal-dark_merged.DL1"
    input_filename = f"dm-finder/cnn/iact_images/input/{particle_type}/{image_type}/" + run_filename + "_images.h5"

    table_individual_run = pd.read_hdf(input_filename)
    print(f"Number of events in Run {run[r]}:", len(table_individual_run))
    table = table.append(table_individual_run, ignore_index = True)

print("Total number of events:", len(table))

# input features: CTA images normalised to 1
X = [[]] * len(table)
for i in range(len(table)):
    X[i] = table["image"][i]
X = np.asarray(X) # / 255
X_shape = np.shape(X)
X = X.reshape(-1, X_shape[1], X_shape[2], 1)

# output label: log10(true energy)
Y = np.log10(np.asarray(table["true_energy"]))

# # hold out the last 3000 events as test data
X_train, X_test = np.split(X, [int(-len(table) / 10)])
Y_train, Y_test = np.split(Y, [int(-len(table) / 10)])

# # display a random cta image
# plt.figure()
# plt.imshow(X[0], cmap = "Greys_r")
# plt.colorbar()
# plt.savefig("dm-finder/cnn/checks/cta_image.png")
# plt.close()

# display total energy distribution of data set
plt.figure()
plt.hist(table["true_energy"], bins=np.logspace(np.log10(np.min(table["true_energy"])),np.log10(np.max(table["true_energy"])), 50))
plt.xlabel("true energy [GeV]")
plt.ylabel("number of events")
plt.xscale("log")
plt.yscale("log")
plt.savefig(f"dm-finder/cnn/iact_images/results/{image_type}/total_energy_distribution.png")
plt.close()

# ----------------------------------------------------------------------
# Define the model
# ----------------------------------------------------------------------
X_shape = np.shape(X_train)
Y_shape = np.shape(Y_train)
input1 = layers.Input(shape = X_shape[1:])

# define a suitable network 
z = Conv2D(4, # number of filters, the dimensionality of the output space
    kernel_size = (3,3), # size of filters 3x3
    activation = "relu")(input1)
zl = [z]

for i in range(5):
    z = Conv2D(16, 
        kernel_size = (3,3), 
        padding="same", # padding, "same" = on, "valid" = off
        activation="relu")(z) 
    zl.append(z)
    z = concatenate(zl[:], axis=-1)

z = GlobalAveragePooling2D()(z)
z = Dense(8,activation="relu")(z)

output = layers.Dense(1, name="energy")(z)

# ----------------------------------------------------------------------
# Train the model
# ----------------------------------------------------------------------

model = keras.models.Model(inputs=input1, outputs=output)

print(model.summary())

#weight estimations, relative values for all attributes:
weight_energy = 1

model.compile(
    loss="mse",
    loss_weights=weight_energy,  
    optimizer=keras.optimizers.Adam(lr=1E-3))

history_path = f"dm-finder/cnn/iact_images/history/{image_type}/" + "history.csv"

fit = model.fit(X_train,
    Y_train,
    batch_size=32,
    epochs=50,
    verbose=2,
    validation_split=0.1,
    callbacks=[CSVLogger(history_path)])

model_path = f"dm-finder/cnn/iact_images/model/{image_type}/" + "model.h5"

model.save(model_path)
