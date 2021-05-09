import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import *
import os 
from keras.callbacks import CSVLogger
import tensorflow as tf
layers = keras.layers

# ---------------------------------------------------
# Load and prepare dataset
# ---------------------------------------------------

# import data
run = 1012
run_filename = f"gamma_20deg_0deg_run{run}___cta-prod5-paranal_desert-2147m-Paranal-dark_merged.DL1"
input_filename = "cta/data/cnn-data-base/" + run_filename + "_cnn.h5"

table = pd.read_hdf(input_filename)

# input features: CTA images normalised to 1
X = [[]] * len(table)
for i in range(len(table)):
    X[i] = table["image"][i]
X = np.asarray(X) / 255
X = X.reshape(-1, 339, 342, 1)

# output label: log10(true energy)
Y = np.log10(np.asarray(table["true_energy"]))

# # hold out the last 3000 events as test data
X_train, X_test = np.split(X, [-3000])
Y_train, Y_test = np.split(Y, [-3000])

# # display a random cta image
# plt.figure()
# plt.imshow(X[0], cmap = "Greys_r")
# plt.colorbar()
# plt.savefig("cta/cnn/checks/cta_image.png")
# plt.close()

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

for i in range(3):
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

history_path = "cta/cnn/history/history.csv"

fit = model.fit(X_train,
    Y_train,
    batch_size=32,
    epochs=5,
    verbose=2,
    validation_split=0.1,
    callbacks=[CSVLogger(history_path)])

model_path = "cta/cnn/model/model.h5"

model.save(model_path)

# ----------------------------------------------------------------------
# Evaluation - this should work as is.
# ----------------------------------------------------------------------

model = keras.models.load_model("cta/cnn/model/model.h5")

title_names = np.array(["Weights of 1"])

losses = model.evaluate(X_test, Y_test, batch_size=128, verbose=0)
# print("Test loss with", title_names[i])
print("%.5e (energy)" % losses)

# predict output for test set and undo feature scaling
yp = model.predict(X_test, batch_size=128)

yp = yp[:, 0]  # remove unnecessary last axis

# energy
# d = 10**(yp - Y_test) - 1
# reso = np.std(d)
# plt.figure()
# plt.hist(d, bins=np.linspace(-0.3, 0.3, 41))
# plt.xlabel("($E_\mathrm{rec} - E_\mathrm{true}) / E_\mathrm{true}$")
# plt.ylabel("#")
# plt.text(0.95, 0.95, "$\sigma = %.3f$" % reso, ha="right", va="top", transform=plt.gca().transAxes)
# plt.grid()
# plt.title(title_names[i])
# # plt.savefig(fname = folder + "hist-energy_{0}.png".format(i), bbox_inches="tight")

x = np.linspace(np.min(Y_test), np.max(Y_test), 100)
plt.figure()
plt.grid(alpha = 0.2)
plt.plot(x, x, color="black")
plt.scatter(Y_test, yp)
plt.xlabel("$\log_{10}(E_\mathrm{true}/\mathrm{GeV})$")
plt.ylabel("$\log_{10}(E_\mathrm{rec}/\mathrm{GeV})$")
plt.savefig("cta/cnn/checks/energy_scattering.png")
