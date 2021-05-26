import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import keras
from keras.layers import *
import os 
from keras.callbacks import CSVLogger
import tensorflow as tf
layers = keras.layers

print("Packages successfully loaded")

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

plt.figure()
plt.imshow(X[0], cmap = "Greys")
plt.savefig(f"dm-finder/cnn/iact_images/results/{image_type}/" + "iact_image_example.png")

X_shape = np.shape(X)
X = X.reshape(-1, X_shape[1], X_shape[2], 1)

# output label: log10(true energy)
Y = np.log10(np.asarray(table["true_energy"]))

# # hold out the last 3000 events as test data
X_train, X_test = np.split(X, [int(-len(table) / 10)])
Y_train, Y_test = np.split(Y, [int(-len(table) / 10)])


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------

model_path = f"dm-finder/cnn/iact_images/model/{image_type}/" + "model.h5"

model = keras.models.load_model(model_path)

losses = model.evaluate(X_test, Y_test, batch_size=128, verbose=0)
# print("Test loss with", title_names[i])
print("%.5e (energy)" % losses)

# predict output for test set and undo feature scaling
yp = model.predict(X_test, batch_size=128)
yp = yp[:, 0]  # remove unnecessary last axis

path = f"dm-finder/cnn/iact_images/results/{image_type}/"

# energy
difference = (yp - Y_test) / Y_test #10**(yp - Y_test) - 1
mean = np.mean(difference)
sigma = np.std(difference)
plt.figure()
plt.title("CTA images")
plt.grid(alpha = 0.2)
plt.hist(difference, bins = np.linspace(np.min(difference), np.max(difference), 40))
plt.yscale("log")
plt.xlabel("($\log_{10}(E_\mathrm{rec}/\mathrm{GeV}) - \log_{10}(E_\mathrm{true}/\mathrm{GeV})) / \log_{10}(E_\mathrm{true} /\mathrm{GeV})$")
plt.ylabel("Number of events")
# plt.text(0.95, 0.95, "$\mu = %.3f$" % mean, ha="right", va="top", transform=plt.gca().transAxes)
# plt.text(0.95, 0.90, "$\sigma = %.3f$" % sigma, ha="right", va="top", transform=plt.gca().transAxes)
plt.vlines(mean, 0.7, 3e3, linestyle = "dashed", label = "$\mu = %.3f$" % mean)
plt.vlines(mean + sigma, 0.7, 3e3, linestyle = "dashdot", label =  "$\sigma = %.3f$" % sigma)
plt.vlines(mean - sigma, 0.7, 3e3, linestyle = "dashdot")
plt.legend()
plt.savefig(f"dm-finder/cnn/iact_images/results/{image_type}/energy_histogram.png")

x = np.linspace(np.min(Y_test), np.max(Y_test), 100)
plt.figure()
plt.title("CTA images")
plt.grid(alpha = 0.2)
plt.plot(x, x, color="black")
plt.scatter(Y_test, yp)
plt.xlabel("$\log_{10}(E_\mathrm{true}/\mathrm{GeV})$")
plt.ylabel("$\log_{10}(E_\mathrm{rec}/\mathrm{GeV})$")
plt.savefig(f"dm-finder/cnn/iact_images/results/{image_type}/energy_scattering.png")

# 2D energy scattering
plt.figure()
plt.title("CTA images")
plt.grid(alpha = 0.2)
plt.plot(x, x, color="black")
# plt.scatter(x,y,edgecolors='none',s=marker_size,c=void_fraction, norm=matplotlib.colors.LogNorm())
plt.hist2d(Y_test, yp, bins=(50, 50), cmap = "viridis", norm=matplotlib.colors.LogNorm())
plt.colorbar()
plt.xlabel("$\log_{10}(E_\mathrm{true}/\mathrm{GeV})$")
plt.ylabel("$\log_{10}(E_\mathrm{rec}/\mathrm{GeV})$")
plt.savefig(path + "energy_scattering_2D.png")