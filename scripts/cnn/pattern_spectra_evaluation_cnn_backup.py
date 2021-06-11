import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import keras
from keras.layers import *
import os
import csv
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
run = np.array([107, 1012, 1024, 1034, 1037, 1054, 1057, 1069, 1073, 1086, 1098]) #1012, 1024, 1034, 1037, 1054, 1057, 1069, 1073, 1086, 1098

table = pd.DataFrame()
for r in range(len(run)):
    run_filename = f"gamma_20deg_0deg_run{run[r]}___cta-prod5-paranal_desert-2147m-Paranal-dark_merged.DL1"
    input_filename = f"dm-finder/cnn/pattern_spectra/input/{particle_type}/{image_type}/" + f"a_{a[0]}_{a[1]}__dl_{dl[0]}_{dl[1]}__dh_{dh[0]}_{dh[1]}__m_{m[0]}_{m[1]}__n_{n[0]}_{n[1]}__f_{f}/" + run_filename + "_ps.h5"

    table_individual_run = pd.read_hdf(input_filename)
    print(f"Number of events in Run {run[r]}:", len(table_individual_run))
    table = table.append(table_individual_run, ignore_index = True)

print("Total number of events:", len(table))

# input features: CTA images normalised to 1
X = [[]] * len(table)
for i in range(len(table)):
    X[i] = table["pattern spectrum"][i]
X = np.asarray(X) # / 255

plt.figure()
plt.imshow(X[0], cmap = "Greys")
plt.savefig(f"dm-finder/cnn/pattern_spectra/results/{image_type}/" + f"a_{a[0]}_{a[1]}__dl_{dl[0]}_{dl[1]}__dh_{dh[0]}_{dh[1]}__m_{m[0]}_{m[1]}__n_{n[0]}_{n[1]}__f_{f}/" + "ps_example.png")

X = X.reshape(-1, 20, 20, 1)

# output label: log10(true energy)
Y = np.log10(np.asarray(table["true_energy"]))

# # hold out the last 3000 events as test data
X_train, X_test = np.split(X, [int(-len(table) / 10)])
Y_train, Y_test = np.split(Y, [int(-len(table) / 10)])

# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------

model_path = f"dm-finder/cnn/pattern_spectra/model/{image_type}/" + f"model_a_{a[0]}_{a[1]}__dl_{dl[0]}_{dl[1]}__dh_{dh[0]}_{dh[1]}__m_{m[0]}_{m[1]}__n_{n[0]}_{n[1]}__f_{f}.h5"

model = keras.models.load_model(model_path)

losses = model.evaluate(X_test, Y_test, batch_size=128, verbose=0)
print("%.5e (energy)" % losses)

# predict output for test set and undo feature scaling
yp = model.predict(X_test, batch_size=128)
yp = yp[:, 0]  # remove unnecessary last axis

# # create csv output file
# Y_test = np.reshape(Y_test, (len(Y_test), 1))
# yp = np.reshape(yp, (len(yp), 1))
# table_output = np.hstack((Y_test, yp))

# path_output = f'dm-finder/cnn/pattern_spectra/output/{image_type}/'

# try:
#     os.makedirs(path_output)
# except OSError:
#     pass #print("Directory could not be created")

# pd.DataFrame(table_output).to_csv(f'dm-finder/cnn/pattern_spectra/output/{image_type}/test_set_energy_a_{a[0]}_{a[1]}__dl_{dl[0]}_{dl[1]}__dh_{dh[0]}_{dh[1]}__m_{m[0]}_{m[1]}__n_{n[0]}_{n[1]}__f_{f}_r.csv', index = None, header = ["log10(E_true / GeV)", "log10(E_rec / GeV)"])


path = f"dm-finder/cnn/pattern_spectra/results/{image_type}/" + f"a_{a[0]}_{a[1]}__dl_{dl[0]}_{dl[1]}__dh_{dh[0]}_{dh[1]}__m_{m[0]}_{m[1]}__n_{n[0]}_{n[1]}__f_{f}/"

# energy
try:
    os.makedirs(path)
except OSError:
    pass #print("Directory could not be created")

difference = (yp - Y_test) / Y_test #10**(yp - Y_test) - 1
mean = np.mean(difference)
sigma = np.std(difference)
plt.figure()
plt.title("pattern spectra")
plt.grid(alpha = 0.2)
plt.hist(difference, bins = np.linspace(np.min(difference), np.max(difference), 40))
plt.yscale("log")
plt.xlabel("($\log_{10}(E_\mathrm{rec}/\mathrm{GeV}) - \log_{10}(E_\mathrm{true}/\mathrm{GeV}))/ \log_{10}(E_\mathrm{true} /\mathrm{GeV})$")
plt.ylabel("Number of events")
# plt.text(0.95, 0.95, "$\mu = %.3f$" % mean, ha="right", va="top", transform=plt.gca().transAxes)
# plt.text(0.95, 0.90, "$\sigma = %.3f$" % sigma, ha="right", va="top", transform=plt.gca().transAxes)
plt.vlines(mean, 0.7, 3e3, linestyle = "dashed", label = "$\mu = %.3f$" % mean, color = "black")
plt.vlines(mean + sigma, 0.7, 3e3, linestyle = "dashdot", label =  "$\sigma = %.3f$" % sigma, color = "black")
plt.vlines(mean - sigma, 0.7, 3e3, linestyle = "dashdot", color = "black")
plt.legend()
plt.savefig(path + "energy_histogram.png")

x = np.linspace(np.min(Y_test), np.max(Y_test), 100)
plt.figure()
plt.title("pattern spectra")
plt.grid(alpha = 0.2)
plt.plot(x, x, color="black")
plt.scatter(Y_test, yp)
plt.xlabel("$\log_{10}(E_\mathrm{true}/\mathrm{GeV})$")
plt.ylabel("$\log_{10}(E_\mathrm{rec}/\mathrm{GeV})$")
plt.savefig(path + "energy_scattering.png")

# 2D energy scattering
plt.figure()
plt.title("pattern spectra")
plt.grid(alpha = 0.2)
plt.plot(x, x, color="black")
# plt.scatter(x,y,edgecolors='none',s=marker_size,c=void_fraction, norm=matplotlib.colors.LogNorm())
plt.hist2d(Y_test, yp, bins=(50, 50), cmap = "viridis", norm=matplotlib.colors.LogNorm())
plt.colorbar()
plt.xlabel("$\log_{10}(E_\mathrm{true}/\mathrm{GeV})$")
plt.ylabel("$\log_{10}(E_\mathrm{rec}/\mathrm{GeV})$")
plt.savefig(path + "energy_scattering_2D.png")