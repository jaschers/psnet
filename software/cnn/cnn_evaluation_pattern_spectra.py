import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import *
layers = keras.layers

# import data
run = np.array([107, 1012, 1024, 1034, 1037, 1054, 1057, 1069, 1073, 1086, 1098]) # 107, 1012, 1024, 1034, 1037, 1054, 1057, 1069, 1073, 1086, 1098
table = pd.DataFrame()
for r in range(len(run)):
    run_filename = f"gamma_20deg_0deg_run{run[r]}___cta-prod5-paranal_desert-2147m-Paranal-dark_merged.DL1"
    input_filename = "dm-finder/cnn/input/pattern-spectra/" + run_filename + "_ps.h5"

    table_individual_run = pd.read_hdf(input_filename)
    print(f"Number of events in Run {run[r]}:", len(table_individual_run))
    table = table.append(table_individual_run, ignore_index = True)

print("Total number of events:", len(table))

# input features: CTA images normalised to 1
X = [[]] * len(table)
for i in range(len(table)):
    X[i] = table["pattern spectrum"][i]
X = np.asarray(X) # / 255
X = X.reshape(-1, 20, 20, 1)

# output label: log10(true energy)
Y = np.log10(np.asarray(table["true_energy"]))

# # hold out the last 3000 events as test data
X_train, X_test = np.split(X, [int(-len(table) / 10)])
Y_train, Y_test = np.split(Y, [int(-len(table) / 10)])


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------

model = keras.models.load_model("dm-finder/cnn/model/model.h5")

title_names = np.array(["Weights of 1"])

losses = model.evaluate(X_train, Y_train, batch_size=128, verbose=0)
# print("Test loss with", title_names[i])
print("%.5e (energy)" % losses)

# predict output for test set and undo feature scaling
yp = model.predict(X_train, batch_size=128)
yp = yp[:, 0]  # remove unnecessary last axis

# energy
difference = (yp - Y_train) / Y_train #10**(yp - Y_train) - 1
reso = np.std(difference)
plt.figure()
plt.hist(difference, bins = np.linspace(np.min(difference), np.max(difference), 40))
plt.yscale("log")
plt.xlabel("($E_\mathrm{rec} - E_\mathrm{true}) / E_\mathrm{true}$")
plt.ylabel("Number of events")
# plt.text(0.95, 0.95, "$\sigma = %.3f$" % reso, ha="right", va="top", transform=plt.gca().transAxes)
plt.grid()
plt.savefig("dm-finder/cnn/results/energy_histogram_ps.png")

x = np.linspace(np.min(Y_train), np.max(Y_train), 100)
plt.figure()
plt.grid(alpha = 0.2)
plt.plot(x, x, color="black")
plt.scatter(Y_train, yp)
plt.xlabel("$\log_{10}(E_\mathrm{true}/\mathrm{GeV})$")
plt.ylabel("$\log_{10}(E_\mathrm{rec}/\mathrm{GeV})$")
plt.savefig("dm-finder/cnn/results/energy_scattering_ps.png")
