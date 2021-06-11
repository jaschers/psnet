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
plt.rcParams.update({'font.size': 14})

# import data
particle_type = "gamma"
image_type = "minimalistic"
run = np.array([1012, 1024, 1034, 1037, 1054, 1057, 1069, 1073, 107, 1086, 1098, 1108, 1117, 1119, 1121, 1146, 1196, 1213, 1232, 1234, 1257, 1258, 1275, 1305, 1308, 1330, 134, 1364, 1368, 1369, 1373, 1394, 1413, 1467, 1475, 1477, 1489, 148, 1514, 1517, 1521, 1531, 1542, 1570, 1613, 1614, 1628, 1642, 1674, 1691, 1703, 1713, 1716, 1749, 1753, 1760, 1780, 1788, 1796, 1798, 1807, 1845, 1862, 1875, 1876, 1945, 1964, 2007, 2079, 2092, 2129, 2139, 214, 2198, 2224, 223, 2254, 2273, 2294, 2299, 2309, 2326, 2331]) # 107, 1012, 1024, 1034, 1037, 1054, 1057, 1069, 1073, 1086, 1098, 
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
plt.imshow(X[0], cmap = "Greys_r")
plt.savefig(f"dm-finder/cnn/iact_images/results/{image_type}/" + "iact_image_example.png")

X_shape = np.shape(X)
X = X.reshape(-1, X_shape[1], X_shape[2], 1)

# output label: log10(true energy)
Y = np.log10(np.asarray(table["true_energy"]))

# # hold out the last 3000 events as test data
X_train, X_test = np.split(X, [int(-len(table) / 10)])
Y_train, Y_test = np.split(Y, [int(-len(table) / 10)])

# remove strange outlier
index = np.argmin(Y_test)
X_test = np.delete(X_test, index, axis = 0)
Y_test = np.delete(Y_test, index)

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

# # energy
# difference = (yp - Y_test) / Y_test #10**(yp - Y_test) - 1
# mean = np.mean(difference)
# sigma = np.std(difference)
# plt.figure()
# plt.title("CTA images")
# plt.grid(alpha = 0.2)
# plt.hist(difference, bins = np.linspace(np.min(difference), np.max(difference), 40))
# # plt.yscale("log")
# plt.xlabel("($\log_{10}(E_\mathrm{rec}/\mathrm{GeV}) - \log_{10}(E_\mathrm{true}/\mathrm{GeV})) / \log_{10}(E_\mathrm{true} /\mathrm{GeV})$")
# plt.ylabel("Number of events")
# plt.text(0.95, 0.95, "$\mu = %.3f$" % mean, ha="right", va="top", transform=plt.gca().transAxes)
# plt.text(0.95, 0.90, "$\sigma = %.3f$" % sigma, ha="right", va="top", transform=plt.gca().transAxes)
# # plt.vlines(mean, 0.7, 3e3, linestyle = "dashed", label = "$\mu = %.3f$" % mean, color = "black")
# # plt.vlines(mean + sigma, 0.7, 3e3, linestyle = "dashdot", label =  "$\sigma = %.3f$" % sigma, color = "black")
# # plt.vlines(mean - sigma, 0.7, 3e3, linestyle = "dashdot", color = "black")
# # plt.legend()
# plt.xlim(-0.4, 0.4)
# plt.savefig(f"dm-finder/cnn/iact_images/results/{image_type}/energy_histogram.png")

x = np.linspace(np.min(Y_test), np.max(Y_test), 100)
# plt.figure()
# plt.title("CTA images")
# plt.grid(alpha = 0.2)
# plt.plot(x, x, color="black")
# plt.scatter(Y_test, yp)
# plt.xlabel("$\log_{10}(E_\mathrm{true}/\mathrm{GeV})$")
# plt.ylabel("$\log_{10}(E_\mathrm{rec}/\mathrm{GeV})$")
# plt.savefig(f"dm-finder/cnn/iact_images/results/{image_type}/energy_scattering.png")

# remove strange outlier
index = np.argmin(yp)
yp = np.delete(yp, index)
Y_test = np.delete(Y_test, index)

# 2D energy scattering
plt.figure()
plt.title("CTA images")
plt.grid(alpha = 0.2)
plt.plot(x, x, color="black")
# plt.scatter(x,y,edgecolors='none',s=marker_size,c=void_fraction, norm=matplotlib.colors.LogNorm())
plt.hist2d(Y_test, yp, bins=(50, 50), cmap = "viridis", norm=matplotlib.colors.LogNorm())
cbar = plt.colorbar()
cbar.set_label('Number of events')
plt.xlabel("$\log_{10}(E_\mathrm{true}/\mathrm{GeV})$")
plt.ylabel("$\log_{10}(E_\mathrm{rec}/\mathrm{GeV})$")
plt.savefig(path + "energy_scattering_2D.png", dpi = 250)

# plot history
history_path = f"dm-finder/cnn/iact_images/history/{image_type}/" + "history.csv"

table_history = pd.read_csv(history_path)

plt.figure()
plt.plot(table_history["epoch"] + 1, table_history["loss"], label = "training")
plt.plot(table_history["epoch"] + 1, table_history["val_loss"], label = "validation")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.savefig(path + "loss.png")
plt.close()

# create csv output file
Y_test = np.reshape(Y_test, (len(Y_test), 1))
yp = np.reshape(yp, (len(yp), 1))
table_output = np.hstack((Y_test, yp))

path_output = f'dm-finder/cnn/iact_images/output/{image_type}/'

try:
    os.makedirs(path_output)
except OSError:
    pass #print("Directory could not be created")

pd.DataFrame(table_output).to_csv(f'dm-finder/cnn/iact_images/output/{image_type}/test_set_energy.csv', index = None, header = ["log10(E_true / GeV)", "log10(E_rec / GeV)"])
