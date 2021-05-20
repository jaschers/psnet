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
run = np.array([107]) # 107, 1012, 1024, 1034, 1037, 1054, 1057, 1069, 1073, 1086, 1098, 
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
print(X)
print("#####################################")
print(X[0])
print("#####################################")
print(X[0][0])
print("#####################################")
print(X[0][0][0])
X = X.reshape(-1, 21, 21, 1)

# output label: log10(true energy)
Y = np.log10(np.asarray(table["true_energy"]))

# # hold out the last 3000 events as test data
X_train, X_test = np.split(X, [int(-len(table) / 10)])
Y_train, Y_test = np.split(Y, [int(-len(table) / 10)])