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

# import data
particle_type = np.array(["gamma_diffuse", "proton"])
image_type = "minimalistic"
# run = np.array([[1010], [10029, 10033]])
run = np.array([[1010, 1027, 1029, 104, 105, 1060, 1062, 1070, 1071, 1077, 1088, 108, 1102, 1115, 1118, 1134, 1140, 1161, 116, 1188, 1200, 1213, 1219, 1225, 1226, 1228, 1232, 1249, 1250, 1251, 1258, 1286, 1289, 1291, 1322, 1325, 132, 1334, 1342, 1344, 1349, 1371, 1374, 1379, 1407, 140, 1420, 1444, 1446, 1449, 1450, 1459, 145, 1472, 1473, 1474, 1477, 1484, 1496, 1537, 1545, 1548, 1573, 1577, 1586, 1631, 1655, 1659, 1668, 1693, 1697, 1702, 1704, 1705, 1708, 1736, 1737, 1744, 1752, 1772, 1791, 17, 180, 1813, 1815, 1817, 1820, 1853, 1868, 1879, 1886, 1887, 192, 1960, 1977, 1983, 1985, 2004, 2006, 2016, 201, 2029, 2041, 2049, 2068, 2087], [10029, 10033, 10037, 10068, 10070, 10076, 10086, 10101, 10102, 10117, 10125, 10144, 10166, 10207, 10217, 10228, 10251, 10255, 10259, 10260, 10262, 10271, 10276, 10278, 10295, 10299, 10310, 10321, 10330, 1034, 10356, 10364, 10365, 10368, 10407, 10416, 10426, 10430, 10431, 1047, 10483, 10503, 10517, 10529, 10555, 1056, 10578, 10609, 10612, 10629, 10633, 10641, 10654, 10662, 10668, 10703, 10725, 10732, 10736, 10738, 10779, 10801, 10812, 10841, 10882, 10900, 10915, 10934, 10949, 10959, 10970, 10988, 10997, 11001, 11008, 11032, 11037, 11045, 11050, 11090, 11096, 11133, 11146, 11168, 11189, 1119, 11211, 11224, 1122, 11239, 11240, 11254, 11263, 11267, 11271, 11272, 1127, 11290, 11301, 11309, 11322, 1134, 11396, 11398, 11405, 11413, 11420, 11425, 11438, 11452, 1145, 11495, 11498, 11499, 11502, 11524, 11525, 1155, 11572, 11576, 11577, 11587, 11609, 11669, 1166, 11734, 11738, 11748, 1176, 11777, 11783, 11810, 11837, 11848, 11868, 11909, 11913, 11943, 11952, 11957, 11968, 11982, 12000, 12002, 12005, 12014, 12022, 12031, 1206, 1210, 12125, 12127, 12155, 12162, 12173, 12181, 12211, 12240, 12246, 12248, 12251, 12295, 12311, 12320, 12338, 12406, 12478, 12479, 12486, 12488, 12522, 1252, 12560, 12565, 1256, 12572, 12573, 12609, 12625, 12638, 12648, 12663, 12697, 12708, 12719, 12779, 12803, 12821, 1283, 12850, 1285, 12861, 12873, 1289, 12916, 12958, 12970, 12982, 13044, 13050, 13065, 13137, 13161, 13168, 1316, 13185, 13197, 13241, 13244, 13252, 13271, 13282, 13320, 13421, 13442, 13466, 13510, 13512, 13612, 13633, 1367, 13682, 13688, 13712, 13733, 13773, 13792, 13793, 13795, 13842, 13845, 13854]], dtype = object) 
table = pd.DataFrame()
events_count = np.array([0, 0])
for p in range(len(particle_type)):
    for r in range(len(run[p])):
        run_filename = f"{particle_type[p]}_20deg_0deg_run{run[p][r]}___cta-prod5-paranal_desert-2147m-Paranal-dark_merged.DL1"
        input_filename = f"dm-finder/cnn/pattern_spectra/input/{particle_type[p]}/{image_type}/" + f"a_{a[0]}_{a[1]}__dl_{dl[0]}_{dl[1]}__dh_{dh[0]}_{dh[1]}__m_{m[0]}_{m[1]}__n_{n[0]}_{n[1]}__f_{f}/" + run_filename + "_ps.h5"

        table_individual_run = pd.read_hdf(input_filename)
        print(f"Number of events in {particle_type[p]} Run {run[p][r]}:", len(table_individual_run))
        if (particle_type[p] == "gamma_diffuse") or particle_type[p] == "gamma":
            events_count[0] += len(table_individual_run)
        if particle_type[p] == "proton":
            events_count[1] += len(table_individual_run)
        table = table.append(table_individual_run, ignore_index = True)

print("______________________________________________")
print("Total number of gamma events:", events_count[0])
print("Total number of proton events:", events_count[1])
print("Total number of events:", len(table))

# table = table.head(10)
# print(table)
# print(table[table["event_id"] == 14205]["pattern spectrum"])
table = table.sample(frac=1).reset_index(drop=True)
# print(table)
# print(table["pattern spectrum"][0])
# print(table[table["event_id"] == 14205]["pattern spectrum"])

# input features: CTA images 
X = [[]] * len(table)
for i in range(len(table)):
    X[i] = table["pattern spectrum"][i]
X = np.asarray(X) # / 255
X_shape = np.shape(X)
X = X.reshape(-1, X_shape[1], X_shape[2], 1)

# output label: log10(true energy)
Y = np.asarray(table["particle"])
Y = keras.utils.to_categorical(Y, 2)
# # hold out the last 3000 events as test data
X_train, X_test = np.split(X, [int(-len(table) / 10)])
Y_train, Y_test = np.split(Y, [int(-len(table) / 10)])

path = f"dm-finder/cnn/pattern_spectra/results/{image_type}/" + f"a_{a[0]}_{a[1]}__dl_{dl[0]}_{dl[1]}__dh_{dh[0]}_{dh[1]}__m_{m[0]}_{m[1]}__n_{n[0]}_{n[1]}__f_{f}/seperation/"

try:
    os.makedirs(path)
except OSError:
    pass #print("Directory could not be created")


# display total energy distribution of data set
plt.figure()
plt.hist(table["true_energy"][np.where(table["particle"] == 1)[0]], bins=np.logspace(np.log10(np.min(table["true_energy"][np.where(table["particle"] == 1)[0]])),np.log10(np.max(table["true_energy"][np.where(table["particle"] == 1)[0]])), 50), label = "gamma", alpha = 0.5)
plt.hist(table["true_energy"][np.where(table["particle"] == 0)[0]], bins=np.logspace(np.log10(np.min(table["true_energy"][np.where(table["particle"] == 0)[0]])),np.log10(np.max(table["true_energy"][np.where(table["particle"] == 0)[0]])), 50), label = "proton", alpha = 0.5)
plt.xlabel("true energy [GeV]")
plt.ylabel("number of events")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.savefig(path + f"total_energy_distribution.png")
plt.close()

# # display a random CTA image
# for i in range(5):
#     plt.figure()
#     plt.title(f"particle - {Y_train[i]}")
#     plt.imshow(X_train[i])
#     plt.savefig(f"dm-finder/cnn/pattern_spectra/results/{image_type}/separation/example_{i}.png")
#     plt.close()

# ----------------------------------------------------------------------
# Define the model
# ----------------------------------------------------------------------
X_shape = np.shape(X_train)
Y_shape = np.shape(Y_train)
input1 = layers.Input(shape = X_shape[1:])

# define a suitable network 
z = Conv2D(4, # number of filters, the dimensionality of the output space
    kernel_size = (3,3),
    padding = "same", # size of filters 3x3
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

# z = Conv2D(8,(3,3),activation="relu")(input1)
# z = Conv2D(16,(3,3),activation="relu")(z)
# z = GlobalAveragePooling2D()(z)

# z = Dense(16,activation="relu")(z)
# # z = Dense(64,activation="relu")(z)

output = Dense(2,activation='softmax', name = "gammaness")(z)

# ----------------------------------------------------------------------
# Train the model
# ----------------------------------------------------------------------

model = keras.models.Model(inputs=input1, outputs=output)

print(model.summary())

#weight estimations, relative values for all attributes:
weight_energy = 1

model.compile(
    loss="categorical_crossentropy",
    loss_weights=weight_energy,  
    optimizer=keras.optimizers.Adam(lr=1E-3))


history_path = f"dm-finder/cnn/pattern_spectra/history/{image_type}/separation/"
try:
    os.makedirs(history_path)
except OSError:
    pass #print("Directory could not be created")


# start timer
start_time = time.time()

fit = model.fit(X_train,
    Y_train,
    batch_size=32,
    epochs=50,
    verbose=2,
    validation_split=0.1,
    callbacks=[CSVLogger(history_path + f"history_a_{a[0]}_{a[1]}__dl_{dl[0]}_{dl[1]}__dh_{dh[0]}_{dh[1]}__m_{m[0]}_{m[1]}__n_{n[0]}_{n[1]}__f_{f}.csv")])

# end timer and print training time
print("Time spend for training the CNN: ", time.time() - start_time)

model_path = f"dm-finder/cnn/pattern_spectra/model/{image_type}/separation/" + "model_a_{a[0]}_{a[1]}__dl_{dl[0]}_{dl[1]}__dh_{dh[0]}_{dh[1]}__m_{m[0]}_{m[1]}__n_{n[0]}_{n[1]}__f_{f}.h5"

model.save(model_path)

# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------

model_path = f"dm-finder/cnn/pattern_spectra/model/{image_type}/separation/" + "model_a_{a[0]}_{a[1]}__dl_{dl[0]}_{dl[1]}__dh_{dh[0]}_{dh[1]}__m_{m[0]}_{m[1]}__n_{n[0]}_{n[1]}__f_{f}.h5"

model = keras.models.load_model(model_path)

losses = model.evaluate(X_test, Y_test, batch_size=128, verbose=0)
# print("Test loss with", title_names[i])
print("%.5e (energy)" % losses)

# predict output for test set and undo feature scaling
yp = model.predict(X_test, batch_size=128)
yp = yp[:, 0]  # remove unnecessary last axis


plt.figure()
plt.hist(yp[np.where(Y_test[:, 0] == 1)[0]], bins = 50, alpha = 0.5, label = "gamma")
plt.hist(yp[np.where(Y_test[:, 1] == 1)[0]], bins = 50, alpha = 0.5, label = "proton")
plt.xlabel("Gammaness")
plt.ylabel("Number of events")
plt.yscale("log")
plt.legend()
plt.savefig(path + f"gammaness.png")
plt.close()

# # create csv output file
# Y_test = np.reshape(Y_test, (len(Y_test), 2))
# yp = np.reshape(yp, (len(yp), 2))
# table_output = np.hstack((Y_test, yp))

# path_output = f'dm-finder/cnn/pattern_spectra/output/{image_type}/separation/'

# try:
#     os.makedirs(path_output)
# except OSError:
#     pass #print("Directory could not be created")

# pd.DataFrame(table_output).to_csv(path_output + "test_set_gammaness.csv", index = None, header = ["gammaness_true", "gammaness_rec"])

# path = f"dm-finder/cnn/pattern_spectra/results/{image_type}/separation/"

