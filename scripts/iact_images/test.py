import numpy as np
import csv
import pandas as pd

# pattern spectra properties
a = np.array([9, 0])
dl = np.array([0, 0])
dh = np.array([10, 100000])
m = np.array([2, 0])
n = np.array([20, 20])
f = 3

particle_type = "gamma" # gamma/proton/gamma_diffuse
image_type = "minimalistic" # default/minimalistic
run = 1012

run_filename = f"{particle_type}_20deg_0deg_run{run}___cta-prod5-paranal_desert-2147m-Paranal-dark_merged.DL1"
filename = f"dm-finder/cnn/iact_images/input/{particle_type}/{image_type}/" + run_filename + "_images.h5"
filename_8bit = f"dm-finder/cnn/iact_images/input/{particle_type}/{image_type}/" + run_filename + "_images_8bit.h5"
filename_ps = f"dm-finder/cnn/pattern_spectra/input/{particle_type}/{image_type}/" + f"/a_{a[0]}_{a[1]}__dl_{dl[0]}_{dl[1]}__dh_{dh[0]}_{dh[1]}__m_{m[0]}_{m[1]}__n_{n[0]}_{n[1]}__f_{f}/" + run_filename + "_ps.h5"



table = pd.read_hdf(filename)
table_8bit = pd.read_hdf(filename_8bit)
table_ps = pd.read_hdf(filename_ps)

# print(table.keys())
# print(table_8bit.keys())

# print(np.unique(table["obs_id"] == table_8bit["obs_id"]))
# print(np.unique(table["event_id"] == table_8bit["event_id"]))
# print(np.unique(table["true_energy"] == table_8bit["true_energy"]))
# print(np.unique(table["particle"] == table_8bit["particle"]))
# print(type(table["image"]), type(table_8bit["image"]))
# print(type(table["image"][0]), type(table_8bit["image"][0]))
# print(type(table["image"][0][0]), type(table_8bit["image"][0][0]))
# print(type(table["image"][0][0][0]), type(table_8bit["image"][0][0][0]))
# print(type(table["image"][0][0][0]), type(table_8bit["image"][0][0][0]))
# print(table["image"] == table_8bit["image"])
# print(table_8bit["image"][0][0][32])
# print(table_8bit["image"][0][0][32].astype(np.int8))
print(type(table_ps["pattern spectrum"][0][0][10]))
print(table_ps["pattern spectrum"][0][10])
print(table_ps["pattern spectrum"][0][10].astype(np.float32))
print(table_ps["pattern spectrum"][0][10].astype(np.float16))
