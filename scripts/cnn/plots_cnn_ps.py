import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import pandas as pd
import os
import sys
# np.set_printoptions(threshold=sys.maxsize)
# ---------------------------------------------------
# Load cnn output
# ---------------------------------------------------

particle_type = "gamma"
image_type = "minimalistic"
input_cnn = "pattern_spectra"

pattern_spectra = np.array(["a_9_0__dl_0_0__dh_10_100000__m_2_0__n_15_15__f_3", "a_9_0__dl_0_0__dh_10_100000__m_2_0__n_20_20__f_3", "a_9_0__dl_0_0__dh_10_100000__m_2_0__n_30_30__f_3"]) # a_9_0__dl_0_0__dh_10_100000__m_2_0__n_30_30__f_3, a_10_0__dl_0_0__dh_10_100000__m_2_0__n_20_20__f_3

# plot_label = np.array(["attributes: (moment of inertia) / (area*area), area", "attributes: compactnes, area", "attributes: sum grey levels, area"])
plot_label = np.array(["size: 15 x 15", "size: 20 x 20", "size: 30 x 30"])

skip = 3
plt.figure()
plt.grid(alpha = 0.2)
for i in range(len(pattern_spectra)):
    filename = f"dm-finder/cnn/{input_cnn}/results/{image_type}/" + f"{pattern_spectra[i]}/" + "energy_resolution.csv"

    table = pd.read_csv(filename)

    if i == 0:
        bins = pd.concat([table["E_min"], table["E_max"].iloc[[-1]]], ignore_index = True)
        bins_central = np.array([])
        for b in range(len(bins) - 1):
            bins_central = np.append(bins_central, bins[b] + (bins[b+1] - bins[b]) / 2)

    plt.errorbar(bins_central[skip:], table["((E_rec - E_true) / E_true)_68"][skip:], xerr = (bins[skip:-1] - bins_central[skip:], bins_central[skip:] - bins[skip+1:]), linestyle = "", capsize = 3.0, marker = ".", label = f"{plot_label[i]}")

path = f"dm-finder/cnn/{input_cnn}/results/{image_type}/"

plt.xlabel("Energy [TeV]")
plt.ylabel("$(\Delta E / E)_{68}$")
plt.xscale("log")
plt.legend()
plt.savefig(path + "energy_resolution_comparison_size.png", dpi = 250)
plt.close()
