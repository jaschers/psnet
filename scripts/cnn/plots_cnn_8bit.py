import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import pandas as pd
import os
import sys

plt.rcParams.update({'font.size': 14})

# np.set_printoptions(threshold=sys.maxsize)
# ---------------------------------------------------
# Load cnn output
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
input_cnn = np.array(["iact_images", "iact_images", "pattern_spectra"])
label = np.array(["CTA images", "CTA images (8-bit)", "pattern spectra"])
sigma = [[]] * len(input_cnn)

for c in range(len(input_cnn)):
    if (input_cnn[c] == "iact_images") and (c == 0):
        filename_output = f'dm-finder/cnn/{input_cnn[c]}/output/{image_type}/test_set_energy_2.csv' # 'test_set_energy_a_{a[0]}_{a[1]}__dl_{dl[0]}_{dl[1]}__dh_{dh[0]}_{dh[1]}__m_{m[0]}_{m[1]}__n_{n[0]}_{n[1]}__f_{f}.csv'
    elif (input_cnn[c] == "iact_images") and (c == 1):
        filename_output = f'dm-finder/cnn/{input_cnn[c]}/output/{image_type}/test_set_energy_8bit_2.csv' # 'test_set_energy_a_{a[0]}_{a[1]}__dl_{dl[0]}_{dl[1]}__dh_{dh[0]}_{dh[1]}__m_{m[0]}_{m[1]}__n_{n[0]}_{n[1]}__f_{f}.csv'
    elif input_cnn[c] == "pattern_spectra":
        filename_output = f'dm-finder/cnn/{input_cnn[c]}/output/{image_type}/test_set_energy_a_{a[0]}_{a[1]}__dl_{dl[0]}_{dl[1]}__dh_{dh[0]}_{dh[1]}__m_{m[0]}_{m[1]}__n_{n[0]}_{n[1]}__f_{f}.csv'


    table_output = pd.read_csv(filename_output)

    table_output = table_output.sort_values(by = ["log10(E_true / GeV)"])

    # convert energy to TeV
    energy_true = np.asarray((10**table_output["log10(E_true / GeV)"] * 1e-3))
    energy_rec = np.asarray((10**table_output["log10(E_rec / GeV)"] * 1e-3))
    # print(energy_true)

    bins = np.logspace(np.log10(np.min(energy_true)), np.log10(np.max(energy_true)), 16)
    indices = np.array([], dtype = int)
    for i in range(len(bins) - 2):
        index = np.max(np.where(energy_true < bins[i+1])) + 1
        indices = np.append(indices, index)

    energy_true_binned = np.split(energy_true, indices)
    energy_rec_binned = np.split(energy_rec, indices)

    if (input_cnn[c] == "iact_images"):
        path = f"dm-finder/cnn/{input_cnn[c]}/results/{image_type}/" # + f"a_{a[0]}_{a[1]}__dl_{dl[0]}_{dl[1]}__dh_{dh[0]}_{dh[1]}__m_{m[0]}_{m[1]}__n_{n[0]}_{n[1]}__f_{f}/"
    elif input_cnn[c] == "pattern_spectra":
        path = f"dm-finder/cnn/{input_cnn[c]}/results/{image_type}/" + f"a_{a[0]}_{a[1]}__dl_{dl[0]}_{dl[1]}__dh_{dh[0]}_{dh[1]}__m_{m[0]}_{m[1]}__n_{n[0]}_{n[1]}__f_{f}/"

    # energy
    try:
        os.makedirs(path)
    except OSError:
        pass #print("Directory could not be created")

    relative_energy_error_single = (energy_rec - energy_true) / energy_true #10**(energy_rec - energy_true) - 1
    mean = np.mean(relative_energy_error_single)
    sigma_total = np.std(relative_energy_error_single)
    plt.figure()
    plt.title(f"{input_cnn[c]}")
    plt.grid(alpha = 0.2)
    plt.hist(relative_energy_error_single, bins = np.linspace(np.min(relative_energy_error_single), np.max(relative_energy_error_single), 40))
    # plt.yscale("log")
    plt.xlabel("($E_\mathrm{rec} - E_\mathrm{true})/ E_\mathrm{true}$")
    plt.ylabel("Number of events")
    plt.text(0.95, 0.95, "$\mu = %.3f$" % mean, ha="right", va="top", transform=plt.gca().transAxes)
    plt.text(0.95, 0.90, "$\sigma = %.3f$" % sigma_total, ha="right", va="top", transform=plt.gca().transAxes)
    # plt.vlines(mean, 0.7, 3e3, linestyle = "dashed", label = "$\mu = %.3f$" % mean, color = 'black')
    # plt.vlines(mean + sigma, 0.7, 3e3, linestyle = "dashdot", label =  "$\sigma = %.3f$" % sigma, color = 'black')
    # plt.vlines(mean - sigma, 0.7, 3e3, linestyle = "dashdot", color = 'black')
    # plt.legend()
    if c == 1:
        plt.savefig(path + "energy_total_histogram_8bit.png", dpi = 250)
    else:
        plt.savefig(path + "energy_total_histogram.png", dpi = 250)

    fig, ax = plt.subplots(3, 3)
    ax = ax.ravel()
    fig.suptitle(f"{label[c]}")
    for i in range(len(energy_true_binned) - 6):
        relative_energy_error = (energy_rec_binned[i+4] - energy_true_binned[i+4]) / energy_true_binned[i+4] 
        median = np.median(relative_energy_error)
        ax[i].set_title(f"{np.round(bins[i+4], 2)} TeV < $E_{{true}}$ < {np.round(bins[i+4+1], 2)} TeV", fontsize = 6)
        ax[i].grid(alpha = 0.2)
        ax[i].hist(relative_energy_error, bins = np.linspace(np.min(relative_energy_error), np.max(relative_energy_error), 40))
        # ax[i].yscale("log")
        # ax[i].text(0.95, 0.95, "median$ = %.3f$" % median, ha="right", va="top", transform=plt.gca().transAxes)
        # ax[i].text(0.95, 0.90, "$\sigma_single = %.3f$" % sigma_single, ha="right", va="top", transform=plt.gca().transAxes)
        ymin, ymax = ax[i].get_ylim()
        ax[i].vlines(median, ymin, ymax, color = "black", linestyle = '--', linewidth = 0.7,  label = "median$ = %.3f$" % median)
        ax[i].tick_params(axis='both', which='major', labelsize = 6)
        ax[i].legend(fontsize = 6)
        ymax = 1.3 * ymax
        ax[i].set_ylim(ymin, ymax)
        # ax[i].vlines(median, 0.7, 3e3, linestyle = "dashed", label = "$\mu = %.3f$" % median, color = 'black')
        # ax[i].vlines(median + sigma_single, 0.7, 3e3, linestyle = "dashdot", label =  "$\sigma_single = %.3f$" % sigma_single, color = 'black')
        # ax[i].vlines(median - sigma_single, 0.7, 3e3, linestyle = "dashdot", color = 'black')
        # ax[i].legend()
    ax[-2].set_xlabel("$(E_\mathrm{rec} - E_\mathrm{true}) / E_\mathrm{true}$", fontsize = 6)
    ax[3].set_ylabel("Number of events", fontsize = 6)
    plt.tight_layout()

    if c == 1:
        plt.savefig(path + "energy_binned_histogram_8bit.png", dpi = 250)
    else:
        plt.savefig(path + "energy_binned_histogram.png", dpi = 250)

    plt.close()


    sigma_collection = np.array([])
    fig, ax = plt.subplots(5, 3)
    ax = ax.ravel()
    fig.suptitle(f"{label[c]}")
    for i in range(len(energy_true_binned)):
        relative_energy_error = (energy_rec_binned[i] - energy_true_binned[i]) / energy_true_binned[i] 
        median = np.median(relative_energy_error)

        relative_energy_error_corrected = np.abs((energy_rec_binned[i] - energy_true_binned[i] - median) / energy_true_binned[i]) 
        relative_energy_error_corrected = np.sort(relative_energy_error_corrected)
        index_68 = int(len(relative_energy_error_corrected) * 0.68)

        sigma_single = relative_energy_error_corrected[index_68]
        sigma_collection = np.append(sigma_collection, sigma_single)
        
        ax[i].set_title(f"{np.round(bins[i], 2)} TeV < E < {np.round(bins[i+1], 2)} TeV", fontsize = 6)
        ax[i].grid(alpha = 0.2)
        ax[i].hist(relative_energy_error_corrected, bins = np.linspace(np.min(relative_energy_error_corrected), np.max(relative_energy_error_corrected), 40))
        # ax[i].yscale("log")
        # ax[i].text(0.95, 0.95, "median$ = %.3f$" % median, ha="right", va="top", transform=plt.gca().transAxes)
        # ax[i].text(0.95, 0.90, "$\sigma_single = %.3f$" % sigma_single, ha="right", va="top", transform=plt.gca().transAxes)
        ymin, ymax = ax[i].get_ylim()
        ax[i].vlines(sigma_single, ymin, ymax, color = "black", linestyle = '--', linewidth = 0.7,  label = "$\sigma_{68} = %.3f$" % sigma_single)
        ax[i].tick_params(axis='both', which='major', labelsize = 6)
        ax[i].legend(fontsize = 6)
        ymax = 1.3 * ymax
        ax[i].set_ylim(ymin, ymax)
        # ax[i].vlines(median, 0.7, 3e3, linestyle = "dashed", label = "$\mu = %.3f$" % median, color = 'black')
        # ax[i].vlines(median + sigma_single, 0.7, 3e3, linestyle = "dashdot", label =  "$\sigma_single = %.3f$" % sigma_single, color = 'black')
        # ax[i].vlines(median - sigma_single, 0.7, 3e3, linestyle = "dashdot", color = 'black')
        # ax[i].legend()
    ax[-2].set_xlabel("$(E_\mathrm{rec} - E_\mathrm{true}) / E_\mathrm{true}$", fontsize = 6)
    ax[3].set_ylabel("Number of events", fontsize = 6)
    plt.tight_layout()

    if c == 1:
        plt.savefig(path + "energy_binned_histogram_corrected_8bit.png", dpi = 250)
    else:
        plt.savefig(path + "energy_binned_histogram_corrected.png", dpi = 250)

    plt.close()

    sigma[c] = sigma_collection

    table_sigma = np.hstack((np.reshape(bins[:-1], (len(bins[:-1]), 1)), np.reshape(bins[1:], (len(bins[1:]), 1)), np.reshape(sigma_collection, (len(sigma_collection), 1))))

    #pd.DataFrame(table_sigma).to_csv(path + "energy_resolution.csv", index = None, header = ["E_min", "E_max", "((E_rec - E_true) / E_true)_68"])


    # plt.figure()
    # plt.plot(bins[:-1], median[:])
    # plt.xlabel("Energy [TeV]")
    # plt.ylabel("Median")
    # plt.xscale("log")
    # plt.savefig(path + "median.png")
    # plt.close()

bins_central = np.array([])
for i in range(len(bins) - 1):
    bins_central = np.append(bins_central, bins[i] + (bins[i+1] - bins[i]) / 2)

skip = 4
plt.figure()
plt.title("Energy resoultion", color = "white")
plt.grid(alpha = 0.2)
plt.errorbar(bins_central[skip:-2], sigma[0][skip:-2], xerr = (bins[skip:-2-1] - bins_central[skip:-2], bins_central[skip:-2] - bins[skip+1:-2]), linestyle = "", capsize = 3.0, marker = ".", label = "CTA images (original)")
plt.errorbar(bins_central[skip:-2], sigma[1][skip:-2], xerr = (bins[skip:-2-1] - bins_central[skip:-2], bins_central[skip:-2] - bins[skip+1:-2]), linestyle = "", capsize = 3.0, marker = ".", label = "CTA images (8-bit)")
plt.errorbar(bins_central[skip:-2], sigma[2][skip:-2], xerr = (bins[skip:-2-1] - bins_central[skip:-2], bins_central[skip:-2] - bins[skip+1:-2]), linestyle = "", capsize = 3.0, marker = ".", label = "pattern spectra")

# plt.hlines(0.24, np.min(bins_central[skip:]), np.max(bins_central[skip:]), label = "Grespan et al. - no cut", linestyle = "--", color = "black")
# plt.plot(bins[2:], sigma[2:])
plt.xlabel("Energy [TeV]")
plt.ylabel("$(\Delta E / E_\mathrm{true})_{68}$")
#plt.ylabel("$((E_\mathrm{rec} - E_\mathrm{true}) / E_\mathrm{true})_{68}$")
plt.xscale("log")
plt.legend()
plt.tight_layout()
plt.savefig(path + "energy_resolution_8bit_2.1_title.png", dpi = 250)
plt.close()


    ##############################################################################################

    # # display total energy distribution of data set
    # plt.figure()
    # plt.hist(energy_true, bins=np.logspace(np.log10(np.min(energy_true)),np.log10(np.max(energy_true)), 50))
    # plt.plot(bins, np.ones(len(bins)) * 10, linestyle = "", marker = "o")
    # plt.xlabel("true energy [TeV]")
    # plt.ylabel("number of events")
    # plt.xscale("log")
    # plt.yscale("log")
    # plt.savefig(path + "total_energy_distribution_test_set.png")
    # plt.close()

    # for i in range(len(energy_true_binned)):
    #     relative_energy_error = (energy_rec_binned[i] - energy_true_binned[i]) / energy_true_binned[i] 
    #     mean = np.mean(relative_energy_error)
    #     sigma = np.std(relative_energy_error)
    #     plt.figure()
    #     plt.title(f"pattern spectra - E < {np.round(bins[i], 2)}")
    #     plt.grid(alpha = 0.2)
    #     plt.hist(relative_energy_error, bins = np.linspace(np.min(relative_energy_error_single), np.max(relative_energy_error_single), 20))
    #     # plt.yscale("log")
    #     plt.xlabel("($\log_{10}(E_\mathrm{rec}/\mathrm{GeV}) - \log_{10}(E_\mathrm{true}/\mathrm{GeV}))/ \log_{10}(E_\mathrm{true} /\mathrm{GeV})$")
    #     plt.ylabel("Number of events")
    #     plt.text(0.95, 0.95, "$\mu = %.3f$" % mean, ha="right", va="top", transform=plt.gca().transAxes)
    #     plt.text(0.95, 0.90, "$\sigma = %.3f$" % sigma, ha="right", va="top", transform=plt.gca().transAxes)
    #     # plt.vlines(mean, 0.7, 3e3, linestyle = "dashed", label = "$\mu = %.3f$" % mean, color = 'black')
    #     # plt.vlines(mean + sigma, 0.7, 3e3, linestyle = "dashdot", label =  "$\sigma = %.3f$" % sigma, color = 'black')
    #     # plt.vlines(mean - sigma, 0.7, 3e3, linestyle = "dashdot", color = 'black')
    #     # plt.legend()
    #     plt.savefig(path + f"energy_histogram_Emax_{np.round(bins[i], 2)}.png")

    # relative_energy_error_single = (energy_rec - energy_true) / energy_true #10**(energy_rec - energy_true) - 1
    # mean = np.mean(relative_energy_error_single)
    # sigma = np.std(relative_energy_error_single)
    # plt.figure()
    # plt.title("pattern spectra")
    # plt.grid(alpha = 0.2)
    # plt.hist(relative_energy_error_single, bins = np.linspace(np.min(relative_energy_error_single), np.max(relative_energy_error_single), 40))
    # plt.yscale("log")
    # plt.xlabel("($\log_{10}(E_\mathrm{rec}/\mathrm{GeV}) - \log_{10}(E_\mathrm{true}/\mathrm{GeV}))/ \log_{10}(E_\mathrm{true} /\mathrm{GeV})$")
    # plt.ylabel("Number of events")
    # # plt.text(0.95, 0.95, "$\mu = %.3f$" % mean, ha="right", va="top", transform=plt.gca().transAxes)
    # # plt.text(0.95, 0.90, "$\sigma = %.3f$" % sigma, ha="right", va="top", transform=plt.gca().transAxes)
    # plt.vlines(mean, 0.7, 3e3, linestyle = "dashed", label = "$\mu = %.3f$" % mean, color = 'black')
    # plt.vlines(mean + sigma, 0.7, 3e3, linestyle = "dashdot", label =  "$\sigma = %.3f$" % sigma, color = 'black')
    # plt.vlines(mean - sigma, 0.7, 3e3, linestyle = "dashdot", color = 'black')
    # plt.legend()
    # plt.savefig(path + "energy_histogram.png")

    # x = np.linspace(np.min(energy_true), np.max(energy_true), 100)
    # plt.figure()
    # plt.title("pattern spectra")
    # plt.grid(alpha = 0.2)
    # plt.plot(x, x, color="black")
    # plt.scatter(energy_true, energy_rec)
    # plt.xlabel("$\log_{10}(E_\mathrm{true}/\mathrm{GeV})$")
    # plt.ylabel("$\log_{10}(E_\mathrm{rec}/\mathrm{GeV})$")
    # plt.savefig(path + "energy_scattering.png")

    # # 2D energy scattering
    # plt.figure()
    # plt.title("pattern spectra")
    # plt.grid(alpha = 0.2)
    # plt.plot(x, x, color="black")
    # # plt.scatter(x,y,edgecolors='none',s=marker_size,c=void_fraction, norm=matplotlib.colors.LogNorm())
    # plt.hist2d(energy_true, energy_rec, bins=(50, 50), cmap = "viridis", norm=matplotlib.colors.LogNorm())
    # plt.colorbar()
    # plt.xlabel("$\log_{10}(E_\mathrm{true}/\mathrm{GeV})$")
    # plt.ylabel("$\log_{10}(E_\mathrm{rec}/\mathrm{GeV})$")
    # plt.savefig(path + "energy_scattering_2D.png")

    # # plot history
    # history_path = f"dm-finder/cnn/{input_cnn[c]}/history/{image_type}/" + f"history_a_{a[0]}_{a[1]}__dl_{dl[0]}_{dl[1]}__dh_{dh[0]}_{dh[1]}__m_{m[0]}_{m[1]}__n_{n[0]}_{n[1]}__f_{f}_mu_0.009.csv"

    # table_history = pd.read_csv(history_path)

    # plt.figure()
    # plt.plot(table_history["epoch"] + 1, table_history["loss"], label = "training")
    # plt.plot(table_history["epoch"] + 1, table_history["val_loss"], label = "validation")
    # plt.xlabel("epoch")
    # plt.ylabel("loss")
    # plt.legend()
    # plt.savefig(path + "loss.png")
    # plt.close()