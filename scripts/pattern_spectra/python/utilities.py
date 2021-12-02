import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math

def ExtractPatternSpectraMean(number_energy_ranges, size, pattern_spectra_binned):
    pattern_spectra_sum = np.zeros(shape = (number_energy_ranges, size[0], size[1]))
    pattern_spectra_mean = np.zeros(shape = (number_energy_ranges, size[0], size[1]))
    for i in range(number_energy_ranges):
        for j in range(len(pattern_spectra_binned[i])):
            pattern_spectra_sum[i] += pattern_spectra_binned[i][j]

        # calculate normed pattern spectra sum
        pattern_spectra_mean[i] = pattern_spectra_sum[i] / len(pattern_spectra_binned[i])
        # pattern_spectra_mean[i] = pattern_spectra_mean[i] / np.max(pattern_spectra_mean[i])

    return pattern_spectra_mean

def ExtractPatternSpectraDifference(number_energy_ranges, size, pattern_spectra_mean):
    pattern_spectra_mean_difference = np.zeros(shape = (number_energy_ranges, size[0], size[1]))
    for i in range(number_energy_ranges):
        # calculate normed patter spectra sum minus pattern spectra of first energy range
        pattern_spectra_mean_difference[i] = (pattern_spectra_mean[i] - pattern_spectra_mean[0]) 
        # pattern_spectra_mean_difference[i] = (pattern_spectra_mean[i] - pattern_spectra_mean[0]) / pattern_spectra_mean[0]
        # pattern_spectra_mean_difference[i] = np.nan_to_num(pattern_spectra_mean_difference[i], nan = 0, posinf = 0)
    # print("pattern_spectra_mean[8]", np.min(pattern_spectra_mean[8]), np.max(pattern_spectra_mean[8]), pattern_spectra_mean[8].flatten()[np.argmax(pattern_spectra_mean_difference[8])], pattern_spectra_mean[0].flatten()[np.argmax(pattern_spectra_mean_difference[8])])
    # print("pattern_spectra_mean_difference[8]", np.min(pattern_spectra_mean_difference[8]), np.max(pattern_spectra_mean_difference[8]), np.argmax(pattern_spectra_mean_difference[8]))

    return pattern_spectra_mean_difference

def ExtractPatternSpectraMinMax(number_energy_ranges, pattern_spectra):
    for i in range(number_energy_ranges):
        if i == 1:
            pattern_spectra_min = np.min(pattern_spectra[i])
            pattern_spectra_max = np.max(pattern_spectra[i])
        elif i > 1:
            if np.min(pattern_spectra[i]) < pattern_spectra_min:
                pattern_spectra_min = np.min(pattern_spectra[i])
            if np.max(pattern_spectra[i]) > pattern_spectra_max:
                pattern_spectra_max = np.max(pattern_spectra[i])
    
    return(pattern_spectra_min, pattern_spectra_max)

def PlotPatternSpectra(number_energy_ranges, pattern_spectra_mean, pattern_spectra_mean_min, pattern_spectra_mean_max, pattern_spectra_mean_difference, pattern_spectra_mean_difference_min, pattern_spectra_mean_difference_max, bins, particle_type, path):
    fig_mean, ax_mean = plt.subplots(int(math.ceil(np.sqrt(number_energy_ranges))), int(math.ceil(np.sqrt(number_energy_ranges))))
    ax_mean = ax_mean.ravel()
    fig_difference, ax_difference = plt.subplots(int(math.ceil(np.sqrt(number_energy_ranges))), int(math.ceil(np.sqrt(number_energy_ranges))))
    ax_difference = ax_difference.ravel()

    if abs(pattern_spectra_mean_difference_min) > abs(pattern_spectra_mean_difference_max):
        pattern_spectra_mean_difference_max = abs(pattern_spectra_mean_difference_min)
    if abs(pattern_spectra_mean_difference_min) < abs(pattern_spectra_mean_difference_max):
        pattern_spectra_mean_difference_min = - pattern_spectra_mean_difference_max 


    for i in range(number_energy_ranges):
        # plot pattern spectra sum
        fig_mean.suptitle(f"pattern spectra mean - {particle_type}")   
        ax_mean[i].set_title(f"{np.round(bins[i], 1)} - {np.round(bins[i+1], 1)} TeV", fontdict = {"fontsize" : 10})  
        im = ax_mean[i].imshow(pattern_spectra_mean[i], cmap = "afmhot_r")
        im.set_clim(pattern_spectra_mean_min, pattern_spectra_mean_max)
        ax_mean[i].set_xticks([])
        ax_mean[i].set_yticks([])
        divider = make_axes_locatable(ax_mean[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig_mean.colorbar(im, cax=cax, orientation="vertical")

        # plot pattern spectra difference
        fig_difference.suptitle(f"pattern spectra mean difference - {particle_type}")   
        ax_difference[i].set_title(f"{np.round(bins[i], 1)} - {np.round(bins[i+1], 1)} TeV", fontdict = {"fontsize" : 10})
        im = ax_difference[i].imshow(pattern_spectra_mean_difference[i], cmap = "RdBu", norm = SymLogNorm(linthresh = 0.1, vmin = pattern_spectra_mean_difference_min, vmax = pattern_spectra_mean_difference_max, base = 10))
        im.set_clim(pattern_spectra_mean_difference_min, pattern_spectra_mean_difference_max)
        ax_difference[i].set_xticks([])
        ax_difference[i].set_yticks([])
        divider = make_axes_locatable(ax_difference[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig_difference.colorbar(im, cax=cax, orientation="vertical")

    fig_mean.tight_layout()
    fig_mean.savefig(path + "pattern_spectra_mean.png", dpi = 250)
    fig_difference.tight_layout()
    fig_difference.savefig(path + "pattern_spectra_mean_difference.png", dpi = 250)
    # plt.show()

def PlotPatternSpectraComparison(number_energy_ranges, pattern_spectra_mean_gamma_proton, bins, particle_type, path):
    fig, ax = plt.subplots(int(math.ceil(np.sqrt(number_energy_ranges))), int(math.ceil(np.sqrt(number_energy_ranges))))
    ax = ax.ravel()
    pattern_spectra_mean_gamma_proton_difference = pattern_spectra_mean_gamma_proton[0] - pattern_spectra_mean_gamma_proton[1]
    pattern_spectra_mean_difference_min = np.min(pattern_spectra_mean_gamma_proton_difference)
    pattern_spectra_mean_difference_max = np.max(pattern_spectra_mean_gamma_proton_difference)
    if abs(pattern_spectra_mean_difference_min) > abs(pattern_spectra_mean_difference_max):
        pattern_spectra_mean_difference_max = abs(pattern_spectra_mean_difference_min)
    if abs(pattern_spectra_mean_difference_min) < abs(pattern_spectra_mean_difference_max):
        pattern_spectra_mean_difference_min = - pattern_spectra_mean_difference_max 


    for i in range(number_energy_ranges):

        # plot pattern spectra difference
        fig.suptitle(f"pattern spectra mean difference - {particle_type[0]} - {particle_type[1]}")   
        ax[i].set_title(f"{np.round(bins[i], 1)} - {np.round(bins[i+1], 1)} TeV", fontdict = {"fontsize" : 10})
        im = ax[i].imshow(pattern_spectra_mean_gamma_proton_difference[i], cmap = "RdBu", norm = SymLogNorm(linthresh = 0.01, base = 10))
        im.set_clim(pattern_spectra_mean_difference_min, pattern_spectra_mean_difference_max)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        divider = make_axes_locatable(ax[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax, orientation="vertical")

    fig.tight_layout()
    fig.savefig(path + f"pattern_spectra_{particle_type[0]}_{particle_type[1]}_comparison.png", dpi = 250)
    # plt.show()

def PlotPatternSpectraComparisonTotal(pattern_spectra_total_sum_normed_gamma_proton, particle_type, attributes, path):
    pattern_spectra_sum_total_normed_gamma_proton_difference = pattern_spectra_total_sum_normed_gamma_proton[0] - pattern_spectra_total_sum_normed_gamma_proton[1]
    pattern_spectra_sum_total_difference_min = np.min(pattern_spectra_sum_total_normed_gamma_proton_difference)
    pattern_spectra_sum_total_difference_max = np.max(pattern_spectra_sum_total_normed_gamma_proton_difference)

    if abs(pattern_spectra_sum_total_difference_min) > abs(pattern_spectra_sum_total_difference_max):
        pattern_spectra_sum_total_difference_max = abs(pattern_spectra_sum_total_difference_min)
    if abs(pattern_spectra_sum_total_difference_min) < abs(pattern_spectra_sum_total_difference_max):
        pattern_spectra_sum_total_difference_min = - pattern_spectra_sum_total_difference_max 


    plt.figure()
    plt.title(f"pattern spectra total mean difference - {particle_type[0]} - {particle_type[1]}", fontsize = 10)
    im = plt.imshow(pattern_spectra_sum_total_normed_gamma_proton_difference, cmap = "RdBu", norm = SymLogNorm(linthresh = 0.04, base = 10))
    im.set_clim(pattern_spectra_sum_total_difference_min, pattern_spectra_sum_total_difference_max)
    plt.xlabel(f"attribute {attributes[0]}", fontsize = 18)
    plt.ylabel(f"attribute {attributes[1]}", fontsize = 18)
    plt.xticks([])
    plt.yticks([])
    cb = plt.colorbar()
    cb.set_label(label='pixel flux', size = 18)
    cb.ax.tick_params(labelsize = 18) 
    plt.tight_layout()
    plt.savefig(path + f"pattern_spectra_{particle_type[0]}_{particle_type[1]}_comparison_total.png", dpi = 250)
    plt.close()
