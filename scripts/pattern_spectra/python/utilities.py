import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math

def ExtractPatternSpectraSum(number_energy_ranges, size, pattern_spectra_binned):
    pattern_spectra_sum = np.zeros(shape = (number_energy_ranges, size[0], size[1]))
    pattern_spectra_sum_normed = np.zeros(shape = (number_energy_ranges, size[0], size[1]))
    for i in range(number_energy_ranges):
        for j in range(len(pattern_spectra_binned[i])):
            pattern_spectra_sum[i] += pattern_spectra_binned[i][j]
        # calculate normed pattern spectra sum
        pattern_spectra_sum_normed[i] = (pattern_spectra_sum[i] - np.min(pattern_spectra_sum[i]))
        pattern_spectra_sum_normed[i] = pattern_spectra_sum_normed[i] / np.max(pattern_spectra_sum_normed[i])

    return pattern_spectra_sum_normed

def ExtractPatternSpectraDifference(number_energy_ranges, size, pattern_spectra_sum_normed):
    pattern_spectra_sum_difference = np.zeros(shape = (number_energy_ranges, size[0], size[1]))
    for i in range(number_energy_ranges):
        # calculate normed patter spectra sum minus pattern spectra of first energy range
        pattern_spectra_sum_difference[i] = pattern_spectra_sum_normed[i] - pattern_spectra_sum_normed[0]
        if i == 1:
            pattern_spectra_sum_difference_min = np.min(pattern_spectra_sum_difference[i])
            pattern_spectra_sum_difference_max = np.max(pattern_spectra_sum_difference[i])
        elif i > 1:
            if np.min(pattern_spectra_sum_difference[i]) < pattern_spectra_sum_difference_min:
                pattern_spectra_sum_difference_min = np.min(pattern_spectra_sum_difference[i])
            if np.max(pattern_spectra_sum_difference[i]) > pattern_spectra_sum_difference_max:
                pattern_spectra_sum_difference_max = np.max(pattern_spectra_sum_difference[i])
    
    return pattern_spectra_sum_difference

def ExtractPatternSpectraDiffernceMinMax(number_energy_ranges, pattern_spectra_sum_difference):
    for i in range(number_energy_ranges):
        if i == 1:
            pattern_spectra_sum_difference_min = np.min(pattern_spectra_sum_difference[i])
            pattern_spectra_sum_difference_max = np.max(pattern_spectra_sum_difference[i])
        elif i > 1:
            if np.min(pattern_spectra_sum_difference[i]) < pattern_spectra_sum_difference_min:
                pattern_spectra_sum_difference_min = np.min(pattern_spectra_sum_difference[i])
            if np.max(pattern_spectra_sum_difference[i]) > pattern_spectra_sum_difference_max:
                pattern_spectra_sum_difference_max = np.max(pattern_spectra_sum_difference[i])
    
    return(pattern_spectra_sum_difference_min, pattern_spectra_sum_difference_max)

def PlotPatternSpectra(number_energy_ranges, pattern_spectra_sum_normed, pattern_spectra_sum_difference, pattern_spectra_sum_difference_min, pattern_spectra_sum_difference_max, bins, particle_type, path):
    fig_normed, ax_normed = plt.subplots(int(math.ceil(np.sqrt(number_energy_ranges))), int(math.ceil(np.sqrt(number_energy_ranges))))
    ax_normed = ax_normed.ravel()
    fig_difference, ax_difference = plt.subplots(int(math.ceil(np.sqrt(number_energy_ranges))), int(math.ceil(np.sqrt(number_energy_ranges))))
    ax_difference = ax_difference.ravel()

    if abs(pattern_spectra_sum_difference_min) > abs(pattern_spectra_sum_difference_max):
        pattern_spectra_sum_difference_max = abs(pattern_spectra_sum_difference_min)
    if abs(pattern_spectra_sum_difference_min) < abs(pattern_spectra_sum_difference_max):
        pattern_spectra_sum_difference_min = - pattern_spectra_sum_difference_max 

    for i in range(number_energy_ranges):
        # plot pattern spectra sum
        fig_normed.suptitle(f"pattern spectra sum - {particle_type}")   
        ax_normed[i].set_title(f"{np.round(bins[i], 1)} - {np.round(bins[i+1], 1)} TeV", fontdict = {"fontsize" : 10})  
        im = ax_normed[i].imshow(pattern_spectra_sum_normed[i])
        ax_normed[i].set_xticks([])
        ax_normed[i].set_yticks([])
        divider = make_axes_locatable(ax_normed[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig_normed.colorbar(im, cax=cax, orientation="vertical")

        # plot pattern spectra difference
        fig_difference.suptitle(f"pattern spectra sum difference - {particle_type}")   
        ax_difference[i].set_title(f"{np.round(bins[i], 1)} - {np.round(bins[i+1], 1)} TeV", fontdict = {"fontsize" : 10})
        im = ax_difference[i].imshow(pattern_spectra_sum_difference[i], cmap = "RdBu", norm = SymLogNorm(linthresh = 0.01, vmin = pattern_spectra_sum_difference_min, vmax = pattern_spectra_sum_difference_max, base = 10))
        im.set_clim(pattern_spectra_sum_difference_min, pattern_spectra_sum_difference_max)
        ax_difference[i].set_xticks([])
        ax_difference[i].set_yticks([])
        divider = make_axes_locatable(ax_difference[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig_difference.colorbar(im, cax=cax, orientation="vertical")

    fig_normed.tight_layout()
    fig_normed.savefig(path + "pattern_spectra_sum_normed.png", dpi = 250)
    fig_difference.tight_layout()
    fig_difference.savefig(path + "pattern_spectra_sum_difference.png", dpi = 250)
    # plt.show()

def PlotPatternSpectraComparison(number_energy_ranges, pattern_spectra_sum_normed_gamma_proton, bins, particle_type, path):
    fig, ax = plt.subplots(int(math.ceil(np.sqrt(number_energy_ranges))), int(math.ceil(np.sqrt(number_energy_ranges))))
    ax = ax.ravel()
    pattern_spectra_sum_normed_gamma_proton_difference = pattern_spectra_sum_normed_gamma_proton[0] - pattern_spectra_sum_normed_gamma_proton[1]
    pattern_spectra_sum_difference_min = np.min(pattern_spectra_sum_normed_gamma_proton_difference)
    pattern_spectra_sum_difference_max = np.max(pattern_spectra_sum_normed_gamma_proton_difference)
    if abs(pattern_spectra_sum_difference_min) > abs(pattern_spectra_sum_difference_max):
        pattern_spectra_sum_difference_max = abs(pattern_spectra_sum_difference_min)
    if abs(pattern_spectra_sum_difference_min) < abs(pattern_spectra_sum_difference_max):
        pattern_spectra_sum_difference_min = - pattern_spectra_sum_difference_max 


    for i in range(number_energy_ranges):

        # plot pattern spectra difference
        fig.suptitle(f"pattern spectra sum difference - {particle_type[0]} - {particle_type[1]}")   
        ax[i].set_title(f"{np.round(bins[i], 1)} - {np.round(bins[i+1], 1)} TeV", fontdict = {"fontsize" : 10})
        im = ax[i].imshow(pattern_spectra_sum_normed_gamma_proton_difference[i], cmap = "RdBu", norm = SymLogNorm(linthresh = 0.01, base = 10))
        im.set_clim(pattern_spectra_sum_difference_min, pattern_spectra_sum_difference_max)
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
    plt.title(f"pattern spectra total sum difference - {particle_type[0]} - {particle_type[1]}", fontsize = 10)
    im = plt.imshow(pattern_spectra_sum_total_normed_gamma_proton_difference, cmap = "RdBu", norm = SymLogNorm(linthresh = 0.01, base = 10))
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
