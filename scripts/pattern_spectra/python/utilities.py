import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math

def PlotPatternSpectra(number_energy_ranges, size, bins, pattern_spectra_binned, path):
    pattern_spectra_sum = np.zeros(shape = (number_energy_ranges, size[0], size[1]))
    pattern_spectra_sum_normed = np.zeros(shape = (number_energy_ranges, size[0], size[1]))
    pattern_spectra_sum_difference = np.zeros(shape = (number_energy_ranges, size[0], size[1]))

    fig_normed, ax_normed = plt.subplots(int(math.ceil(np.sqrt(number_energy_ranges))), int(math.ceil(np.sqrt(number_energy_ranges))))
    ax_normed = ax_normed.ravel()
    fig_difference, ax_difference = plt.subplots(int(math.ceil(np.sqrt(number_energy_ranges))), int(math.ceil(np.sqrt(number_energy_ranges))))
    ax_difference = ax_difference.ravel()

    for i in range(number_energy_ranges):
        for j in range(len(pattern_spectra_binned[i])):
            pattern_spectra_sum[i] += pattern_spectra_binned[i][j]

        # calculate normed pattern spectra sum
        pattern_spectra_sum_normed[i] = (pattern_spectra_sum[i] - np.min(pattern_spectra_sum[i]))
        pattern_spectra_sum_normed[i] /= np.max(pattern_spectra_sum_normed[i])
        # calculate normed patter spectra sum minus pattern spectra of first energy range
        pattern_spectra_sum_difference[i] = pattern_spectra_sum_normed[i] - pattern_spectra_sum_normed[0]

        # pattern_spectra_sum_difference[i] = (pattern_spectra_sum_difference[i] - np.min(pattern_spectra_sum_difference[i])) / np.max(pattern_spectra_sum_difference[i])
        # pattern_spectra_sum_difference[i] /= np.max(pattern_spectra_sum_difference[i])

        # plot pattern spectra sum
        fig_normed.suptitle("pattern spectra sum")   
        ax_normed[i].set_title(f"{np.round(bins[i], 1)} - {np.round(bins[i+1], 1)} TeV", fontdict = {"fontsize" : 10})  
        im = ax_normed[i].imshow((pattern_spectra_sum_normed[i]))
        ax_normed[i].set_xticks([])
        ax_normed[i].set_yticks([])
        divider = make_axes_locatable(ax_normed[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig_normed.colorbar(im, cax=cax, orientation="vertical")

        # plot pattern spectra difference
        fig_difference.suptitle("pattern spectra sum difference")   
        ax_difference[i].set_title(f"{np.round(bins[i], 1)} - {np.round(bins[i+1], 1)} TeV", fontdict = {"fontsize" : 10})  
        im = ax_difference[i].imshow((pattern_spectra_sum_difference[i]))
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