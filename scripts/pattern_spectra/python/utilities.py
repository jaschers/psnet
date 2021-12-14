import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import SymLogNorm, LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math

def PlotEnergyDistribution(table, path):

    table_gamma = np.asarray(table[table["particle"] == 1].reset_index(drop = True)["true_energy"])
    table_proton = np.asarray(table[table["particle"] == 0].reset_index(drop = True)["true_energy"])

    if np.min(table_gamma) < np.min(table_proton):
        table_min = np.min(table_gamma)
    else:
        table_min = np.min(table_proton)
    if np.max(table_gamma) > np.max(table_proton):
        table_max = np.max(table_gamma)
    else:
        table_max = np.max(table_proton)

    plt.hist(table_gamma, bins = np.logspace(np.log10(table_min), np.log10(table_max), 30), alpha = 0.5, label = "gamma")
    plt.hist(table_proton, bins = np.logspace(np.log10(table_min), np.log10(table_max), 30), alpha = 0.5, label = "proton")
    plt.xlabel("True energy [GeV]")
    plt.ylabel("Number of events")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.savefig(path + "energy_distribution.png", dpi = 250)
    plt.close()

def cstm_PuBu(x):
    return plt.cm.PuBu((np.clip(x,2,10)-2)/8.)

def cstm_RdBu(x):
    return plt.cm.RdBu((np.clip(x,2,10)-2)/8.)

def ExtractPatternSpectraMean(number_energy_ranges, size, pattern_spectra_binned):
    print("Extract mean pattern spectra...")
    pattern_spectra_mean = np.zeros(shape = (number_energy_ranges, size[0], size[1]))
    for i in range(number_energy_ranges):
        pattern_spectra_mean[i] = np.mean(pattern_spectra_binned[i], axis = 0)

    return pattern_spectra_mean

def ExtractPatternSpectraMedian(number_energy_ranges, size, pattern_spectra_binned):
    print("Extract median pattern spectra...")
    pattern_spectra_median = np.zeros(shape = (number_energy_ranges, size[0], size[1]))
    for i in range(number_energy_ranges):
        pattern_spectra_median[i] = np.median(pattern_spectra_binned[i], axis = 0)

    return pattern_spectra_median

def ExtractPatternSpectraVariance(number_energy_ranges, size, pattern_spectra_binned):
    print("Extract variance of pattern spectra...")
    pattern_spectra_variance = np.zeros(shape = (number_energy_ranges, size[0], size[1]))
    for i in range(number_energy_ranges):
        pattern_spectra_variance[i] = np.var(pattern_spectra_binned[i], axis = 0)

    return pattern_spectra_variance

def ExtractPatternSpectraDifference(number_energy_ranges, size, pattern_spectra):
    pattern_spectra_difference = np.zeros(shape = (number_energy_ranges, size[0], size[1]))
    for i in range(number_energy_ranges):
        # calculate normed patter spectra sum minus pattern spectra of first energy range
        pattern_spectra_difference[i] = (pattern_spectra[i] - pattern_spectra[0]) 
        # pattern_spectra_difference[i] = (pattern_spectra[i] - pattern_spectra[0]) / pattern_spectra[0]
        # pattern_spectra_difference[i] = np.nan_to_num(pattern_spectra_difference[i], nan = 0, posinf = 0)
    # print("pattern_spectra[8]", np.min(pattern_spectra[8]), np.max(pattern_spectra[8]), pattern_spectra[8].flatten()[np.argmax(pattern_spectra_difference[8])], pattern_spectra[0].flatten()[np.argmax(pattern_spectra_difference[8])])
    # print("pattern_spectra_difference[8]", np.min(pattern_spectra_difference[8]), np.max(pattern_spectra_difference[8]), np.argmax(pattern_spectra_difference[8]))

    return pattern_spectra_difference

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

def PlotPatternSpectraMean(number_energy_ranges, pattern_spectra_mean, pattern_spectra_mean_min, pattern_spectra_mean_max, pattern_spectra_mean_difference, pattern_spectra_mean_difference_min, pattern_spectra_mean_difference_max, bins, particle_type, cmap, attributes, path):
    print("Plot mean pattern spectra...")
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
        im = ax_mean[i].imshow(pattern_spectra_mean[i], cmap = cmap) #, norm = SymLogNorm(linthresh = 0.1, base = 10))
        im.set_clim(pattern_spectra_mean_min, pattern_spectra_mean_max)
        ax_mean[i].set_xticks([])
        ax_mean[i].set_yticks([])
        divider = make_axes_locatable(ax_mean[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar_mean = fig_mean.colorbar(im, cax = cax, orientation = "vertical")
        cbar_mean.set_label(label = "log$_{10}$(flux)")
        ax_mean[i].annotate('', xy=(0, -0.1), xycoords='axes fraction', xytext=(1, -0.1), arrowprops=dict(arrowstyle="<-", color='black'))
        ax_mean[i].annotate('', xy=(-0.1, 1), xycoords='axes fraction', xytext=(-0.1, 0), arrowprops=dict(arrowstyle="<-", color='black'))
        ax_mean[i].set_xlabel(f"a {attributes[0]}", labelpad = 10)
        ax_mean[i].set_ylabel(f"a {attributes[1]}", labelpad = 10)

        # plot pattern spectra difference
        fig_difference.suptitle(f"pattern spectra mean difference - {particle_type}")   
        ax_difference[i].set_title(f"{np.round(bins[i], 1)} - {np.round(bins[i+1], 1)} TeV", fontdict = {"fontsize" : 10})
        im = ax_difference[i].imshow(pattern_spectra_mean_difference[i], cmap = "RdBu") #, norm = SymLogNorm(linthresh = 0.1, vmin = pattern_spectra_mean_difference_min, vmax = pattern_spectra_mean_difference_max, base = 10))
        im.set_clim(pattern_spectra_mean_difference_min, pattern_spectra_mean_difference_max)
        ax_difference[i].set_xticks([])
        ax_difference[i].set_yticks([])
        divider = make_axes_locatable(ax_difference[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar_difference = fig_difference.colorbar(im, cax=cax, orientation="vertical")
        cbar_difference.set_label(label = "log$_{10}$(flux)")
        ax_difference[i].annotate('', xy=(0, -0.1), xycoords='axes fraction', xytext=(1, -0.1), arrowprops=dict(arrowstyle="<-", color='black'))
        ax_difference[i].annotate('', xy=(-0.1, 1), xycoords='axes fraction', xytext=(-0.1, 0), arrowprops=dict(arrowstyle="<-", color='black'))
        ax_difference[i].set_xlabel(f"a {attributes[0]}", labelpad = 10)
        ax_difference[i].set_ylabel(f"a {attributes[1]}", labelpad = 10)

    fig_mean.tight_layout()
    fig_mean.savefig(path + "pattern_spectra_mean.png", dpi = 250)
    fig_difference.tight_layout()
    fig_difference.savefig(path + "pattern_spectra_mean_difference.png", dpi = 250)
    plt.close()

def PlotPatternSpectraMedian(number_energy_ranges, pattern_spectra_median, pattern_spectra_median_min, pattern_spectra_median_max, pattern_spectra_median_difference, pattern_spectra_median_difference_min, pattern_spectra_median_difference_max, bins, particle_type, cmap, attributes, path):
    print("Plot median pattern spectra...")
    fig_median, ax_median = plt.subplots(int(math.ceil(np.sqrt(number_energy_ranges))), int(math.ceil(np.sqrt(number_energy_ranges))))
    ax_median = ax_median.ravel()
    fig_difference, ax_difference = plt.subplots(int(math.ceil(np.sqrt(number_energy_ranges))), int(math.ceil(np.sqrt(number_energy_ranges))))
    ax_difference = ax_difference.ravel()

    if abs(pattern_spectra_median_difference_min) > abs(pattern_spectra_median_difference_max):
        pattern_spectra_median_difference_max = abs(pattern_spectra_median_difference_min)
    if abs(pattern_spectra_median_difference_min) < abs(pattern_spectra_median_difference_max):
        pattern_spectra_median_difference_min = - pattern_spectra_median_difference_max 


    for i in range(number_energy_ranges):
        # plot pattern spectra sum
        fig_median.suptitle(f"pattern spectra median - {particle_type}")   
        ax_median[i].set_title(f"{np.round(bins[i], 1)} - {np.round(bins[i+1], 1)} TeV", fontdict = {"fontsize" : 10})
        im = ax_median[i].imshow(pattern_spectra_median[i], cmap = cmap) #, norm = SymLogNorm(linthresh = 0.1, base = 10))
        # im = ax_median[i].imshow(pattern_spectra_median[i], cmap = cmap)
        im.set_clim(pattern_spectra_median_min, pattern_spectra_median_max)
        ax_median[i].set_xticks([])
        ax_median[i].set_yticks([])
        divider = make_axes_locatable(ax_median[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar_median = fig_median.colorbar(im, cax = cax, orientation = "vertical")
        cbar_median.set_label(label = "log$_{10}$(flux)")
        ax_median[i].annotate('', xy=(0, -0.1), xycoords='axes fraction', xytext=(1, -0.1), arrowprops=dict(arrowstyle="<-", color='black'))
        ax_median[i].annotate('', xy=(-0.1, 1), xycoords='axes fraction', xytext=(-0.1, 0), arrowprops=dict(arrowstyle="<-", color='black'))
        ax_median[i].set_xlabel(f"a {attributes[0]}", labelpad = 10)
        ax_median[i].set_ylabel(f"a {attributes[1]}", labelpad = 10)

        # plot pattern spectra difference
        fig_difference.suptitle(f"pattern spectra median difference - {particle_type}")   
        ax_difference[i].set_title(f"{np.round(bins[i], 1)} - {np.round(bins[i+1], 1)} TeV", fontdict = {"fontsize" : 10})
        im = ax_difference[i].imshow(pattern_spectra_median_difference[i], cmap = "RdBu") #, norm = SymLogNorm(linthresh = 0.1, vmin = pattern_spectra_median_difference_min, vmax = pattern_spectra_median_difference_max, base = 10))
        # im = ax_difference[i].imshow(pattern_spectra_median_difference[i], cmap = "RdBu")
        im.set_clim(pattern_spectra_median_difference_min, pattern_spectra_median_difference_max)
        ax_difference[i].set_xticks([])
        ax_difference[i].set_yticks([])
        divider = make_axes_locatable(ax_difference[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar_difference = fig_difference.colorbar(im, cax=cax, orientation="vertical")
        cbar_difference.set_label(label = "log$_{10}$(flux)")
        ax_difference[i].annotate('', xy=(0, -0.1), xycoords='axes fraction', xytext=(1, -0.1), arrowprops=dict(arrowstyle="<-", color='black'))
        ax_difference[i].annotate('', xy=(-0.1, 1), xycoords='axes fraction', xytext=(-0.1, 0), arrowprops=dict(arrowstyle="<-", color='black'))
        ax_difference[i].set_xlabel(f"a {attributes[0]}", labelpad = 10)
        ax_difference[i].set_ylabel(f"a {attributes[1]}", labelpad = 10)

    fig_median.tight_layout()
    fig_median.savefig(path + "pattern_spectra_median.png", dpi = 250)
    fig_difference.tight_layout()
    fig_difference.savefig(path + "pattern_spectra_median_difference.png", dpi = 250)
    plt.close()

def PlotPatternSpectraVariance(number_energy_ranges, pattern_spectra_variance, pattern_spectra_variance_min, pattern_spectra_variance_max, pattern_spectra_variance_difference, pattern_spectra_variance_difference_min, pattern_spectra_variance_difference_max, bins, particle_type, cmap, attributes, path):
    print("Plot variance of pattern spectra ...")
    fig_variance, ax_variance = plt.subplots(int(math.ceil(np.sqrt(number_energy_ranges))), int(math.ceil(np.sqrt(number_energy_ranges))))
    ax_variance = ax_variance.ravel()
    fig_difference, ax_difference = plt.subplots(int(math.ceil(np.sqrt(number_energy_ranges))), int(math.ceil(np.sqrt(number_energy_ranges))))
    ax_difference = ax_difference.ravel()

    if abs(pattern_spectra_variance_difference_min) > abs(pattern_spectra_variance_difference_max):
        pattern_spectra_variance_difference_max = abs(pattern_spectra_variance_difference_min)
    if abs(pattern_spectra_variance_difference_min) < abs(pattern_spectra_variance_difference_max):
        pattern_spectra_variance_difference_min = - pattern_spectra_variance_difference_max 


    for i in range(number_energy_ranges):
        # plot pattern spectra sum
        fig_variance.suptitle(f"pattern spectra variance - {particle_type}")   
        ax_variance[i].set_title(f"{np.round(bins[i], 1)} - {np.round(bins[i+1], 1)} TeV", fontdict = {"fontsize" : 10})
        im = ax_variance[i].imshow(pattern_spectra_variance[i], cmap = cmap) #, norm = SymLogNorm(linthresh = 0.1, base = 10))
        # im = ax_variance[i].imshow(pattern_spectra_variance[i], cmap = cmap)
        im.set_clim(pattern_spectra_variance_min, pattern_spectra_variance_max)
        ax_variance[i].set_xticks([])
        ax_variance[i].set_yticks([])
        divider = make_axes_locatable(ax_variance[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar_variance = fig_variance.colorbar(im, cax = cax, orientation = "vertical")
        cbar_variance.set_label(label = "log$_{10}$(flux)")
        ax_variance[i].annotate('', xy=(0, -0.1), xycoords='axes fraction', xytext=(1, -0.1), arrowprops=dict(arrowstyle="<-", color='black'))
        ax_variance[i].annotate('', xy=(-0.1, 1), xycoords='axes fraction', xytext=(-0.1, 0), arrowprops=dict(arrowstyle="<-", color='black'))
        ax_variance[i].set_xlabel(f"a {attributes[0]}", labelpad = 10)
        ax_variance[i].set_ylabel(f"a {attributes[1]}", labelpad = 10)

        # plot pattern spectra difference
        fig_difference.suptitle(f"pattern spectra variance difference - {particle_type}")   
        ax_difference[i].set_title(f"{np.round(bins[i], 1)} - {np.round(bins[i+1], 1)} TeV", fontdict = {"fontsize" : 10})
        im = ax_difference[i].imshow(pattern_spectra_variance_difference[i], cmap = "RdBu") #, norm = SymLogNorm(linthresh = 0.1, vmin = pattern_spectra_variance_difference_min, vmax = pattern_spectra_variance_difference_max, base = 10))
        # im = ax_difference[i].imshow(pattern_spectra_variance_difference[i], cmap = "RdBu")
        im.set_clim(pattern_spectra_variance_difference_min, pattern_spectra_variance_difference_max)
        ax_difference[i].set_xticks([])
        ax_difference[i].set_yticks([])
        divider = make_axes_locatable(ax_difference[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar_difference = fig_difference.colorbar(im, cax=cax, orientation="vertical")
        cbar_difference.set_label(label = "log$_{10}$(flux)")
        ax_difference[i].annotate('', xy=(0, -0.1), xycoords='axes fraction', xytext=(1, -0.1), arrowprops=dict(arrowstyle="<-", color='black'))
        ax_difference[i].annotate('', xy=(-0.1, 1), xycoords='axes fraction', xytext=(-0.1, 0), arrowprops=dict(arrowstyle="<-", color='black'))
        ax_difference[i].set_xlabel(f"a {attributes[0]}", labelpad = 10)
        ax_difference[i].set_ylabel(f"a {attributes[1]}", labelpad = 10)

    fig_variance.tight_layout()
    fig_variance.savefig(path + "pattern_spectra_variance.png", dpi = 250)
    fig_difference.tight_layout()
    fig_difference.savefig(path + "pattern_spectra_variance_difference.png", dpi = 250)
    plt.close()


def PlotPatternSpectraMeanComparison(number_energy_ranges, pattern_spectra_mean_gamma_proton, bins, particle_type, attributes, path):
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
        im = ax[i].imshow(pattern_spectra_mean_gamma_proton_difference[i], cmap = "RdBu") #, norm = SymLogNorm(linthresh = 0.01, base = 10))
        im.set_clim(pattern_spectra_mean_difference_min, pattern_spectra_mean_difference_max)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        divider = make_axes_locatable(ax[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax, orientation="vertical")
        cbar.set_label(label = "log$_{10}$(flux)")
        ax[i].annotate('', xy=(0, -0.1), xycoords='axes fraction', xytext=(1, -0.1), arrowprops=dict(arrowstyle="<-", color='black'))
        ax[i].annotate('', xy=(-0.1, 1), xycoords='axes fraction', xytext=(-0.1, 0), arrowprops=dict(arrowstyle="<-", color='black'))
        ax[i].set_xlabel(f"a {attributes[0]}", labelpad = 10)
        ax[i].set_ylabel(f"a {attributes[1]}", labelpad = 10)

    fig.tight_layout()
    fig.savefig(path + f"pattern_spectra_{particle_type[0]}_{particle_type[1]}_mean_comparison.png", dpi = 250)
    plt.close()

def PlotPatternSpectraMeanComparisonTotal(pattern_spectra_total_median_gamma_proton, particle_type, attributes, path):
    pattern_spectra_total_mean_gamma_proton_difference = pattern_spectra_total_median_gamma_proton[0] - pattern_spectra_total_median_gamma_proton[1]
    pattern_spectra_sum_total_difference_min = np.min(pattern_spectra_total_mean_gamma_proton_difference)
    pattern_spectra_sum_total_difference_max = np.max(pattern_spectra_total_mean_gamma_proton_difference)

    if abs(pattern_spectra_sum_total_difference_min) > abs(pattern_spectra_sum_total_difference_max):
        pattern_spectra_sum_total_difference_max = abs(pattern_spectra_sum_total_difference_min)
    if abs(pattern_spectra_sum_total_difference_min) < abs(pattern_spectra_sum_total_difference_max):
        pattern_spectra_sum_total_difference_min = - pattern_spectra_sum_total_difference_max 


    plt.figure()
    plt.title(f"pattern spectra total mean difference - {particle_type[0]} - {particle_type[1]}", fontsize = 10)
    im = plt.imshow(pattern_spectra_total_mean_gamma_proton_difference, cmap = "RdBu") #, norm = SymLogNorm(linthresh = 0.04, base = 10))
    im.set_clim(pattern_spectra_sum_total_difference_min, pattern_spectra_sum_total_difference_max)
    plt.xlabel(f"attribute {attributes[0]}", fontsize = 18)
    plt.ylabel(f"attribute {attributes[1]}", fontsize = 18)
    plt.xticks([])
    plt.yticks([])
    cb = plt.colorbar()
    cb.set_label(label = "log$_{10}$(flux)", size = 18)
    cb.ax.tick_params(labelsize = 18) 
    plt.tight_layout()
    plt.savefig(path + f"pattern_spectra_{particle_type[0]}_{particle_type[1]}_mean_comparison_total.png", dpi = 250)
    plt.close()

def PlotPatternSpectraMedianComparison(number_energy_ranges, pattern_spectra_median_gamma_proton, bins, particle_type, attributes, path):
    fig, ax = plt.subplots(int(math.ceil(np.sqrt(number_energy_ranges))), int(math.ceil(np.sqrt(number_energy_ranges))))
    ax = ax.ravel()
    pattern_spectra_median_gamma_proton_difference = pattern_spectra_median_gamma_proton[0] - pattern_spectra_median_gamma_proton[1]
    pattern_spectra_median_difference_min = np.min(pattern_spectra_median_gamma_proton_difference)
    pattern_spectra_median_difference_max = np.max(pattern_spectra_median_gamma_proton_difference)
    if abs(pattern_spectra_median_difference_min) > abs(pattern_spectra_median_difference_max):
        pattern_spectra_median_difference_max = abs(pattern_spectra_median_difference_min)
    if abs(pattern_spectra_median_difference_min) < abs(pattern_spectra_median_difference_max):
        pattern_spectra_median_difference_min = - pattern_spectra_median_difference_max 


    for i in range(number_energy_ranges):

        # plot pattern spectra difference
        fig.suptitle(f"pattern spectra median difference - {particle_type[0]} - {particle_type[1]}")   
        ax[i].set_title(f"{np.round(bins[i], 1)} - {np.round(bins[i+1], 1)} TeV", fontdict = {"fontsize" : 10})
        im = ax[i].imshow(pattern_spectra_median_gamma_proton_difference[i], cmap = "RdBu") #, norm = SymLogNorm(linthresh = 0.01, base = 10))
        im.set_clim(pattern_spectra_median_difference_min, pattern_spectra_median_difference_max)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        divider = make_axes_locatable(ax[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax, orientation="vertical")
        cbar.set_label(label = "log$_{10}$(flux)")
        ax[i].annotate('', xy=(0, -0.1), xycoords='axes fraction', xytext=(1, -0.1), arrowprops=dict(arrowstyle="<-", color='black'))
        ax[i].annotate('', xy=(-0.1, 1), xycoords='axes fraction', xytext=(-0.1, 0), arrowprops=dict(arrowstyle="<-", color='black'))
        ax[i].set_xlabel(f"a {attributes[0]}", labelpad = 10)
        ax[i].set_ylabel(f"a {attributes[1]}", labelpad = 10)

    fig.tight_layout()
    fig.savefig(path + f"pattern_spectra_{particle_type[0]}_{particle_type[1]}_median_comparison.png", dpi = 250)
    plt.close()

def PlotPatternSpectraMedianComparisonTotal(pattern_spectra_total_median_gamma_proton, particle_type, attributes, path):
    pattern_spectra_total_median_gamma_proton_difference = pattern_spectra_total_median_gamma_proton[0] - pattern_spectra_total_median_gamma_proton[1]
    pattern_spectra_sum_total_difference_min = np.min(pattern_spectra_total_median_gamma_proton_difference)
    pattern_spectra_sum_total_difference_max = np.max(pattern_spectra_total_median_gamma_proton_difference)

    if abs(pattern_spectra_sum_total_difference_min) > abs(pattern_spectra_sum_total_difference_max):
        pattern_spectra_sum_total_difference_max = abs(pattern_spectra_sum_total_difference_min)
    if abs(pattern_spectra_sum_total_difference_min) < abs(pattern_spectra_sum_total_difference_max):
        pattern_spectra_sum_total_difference_min = - pattern_spectra_sum_total_difference_max 


    plt.figure()
    plt.title(f"pattern spectra total median difference - {particle_type[0]} - {particle_type[1]}", fontsize = 10)
    im = plt.imshow(pattern_spectra_total_median_gamma_proton_difference, cmap = "RdBu") #, norm = SymLogNorm(linthresh = 0.04, base = 10))
    im.set_clim(pattern_spectra_sum_total_difference_min, pattern_spectra_sum_total_difference_max)
    plt.xlabel(f"attribute {attributes[0]}", fontsize = 18)
    plt.ylabel(f"attribute {attributes[1]}", fontsize = 18)
    plt.xticks([])
    plt.yticks([])
    cb = plt.colorbar()
    cb.set_label(label = "log$_{10}$(flux)", size = 18)
    cb.ax.tick_params(labelsize = 18) 
    plt.tight_layout()
    plt.savefig(path + f"pattern_spectra_{particle_type[0]}_{particle_type[1]}_median_comparison_total.png", dpi = 250)
    plt.close()

def PlotPatternSpectraVarianceComparison(number_energy_ranges, pattern_spectra_variance_gamma_proton, bins, particle_type, attributes, path):
    fig, ax = plt.subplots(int(math.ceil(np.sqrt(number_energy_ranges))), int(math.ceil(np.sqrt(number_energy_ranges))))
    ax = ax.ravel()
    pattern_spectra_variance_gamma_proton_difference = pattern_spectra_variance_gamma_proton[0] - pattern_spectra_variance_gamma_proton[1]
    pattern_spectra_variance_difference_min = np.min(pattern_spectra_variance_gamma_proton_difference)
    pattern_spectra_variance_difference_max = np.max(pattern_spectra_variance_gamma_proton_difference)
    if abs(pattern_spectra_variance_difference_min) > abs(pattern_spectra_variance_difference_max):
        pattern_spectra_variance_difference_max = abs(pattern_spectra_variance_difference_min)
    if abs(pattern_spectra_variance_difference_min) < abs(pattern_spectra_variance_difference_max):
        pattern_spectra_variance_difference_min = - pattern_spectra_variance_difference_max 


    for i in range(number_energy_ranges):

        # plot pattern spectra difference
        fig.suptitle(f"pattern spectra variance difference - {particle_type[0]} - {particle_type[1]}")   
        ax[i].set_title(f"{np.round(bins[i], 1)} - {np.round(bins[i+1], 1)} TeV", fontdict = {"fontsize" : 10})
        im = ax[i].imshow(pattern_spectra_variance_gamma_proton_difference[i], cmap = "RdBu") #, norm = SymLogNorm(linthresh = 0.01, base = 10))
        im.set_clim(pattern_spectra_variance_difference_min, pattern_spectra_variance_difference_max)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        divider = make_axes_locatable(ax[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax, orientation="vertical")
        cbar.set_label(label = "log$_{10}$(flux)")
        ax[i].annotate('', xy=(0, -0.1), xycoords='axes fraction', xytext=(1, -0.1), arrowprops=dict(arrowstyle="<-", color='black'))
        ax[i].annotate('', xy=(-0.1, 1), xycoords='axes fraction', xytext=(-0.1, 0), arrowprops=dict(arrowstyle="<-", color='black'))
        ax[i].set_xlabel(f"a {attributes[0]}", labelpad = 10)
        ax[i].set_ylabel(f"a {attributes[1]}", labelpad = 10)

    fig.tight_layout()
    fig.savefig(path + f"pattern_spectra_{particle_type[0]}_{particle_type[1]}_variance_comparison.png", dpi = 250)
    plt.close()

def PlotPatternSpectraVarianceComparisonTotal(pattern_spectra_total_variance_gamma_proton, particle_type, attributes, path):
    pattern_spectra_total_variance_gamma_proton_difference = pattern_spectra_total_variance_gamma_proton[0] - pattern_spectra_total_variance_gamma_proton[1]
    pattern_spectra_sum_total_difference_min = np.min(pattern_spectra_total_variance_gamma_proton_difference)
    pattern_spectra_sum_total_difference_max = np.max(pattern_spectra_total_variance_gamma_proton_difference)

    if abs(pattern_spectra_sum_total_difference_min) > abs(pattern_spectra_sum_total_difference_max):
        pattern_spectra_sum_total_difference_max = abs(pattern_spectra_sum_total_difference_min)
    if abs(pattern_spectra_sum_total_difference_min) < abs(pattern_spectra_sum_total_difference_max):
        pattern_spectra_sum_total_difference_min = - pattern_spectra_sum_total_difference_max 


    plt.figure()
    plt.title(f"pattern spectra total variance difference - {particle_type[0]} - {particle_type[1]}", fontsize = 10)
    im = plt.imshow(pattern_spectra_total_variance_gamma_proton_difference, cmap = "RdBu") #, norm = SymLogNorm(linthresh = 0.04, base = 10))
    im.set_clim(pattern_spectra_sum_total_difference_min, pattern_spectra_sum_total_difference_max)
    plt.xlabel(f"attribute {attributes[0]}", fontsize = 18)
    plt.ylabel(f"attribute {attributes[1]}", fontsize = 18)
    plt.xticks([])
    plt.yticks([])
    cb = plt.colorbar()
    cb.set_label(label = "log$_{10}$(flux)", size = 18)
    cb.ax.tick_params(labelsize = 18) 
    plt.tight_layout()
    plt.savefig(path + f"pattern_spectra_{particle_type[0]}_{particle_type[1]}_variance_comparison_total.png", dpi = 250)
    plt.close()

def PlotPatternSpectraPixelDistribution(pattern_spectra, pattern_spectra_binned, number_energy_ranges, bins,particle_type, path):
    print(f"Extracting pixel distribution...")

    shape = np.shape(pattern_spectra)

    fig, ax = plt.subplots(shape[1], shape[2])
    fig.suptitle(f"Pixel distribution {particle_type} - full pattern spectrum")
    ax = ax.ravel()
    count = 0
    for i in range(shape[1]):
        for j in range(shape[2]):
            # create a histogram for each pattern spectrum pixel
            ax[count].hist(pattern_spectra[:,i,j], range = (np.min(pattern_spectra), np.max(pattern_spectra)))
            ax[count].set_xlim(np.min(pattern_spectra) - 1, np.max(pattern_spectra) + 1)
            ylim = ax[count].get_ylim()
            ax[count].vlines(np.mean(pattern_spectra[:,i,j]), ylim[0], ylim[1], color = "C1", linestyle = "--", linewidth = 0.5)
            ax[count].vlines(np.median(pattern_spectra[:,i,j]), ylim[0], ylim[1], color = "C2", linestyle = "--", linewidth = 0.5)
            if count < 379:
                ax[count].set_xticks([])
            ax[count].tick_params(axis = 'x', labelsize = 6)
            # ax[count].set_yscale("log")
            ax[count].set_yticks([])
            count += 1
    #plt.tight_layout()
    fig.text(0.5, 0.04, "log$_{10}$(flux)", ha="center")
    fig.text(0.04, 0.5, "Number of events (linear scale)", va="center", rotation="vertical")
    plt.savefig(path + f"pattern_spectra_PixelDistribution_full.png", dpi = 250)
    plt.close()

    fig, ax = plt.subplots(11, 8) # only show 'interesting' distributions
    fig.suptitle(f"Pixel distribution {particle_type} - snippet of pattern spectrum")
    ax = ax.ravel()
    count = 0
    for i in range(shape[1]):
        for j in range(shape[2]):
            if j >= 5 and j <= 12 and i >= 0 and i <= 10: #i = y-axis, j = x-axis
                # create a histogram for each pattern spectrum pixel
                ax[count].hist(pattern_spectra[:,i,j], range = (np.min(pattern_spectra), np.max(pattern_spectra)))
                ax[count].set_xlim(np.min(pattern_spectra) - 1, np.max(pattern_spectra) + 1)
                ylim = ax[count].get_ylim()
                ax[count].vlines(np.mean(pattern_spectra[:,i,j]), ylim[0], ylim[1], color = "C1", linestyle = "--", linewidth = 0.8, label = "mean")
                ax[count].vlines(np.median(pattern_spectra[:,i,j]), ylim[0], ylim[1], color = "C2", linestyle = "--", linewidth = 0.8, label = "median")
                if count < 80: # 11 * 8 - 8 = 79
                    ax[count].set_xticks([])
                ax[count].tick_params(axis = 'x', labelsize = 8)
                # ax[count].set_yscale("log")
                ax[count].set_yticks([])
                count += 1
    #plt.tight_layout()
    plt.legend(fontsize = 4)
    fig.text(0.5, 0.04, "log$_{10}$(flux)", ha = "center")
    fig.text(0.04, 0.5, "Number of events (linear scale)", va = "center", rotation = "vertical")
    plt.savefig(path + f"pattern_spectra_PixelDistribution_selection.png", dpi = 250)
    plt.close()

    # plot the pixel distrubtion for the different energy ranges
    for k in range(number_energy_ranges):
        fig, ax = plt.subplots(11, 8) # only show 'interesting' distributions
        fig.suptitle(f"Pixel distribution {particle_type} - snippet of pattern spectrum ({np.round(bins[k], 1)} - {np.round(bins[k+1], 1)} TeV)")
        ax = ax.ravel()
        count = 0
        pattern_spectra_binned_k = pattern_spectra_binned[k]
        for i in range(shape[1]):
            for j in range(shape[2]):
                if j >= 5 and j <= 12 and i >= 0 and i <= 10: #i = y-axis, j = x-axis
                    # create a histogram for each pattern spectrum pixel
                    ax[count].hist(pattern_spectra_binned_k[:,i,j], range = (np.min(pattern_spectra_binned_k), np.max(pattern_spectra_binned_k)))
                    ax[count].set_xlim(np.min(pattern_spectra_binned_k) - 1, np.max(pattern_spectra_binned_k) + 1)
                    ylim = ax[count].get_ylim()
                    ax[count].vlines(np.mean(pattern_spectra_binned_k[:,i,j]), ylim[0], ylim[1], color = "C1", linestyle = "--", linewidth = 0.8, label = "mean")
                    ax[count].vlines(np.median(pattern_spectra_binned_k[:,i,j]), ylim[0], ylim[1], color = "C2", linestyle = "--", linewidth = 0.8, label = "median")
                    if count < 80: # 11 * 8 - 8 = 79
                        ax[count].set_xticks([])
                    ax[count].tick_params(axis = 'x', labelsize = 8)
                    # ax[count].set_yscale("log")
                    ax[count].set_yticks([])
                    count += 1
        #plt.tight_layout()
        plt.legend(fontsize = 4)
        fig.text(0.5, 0.04, "log$_{10}$(flux)", ha = "center")
        fig.text(0.04, 0.5, "Number of events (linear scale)", va = "center", rotation = "vertical")
        plt.savefig(path + f"pattern_spectra_PixelDistribution_selection_{np.round(bins[k], 1)}_{np.round(bins[k+1], 1)}TeV.png", dpi = 250)
        plt.close()


def PlotPatternSpectraTotal(pattern_spectra_total_mean, pattern_spectra_total_median, pattern_spectra_total_variance, particle_type, cmap, attributes, path):

    plt.figure()
    plt.title(f"pattern spectra total mean - {particle_type}", fontsize = 10)
    im = plt.imshow(pattern_spectra_total_mean, cmap = cmap) #, norm = SymLogNorm(linthresh = 0.04, base = 10))
    plt.xlabel(f"attribute {attributes[0]}", fontsize = 18)
    plt.ylabel(f"attribute {attributes[1]}", fontsize = 18)
    plt.xticks([])
    plt.yticks([])
    cb = plt.colorbar()
    cb.set_label(label = "log$_{10}$(flux)", size = 18)
    cb.ax.tick_params(labelsize = 18) 
    plt.tight_layout()
    plt.savefig(path + f"pattern_spectra_mean_total.png", dpi = 250)
    plt.close()

    plt.figure()
    plt.title(f"pattern spectra total median - {particle_type}", fontsize = 10)
    im = plt.imshow(pattern_spectra_total_median, cmap = cmap) #, norm = SymLogNorm(linthresh = 0.04, base = 10))
    plt.xlabel(f"attribute {attributes[0]}", fontsize = 18)
    plt.ylabel(f"attribute {attributes[1]}", fontsize = 18)
    plt.xticks([])
    plt.yticks([])
    cb = plt.colorbar()
    cb.set_label(label = "log$_{10}$(flux)", size = 18)
    cb.ax.tick_params(labelsize = 18) 
    plt.tight_layout()
    plt.savefig(path + f"pattern_spectra_median_total.png", dpi = 250)
    plt.close()

    plt.figure()
    plt.title(f"pattern spectra total variance - {particle_type}", fontsize = 10)
    im = plt.imshow(pattern_spectra_total_variance, cmap = cmap) #, norm = SymLogNorm(linthresh = 0.04, base = 10))
    plt.xlabel(f"attribute {attributes[0]}", fontsize = 18)
    plt.ylabel(f"attribute {attributes[1]}", fontsize = 18)
    plt.xticks([])
    plt.yticks([])
    cb = plt.colorbar()
    cb.set_label(label = "log$_{10}$(flux)", size = 18)
    cb.ax.tick_params(labelsize = 18) 
    plt.tight_layout()
    plt.savefig(path + f"pattern_spectra_variance_total.png", dpi = 250)
    plt.close()


    pattern_spectra_total_mean_median = (pattern_spectra_total_mean - pattern_spectra_total_median) / pattern_spectra_total_mean
    pattern_spectra_total_mean_median = np.nan_to_num(pattern_spectra_total_mean_median, nan = 0.0) 
    pattern_spectra_total_mean_median_min = np.min(pattern_spectra_total_mean_median)
    pattern_spectra_total_mean_median_max = np.max(pattern_spectra_total_mean_median)
    if abs(pattern_spectra_total_mean_median_min) > abs(pattern_spectra_total_mean_median_max):
        pattern_spectra_total_mean_median_max = abs(pattern_spectra_total_mean_median_min)
    if abs(pattern_spectra_total_mean_median_min) < abs(pattern_spectra_total_mean_median_max):
        pattern_spectra_total_mean_median_min = - pattern_spectra_total_mean_median_max 

    plt.figure()
    plt.title(f"pattern spectra total (mean - median) / mean - {particle_type}", fontsize = 10)
    im = plt.imshow(pattern_spectra_total_mean_median, cmap = "RdBu") #, norm = SymLogNorm(linthresh = 0.04, base = 10))
    im.set_clim(pattern_spectra_total_mean_median_min, pattern_spectra_total_mean_median_max)
    plt.xlabel(f"attribute {attributes[0]}", fontsize = 18)
    plt.ylabel(f"attribute {attributes[1]}", fontsize = 18)
    plt.xticks([])
    plt.yticks([])
    cb = plt.colorbar()
    cb.set_label(label = "log$_{10}$(flux)", size = 18)
    cb.ax.tick_params(labelsize = 18) 
    plt.tight_layout()
    plt.savefig(path + f"pattern_spectra_meanVSmedian_total.png", dpi = 250)
    plt.close()