import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math

cmap_mean_pattern_spectra = LinearSegmentedColormap.from_list("", ["#f6f6f6", "#00C6B4", "#143d59"])
cmap_mean_ps_diff = LinearSegmentedColormap.from_list("", ["#93000F", "#f6f6f6", "#143d59"])

def PlotEnergyDistribution(table, energy_range, path):
    table_gamma = np.asarray(table[table["particle"] == 1].reset_index(drop = True)["true_energy"])
    table_proton = np.asarray(table[table["particle"] == 0].reset_index(drop = True)["true_energy"])

    plt.figure()
    plt.hist(table_gamma, bins = np.logspace(np.log10(energy_range[0] * 1e3), np.log10(energy_range[1] * 1e3), 51), alpha = 0.5, label = "gamma")
    plt.hist(table_proton, bins = np.logspace(np.log10(energy_range[0] * 1e3), np.log10(energy_range[1] * 1e3), 51), alpha = 0.5, label = "proton")
    plt.xlabel("True energy [GeV]")
    plt.ylabel("Number of events")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.savefig(path, dpi = 250)
    plt.close()

def cstm_PuBu(x):
    return plt.cm.PuBu((np.clip(x,2,10)-2)/8.)

def cstm_RdBu(x):
    return plt.cm.RdBu((np.clip(x,2,10)-2)/8.)

def ExtractPatternSpectraMean(number_energy_ranges, size, ps_binned):
    print("Extracting mean pattern spectra...")
    ps_mean = np.zeros(shape = (number_energy_ranges, size[0], size[1]))
    for i in range(number_energy_ranges):
        ps_mean[i] = np.mean(ps_binned[i], axis = 0)

    return ps_mean

def ExtractPatternSpectraMedian(number_energy_ranges, size, ps_binned):
    print("Extracting median pattern spectra...")
    ps_median = np.zeros(shape = (number_energy_ranges, size[0], size[1]))
    for i in range(number_energy_ranges):
        ps_median[i] = np.median(ps_binned[i], axis = 0)

    return ps_median

def ExtractPatternSpectraVariance(number_energy_ranges, size, ps_binned):
    print("Extracting variance of pattern spectra...")
    ps_variance = np.zeros(shape = (number_energy_ranges, size[0], size[1]))
    for i in range(number_energy_ranges):
        ps_variance[i] = np.var(ps_binned[i], axis = 0)

    return ps_variance

def ExtractPatternSpectraDifference(number_energy_ranges, size, pattern_spectra):
    ps_diff = np.zeros(shape = (number_energy_ranges, size[0], size[1]))
    for i in range(number_energy_ranges):
        # calculate normed patter spectra sum minus pattern spectra of first energy range
        ps_diff[i] = (pattern_spectra[i] - pattern_spectra[0]) 

    return ps_diff

def ExtractPatternSpectraMinMax(number_energy_ranges, pattern_spectra):
    for i in range(number_energy_ranges):
        if i == 1:
            ps_min = np.min(pattern_spectra[i])
            ps_max = np.max(pattern_spectra[i])
        elif i > 1:
            if np.min(pattern_spectra[i]) < ps_min:
                ps_min = np.min(pattern_spectra[i])
            if np.max(pattern_spectra[i]) > ps_max:
                ps_max = np.max(pattern_spectra[i])
    
    return(ps_min, ps_max)

def PlotPatternSpectraMean(number_energy_ranges, ps_mean, ps_mean_min, ps_mean_max, ps_mean_diff, ps_mean_diff_min, ps_mean_diff_max, bins, particle_type, attributes, path):
    print("Plotting mean pattern spectra...")
    fig_mean, ax_mean = plt.subplots(int(math.ceil(np.sqrt(number_energy_ranges))), int(math.ceil(np.sqrt(number_energy_ranges))))
    ax_mean = ax_mean.ravel()
    fig_diff, ax_diff = plt.subplots(int(math.ceil(np.sqrt(number_energy_ranges))), int(math.ceil(np.sqrt(number_energy_ranges))))
    ax_diff = ax_diff.ravel()

    if abs(ps_mean_diff_min) > abs(ps_mean_diff_max):
        ps_mean_diff_max = abs(ps_mean_diff_min)
    if abs(ps_mean_diff_min) < abs(ps_mean_diff_max):
        ps_mean_diff_min = - ps_mean_diff_max 


    for i in range(number_energy_ranges):
        # plot pattern spectra sum
        fig_mean.suptitle(f"pattern spectra mean - {particle_type}")   
        ax_mean[i].set_title(f"{np.round(bins[i], 1)} - {np.round(bins[i+1], 1)} TeV", fontdict = {"fontsize" : 10})
        im = ax_mean[i].imshow(ps_mean[i], cmap = cmap_mean_pattern_spectra)
        im.set_clim(ps_mean_min, ps_mean_max)
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
        fig_diff.suptitle(f"pattern spectra mean difference - {particle_type}")   
        ax_diff[i].set_title(f"{np.round(bins[i], 1)} - {np.round(bins[i+1], 1)} TeV", fontdict = {"fontsize" : 10})
        im = ax_diff[i].imshow(ps_mean_diff[i], cmap = cmap_mean_ps_diff) 
        im.set_clim(ps_mean_diff_min, ps_mean_diff_max)
        ax_diff[i].set_xticks([])
        ax_diff[i].set_yticks([])
        divider = make_axes_locatable(ax_diff[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar_diff = fig_diff.colorbar(im, cax=cax, orientation="vertical")
        cbar_diff.set_label(label = "log$_{10}$(flux)")
        ax_diff[i].annotate('', xy=(0, -0.1), xycoords='axes fraction', xytext=(1, -0.1), arrowprops=dict(arrowstyle="<-", color='black'))
        ax_diff[i].annotate('', xy=(-0.1, 1), xycoords='axes fraction', xytext=(-0.1, 0), arrowprops=dict(arrowstyle="<-", color='black'))
        ax_diff[i].set_xlabel(f"a {attributes[0]}", labelpad = 10)
        ax_diff[i].set_ylabel(f"a {attributes[1]}", labelpad = 10)

    fig_mean.tight_layout()
    fig_mean.savefig(path + "pattern_spectra_mean.pdf", dpi = 250)
    fig_diff.tight_layout()
    fig_diff.savefig(path + "pattern_spectra_mean_diff.pdf", dpi = 250)
    plt.close()

def PlotPatternSpectraMedian(number_energy_ranges, ps_median, ps_median_min, ps_median_max, ps_median_diff, ps_median_diff_min, ps_median_diff_max, bins, particle_type, attributes, path):
    print("Plotting median pattern spectra...")
    fig_median, ax_median = plt.subplots(int(math.ceil(np.sqrt(number_energy_ranges))), int(math.ceil(np.sqrt(number_energy_ranges))))
    ax_median = ax_median.ravel()
    fig_diff, ax_diff = plt.subplots(int(math.ceil(np.sqrt(number_energy_ranges))), int(math.ceil(np.sqrt(number_energy_ranges))))
    ax_diff = ax_diff.ravel()

    if abs(ps_median_diff_min) > abs(ps_median_diff_max):
        ps_median_diff_max = abs(ps_median_diff_min)
    if abs(ps_median_diff_min) < abs(ps_median_diff_max):
        ps_median_diff_min = - ps_median_diff_max 

    for i in range(number_energy_ranges):
        # plot pattern spectra sum
        fig_median.suptitle(f"pattern spectra median - {particle_type}")   
        ax_median[i].set_title(f"{np.round(bins[i], 1)} - {np.round(bins[i+1], 1)} TeV", fontdict = {"fontsize" : 10})
        im = ax_median[i].imshow(ps_median[i], cmap = cmap_mean_pattern_spectra) 
        im.set_clim(ps_median_min, ps_median_max)
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
        fig_diff.suptitle(f"pattern spectra median difference - {particle_type}")   
        ax_diff[i].set_title(f"{np.round(bins[i], 1)} - {np.round(bins[i+1], 1)} TeV", fontdict = {"fontsize" : 10})
        im = ax_diff[i].imshow(ps_median_diff[i], cmap = cmap_mean_ps_diff) 
        im.set_clim(ps_median_diff_min, ps_median_diff_max)
        ax_diff[i].set_xticks([])
        ax_diff[i].set_yticks([])
        divider = make_axes_locatable(ax_diff[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar_diff = fig_diff.colorbar(im, cax=cax, orientation="vertical")
        cbar_diff.set_label(label = "log$_{10}$(flux)")
        ax_diff[i].annotate('', xy=(0, -0.1), xycoords='axes fraction', xytext=(1, -0.1), arrowprops=dict(arrowstyle="<-", color='black'))
        ax_diff[i].annotate('', xy=(-0.1, 1), xycoords='axes fraction', xytext=(-0.1, 0), arrowprops=dict(arrowstyle="<-", color='black'))
        ax_diff[i].set_xlabel(f"a {attributes[0]}", labelpad = 10)
        ax_diff[i].set_ylabel(f"a {attributes[1]}", labelpad = 10)

    fig_median.tight_layout()
    fig_median.savefig(path + "pattern_spectra_median.pdf", dpi = 250)
    fig_diff.tight_layout()
    fig_diff.savefig(path + "pattern_spectra_median_diff.pdf", dpi = 250)
    plt.close()

def PlotPatternSpectraVariance(number_energy_ranges, ps_variance, ps_variance_min, ps_variance_max, ps_variance_diff, ps_variance_diff_min, ps_variance_diff_max, bins, particle_type, attributes, path):
    print("Plotting variance of pattern spectra ...")
    fig_variance, ax_variance = plt.subplots(int(math.ceil(np.sqrt(number_energy_ranges))), int(math.ceil(np.sqrt(number_energy_ranges))))
    ax_variance = ax_variance.ravel()
    fig_diff, ax_diff = plt.subplots(int(math.ceil(np.sqrt(number_energy_ranges))), int(math.ceil(np.sqrt(number_energy_ranges))))
    ax_diff = ax_diff.ravel()

    if abs(ps_variance_diff_min) > abs(ps_variance_diff_max):
        ps_variance_diff_max = abs(ps_variance_diff_min)
    if abs(ps_variance_diff_min) < abs(ps_variance_diff_max):
        ps_variance_diff_min = - ps_variance_diff_max 


    for i in range(number_energy_ranges):
        # plot pattern spectra sum
        fig_variance.suptitle(f"pattern spectra variance - {particle_type}")   
        ax_variance[i].set_title(f"{np.round(bins[i], 1)} - {np.round(bins[i+1], 1)} TeV", fontdict = {"fontsize" : 10})
        im = ax_variance[i].imshow(ps_variance[i], cmap = cmap_mean_pattern_spectra)
        im.set_clim(ps_variance_min, ps_variance_max)
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
        fig_diff.suptitle(f"pattern spectra variance difference - {particle_type}")   
        ax_diff[i].set_title(f"{np.round(bins[i], 1)} - {np.round(bins[i+1], 1)} TeV", fontdict = {"fontsize" : 10})
        im = ax_diff[i].imshow(ps_variance_diff[i], cmap = cmap_mean_ps_diff)
        im.set_clim(ps_variance_diff_min, ps_variance_diff_max)
        ax_diff[i].set_xticks([])
        ax_diff[i].set_yticks([])
        divider = make_axes_locatable(ax_diff[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar_diff = fig_diff.colorbar(im, cax=cax, orientation="vertical")
        cbar_diff.set_label(label = "log$_{10}$(flux)")
        ax_diff[i].annotate('', xy=(0, -0.1), xycoords='axes fraction', xytext=(1, -0.1), arrowprops=dict(arrowstyle="<-", color='black'))
        ax_diff[i].annotate('', xy=(-0.1, 1), xycoords='axes fraction', xytext=(-0.1, 0), arrowprops=dict(arrowstyle="<-", color='black'))
        ax_diff[i].set_xlabel(f"a {attributes[0]}", labelpad = 10)
        ax_diff[i].set_ylabel(f"a {attributes[1]}", labelpad = 10)

    fig_variance.tight_layout()
    fig_variance.savefig(path + "pattern_spectra_variance.pdf", dpi = 250)
    fig_diff.tight_layout()
    fig_diff.savefig(path + "pattern_spectra_variance_diff.pdf", dpi = 250)
    plt.close()


def PlotPatternSpectraMeanComparison(number_energy_ranges, ps_mean_gamma_proton, bins, particle_type, attributes, path):
    fig, ax = plt.subplots(int(math.ceil(np.sqrt(number_energy_ranges))), int(math.ceil(np.sqrt(number_energy_ranges))))
    ax = ax.ravel()
    ps_mean_gamma_proton_diff = ps_mean_gamma_proton[0] - ps_mean_gamma_proton[1]
    ps_mean_diff_min = np.min(ps_mean_gamma_proton_diff)
    ps_mean_diff_max = np.max(ps_mean_gamma_proton_diff)
    if abs(ps_mean_diff_min) > abs(ps_mean_diff_max):
        ps_mean_diff_max = abs(ps_mean_diff_min)
    if abs(ps_mean_diff_min) < abs(ps_mean_diff_max):
        ps_mean_diff_min = - ps_mean_diff_max 


    for i in range(number_energy_ranges):
        # plot pattern spectra difference
        fig.suptitle(f"pattern spectra mean difference - {particle_type[0]} - {particle_type[1]}")   
        ax[i].set_title(f"{np.round(bins[i], 1)} - {np.round(bins[i+1], 1)} TeV", fontdict = {"fontsize" : 10})
        im = ax[i].imshow(ps_mean_gamma_proton_diff[i], cmap = cmap_mean_ps_diff) 
        im.set_clim(ps_mean_diff_min, ps_mean_diff_max)
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
    fig.savefig(path + f"pattern_spectra_{particle_type[0]}_{particle_type[1]}_mean_comparison.pdf", dpi = 250)
    plt.close()

def PlotPatternSpectraMeanComparisonTotal(ps_total_median_gamma_proton, particle_type, attributes, path):
    ps_total_mean_gamma_proton_diff = ps_total_median_gamma_proton[0] - ps_total_median_gamma_proton[1]
    ps_sum_total_diff_min = np.min(ps_total_mean_gamma_proton_diff)
    ps_sum_total_diff_max = np.max(ps_total_mean_gamma_proton_diff)

    if abs(ps_sum_total_diff_min) > abs(ps_sum_total_diff_max):
        ps_sum_total_diff_max = abs(ps_sum_total_diff_min)
    if abs(ps_sum_total_diff_min) < abs(ps_sum_total_diff_max):
        ps_sum_total_diff_min = - ps_sum_total_diff_max 


    plt.figure()
    plt.title(f"pattern spectra total mean difference - {particle_type[0]} - {particle_type[1]}", fontsize = 10)
    im = plt.imshow(ps_total_mean_gamma_proton_diff, cmap = cmap_mean_ps_diff)
    im.set_clim(ps_sum_total_diff_min, ps_sum_total_diff_max)
    plt.xlabel(f"attribute {attributes[0]}", fontsize = 18)
    plt.ylabel(f"attribute {attributes[1]}", fontsize = 18)
    plt.xticks([])
    plt.yticks([])
    cb = plt.colorbar()
    cb.set_label(label = "log$_{10}$(flux)", size = 18)
    cb.ax.tick_params(labelsize = 18) 
    plt.tight_layout()
    plt.savefig(path + f"pattern_spectra_{particle_type[0]}_{particle_type[1]}_mean_comparison_total.pdf", dpi = 250)
    plt.close()

def PlotPatternSpectraMedianComparison(number_energy_ranges, ps_median_gamma_proton, bins, particle_type, attributes, path):
    fig, ax = plt.subplots(int(math.ceil(np.sqrt(number_energy_ranges))), int(math.ceil(np.sqrt(number_energy_ranges))))
    ax = ax.ravel()
    ps_median_gamma_proton_diff = ps_median_gamma_proton[0] - ps_median_gamma_proton[1]
    ps_median_diff_min = np.min(ps_median_gamma_proton_diff)
    ps_median_diff_max = np.max(ps_median_gamma_proton_diff)
    if abs(ps_median_diff_min) > abs(ps_median_diff_max):
        ps_median_diff_max = abs(ps_median_diff_min)
    if abs(ps_median_diff_min) < abs(ps_median_diff_max):
        ps_median_diff_min = - ps_median_diff_max 

    for i in range(number_energy_ranges):
        # plot pattern spectra difference
        fig.suptitle(f"pattern spectra median difference - {particle_type[0]} - {particle_type[1]}")   
        ax[i].set_title(f"{np.round(bins[i], 1)} - {np.round(bins[i+1], 1)} TeV", fontdict = {"fontsize" : 10})
        im = ax[i].imshow(ps_median_gamma_proton_diff[i], cmap = cmap_mean_ps_diff, norm = SymLogNorm(linthresh = 0.01, base = 10))
        im.set_clim(ps_median_diff_min, ps_median_diff_max)
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
    fig.savefig(path + f"pattern_spectra_{particle_type[0]}_{particle_type[1]}_median_comparison.pdf", dpi = 250)
    plt.close()

def PlotPatternSpectraMedianComparisonTotal(ps_total_median_gamma_proton, particle_type, attributes, path):
    ps_total_median_gamma_proton_diff = ps_total_median_gamma_proton[0] - ps_total_median_gamma_proton[1]
    ps_sum_total_diff_min = np.min(ps_total_median_gamma_proton_diff)
    ps_sum_total_diff_max = np.max(ps_total_median_gamma_proton_diff)

    if abs(ps_sum_total_diff_min) > abs(ps_sum_total_diff_max):
        ps_sum_total_diff_max = abs(ps_sum_total_diff_min)
    if abs(ps_sum_total_diff_min) < abs(ps_sum_total_diff_max):
        ps_sum_total_diff_min = - ps_sum_total_diff_max 

    plt.figure()
    plt.title(f"pattern spectra total median difference - {particle_type[0]} - {particle_type[1]}", fontsize = 10)
    im = plt.imshow(ps_total_median_gamma_proton_diff, cmap = cmap_mean_ps_diff, norm = SymLogNorm(linthresh = 0.01, base = 10))
    im.set_clim(ps_sum_total_diff_min, ps_sum_total_diff_max)
    plt.xlabel(f"attribute {attributes[0]}", fontsize = 18)
    plt.ylabel(f"attribute {attributes[1]}", fontsize = 18)
    plt.xticks([])
    plt.yticks([])
    cb = plt.colorbar()
    cb.set_label(label = "log$_{10}$(flux)", size = 18)
    cb.ax.tick_params(labelsize = 18) 
    plt.tight_layout()
    plt.savefig(path + f"pattern_spectra_{particle_type[0]}_{particle_type[1]}_median_comparison_total.pdf", dpi = 250)
    plt.close()

def PlotPatternSpectraVarianceComparison(number_energy_ranges, ps_variance_gamma_proton, bins, particle_type, attributes, path):
    fig, ax = plt.subplots(int(math.ceil(np.sqrt(number_energy_ranges))), int(math.ceil(np.sqrt(number_energy_ranges))))
    ax = ax.ravel()
    ps_variance_gamma_proton_diff = ps_variance_gamma_proton[0] - ps_variance_gamma_proton[1]
    ps_variance_diff_min = np.min(ps_variance_gamma_proton_diff)
    ps_variance_diff_max = np.max(ps_variance_gamma_proton_diff)
    if abs(ps_variance_diff_min) > abs(ps_variance_diff_max):
        ps_variance_diff_max = abs(ps_variance_diff_min)
    if abs(ps_variance_diff_min) < abs(ps_variance_diff_max):
        ps_variance_diff_min = - ps_variance_diff_max 


    for i in range(number_energy_ranges):
        # plot pattern spectra difference
        fig.suptitle(f"pattern spectra variance difference - {particle_type[0]} - {particle_type[1]}")   
        ax[i].set_title(f"{np.round(bins[i], 1)} - {np.round(bins[i+1], 1)} TeV", fontdict = {"fontsize" : 10})
        im = ax[i].imshow(ps_variance_gamma_proton_diff[i], cmap = cmap_mean_ps_diff) 
        im.set_clim(ps_variance_diff_min, ps_variance_diff_max)
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
    fig.savefig(path + f"pattern_spectra_{particle_type[0]}_{particle_type[1]}_variance_comparison.pdf", dpi = 250)
    plt.close()

def PlotPatternSpectraVarianceComparisonTotal(ps_total_variance_gamma_proton, particle_type, attributes, path):
    ps_total_variance_gamma_proton_diff = ps_total_variance_gamma_proton[0] - ps_total_variance_gamma_proton[1]
    ps_sum_total_diff_min = np.min(ps_total_variance_gamma_proton_diff)
    ps_sum_total_diff_max = np.max(ps_total_variance_gamma_proton_diff)

    if abs(ps_sum_total_diff_min) > abs(ps_sum_total_diff_max):
        ps_sum_total_diff_max = abs(ps_sum_total_diff_min)
    if abs(ps_sum_total_diff_min) < abs(ps_sum_total_diff_max):
        ps_sum_total_diff_min = - ps_sum_total_diff_max 

    plt.figure()
    plt.title(f"pattern spectra total variance difference - {particle_type[0]} - {particle_type[1]}", fontsize = 10)
    im = plt.imshow(ps_total_variance_gamma_proton_diff, cmap = cmap_mean_ps_diff) 
    im.set_clim(ps_sum_total_diff_min, ps_sum_total_diff_max)
    plt.xlabel(f"attribute {attributes[0]}", fontsize = 18)
    plt.ylabel(f"attribute {attributes[1]}", fontsize = 18)
    plt.xticks([])
    plt.yticks([])
    cb = plt.colorbar()
    cb.set_label(label = "log$_{10}$(flux)", size = 18)
    cb.ax.tick_params(labelsize = 18) 
    plt.tight_layout()
    plt.savefig(path + f"pattern_spectra_{particle_type[0]}_{particle_type[1]}_variance_comparison_total.pdf", dpi = 250)
    plt.close()

def PlotPatternSpectraPixelDistribution(pattern_spectra, ps_binned, number_energy_ranges, bins,particle_type, path):
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
            ax[count].set_yticks([])
            count += 1
    fig.text(0.5, 0.04, "log$_{10}$(flux)", ha="center")
    fig.text(0.04, 0.5, "Number of events (linear scale)", va="center", rotation="vertical")
    plt.savefig(path + f"pattern_spectra_PixelDistribution_full.pdf", dpi = 250)
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
                ax[count].set_yticks([])
                count += 1
    plt.legend(fontsize = 4)
    fig.text(0.5, 0.04, "log$_{10}$(flux)", ha = "center")
    fig.text(0.04, 0.5, "Number of events (linear scale)", va = "center", rotation = "vertical")
    plt.savefig(path + f"pattern_spectra_PixelDistribution_selection.pdf", dpi = 250)
    plt.close()

    # plot the pixel distrubtion for the different energy ranges
    for k in range(number_energy_ranges):
        fig, ax = plt.subplots(11, 8) # only show 'interesting' distributions
        fig.suptitle(f"Pixel distribution {particle_type} - snippet of pattern spectrum ({np.round(bins[k], 1)} - {np.round(bins[k+1], 1)} TeV)")
        ax = ax.ravel()
        count = 0
        ps_binned_k = ps_binned[k]
        for i in range(shape[1]):
            for j in range(shape[2]):
                if j >= 5 and j <= 12 and i >= 0 and i <= 10: #i = y-axis, j = x-axis
                    # create a histogram for each pattern spectrum pixel
                    hist = ax[count].hist(ps_binned_k[:,i,j], range = (np.min(ps_binned_k), np.max(ps_binned_k)))
                    ax[count].set_xlim(np.min(ps_binned_k) - 1, np.max(ps_binned_k) + 1)
                    ylim = ax[count].get_ylim()
                    ax[count].vlines(np.mean(ps_binned_k[:,i,j]), ylim[0], ylim[1], color = "C1", linestyle = "--", linewidth = 0.8, label = "mean")
                    ax[count].vlines(np.median(ps_binned_k[:,i,j]), ylim[0], ylim[1], color = "C2", linestyle = "--", linewidth = 0.8, label = "median")
                    if count < 80: # 11 * 8 - 8 = 79
                        ax[count].set_xticks([])
                    ax[count].tick_params(axis = 'x', labelsize = 8)
                    ax[count].set_yticks([])
                    count += 1
        plt.legend(fontsize = 4)
        fig.text(0.5, 0.04, "log$_{10}$(flux)", ha = "center")
        fig.text(0.04, 0.5, "Number of events (linear scale)", va = "center", rotation = "vertical")
        plt.savefig(path + f"pattern_spectra_PixelDistribution_selection_{np.round(bins[k], 1)}_{np.round(bins[k+1], 1)}TeV.pdf", dpi = 250)
        plt.close()




def PlotPatternSpectraTotal(ps_total_mean, ps_total_median, ps_total_variance, particle_type, attributes, path):
    plt.figure()
    plt.title(f"pattern spectra total mean - {particle_type}", fontsize = 10)
    im = plt.imshow(ps_total_mean, cmap = cmap_mean_pattern_spectra) 
    plt.xlabel(f"attribute {attributes[0]}", fontsize = 18)
    plt.ylabel(f"attribute {attributes[1]}", fontsize = 18)
    plt.xticks([])
    plt.yticks([])
    cb = plt.colorbar()
    cb.set_label(label = "log$_{10}$(flux)", size = 18)
    cb.ax.tick_params(labelsize = 18) 
    plt.tight_layout()
    plt.savefig(path + f"pattern_spectra_mean_total.pdf", dpi = 250)
    plt.close()

    plt.figure()
    plt.title(f"pattern spectra total median - {particle_type}", fontsize = 10)
    im = plt.imshow(ps_total_median, cmap = cmap_mean_pattern_spectra) 
    plt.xlabel(f"attribute {attributes[0]}", fontsize = 18)
    plt.ylabel(f"attribute {attributes[1]}", fontsize = 18)
    plt.xticks([])
    plt.yticks([])
    cb = plt.colorbar()
    cb.set_label(label = "log$_{10}$(flux)", size = 18)
    cb.ax.tick_params(labelsize = 18) 
    plt.tight_layout()
    plt.savefig(path + f"pattern_spectra_median_total.pdf", dpi = 250)
    plt.close()

    plt.figure()
    plt.title(f"pattern spectra total variance - {particle_type}", fontsize = 10)
    im = plt.imshow(ps_total_variance, cmap = cmap_mean_pattern_spectra) 
    plt.xlabel(f"attribute {attributes[0]}", fontsize = 18)
    plt.ylabel(f"attribute {attributes[1]}", fontsize = 18)
    plt.xticks([])
    plt.yticks([])
    cb = plt.colorbar()
    cb.set_label(label = "log$_{10}$(flux)", size = 18)
    cb.ax.tick_params(labelsize = 18) 
    plt.tight_layout()
    plt.savefig(path + f"pattern_spectra_variance_total.pdf", dpi = 250)
    plt.close()


    ps_total_mean_median = (ps_total_mean - ps_total_median) / ps_total_mean
    ps_total_mean_median = np.nan_to_num(ps_total_mean_median, nan = 0.0) 
    ps_total_mean_median_min = np.min(ps_total_mean_median)
    ps_total_mean_median_max = np.max(ps_total_mean_median)
    if abs(ps_total_mean_median_min) > abs(ps_total_mean_median_max):
        ps_total_mean_median_max = abs(ps_total_mean_median_min)
    if abs(ps_total_mean_median_min) < abs(ps_total_mean_median_max):
        ps_total_mean_median_min = - ps_total_mean_median_max 

    plt.figure()
    plt.title(f"pattern spectra total (mean - median) / mean - {particle_type}", fontsize = 10)
    im = plt.imshow(ps_total_mean_median, cmap = cmap_mean_ps_diff) 
    im.set_clim(ps_total_mean_median_min, ps_total_mean_median_max)
    plt.xlabel(f"attribute {attributes[0]}", fontsize = 18)
    plt.ylabel(f"attribute {attributes[1]}", fontsize = 18)
    plt.xticks([])
    plt.yticks([])
    cb = plt.colorbar()
    cb.set_label(label = "log$_{10}$(flux)", size = 18)
    cb.ax.tick_params(labelsize = 18) 
    plt.tight_layout()
    plt.savefig(path + f"pattern_spectra_meanVSmedian_total.pdf", dpi = 250)
    plt.close()

def PlotPatternSpectrum(pattern_spectrum, attributes, path):
    plt.figure()
    plt.imshow(pattern_spectrum, cmap = "Greys_r")
    plt.annotate('', xy=(0, -0.05), xycoords='axes fraction', xytext=(1, -0.05), arrowprops=dict(arrowstyle="<-", color='black'))
    plt.annotate('', xy=(-0.05, 1), xycoords='axes fraction', xytext=(-0.05, 0), arrowprops=dict(arrowstyle="<-", color='black'))
    plt.xlabel(f"attribute {attributes[1]}", labelpad = 20, fontsize = 16)
    plt.ylabel(f"attribute {attributes[0]}", labelpad = 20, fontsize = 16)
    cbar = plt.colorbar()
    cbar.set_label(label = r"log$_{10}$($\Phi$)", fontsize = 16)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.tight_layout()
    plt.savefig(path)
    plt.close()