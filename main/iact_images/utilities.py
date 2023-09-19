import matplotlib.pyplot as plt
from ctapipe.visualization import CameraDisplay
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

def GetEventImage(image, camera_geometry, cmap = "Greys", show_frame = False, colorbar = False, clean_image = False, savefig = False):
    plt.figure()
    disp = CameraDisplay(camera_geometry, cmap = cmap, show_frame = show_frame)
    disp.image = image

    if colorbar == True:
        disp.add_colorbar()
    
    if clean_image == True:
        plt.title("")
        plt.axis("off")
        plt.tick_params(axis = "both", left = "off", top = "off", right = "off", bottom = "off", labelleft = "off", labeltop = "off", labelright = "off", labelbottom = "off")
        bbox_inches = "tight"
        pad_inches = 0.0
    else:
        bbox_inches = None
        pad_inches = 0.1

    if savefig != False:
        plt.savefig(savefig, bbox_inches = bbox_inches, pad_inches = pad_inches)
    
    plt.close()


def GetEventImageBasic(image, cmap = "Greys", show_frame = False, colorbar = False, clean_image = False, savefig = False):
    plt.figure()
    plt.imshow(image, cmap = cmap) 

    if colorbar == True:
        cbar = plt.colorbar()
        cbar.set_label(label = "Photoelectrons")
    
    if clean_image == True:
        plt.axis("off")
        plt.tick_params(axis = "both", left = "off", top = "off", right = "off", bottom = "off", labelleft = "off", labeltop = "off", labelright = "off", labelbottom = "off")
        bbox_inches = "tight"
        pad_inches = 0.0
    else:
        plt.xlabel("x")
        plt.ylabel("y")
        bbox_inches = None
        pad_inches = 0.1

    plt.tight_layout()
    if savefig != False:
        plt.savefig(savefig, bbox_inches = bbox_inches, pad_inches = pad_inches)
    
    plt.close()


def GetEventImageBasicSmall(image, cmap = "Greys", show_frame = False, colorbar = False, clean_image = False, savefig = False):
    dpi = 80
    height, width = np.array(image.shape, dtype=float) / dpi
    
    fig = plt.figure(figsize=(width, height), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')

    ax.imshow(image, interpolation='none', cmap = cmap)
    fig.savefig(savefig, dpi=dpi)

    plt.close("all")

def PlotImages(number_energy_ranges, size, bins, image_binned, path):
    image_sum = np.zeros(shape = (number_energy_ranges, size[0], size[1]))
    image_sum_normed = np.zeros(shape = (number_energy_ranges, size[0], size[1]))
    image_sum_difference = np.zeros(shape = (number_energy_ranges, size[0], size[1]))

    fig_normed, ax_normed = plt.subplots(int(np.sqrt(number_energy_ranges)), int(np.sqrt(number_energy_ranges)))
    ax_normed = ax_normed.ravel()
    fig_difference, ax_difference = plt.subplots(int(np.sqrt(number_energy_ranges)), int(np.sqrt(number_energy_ranges)))
    ax_difference = ax_difference.ravel()

    for i in range(number_energy_ranges):
        for j in range(len(image_binned[i])):
            image_sum[i] += image_binned[i][j]

        # calculate normed pattern spectra sum
        image_sum_normed[i] = (image_sum[i] - np.min(image_sum[i]))
        image_sum_normed[i] /= np.max(image_sum_normed[i])
        # calculate normed patter spectra sum minus pattern spectra of first energy range
        image_sum_difference[i] = image_sum_normed[i] - image_sum_normed[0]

        # plot pattern spectra sum
        fig_normed.suptitle("CTA images sum")   
        ax_normed[i].set_title(f"{np.round(bins[i], 1)} - {np.round(bins[i+1], 1)} TeV", fontdict = {"fontsize" : 10})  
        im = ax_normed[i].imshow((image_sum_normed[i]))
        ax_normed[i].set_xticks([])
        ax_normed[i].set_yticks([])
        divider = make_axes_locatable(ax_normed[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig_normed.colorbar(im, cax=cax, orientation="vertical")

        # plot pattern spectra difference
        fig_difference.suptitle("CTA images sum difference")   
        ax_difference[i].set_title(f"{np.round(bins[i], 1)} - {np.round(bins[i+1], 1)} TeV", fontdict = {"fontsize" : 10})  
        im = ax_difference[i].imshow((image_sum_difference[i]))
        ax_difference[i].set_xticks([])
        ax_difference[i].set_yticks([])
        divider = make_axes_locatable(ax_difference[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig_difference.colorbar(im, cax=cax, orientation="vertical")

    fig_normed.tight_layout()
    fig_normed.savefig(path + "image_sum_normed.png", dpi = 250)
    fig_difference.tight_layout()
    fig_difference.savefig(path + "image_sum_difference.png", dpi = 250)

def PlotHillasParametersDistribution(table, hillas_parameter, energy_range, number_energy_ranges, particle_type):
    # prepare energy binning
    sst_energy_min = energy_range[0] # TeV
    sst_energy_max = energy_range[1] # TeV
    bins_energy = np.logspace(np.log10(np.min(sst_energy_min)), np.log10(np.max(sst_energy_max)), number_energy_ranges + 1) 

    # prepare hillas_ellipticity binning
    bins_ellipticity = np.linspace(np.min(table[hillas_parameter]), np.max(table[hillas_parameter]), 50)

    os.makedirs(f"data/{particle_type}/info/hillas", exist_ok = True)

    fig, ax = plt.subplots(int(np.sqrt(number_energy_ranges)), int(np.sqrt(number_energy_ranges)))
    fig.set_size_inches(12, 8)
    ax = ax.ravel()
    for i in range(number_energy_ranges):
        table_copy = table.copy()
        table_copy.drop(table_copy.loc[table_copy["true_energy"] <= bins_energy[i]].index, inplace=True)
        table_copy.drop(table_copy.loc[table_copy["true_energy"] >= bins_energy[i+1]].index, inplace=True)
        table_copy.reset_index()
        mean = np.mean(table_copy[hillas_parameter])
        ax[i].set_title(f"{np.round(bins_energy[i], 1)} - {np.round(bins_energy[i+1], 1)} TeV")
        ax[i].hist(table_copy[hillas_parameter], bins = bins_ellipticity)
        if hillas_parameter == "hillas_intensity":
            ax[i].set_yscale('log')
        ymin, ymax = ax[i].get_ylim()
        ax[i].vlines(mean, ymin, ymax, color = "black", linestyle = "--", label = f"$\mu = {np.round(mean, 1)}$")
        ax[i].set_ylim(ymin, ymax)
        ax[i].legend()
    ax[-2].set_xlabel(hillas_parameter)
    ax[3].set_ylabel("number of events")
    plt.tight_layout()
    plt.savefig(f"data/{particle_type}/info/hillas/{hillas_parameter}_dist.png", dpi = 250)


def reshape_image(image):
    image = np.append(image, np.ones(4 * 8 * 8) * np.min(image))
    image = np.split(image, 36)
    image = np.reshape(image, newshape = (-1, 8, 8)).astype('float32')
    image = np.rot90(image, k = 1, axes = (1, 2))
    mask = np.array([32, 22, 10, 4, 16, 33, 30, 20, 8, 2, 14, 26, 28, 18, 6, 0, 12, 24, 29, 19, 7, 1, 13, 25, 31, 21, 9, 3, 15, 27, 34, 23, 11, 5, 17, 35])
    image = image[mask]
    image = image.reshape(6,6,8,8).transpose(0,2,1,3).reshape(48,48)
    image = np.round(image, 1)
    return(image)


def plot_energy_dist(table, particle_type, run):
    plt.figure()
    plt.grid(alpha = 0.2)
    plt.hist(table["true_energy"].to("TeV").value)
    plt.xlabel("True energy [TeV]")
    plt.ylabel("Number of events")
    plt.yscale("log")
    plt.savefig(f"data/{particle_type}/info/energy_distribution/energy_distribution_run{run}_alpha.pdf", dpi = 500)
    plt.close()


def plot_subarray_layout(subarray, path, run):
    plt.figure()
    subarray.peek()
    plt.savefig(path + f"telescope_subarray_layout_run{run}.pdf", dpi = 500)
    plt.close()