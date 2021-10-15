import matplotlib.pyplot as plt
from ctapipe.visualization import CameraDisplay
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


def GetEventImage(image, camera_geometry, cmap = "Greys", show_frame = False, colorbar = False, clean_image = False, savefig = False):
    """[summary]

    Args:
        image ([numpy.ndarray]): [description]
        camera_geometry ([ctapipe.instrument.camera.geometry.CameraGeometry]): [description]
        cmap (str, optional): [description]. Defaults to "Greys".
        show_frame (bool, optional): [description]. Defaults to False.
        colorbar (bool, optional): [description]. Defaults to False.
        clean_image (bool, optional): [description]. Defaults to False.
        savefig (bool, optional): [description]. Defaults to False.
    """
    plt.figure()
    # plt.style.use('dark_background')
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
    #plt.show()

def GetEventImageBasic(image, cmap = "Greys", show_frame = False, colorbar = False, clean_image = False, savefig = False):
    """[summary]

    Args:
        image ([numpy.ndarray]): [description]
        cmap (str, optional): [description]. Defaults to "Greys".
        show_frame (bool, optional): [description]. Defaults to False.
        colorbar (bool, optional): [description]. Defaults to False.
        clean_image (bool, optional): [description]. Defaults to False.
        savefig (bool, optional): [description]. Defaults to False.
    """
    plt.figure()
    plt.imshow(image, cmap = cmap) 

    if colorbar == True:
        cbar = plt.colorbar()
        cbar.set_label(label = "photon count")
    
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
    #plt.show()


def GetEventImageBasicSmall(image, cmap = "Greys", show_frame = False, colorbar = False, clean_image = False, savefig = False):
    """[summary]

    Args:
        image ([numpy.ndarray]): [description]
        cmap (str, optional): [description]. Defaults to "Greys".
        show_frame (bool, optional): [description]. Defaults to False.
        colorbar (bool, optional): [description]. Defaults to False.
        clean_image (bool, optional): [description]. Defaults to False.
        savefig (bool, optional): [description]. Defaults to False.
    """
    dpi = 80
    height, width = np.array(image.shape, dtype=float) / dpi
    
    fig = plt.figure(figsize=(width, height), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')

    ax.imshow(image, interpolation='none', cmap = cmap)
    fig.savefig(savefig, dpi=dpi)

    plt.close("all")
    #plt.show()

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

        # image_sum_difference[i] = (image_sum_difference[i] - np.min(image_sum_difference[i])) / np.max(image_sum_difference[i])
        # image_sum_difference[i] /= np.max(image_sum_difference[i])

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
    # plt.show()