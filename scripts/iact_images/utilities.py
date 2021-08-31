import matplotlib.pyplot as plt
from ctapipe.visualization import CameraDisplay
import numpy as np

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