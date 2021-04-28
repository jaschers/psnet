import matplotlib.pyplot as plt
from ctapipe.visualization import CameraDisplay

def ShowEventImage(image, camera_geometry, cmap = "Greys", show_frame = False, colorbar = False, clean_image = False, savefig = False):
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
    
    plt.show()