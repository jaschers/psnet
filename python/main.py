import numpy as np
import matplotlib.pyplot as plt
from ctapipe.io import EventSource
from ctapipe.io import read_table  
from ctapipe.instrument import SubarrayDescription 
from ctapipe.visualization import CameraDisplay
from utilities import ShowEventImage
from astropy.table import Table, join, vstack
import sys
np.set_printoptions(threshold=sys.maxsize)


# load data 
input_filename = "cta/data/event-files/gamma_20deg_0deg_run1069___cta-prod5-paranal_desert-2147m-Paranal-dark_merged.DL1.h5"

source = EventSource(input_filename)
print(f"Total number of events: {len(source)}")

# get telescope subarray description
subarray = SubarrayDescription.from_hdf(input_filename)
subarray.info()
plt.figure()
subarray.peek()
plt.savefig(f"cta/data/info/telescope_subarray_layout_run1069.png")

# get images and corresponding obs_id + event_id of each event
images_table_1 = vstack([read_table(input_filename, f"/dl1/event/telescope/images/tel_{t:03}") for t in range(30, 100)])
images_table_2 = vstack([read_table(input_filename, f"/dl1/event/telescope/images/tel_{t:03}") for t in range(131, 181)])
images_table = vstack([images_table_1, images_table_2])

# get true energy of each event
simulated_parameter_table = read_table(input_filename, "/simulation/event/subarray/shower")

# combine both tables
complete_table = join(left = images_table, right = simulated_parameter_table, keys=['obs_id', 'event_id'])
#complete_table.info("stats", out=None).pprint_all()

obs_id = complete_table["obs_id"]
event_id = complete_table["event_id"]
tel_id = complete_table["tel_id"]
images = complete_table["image"]
true_energy = complete_table["true_energy"].to("GeV")

# define telescope geometry to the SST geometry (telescope id 30)
SST_camera_geometry = source.subarray.tel[30].camera.geometry

# save images 
for i in range(10):
    ShowEventImage(images[i], SST_camera_geometry, clean_image = False, savefig = f"cta/data/images/tests/obs_id{obs_id[i]}__event_id{event_id[i]}__tel_id{tel_id[i]}.png", colorbar = True, cmap = "viridis")
    print("_______________________________")
    print("obs_id, event_id, tel_id:", obs_id[i], event_id[i], tel_id[i])
    print("true energy:", np.round(true_energy[i], 2))
    print("min/max pixel:", np.min(images_table["image"][i]), np.max(images_table["image"][i]))
    


# close event file
source.close()
##################### more options ########################

# # show telescope array structure
# source.subarray.peek()
# plt.show()