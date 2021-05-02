import numpy as np
import matplotlib.pyplot as plt
from ctapipe.io import EventSource, read_table
from ctapipe.instrument import SubarrayDescription 
from ctapipe.visualization import CameraDisplay
from utilities import ShowEventImage
from astropy.table import Table, join, vstack
from astropy.io import ascii
import sys
import os
from tqdm import tqdm
np.set_printoptions(threshold=sys.maxsize)


# load data
run = 1012 
input_filename = f"gamma_20deg_0deg_run{run}___cta-prod5-paranal_desert-2147m-Paranal-dark_merged.DL1"
input_directory = "cta/data/event-files/" + input_filename + ".h5"

source = EventSource(input_directory)
print(f"Total number of events: {len(source)}")

# get telescope subarray description
subarray = SubarrayDescription.from_hdf(input_directory)
subarray.info()

# create figure of telescope subarray layout
plt.figure()
subarray.peek()
plt.savefig(f"cta/data/info/telescope_subarray_layout_run{run}.png")

# get tables for all SST telescopes that include images and corresponding obs_id + event_id for each event
images_table_1 = vstack([read_table(input_directory, f"/dl1/event/telescope/images/tel_{t:03}") for t in range(30, 100)])
images_table_2 = vstack([read_table(input_directory, f"/dl1/event/telescope/images/tel_{t:03}") for t in range(131, 181)])
images_table = vstack([images_table_1, images_table_2])

# get true energy of each event
simulated_parameter_table = read_table(input_directory, "/simulation/event/subarray/shower")

# combine both tables and remove unnecessary columns
complete_table = join(left = images_table, right = simulated_parameter_table, keys=["obs_id", "event_id"])
complete_table.keep_columns(["obs_id", "event_id", "tel_id", "image", "true_energy"])

# convert energy from TeV to GeV
complete_table["true_energy"] = complete_table["true_energy"].to("GeV")

# define telescope geometry to the SST geometry (telescope id 30)
SST_camera_geometry = source.subarray.tel[30].camera.geometry

# # save images individual telescope images
# for i in range(30):
#     ShowEventImage(complete_table["image"][i], SST_camera_geometry, clean_image = False, savefig = f"cta/data/images/tests/obs_id_{complete_table['obs_id'][i]}__event_id_{complete_table['event_id'][i]}__tel_id_{complete_table['tel_id'][i]}.png", colorbar = True, cmap = "Greys")
#     print("_______________________________")
#     print("obs_id, event_id, tel_id:", complete_table["obs_id"][i], complete_table["event_id"][i], complete_table["tel_id"][i])
#     print("true energy:", complete_table["true_energy"][i])
#     print("min/max pixel:", np.min(complete_table["image"][i]), np.max(complete_table["image"][i]))

# group table by same obs_id and event_id
complete_table_by_obs_id_event_id = complete_table.group_by(["obs_id", "event_id", "true_energy"])

# create new table in which we will add a combined image of all telescopes for each individual event
complete_table_tel_combined = complete_table_by_obs_id_event_id.groups.keys

# save data into a .csv file
ascii.write(complete_table_tel_combined, f"cta/data/csv/gamma_20deg_0deg_run{run}___cta-prod5-paranal_desert-2147m-Paranal-dark_merged.DL1.csv", format = "csv", fast_writer = False, overwrite = True)

image_combined = [[]] * len(complete_table_tel_combined)
# image_combined = np.zeros(shape = (len(complete_table_tel_combined), 2048))
# print(image_combined)

k = 0
for key, group in zip(complete_table_by_obs_id_event_id.groups.keys, complete_table_by_obs_id_event_id.groups):
    image_combined[k] = group["image"].groups.aggregate(np.add)
    k = k + 1
    # print(f"****** obs_id_{key['obs_id']}, event_id_{key['event_id']} *******")
    # print(group["image"])
    # print(group["image"].groups.aggregate(np.add))

# add combined images to the table
complete_table_tel_combined["image combined"] = image_combined


# save combined telescope images
for i in tqdm(range(len(complete_table_tel_combined))):
    # create directory in which the images will be saved
    path = f"cta/data/images/{input_filename}/obs_id_{complete_table_tel_combined['obs_id'][i]}/tif"
    try:
        os.makedirs(path)
    except OSError:
        pass

    # save image
    ShowEventImage(complete_table_tel_combined["image combined"][i][0], SST_camera_geometry, clean_image = True, savefig = f"cta/data/images/{input_filename}/obs_id_{complete_table_tel_combined['obs_id'][i]}/tif/obs_id_{complete_table_tel_combined['obs_id'][i]}__event_id_{complete_table_tel_combined['event_id'][i]}.tif", colorbar = False, cmap = "Greys")

    # # print information
    # print("_______________________________")
    # print("obs_id, event_id:", complete_table_tel_combined["obs_id"][i], complete_table_tel_combined["event_id"][i])
    # print("true energy:", np.round(complete_table_tel_combined["true_energy"][i], 10))
    # print("min/max pixel:", np.min(complete_table_tel_combined["image combined"][i][0]), np.max(complete_table_tel_combined["image combined"][i][0]))

# convert png images to pgm
for i in tqdm(range(len(complete_table_tel_combined))):
    # create directory in which the images will be saved
    path = f"cta/data/images/{input_filename}/obs_id_{complete_table_tel_combined['obs_id'][i]}/pgm"
    try:
        os.makedirs(path)
    except OSError:
        pass

    os.system(f"convert cta/data/images/{input_filename}/obs_id_{complete_table_tel_combined['obs_id'][i]}/tif/obs_id_{complete_table_tel_combined['obs_id'][i]}__event_id_{complete_table_tel_combined['event_id'][i]}.tif cta/data/images/{input_filename}/obs_id_{complete_table_tel_combined['obs_id'][i]}/pgm/obs_id_{complete_table_tel_combined['obs_id'][i]}__event_id_{complete_table_tel_combined['event_id'][i]}.pgm")

# close event file
source.close()
