import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ctapipe.io import EventSource, read_table
from ctapipe.instrument import SubarrayDescription 
from ctapipe.visualization import CameraDisplay
from utilities import GetEventImage, GetEventImageBasic, GetEventImageBasicSmall
from astropy.table import Table, join, vstack
from astropy.io import ascii
import sys
import os
from tqdm import tqdm
np.set_printoptions(threshold=sys.maxsize)

plt.rcParams.update({'font.size': 14})

# ----------------------------------------------------------------------
# load and prepare data set
# ----------------------------------------------------------------------

# load data
particle_type = "gamma" # gamma/proton/gamma_diffuse
image_type = "minimalistic" # default/minimalistic
run = np.array([1012, 1024, 1034, 1037, 1054, 1057, 1069, 1073, 107, 1086, 1098, 1108, 1117, 1119, 1121, 1146, 1196, 1213, 1232, 1234, 1257, 1258, 1275, 1305, 1308, 1330, 134, 1364, 1368, 1369, 1373, 1394, 1413, 1467, 1475, 1477, 1489, 148, 1514, 1517, 1521, 1531, 1542, 1570, 1613, 1614, 1628, 1642, 1674, 1691, 1703, 1713, 1716, 1749, 1753, 1760, 1780, 1788, 1796, 1798, 1807, 1845, 1862, 1875, 1876, 1945, 1964, 2007, 2079, 2092, 2129, 2139, 214, 2198, 2224, 223, 2254, 2273, 2294, 2299, 2309, 2326, 2331]) #gamma 1012, 1024, 1034, 1037, 1054, 1057, 1069, 1073, 107, 1086, 1098, 1108, 1117, 1119, 1121, 1146, 1196, 1213, 1232, 1234, 1257, 1258, 1275, 1305, 1308, 1330, 134, 1364, 1368, 1369, 1373, 1394, 1413, 1467, 1475, 1477, 1489, 148, 1514, 1517, 1521, 1531, 1542, 1570, 1581, 1613, 1614, 1628, 1642, 1674, 1691, 1703, 1713, 1716, 1749, 1753, 1760, 1780, 1788, 1796, 1798, 1807, 1845, 1862, 1875, 1876, 1945, 1964, 2007, 2079, 2092, 2129, 2139, 214, 2198, 2224, 223, 2254, 2273, 2294, 2299, 2309, 2326, 2331, 2522, 2668, 2673, 2677, 2805, 2921, 2953, 3334, 3375, 3667 
# run = np.array([10029, 10033, 10037, 10068, 10070, 10076, 10086, 10101, 10102, 10117, 10125, 10144, 10166, 10207, 10217, 10228, 10251, 10255, 10259, 10260, 10262, 10271, 10276, 10278, 10295, 10299, 10310, 10321, 10330, 1034, 10356, 10364, 10365, 10368, 10407, 10416, 10426, 10430, 10431, 1047, 10483, 10503, 10517, 10529, 10555, 1056, 10578, 10609, 10612, 10629, 10633, 10641, 10654, 10662, 10668, 10703, 10725, 10732, 10736, 10738, 10779, 10801, 10812, 10841, 10882, 10900, 10915, 10934, 10949, 10959, 10970, 10988, 10997, 11001, 11008, 11032, 11037, 11045, 11050, 11090, 11096, 11133, 11146, 11168, 11189, 1119, 11211, 11224, 1122, 11239, 11240, 11254, 11263, 11267, 11271, 11272, 1127, 11290, 11301, 11309, 11322, 1134, 11396, 11398, 11405, 11413, 11420, 11425, 11438, 11452, 1145, 11495, 11498, 11499, 11502, 11524, 11525, 1155, 11572, 11576, 11577, 11587, 11609, 11669, 1166, 11734, 11738, 11748, 1176, 11777, 11783, 11810, 11837, 11848, 11868, 11909, 11913, 11943, 11952, 11957, 11968, 11982, 12000, 12002, 12005, 12014, 12022, 12031, 1206, 1210, 12125, 12127, 12155, 12162, 12173, 12181, 12211, 12240, 12246, 12248, 12251, 12295, 12311, 12320, 12338, 12406, 12478, 12479, 12486, 12488, 12522, 1252, 12560, 12565, 1256, 12572, 12573, 12609, 12625, 12638, 12648, 12663, 12697, 12708, 12719, 12779, 12803, 12821, 1283, 12850, 1285, 12861, 12873, 1289, 12916, 12958, 12970, 12982, 13044, 13050, 13065, 13137, 13161, 13168, 1316, 13185, 13197, 13241, 13244, 13252, 13271, 13282, 13320, 13421, 13442, 13466, 13510, 13512, 13612, 13633, 1367, 13682, 13688, 13712, 13733, 13773, 13792, 13793, 13795, 13842, 13845, 13854, 13882, 13945, 13947, 13949, 13953, 13971, 13978, 13993, 14044, 14069, 14082, 14101, 14112, 14132, 14144, 14192, 14193, 14216, 14223, 14235, 14236, 14239, 14243, 14306, 1433, 14352, 14360, 14424, 14442, 14523, 14595, 14596, 14598, 1459, 14601, 14621, 14626, 14629, 14630, 14656, 1465, 14705, 14714, 14761, 14782, 14843, 14871, 1487, 14987, 14991, 15066, 15069, 15086, 15102, 15146, 15155, 15226, 15240, 15336, 15354, 15379, 15407, 15412, 1544, 15463, 15480, 15513, 15540, 15543, 15546, 15578, 15611, 15621, 15645, 15702, 15793, 15831, 15892, 15899, 15991, 15997, 16019, 16034, 1605, 16108, 1612, 16157, 16194, 16229, 16230, 16245, 1624, 16256, 16287, 16288, 16292, 16332, 1637, 16402, 16413, 16483, 16515, 16520, 16529, 1654, 16624, 16635, 16687, 16706, 1682, 16890, 16925, 17029, 17049, 17053, 17337, 17344, 17371, 17403, 1748, 17609, 1763, 17679, 17693, 17694, 17695, 17699, 17733, 17744, 17771, 17815, 17833, 17886, 17992, 18075, 1811, 18212, 18220, 18225, 18294, 18297, 18317, 18414, 18447, 18451, 18579, 18643, 18654, 18668, 18732, 18766, 18771, 18774, 18815, 18823, 18860, 18892, 18968, 19005, 19008, 19023, 19046, 19072, 19111, 19168, 19211, 19220, 19296, 19337, 19369, 19399, 19456, 19477, 19489, 19498, 19525, 19532, 19539, 19560, 1979, 19836, 19928, 19943, 1998, 19990, 20019, 20025, 20073, 2027, 20406, 2042, 20432, 20625, 20650, 20776, 20780, 20830, 20877, 2087, 20930, 21072, 2107, 21083, 21316, 21493, 21521, 21545, 21600, 21787, 21829, 21906, 21961, 22046, 22070, 22217, 22218, 22239, 22273, 22289, 22338, 22385, 22390, 22414, 22499, 2258, 2262, 22731, 22832, 22924, 23109, 23141, 23290, 23542, 23600, 23943, 2402, 24073, 24175, 24811, 24812, 24856, 25225, 2530, 26114, 26180, 26300, 27029, 2702, 27134, 27408, 27643, 27922, 28145, 28157, 28198, 28491, 28775, 2886, 28920, 28975, 29223, 29350, 30153, 3060, 30701, 31190, 31892, 32069, 3253, 32788, 33006, 33214, 34629]) # proton
# run = np.array([1010, 1027, 1029, 104, 105, 1060, 1062, 1070, 1071, 1077, 1088, 108, 1102, 1115, 1118, 1134, 1140, 1161, 116, 1188, 1200, 1213, 1219, 1225, 1226, 1228, 1232, 1249, 1250, 1251, 1258, 1286, 1289, 1291, 1322, 1325, 132, 1334, 1342, 1344, 1349, 1371, 1374, 1379, 1407, 140, 1420, 1444, 1446, 1449, 1450, 1459, 145, 1472, 1473, 1474, 1477, 1484, 1496, 1537, 1545, 1548, 1573, 1577, 1586, 1631, 1655, 1659, 1668, 1693, 1697, 1702, 1704, 1705, 1708, 1736, 1737, 1744, 1752, 1772, 1791, 17, 180, 1813, 1815, 1817, 1820, 1853, 1868, 1879, 1886, 1887, 192, 1960, 1977, 1983, 1985, 2004, 2006, 2016, 201, 2029, 2041, 2049, 2068, 2087, 2095, 2098, 2104, 212, 214, 2151, 2175, 2203, 2219, 2222, 2243, 224, 2260, 2262, 2278, 2293, 2295, 229, 2329, 2333, 2337, 2338, 2354, 2379, 237, 2384, 2390, 2406, 2407, 2417, 2443, 2456, 2503, 2519, 2546, 2556, 256, 2571, 2592, 259, 25, 2606, 2608, 2615, 2640, 2656, 2665, 2672, 2692, 2711, 2726, 2727, 2736, 2766, 2787, 2805, 285, 2879, 2890, 2901, 2911, 291, 2921, 2973, 2978, 2995, 3021, 3023, 3037, 3041, 3088, 3099, 3164, 3175, 3181, 3196, 3222, 3267, 3271, 3316, 3331, 3407, 3473, 3508, 3564, 3582, 3618, 3627, 3658, 3853, 3877, 38, 3938, 3958, 3966, 3969, 3989, 4005, 401, 4030, 4099, 410, 4238, 4252, 4293, 4370, 4401, 446, 4629, 4710, 4760, 4871, 4887, 5038, 5235, 556, 5816, 6771, 6821]) # gamma_diffuse
for r in range(len(run)): #len(run)
    print("Run", run[r])

    if particle_type == "gamma_diffuse":
        input_filename = f"gamma_20deg_0deg_run{run[r]}___cta-prod5-paranal_desert-2147m-Paranal-dark_cone10_merged.DL1"
    else:
        input_filename = f"{particle_type}_20deg_0deg_run{run[r]}___cta-prod5-paranal_desert-2147m-Paranal-dark_merged.DL1"
    input_directory = f"dm-finder/data/{particle_type}/event_files/" + input_filename + ".h5"

    input_filename = f"{particle_type}_20deg_0deg_run{run[r]}___cta-prod5-paranal_desert-2147m-Paranal-dark_merged.DL1"

    source = EventSource(input_directory)
    print(f"Total number of events: {len(source)}")

    # get telescope subarray description
    subarray = SubarrayDescription.from_hdf(input_directory)
    subarray.info()

    # create figure of telescope subarray layout
    path_array_layout = f"dm-finder/data/{particle_type}/info/array_layout/"
    path_energy_distribution = f"dm-finder/data/{particle_type}/info/energy_distribution/"
    try:
        os.makedirs(path_array_layout)
        os.makedirs(path_energy_distribution)
    except OSError:
        pass

    plt.figure()
    subarray.peek()
    plt.savefig(f"dm-finder/data/{particle_type}/info/array_layout/telescope_subarray_layout_run{run[r]}.png")
    plt.close()

    # get tables for all SST telescopes that include images and corresponding obs_id + event_id for each event
    sst_tel_id = np.append(range(30, 100), range(131, 181))
    images_table= vstack([read_table(input_directory, f"/dl1/event/telescope/images/tel_{t:03}") for t in sst_tel_id])

    # get true energy of each event
    simulated_parameter_table = read_table(input_directory, "/simulation/event/subarray/shower")

    # combine both tables and remove unnecessary columns
    complete_table = join(left = images_table, right = simulated_parameter_table, keys=["obs_id", "event_id"])
    complete_table.keep_columns(["obs_id", "event_id", "tel_id", "image", "true_energy"])

    # convert energy from TeV to GeV
    complete_table["true_energy"] = complete_table["true_energy"].to("GeV")

    # define telescope geometry to the SST geometry (telescope id 30)
    sst_camera_geometry = source.subarray.tel[30].camera.geometry

    ############################### save individual telescope images ###############################
    # for i in range(30):
    #     GetEventImage(complete_table["image"][i], sst_camera_geometry, clean_image = False, savefig = f"cta/data/images/tests/obs_id_{complete_table['obs_id'][i]}__event_id_{complete_table['event_id'][i]}__tel_id_{complete_table['tel_id'][i]}.png", colorbar = True, cmap = "Greys")
    #     print("_______________________________")
    #     print("obs_id, event_id, tel_id:", complete_table["obs_id"][i], complete_table["event_id"][i], complete_table["tel_id"][i])
    #     print("true energy:", complete_table["true_energy"][i])
    #     print("min/max pixel:", np.min(complete_table["image"][i]), np.max(complete_table["image"][i]))
    ################################################################################################

    # group table by same obs_id and event_id
    complete_table_by_obs_id_event_id = complete_table.group_by(["obs_id", "event_id", "true_energy"])

    # create new table in which we will add a combined images of telescopes originiting from the same event
    complete_table_tel_combined = complete_table_by_obs_id_event_id.groups.keys

    # create empty image combined list to be filled in
    image_combined = [[]] * len(complete_table_tel_combined)

    # combine all images of telescopes originiting from the same event and put them into the list
    k = 0
    for key, group in zip(complete_table_by_obs_id_event_id.groups.keys, complete_table_by_obs_id_event_id.groups):
        image_combined[k] = group["image"].groups.aggregate(np.add)
        k += 1
    
    # # compute sum of all 'photon' counts in each combined image 
    # sum_image_combined = np.sum(image_combined, axis = 2)

    # # add combined sum of all 'photon' counts list to the table
    # complete_table_tel_combined["sum_image"] = sum_image_combined

    # save data into a .csv file
    ascii.write(complete_table_tel_combined, f"dm-finder/data/{particle_type}/tables/{particle_type}_20deg_0deg_run{run[r]}___cta-prod5-paranal_desert-2147m-Paranal-dark_merged.DL1.csv", format = "csv", fast_writer = False, overwrite = True)

    # add combined images list to the table
    complete_table_tel_combined["image combined"] = image_combined

    # display and save energy distribution
    plt.figure()
    plt.grid(alpha = 0.2)
    plt.hist(complete_table_tel_combined["true_energy"].to("TeV").value)
    plt.xlabel("True energy [TeV]")
    plt.ylabel("Number of events")
    plt.yscale("log")
    plt.savefig(f"dm-finder/data/{particle_type}/info/energy_distribution/energy_distribution_run{run[r]}.png")
    plt.close()

    # print(complete_table_tel_combined)

    # open csv file and prepare table for filling in images 
    run_filename = f"{particle_type}_20deg_0deg_run{run[r]}___cta-prod5-paranal_desert-2147m-Paranal-dark_merged.DL1"
    table_path = f"dm-finder/data/{particle_type}/tables/" + run_filename + ".csv"

    table = pd.read_csv(table_path)

    # extract unique obs_id and all event_id
    obs_id_unique = np.unique(table["obs_id"])
    event_id = table["event_id"]

    # add image columns to table to be filled in
    if (particle_type == "gamma") or (particle_type == "gamma_diffuse"):
        table["particle"] = 1
        print("gamma")
    elif particle_type == "proton":
        table["particle"] = 0
        print("proton")
    table["image"] = np.nan
    table["image"] = table["image"].astype(object)


    # save combined telescope images
    for i in tqdm(range(len(complete_table_tel_combined))): # len(complete_table_tel_combined)
        # create directory in which the images will be saved
        path = f"dm-finder/data/{particle_type}/images/{image_type}/{input_filename}/obs_id_{complete_table_tel_combined['obs_id'][i]}/tif"
        try:
            os.makedirs(path)
        except OSError:
            pass

        # print("obs_id:", complete_table_tel_combined["obs_id"][i])
        # print("event_id:", complete_table_tel_combined["event_id"][i])
        # print(complete_table_tel_combined["image combined"][i][0])

        image = complete_table_tel_combined["image combined"][i][0]
        
        image = np.append(image, np.ones(4 * 8 * 8) * np.min(image))
        image = np.split(image, 36)
        image = np.reshape(image, newshape = (-1, 8, 8)).astype('float32')
        image = np.rot90(image, k = 1, axes = (1, 2))
        # mask = np.array([15, 21, 9, 27, 3, 33, 14, 20, 8, 26, 2, 32, 16, 22, 10, 28, 4, 34, 13, 19, 7, 25, 1, 31, 17, 23, 11, 29, 18, 6, 24, 0, 5, 30, 35])
        mask = np.array([32, 22, 10, 4, 16, 33, 30, 20, 8, 2, 14, 26, 28, 18, 6, 0, 12, 24, 29, 19, 7, 1, 13, 25, 31, 21, 9, 3, 15, 27, 34, 23, 11, 5, 17, 35])
        image = image[mask]
        image = image.reshape(6,6,8,8).transpose(0,2,1,3).reshape(48,48)

        # convert it to an 8-bit image (0, 255)
        image = image - np.min(image)
        image = 255 * image / np.max(image) 
        image = image.astype(np.int8)

        # GetEventImageBasicSmall(image, clean_image = True, savefig = f"dm-finder/data/{particle_type}/images/{image_type}/{input_filename}/obs_id_{complete_table_tel_combined['obs_id'][i]}/tif/obs_id_{complete_table_tel_combined['obs_id'][i]}__event_id_{complete_table_tel_combined['event_id'][i]}_8bit.tif", colorbar = False, cmap = "Greys_r")

        # implement image into table
        table["image"][i] = image


    # # convert tif images to pgm
    # for i in tqdm(range(len(complete_table_tel_combined))): #len(complete_table_tel_combined)
    #     # create directory in which the images will be saved
    #     path = f"dm-finder/data/{particle_type}/images/{image_type}/{input_filename}/obs_id_{complete_table_tel_combined['obs_id'][i]}/pgm"
    #     try:
    #         os.makedirs(path)
    #     except OSError:
    #         pass

    #     os.system(f"convert dm-finder/data/{particle_type}/images/{image_type}/{input_filename}/obs_id_{complete_table_tel_combined['obs_id'][i]}/tif/obs_id_{complete_table_tel_combined['obs_id'][i]}__event_id_{complete_table_tel_combined['event_id'][i]}_8bit.tif dm-finder/data/{particle_type}/images/{image_type}/{input_filename}/obs_id_{complete_table_tel_combined['obs_id'][i]}/pgm/obs_id_{complete_table_tel_combined['obs_id'][i]}__event_id_{complete_table_tel_combined['event_id'][i]}_8bit.pgm")

    output_filename = f"dm-finder/cnn/iact_images/input/{particle_type}/{image_type}/" + run_filename + "_images_8bit.h5"

    table.to_hdf(output_filename, key = 'events', mode = 'w', index = False)

    # close event file
    source.close()
