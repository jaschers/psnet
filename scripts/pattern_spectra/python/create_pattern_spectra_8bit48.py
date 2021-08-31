import numpy as np
import re
from io import StringIO
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import pandas as pd
from tqdm import tqdm

plt.rcParams.update({'font.size': 16})


# properties = np.array([9, 0, 0, 0, 10, 100000, 2, 0, 20, 20, 3]) # a1, a2, dl1, dl2, dh1, dh2, m1, m2, n1, n2, f 
# pattern spectra properties
a = np.array([9, 0])
dl = np.array([0, 0])
dh = np.array([10, 100000])
m = np.array([2, 0])
n = np.array([20, 20])
f = 3

particle_type = "gamma"
image_type = "minimalistic"
run = np.array([1796, 1798, 1807, 1845, 1862, 1875, 1876, 1945, 1964, 2007, 2079, 2092, 2129, 2139, 214, 2198, 2224, 223, 2254, 2273, 2294, 2299, 2309, 2326, 2331]) # gamma

# run = np.array([1010, 1027, 1029, 104, 105, 1060, 1062, 1070, 1071, 1077, 1088, 108, 1102, 1115, 1118, 1134, 1140, 1161, 116, 1188, 1200, 1213, 1219, 1225, 1226, 1228, 1232, 1249, 1250, 1251, 1258, 1286, 1289, 1291, 1322, 1325, 132, 1334, 1342, 1344, 1349, 1371, 1374, 1379, 1407, 140, 1420, 1444, 1446, 1449, 1450, 1459, 145, 1472, 1473, 1474, 1477, 1484, 1496, 1537, 1545, 1548, 1573, 1577, 1586, 1631, 1655, 1659, 1668, 1693, 1697, 1702, 1704, 1705, 1708, 1736, 1737, 1744, 1752, 1772, 1791, 17, 180, 1813, 1815, 1817, 1820, 1853, 1868, 1879, 1886, 1887, 192, 1960, 1977, 1983, 1985, 2004, 2006, 2016, 201, 2029, 2041, 2049, 2068, 2087, 2095, 2098, 2104, 212, 214, 2151, 2175, 2203, 2219, 2222, 2243, 224, 2260, 2262, 2278, 2293, 2295, 229, 2329, 2333, 2337, 2338, 2354, 2379, 237, 2384, 2390, 2406, 2407, 2417, 2443, 2456, 2503, 2519, 2546, 2556, 256, 2571, 2592, 259, 25, 2606, 2608, 2615, 2640, 2656, 2665, 2672, 2692, 2711, 2726, 2727, 2736, 2766, 2787, 2805, 285, 2879, 2890, 2901, 2911, 291, 2921, 2973, 2978, 2995, 3021, 3023, 3037, 3041, 3088, 3099, 3164, 3175, 3181, 3196, 3222, 3267, 3271, 3316, 3331, 3407, 3473, 3508, 3564, 3582, 3618, 3627, 3658, 3853, 3877, 38, 3938]) # gamma_diffuse

# run = np.array([10029, 10033, 10037, 10068, 10070, 10076, 10086, 10101, 10102, 10117, 10125, 10144, 10166, 10207, 10217, 10228, 10251, 10255, 10259, 10260, 10262, 10271, 10276, 10278, 10295, 10299, 10310, 10321, 10330, 1034, 10356, 10364, 10365, 10368, 10407, 10416, 10426, 10430, 10431, 1047, 10483, 10503, 10517, 10529, 10555, 1056, 10578, 10609, 10612, 10629, 10633, 10641, 10654, 10662, 10668, 10703, 10725, 10732, 10736, 10738, 10779, 10801, 10812, 10841, 10882, 10900, 10915, 10934, 10949, 10959, 10970, 10988, 10997, 11001, 11008, 11032, 11037, 11045, 11050, 11090, 11096, 11133, 11146, 11168, 11189, 1119, 11211, 11224, 1122, 11239, 11240, 11254, 11263, 11267, 11271, 11272, 1127, 11290, 11301, 11309, 11322, 1134, 11396, 11398, 11405, 11413, 11420, 11425, 11438, 11452, 1145, 11495, 11498, 11499, 11502, 11524, 11525, 1155, 11572, 11576, 11577, 11587, 11609, 11669, 1166, 11734, 11738, 11748, 1176, 11777, 11783, 11810, 11837, 11848, 11868, 11909, 11913, 11943, 11952, 11957, 11968, 11982, 12000, 12002, 12005, 12014, 12022, 12031, 1206, 1210, 12125, 12127, 12155, 12162, 12173, 12181, 12211, 12240, 12246, 12248, 12251, 12295, 12311, 12320, 12338, 12406, 12478, 12479, 12486, 12488, 12522, 1252, 12560, 12565, 1256, 12572, 12573, 12609, 12625, 12638, 12648, 12663, 12697, 12708, 12719, 12779, 12803, 12821, 1283, 12850, 1285, 12861, 12873, 1289, 12916, 12958, 12970, 12982, 13044, 13050, 13065, 13137, 13161, 13168, 1316, 13185, 13197, 13241, 13244, 13252, 13271, 13282, 13320, 13421, 13442, 13466, 13510, 13512, 13612, 13633, 1367, 13682, 13688, 13712, 13733, 13773, 13792, 13793, 13795, 13842, 13845, 13854, 13882, 13945, 13947, 13949, 13953, 13971, 13978, 13993, 14044, 14069, 14082, 14101, 14112, 14132, 14144, 14192, 14193, 14216, 14223, 14235, 14236, 14239, 14243, 14306, 1433, 14352, 14360, 14424, 14442, 14523, 14595, 14596, 14598, 1459, 14601, 14621, 14626, 14629, 14630, 14656, 1465, 14705, 14714]) #proton

for r in range(len(run)): #len(run)
    # read csv file
    print("Run", run[r])
    run_filename = f"{particle_type}_20deg_0deg_run{run[r]}___cta-prod5-paranal_desert-2147m-Paranal-dark_merged.DL1"
    csv_directory = f"dm-finder/data/{particle_type}/tables/" + run_filename + ".csv"

    table = pd.read_csv(csv_directory)

    # extract unique obs_id and all event_id
    obs_id_unique = np.unique(table["obs_id"])
    event_id = table["event_id"]

    if (particle_type == "gamma") or (particle_type == "gamma_diffuse"):
        table["particle"] = 1
    elif particle_type == "proton":
        table["particle"] = 0

    # add image column to table to be filled in
    table["pattern spectrum"] = np.nan
    table["pattern spectrum"] = table["pattern spectrum"].astype(object)

    # for loop to create pattern spectra from cta images with pattern spectra code
    for i in tqdm(range(len(table))): #len(table)
        # create folder 
        path_mat = f"dm-finder/data/{particle_type}/pattern_spectra/{image_type}" + f"/a_{a[0]}_{a[1]}__dl_{dl[0]}_{dl[1]}__dh_{dh[0]}_{dh[1]}__m_{m[0]}_{m[1]}__n_{n[0]}_{n[1]}__f_{f}/" + run_filename + "/obs_id_" + f"{table['obs_id'][i]}/mat/"
        path_tif = f"dm-finder/data/{particle_type}/pattern_spectra/{image_type}" + f"/a_{a[0]}_{a[1]}__dl_{dl[0]}_{dl[1]}__dh_{dh[0]}_{dh[1]}__m_{m[0]}_{m[1]}__n_{n[0]}_{n[1]}__f_{f}/" + run_filename + "/obs_id_" + f"{table['obs_id'][i]}/tif/"
        filename = "obs_id_" + f"{table['obs_id'][i]}" + "__" "event_id_" + f"{table['event_id'][i]}_8bit"

        try:
            os.makedirs(path_mat)
        except OSError:
            pass #print("Directory could not be created")
        
        try:
            os.makedirs(path_tif)
        except OSError:
            pass #print("Directory could not be created")

        # command to create pattern spectra
        command = "./dm-finder/scripts/pattern_spectra/xmaxtree/xmaxtree" + f" dm-finder/data/{particle_type}/images/{image_type}/" + run_filename + "/obs_id_" + f"{table['obs_id'][i]}" + "/pgm/" + "obs_id_" + f"{table['obs_id'][i]}" + "__" "event_id_" + f"{table['event_id'][i]}_8bit.pgm" f" a {a[0]}, {a[1]} dl {dl[0]}, {dl[1]} dh {dh[0]}, {dh[1]} m {m[0]}, {m[1]} n {n[0]}, {n[1]} f {f} nogui e " + path_mat + filename + " &> /dev/null"

        # apply command in terminal
        os.system(command)

        # open pattern spectra file
        file = open(path_mat + filename + ".m", "r")
        # remove unnecessary "Granulometry(:,:)" from the string
        image = file.read()[18:]
        # convert string to numpy array with proper shape and remove unnecessary colomns and rows
        image = "0" + re.sub(" +", " ", image)
        image = np.genfromtxt(StringIO(image))[2:, 2:]
        # take log of the image and replace -inf values with 0
        image = np.log10(image)

        image[image == -np.inf] = 0

        # add image to table
        table["pattern spectrum"][i] = image

        if r == 0 and i < 50:
            # plt.rcParams['figure.facecolor'] = 'black'
            plt.figure()
            plt.imshow(image, cmap = "Greys_r")
            # plt.text(0.12, 0.80, "small", rotation = 90, transform=plt.gcf().transFigure)
            # plt.text(0.12, 0.15, "large", rotation = 90, transform=plt.gcf().transFigure)
            # plt.text(0.65, 0.05, "large", transform=plt.gcf().transFigure)
            # plt.text(0.15, 0.05, "small", transform=plt.gcf().transFigure)
            # plt.text(0.14, 0.5, r"$\rightarrow$", rotation = 90, transform=plt.gcf().transFigure)
            # plt.text(0.5, 0.05, r"$\rightarrow$", transform=plt.gcf().transFigure)
            plt.annotate('', xy=(0, -0.05), xycoords='axes fraction', xytext=(1, -0.05), arrowprops=dict(arrowstyle="<-", color='black'))
            plt.annotate('', xy=(-0.05, 1), xycoords='axes fraction', xytext=(-0.05, 0), arrowprops=dict(arrowstyle="<-", color='black'))
            plt.xlabel("(moment of inertia) / area$^2$", labelpad = 20, fontsize = 16)
            plt.ylabel("area", labelpad = 20, fontsize = 16)
            cbar = plt.colorbar()
            cbar.set_label(label = "log$_{10}$(flux)", fontsize = 16)
            # plt.axis('off')
            plt.xticks([], [])
            plt.yticks([], [])
            plt.tight_layout()
            plt.savefig(path_tif + filename + ".tif")
            plt.close()

    # save tabel as h5 file
    path_cnn_input = f"dm-finder/cnn/pattern_spectra/input/{particle_type}/{image_type}/" + f"/a_{a[0]}_{a[1]}__dl_{dl[0]}_{dl[1]}__dh_{dh[0]}_{dh[1]}__m_{m[0]}_{m[1]}__n_{n[0]}_{n[1]}__f_{f}/" 
    try:
        os.makedirs(path_cnn_input)
    except OSError:
        pass

    output_filename = path_cnn_input + run_filename + "_ps_8bit.h5"
    # print(output_filename)
    table.to_hdf(output_filename, key = 'events', mode = 'w', index = False)
