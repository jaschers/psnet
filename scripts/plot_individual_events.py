import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd
import argparse
import os

plt.rcParams.update({'font.size': 16})

######################################## argparse setup ########################################
script_version=0.1
script_descr="""
This script saves a pattern spectrum and the corresponding CTA image of a specific event as tif image. The pattern spectrum and CTA image must already have been extracted. 
"""

# Open argument parser
parser = argparse.ArgumentParser(description=script_descr, formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-v", "--version", action="version", version=f"v{script_version}")

# Define expected arguments
parser.add_argument("-input", "--input", type = str, required = True, metavar = "-", choices = ["cta", "ps"], help = "plot CTA images or pattern spectra [cta, ps]")
parser.add_argument("-pt", "--particle_type", type = str, metavar = "", choices = ["gamma", "gamma_diffuse", "proton"], help = "particle type [gamma, gamma_diffuse, proton], default: gamma", default = "gamma")
parser.add_argument("-tm", "--telescope_mode", type = str, required = False, metavar = "", choices = ["mono", "stereo_sum_cta", "stereo_sum_ps"], help = "telescope mode [mono, stereo_sum_cta, stereo_sum_ps], default: stereo_sum_cta", default = "stereo_sum_cta")
parser.add_argument("-r", "--run", type = int, metavar = "-", help = "run", required = True)
parser.add_argument("-o", "--obs_id", type = int, metavar = "-", help = "observation ID", required = True)
parser.add_argument("-e", "--event_id", type = int, metavar = "-", help = "event ID", required = True)
parser.add_argument("-t", "--tel_id", type = int, metavar = "-", help = "telescope ID", required = False)
parser.add_argument("-a", "--attribute", type = int, metavar = "", choices = np.arange(0, 19, dtype = int), help = "attribute [0, 1 ... 18] (two required), default: 9 0", default = [9, 0], nargs = 2)
parser.add_argument("-dl", "--domain_lower", type = int, metavar = "", help = "Granulometry: domain - start at <value> <value>, default: 0 0", default = [0, 0], nargs = 2)
parser.add_argument("-dh", "--domain_higher", type = int, metavar = "", help = "Granulometry: domain - end at <value> <value>, default: 10 100000", default = [10, 100000], nargs = 2)
parser.add_argument("-m", "--mapper", type = int, metavar = "", help = "Granulometry: use lambdamappers <mapper1> <mapper2>, default: 2 0", default = [2, 0], nargs = 2)
parser.add_argument("-n", "--size", type = int, metavar = "", help = "Granulometry: size <n1>x<n2>, default: 20 20", default = [20, 20], nargs = 2)
parser.add_argument("-f", "--filter", type = int, metavar = "", help = "Use decision <filter>, default: 3", default = 3, nargs = 1)

args = parser.parse_args()
##########################################################################################

######################################## Define some strings based on the input of the user ########################################

if args.input == "cta":
    print(f"################### Input summary ################### \nInput: CTA images \nParticle type: {args.particle_type} \nRun: {args.run} \nObservation ID: {args.obs_id} \nEvent ID: {args.event_id}")
    if args.tel_id != None:
        print(f"Telescope ID: {args.tel_id}")
    string_input = "iact_images"
    if args.telescope_mode == "stereo_sum_cta":
        string_input_short = "_images"
    elif args.telescope_mode == "mono":
        string_input_short = "_images_mono"
    string_ps_input = ""
    string_table_column = "image"
elif args.input == "ps":
    print(f"################### Input summary ################### \nInput: Pattern spectra \nParticle type: {args.particle_type} \nRun: {args.run} \nObservation ID: {args.obs_id} \nEvent ID: {args.event_id}")
    if args.tel_id != None:
        print(f"Telescope ID: {args.tel_id}")
    print(f"Attribute: {args.attribute} \nDomain lower: {args.domain_lower} \nDomain higher: {args.domain_higher} \nMapper: {args.mapper} \nSize: {args.size} \nFilter: {args.filter}")
    string_input = "pattern_spectra"
    if args.telescope_mode == "stereo_sum_cta":
        string_input_short = "_ps_float"
    elif args.telescope_mode == "mono":
        string_input_short = "_ps_float_mono"
    elif args.telescope_mode == "stereo_sum_ps":
        string_input_short = "_ps_float_stereo_sum"
    string_ps_input = f"a_{args.attribute[0]}_{args.attribute[1]}__dl_{args.domain_lower[0]}_{args.domain_lower[1]}__dh_{args.domain_higher[0]}_{args.domain_higher[1]}__m_{args.mapper[0]}_{args.mapper[1]}__n_{args.size[0]}_{args.size[1]}__f_{args.filter}/"
    string_table_column = "pattern spectrum"
##########################################################################################
run_filename = f"{args.particle_type}_20deg_0deg_run{args.run}___cta-prod5-paranal_desert-2147m-Paranal-dark_merged.DL1"
input_filename = f"dm-finder/cnn/{string_input}/input/{args.particle_type}/" + string_ps_input + run_filename + string_input_short + ".h5"

table = pd.read_hdf(input_filename)

if args.tel_id == None:
    table = table[(table["run"] == args.run) & (table["obs_id"] == args.obs_id) & (table["event_id"] == args.event_id)]
else:
    table = table[(table["run"] == args.run) & (table["obs_id"] == args.obs_id) & (table["event_id"] == args.event_id) & (table["tel_id"] == args.tel_id)]

table.reset_index(inplace = True)

print(table)

if args.input == "ps":
    path_tif = f"dm-finder/data/{args.particle_type}/pattern_spectra" + f"/a_{args.attribute[0]}_{args.attribute[1]}__dl_{args.domain_lower[0]}_{args.domain_lower[1]}__dh_{args.domain_higher[0]}_{args.domain_higher[1]}__m_{args.mapper[0]}_{args.mapper[1]}__n_{args.size[0]}_{args.size[1]}__f_{args.filter}/" + run_filename + "/float" + f"/{args.telescope_mode}" + "/obs_id_" + f"{table['obs_id'][0]}/tif/"

    if args.tel_id == None:
        filename_tif = "obs_id_" + f"{table['obs_id'][0]}" + "__event_id_" + f"{table['event_id'][0]}"
    else:
        filename_tif = "obs_id_" + f"{table['obs_id'][0]}" + "__event_id_" + f"{table['event_id'][0]}" + "__tel_id_" + f"{table['tel_id'][0]}"
        

    os.makedirs(path_tif, exist_ok = True)

    plt.figure()

    if args.tel_id == None:
        plt.title("obs " + f"{table['obs_id'][0]}" + " - event " + f"{table['event_id'][0]}", fontsize = 16, pad = 20)
    else:
        plt.title("obs " + f"{table['obs_id'][0]}" + " - event " + f"{table['event_id'][0]}" + " - tel " + f"{table['tel_id'][0]}", fontsize = 16, pad = 20)

    plt.imshow(table["pattern spectrum"][0], cmap = "Greys_r")
    plt.annotate('', xy=(0, -0.05), xycoords='axes fraction', xytext=(1, -0.05), arrowprops=dict(arrowstyle="<-", color='black'))
    plt.annotate('', xy=(-0.05, 1), xycoords='axes fraction', xytext=(-0.05, 0), arrowprops=dict(arrowstyle="<-", color='black'))
    plt.xlabel(f"attribute {args.attribute[0]}", labelpad = 20, fontsize = 16)
    plt.ylabel(f"attribute {args.attribute[1]}", labelpad = 20, fontsize = 16)
    cbar = plt.colorbar()
    cbar.set_label(label = "log$_{10}$(flux)", fontsize = 16)
    # plt.axis('off')
    plt.xticks([], [])
    plt.yticks([], [])
    plt.tight_layout()
    print(path_tif + filename_tif)
    plt.savefig(path_tif + filename_tif + ".tif", dpi = 150)
    plt.close()

elif args.input == "cta":
    path_tif = f"dm-finder/data/{args.particle_type}/images/" + run_filename + "/float" + f"/{args.telescope_mode}" + "/obs_id_" + f"{table['obs_id'][0]}/tif/"

    if args.tel_id == None:
        filename_tif = "obs_id_" + f"{table['obs_id'][0]}" + "__event_id_" + f"{table['event_id'][0]}"
    else:
        filename_tif = "obs_id_" + f"{table['obs_id'][0]}" + "__event_id_" + f"{table['event_id'][0]}" + "__tel_id_" + f"{table['tel_id'][0]}"

    os.makedirs(path_tif, exist_ok = True)

    plt.figure()
    plt.title("obs " + f"{table['obs_id'][0]}" + " - event " + f"{table['event_id'][0]}" + " - tel " + f"{table['tel_id'][0]}", fontsize = 16, pad = 20)
    plt.imshow(table["image"][0], cmap = "Greys_r") 
    cbar = plt.colorbar()
    cbar.set_label(label = "photon count")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xticks([0, 10, 20, 30, 40])
    plt.yticks([0, 10, 20, 30, 40])
    bbox_inches = None
    pad_inches = 0.1
    plt.tight_layout()
    plt.savefig(path_tif + filename_tif + ".tif", bbox_inches = bbox_inches, pad_inches = pad_inches, dpi = 150)
    plt.close()
