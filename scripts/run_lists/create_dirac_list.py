import numpy as np
import pandas as pd

filename_input = "scripts/run_lists/gamma_run_list_complete.csv"

run = pd.read_csv(filename_input)
run = run.to_numpy().reshape(len(run))
run = np.sort(run)

filename_output = "data/gamma/event_files/Prod5_Paranal_AdvancedBaseline_NSB1x_gamma_North_20deg_ctapipe_v0.10.5_DL1_default.list"

file_output = open(filename_output, "w")

string_beginning_1 = "/vo.cta.in2p3.fr/MC/PROD5/Paranal/gamma/ctapipe-stage1-merge/2462/Data/000xxx/gamma_20deg_0deg_run"
string_beginning_2 = "/vo.cta.in2p3.fr/MC/PROD5/Paranal/gamma/ctapipe-stage1-merge/2462/Data/001xxx/gamma_20deg_0deg_run"
string_end = "___cta-prod5-paranal_desert-2147m-Paranal-dark_merged.DL1.h5"

for i in range(len(run)):
    if run[i] < 1000:
        file_output.write(string_beginning_1 + f"{run[i]}" + string_end + "\n")
    else:
        file_output.write(string_beginning_2 + f"{run[i]}" + string_end + "\n")

file_output.close()