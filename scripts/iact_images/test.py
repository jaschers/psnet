import numpy as np
import csv

a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

print(np.shape(a))

a = np.reshape(a, (len(a), 1))
b = np.reshape(b, (len(b), 1))

c = np.hstack((a, b))

with open('test.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["log10(E_true / GeV)", "log10(E_rec / GeV)"])
    writer.writerows(c)


# ascii.write(complete_table_tel_combined, f"dm-finder/data/{particle_type}/tables/{particle_type}_20deg_0deg_run{run[r]}___cta-prod5-paranal_desert-2147m-Paranal-dark_merged.DL1.csv", format = "csv", fast_writer = False, overwrite = True)
