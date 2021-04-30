import numpy as np
from astropy.table import Table, join, vstack, Column



arr = np.arange(15).reshape(5, 3)
t = Table(arr, names=('a', 'b', 'c'), meta={'keywords': {'key1': 'val1'}})
t['a'] = [1, -2, 3, -4, 5]  # Set all column values
t['a'][2] = 30              # Set row 2 of column 'a'
t[1] = (8, 9, 10)           # Set all row values
t[1]['b'] = -9              # Set column 'b' of row 1
t[0:3]['c'] = 100           # Set column 'c' of rows 0, 1, 2

t[[1, 2]]['a'] = [3., 5.]             # doesn't change table t
t[np.array([1, 2])]['a'] = [3., 5.]   # doesn't change table t
t[np.where(t['a'] > 3)]['a'] = 3.     # doesn't change table t

t['a'][[1, 2]] = [3., 5.]
t['a'][np.array([1, 2])] = [3., 5.]
t['a'][np.where(t['a'] > 3)] = 3.

t['d1'] = np.arange(5)
t['d2'] = [1, 2, 3, 4, 5]
t['d3'] = 6  # all 5 rows set to 6

aa = Column(np.arange(5), name='aa')
t.add_column(aa, index=0)  # Insert before the first table column

print(t.info())
# print(t.colnames)
# print(t.remove_columns(['aa', 'd1', 'd2', 'd3']))

# t.keep_columns(['a', 'b'])

"""
# load data 
input_filename = "cta/data/event-files/gamma_20deg_0deg_run1069___cta-prod5-paranal_desert-2147m-Paranal-dark_merged.DL1.h5"

source = EventSource(input_filename)
print(f"Total number of events: {len(source)}")

# get tables for all SST telescopes that include images and corresponding obs_id + event_id for each event
simulated_parameter_table = read_table(input_filename, "/simulation/event/subarray/shower")[0:5]

print(simulated_parameter_table)
print(simulated_parameter_table["obs_id"])
print(type(simulated_parameter_table))
print(type(simulated_parameter_table["obs_id"]))
print(simulated_parameter_table.remove_columns("obs_id"))
"""