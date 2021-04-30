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

from time import sleep
from tqdm import tqdm
for i in tqdm(range(10000)):
    pass

obs = Table.read("""name    obs_date    mag_b  mag_v
                     M31     2012-01-02  17.0   17.5
                     M31     2012-01-02  17.1   17.4
                     M101    2012-01-02  15.1   13.5
                     M82     2012-02-14  16.2   14.5
                     M31     2012-02-14  16.9   17.3
                     M82     2012-02-14  15.2   15.5
                     M101    2012-02-14  15.0   13.6
                     M82     2012-03-26  15.7   16.5
                     M101    2012-03-26  15.1   13.5
                     M101    2012-03-26  14.8   14.3
                     """, format='ascii')

arr = np.arange(15).reshape(5, 3)
table = Table(arr, names=('name', 'b', 'c'), meta={'keywords': {'key1': 'val1'}})

print(obs)
array = np.zeros((10, 100))
obs["image combined"] = np.zeros((10, 100))
print(obs)

a = [[]]*10
print(a)
# print(array)
# print(array.shape)
# obs.add_column(np.zeros((100, 10)), name = "image combined")
# print(obs)
