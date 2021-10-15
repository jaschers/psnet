import numpy as np
from numba import jit, njit, vectorize
import time

a = 0

@njit()
def function(x, a):
    for i in range(x):
        a += 1
    return a

start = time.time()
function(100000000, a)
end = time.time()
print((end - start), "s")

start = time.time()
function(100000000, a)
end = time.time()
print((end - start), "s")
