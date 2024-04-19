import numpy as np
from my_mpi import norm_mpi, norm_sec

vector = np.array([1 for i in range(2**24)], dtype = np.complex128)

norm_sec(vector)

norm_mpi(vector)