from mpi4py import MPI
import numpy as np
from time import time

def norm_sec(vec):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        t0 = time()
        res = np.sqrt(sum(vec**2))
        t1 = time()
        print(res)
        print("Tiempo secuencial:",t1 - t0)
        return res

def suma_cuadrado_mpi(vec):
    comm = MPI.COMM_WORLD
    n_ranks = comm.Get_size()
    rank = comm.Get_rank()

    datos = None
    if rank == 0:
        datos = vec
    
    result = np.empty(1, dtype = np.complex128)

    porcion = np.empty(vec.size//n_ranks, dtype = np.complex128)

    comm.Scatter(datos, porcion, root = 0)

    porcion = np.array(sum(np.absolute(porcion)**2), dtype = np.complex128)

    comm.Allreduce(porcion, result, op = MPI.SUM)

    return result[0]
