from QuantumRegistrer import *
import numpy as np
from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

reg = QRegistry(4)

reg.aplicar_puerta(Puerta("h"),[0])
reg.aplicar_puerta(Puerta("cx"),[0,1])


print("\n",reg.ket())

medida0 = reg.dmedir(0)

print("\n",reg.ket())

medida1 = reg.dmedir(1)

if rank == 0:
    print(reg.ket())
    print(medida0)
    print(medida1)

