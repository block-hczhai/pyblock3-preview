
import sys
sys.path[:0] = ['..']

import time
import numpy as np
from pyblock3.algebra.mpe import MPE, CachedMPE
from pyblock3.hamiltonian import Hamiltonian
from pyblock3.fcidump import FCIDUMP
import pickle
mpi = True
if mpi:
    from mpi4py import MPI
    mrank = MPI.COMM_WORLD.Get_rank()
    msize = MPI.COMM_WORLD.Get_size()
    comm = MPI.COMM_WORLD
else:
    mrank = 0
    msize = 1

print(mrank, msize)

import ctypes
mkl_rt = ctypes.CDLL('libmkl_rt.so')
mkl_set_num_threads = mkl_rt.MKL_Set_Num_Threads
mkl_get_max_threads = mkl_rt.MKL_Get_Max_Threads

np.random.seed(1000)

scratch = './tmp'
n_threads = 4

import os
if not mpi or mrank == 0:
    if not os.path.isdir(scratch):
        os.mkdir(scratch)
if mpi:
    comm.barrier()
os.environ['TMPDIR'] = scratch
os.environ['OMP_NUM_THREADS'] = str(n_threads)

# fd = '../data/HUBBARD-L8.FCIDUMP'
# fd = '../data/HUBBARD-L16.FCIDUMP'
# fd = '../data/N2.STO3G.FCIDUMP'
# fd = '../data/H8.STO6G.R1.8.FCIDUMP'
# fd = '../data/H10.STO6G.R1.8.FCIDUMP'
# fd = '../my_test/n2/N2.FCIDUMP'
# fd = '../data/N2.CCPVDZ.FCIDUMP'
fd = '../data/CR2.SVP.FCIDUMP'
occ = None
bond_dim = 250

occf = '../data/CR2.SVP.OCC'
occ = [float(x) for x in open(occf, 'r').readline().split()]

hamil = Hamiltonian(FCIDUMP(pg='d2h').read(fd).parallelize(), flat=True)

# os.environ['MKL_NUM_THREADS'] = str(1)

tx = time.perf_counter()
mpo = hamil.build_qc_mpo()
print('rank = %2d MPO (NC) =         ' % mrank, mpo.show_bond_dims())
print('rank = %2d build mpo time = ' % mrank, time.perf_counter() - tx)

tx = time.perf_counter()
mpo, _ = mpo.compress(left=True, cutoff=1E-9, norm_cutoff=1E-9)
print('rank = %2d MPO (compressed) = ' % mrank, mpo.show_bond_dims())
# print('rank = %2d compress mpo time = ' % mrank, time.perf_counter() - tx)

# mkl_set_num_threads(n_threads)
# print(mkl_get_max_threads())

if not mpi or mrank == 0:
    mps = hamil.build_mps(bond_dim, occ=occ)
if mpi:
    mps = comm.bcast(mps if mrank == 0 else None, root=0)
print('MPS = ', mps.show_bond_dims())

bdims = [250] * 5 + [500] * 5
noises = [1E-5] * 2 + [1E-6] * 4 + [1E-7] * 3 + [0]
davthrds = [5E-3] * 4 + [1E-3] * 4 + [1E-4]

dmrg = CachedMPE(mps, mpo, mps, mpi=mpi).dmrg(bdims=bdims, noises=noises,
                                     dav_thrds=davthrds, iprint=2, n_sweeps=10)
ener = dmrg.energies[-1]
print("Energy = %20.12f" % ener)
