
import sys
sys.path[:0] = ['..']

bdims = 250
flat = True
contract = True
fast = flat
iprint = False
dot = 2
cutoff = 1E-9
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

import pyblock3.algebra.funcs as pbalg
from pyblock3.algebra.mpe import MPE, CachedMPE
from pyblock3.hamiltonian import Hamiltonian
from pyblock3.fcidump import FCIDUMP
from pyblock3.symbolic.expr import OpNames
from pyblock3.algebra.mps import MPSInfo, MPS
from functools import reduce
import time
import numpy as np
import numba as nb
import os

np.random.seed(1000)

scratch = './tmp'
n_threads = 4

if not mpi or mrank == 0:
    if not os.path.isdir(scratch):
        os.mkdir(scratch)
if mpi:
    comm.barrier()

os.environ['TMPDIR'] = scratch
os.environ['OMP_NUM_THREADS'] = str(n_threads)
os.environ['MKL_NUM_THREADS'] = str(n_threads)

SPIN, SITE, OP = 1, 2, 16384
@nb.njit(nb.types.Tuple((nb.float64[:], nb.int32[:, :]))(nb.int32, nb.float64, nb.float64))
def generate_hubbard_terms(n_sites, u, t):
    OP_C, OP_D = 0 * OP, 1 * OP
    h_values = []
    h_terms = []
    for i in range(0, n_sites):
        for s in [0, 1]:
            if i - 1 >= 0:
                h_values.append(t)
                h_terms.append([OP_C + i * SITE + s * SPIN,
                                OP_D + (i - 1) * SITE + s * SPIN, -1, -1])
            if i + 1 < n_sites:
                h_values.append(t)
                h_terms.append([OP_C + i * SITE + s * SPIN,
                                OP_D + (i + 1) * SITE + s * SPIN, -1, -1])
            h_values.append(0.5 * u)
            h_terms.append([OP_C + i * SITE + s * SPIN, OP_C + i * SITE + (1 - s) * SPIN,
                            OP_D + i * SITE + (1 - s) * SPIN, OP_D + i * SITE + s * SPIN])
    return np.array(h_values, dtype=np.float64), np.array(h_terms, dtype=np.int32)


@nb.njit(nb.boolean(nb.int32, nb.int32, nb.int32, nb.int32))
def qc_hamil_partition(i, j, k, l):
    if j == -1:
        return mrank == i % msize
    elif k == -1:
        if i <= j:
            return mrank == (j * (j + 1) // 2 + i) % msize
        else:
            return mrank == (i * (i + 1) // 2 + j) % msize
    else:
        x = np.array([i, j, k, l], dtype=np.int32)
        x.sort()
        if x[1] == x[2]:
            return mrank == x[1] % msize
        else:
            i, j = x[0], x[1]
            if i <= j:
                return mrank == (j * (j + 1) // 2 + i) % msize
            else:
                return mrank == (i * (i + 1) // 2 + j) % msize

@nb.njit(nb.types.Tuple((nb.float64[:], nb.int32[:, :]))
         (nb.int32, nb.float64[:, :], nb.float64[:, :, :, :], nb.float64))
def generate_qc_terms(n_sites, h1e, g2e, cutoff=1E-9):
    OP_C, OP_D = 0 * OP, 1 * OP
    h_values = []
    h_terms = []
    for i in range(0, n_sites):
        for j in range(0, n_sites):
            t = h1e[i, j]
            if abs(t) > cutoff and qc_hamil_partition(i, -1, -1, -1):
                for s in [0, 1]:
                    h_values.append(t)
                    h_terms.append([OP_C + i * SITE + s * SPIN,
                                    OP_D + j * SITE + s * SPIN, -1, -1])
    for i in range(0, n_sites):
        for j in range(0, n_sites):
            for k in range(0, n_sites):
                for l in range(0, n_sites):
                    v = g2e[i, j, k, l]
                    if abs(v) > cutoff and qc_hamil_partition(i, j, k, l):
                        for sij in [0, 1]:
                            for skl in [0, 1]:
                                h_values.append(0.5 * v)
                                h_terms.append([OP_C + i * SITE + sij * SPIN,
                                                OP_C + k * SITE + skl * SPIN,
                                                OP_D + l * SITE + skl * SPIN,
                                                OP_D + j * SITE + sij * SPIN])
    if len(h_values) == 0:
        return np.zeros((0, ), dtype=np.float64), np.zeros((0, 4), dtype=np.int32)
    else:
        return np.array(h_values, dtype=np.float64), np.array(h_terms, dtype=np.int32)


def build_hubbard(u=2, t=1, n=8, cutoff=1E-9):
    fcidump = FCIDUMP(pg='c1', n_sites=n, n_elec=n,
                      twos=0, ipg=0, orb_sym=[0] * n)
    hamil = Hamiltonian(fcidump, flat=flat)

    terms = generate_hubbard_terms(n, u, t)
    return hamil, hamil.build_mpo(terms, cutoff=cutoff).to_sparse()


def build_qc(filename, pg='d2h', cutoff=1E-9, max_bond_dim=-1):
    fcidump = FCIDUMP(pg=pg).read(filename)
    if mrank != 0:
        fcidump.const_e = 0
    hamil = Hamiltonian(fcidump, flat=flat)

    tx = time.perf_counter()
    if fcidump.uhf:
        terms = generate_qc_terms(
            fcidump.n_sites, fcidump.h1e[0], fcidump.g2e[0], cutoff)
    else:
        terms = generate_qc_terms(
            fcidump.n_sites, fcidump.h1e, fcidump.g2e, cutoff)
    print('hamil term time = ', time.perf_counter() - tx, len(terms[0]))
    return hamil, hamil.build_mpo(terms, cutoff=cutoff, max_bond_dim=max_bond_dim,
        const=hamil.fcidump.const_e).to_sparse()

# os.environ['MKL_NUM_THREADS'] = str(1)

tx = time.perf_counter()
# fd = '../data/HUBBARD-L8.FCIDUMP'
# fd = '../data/N2.STO3G.FCIDUMP'
# fd = '../../block2-old/data/H4.STO6G.R1.8.FCIDUMP'
# fd = '../data/H4.TEST'
# fd = '../data/H8.STO6G.R1.8.FCIDUMP'
# fd = '../my_test/n2/N2.FCIDUMP'
fd = '../my_test/n2/N2.CCPVDZ.FCIDUMP'
# fd = '../data/CR2.SVP.FCIDUMP'
# fd = '/home/hczhai/work/block2-old/my_test/cuprate3/prepare/REV.FCIDUMP'
# hamil, mpo = build_hubbard(n=32, cutoff=cutoff)
hamil, mpo = build_qc(fd, cutoff=cutoff, max_bond_dim=-9)
print('rank = %4d mpo build time = %.3f' % (mrank, time.perf_counter() - tx))

# os.environ['MKL_NUM_THREADS'] = str(n_threads)

tx = time.perf_counter()
if not mpi or mrank == 0:
    mps_info = MPSInfo(hamil.n_sites, hamil.vacuum, hamil.target, hamil.basis)
    mps_info.set_bond_dimension(bdims)
    mps = MPS.random(mps_info)
    print('mps build time = ', time.perf_counter() - tx)
    mps.opts = dict(cutoff=1E-12, norm_cutoff=1E-12, max_bond_dim=bdims)
if mpi:
    mps = comm.bcast(mps if mrank == 0 else None, root=0)
if flat:
    mps = mps.to_flat()
    mpo = mpo.to_flat()

print('rank = %4d MPO (build) =      ' % mrank, mpo.show_bond_dims())
print('MPS = ', mps.show_bond_dims())
# mpo, _ = mpo.compress(left=True, cutoff=cutoff, norm_cutoff=cutoff)
# comm.barrier()
# for i in range(msize):
#     if i == mrank:
#         print('rank = %4d MPO (compressed) = ' % mrank, mpo.show_bond_dims(), flush=True)
#     comm.barrier()

bdims = [250] * 5 + [500] * 5
noises = [1E-5] * 15 + [0]
davthrds = [5E-3] * 4 + [1E-3] * 4 + [1E-4] * 2 + [1E-5] * 2 + [1E-6] * 2

dmrg = CachedMPE(mps, mpo, mps, mpi=mpi).dmrg(bdims=bdims, noises=noises,
                                     dav_thrds=davthrds, iprint=2, n_sweeps=1)
ener = dmrg.energies[-1]
if mrank == 0:
    print("Energy = %20.12f" % ener)
