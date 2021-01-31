
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

if not os.path.isdir(scratch):
    os.mkdir(scratch)

os.environ['TMPDIR'] = scratch
os.environ['OMP_NUM_THREADS'] = str(n_threads)
os.environ['MKL_NUM_THREADS'] = str(n_threads)

SPIN, SITE, OP = 1, 2, 16384

method = "ij"

if method == "ij":
    @nb.njit(nb.boolean(nb.int32, nb.int32, nb.int32, nb.int32, nb.int32, nb.int32))
    def qc_hamil_partition(i, j, k, l, mrank, msize):
        if k == -1:
            return mrank == i % msize
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
elif method == "ijx":
    @nb.njit(nb.boolean(nb.int32, nb.int32, nb.int32, nb.int32, nb.int32, nb.int32))
    def qc_hamil_partition(i, j, k, l, mrank, msize):
        if k == -1:
            return mrank == i % msize
        else:
            x = np.array([i, j, k, l], dtype=np.int32)
            x.sort()
            if 59 - x[2] >= x[1]:
                i, j = x[2], x[3]
            else:
                i, j = x[0], x[1]
            if i <= j:
                return mrank == (j * (j + 1) // 2 + i) % msize
            else:
                return mrank == (i * (i + 1) // 2 + j) % msize
elif method == "kl":
    @nb.njit(nb.boolean(nb.int32, nb.int32, nb.int32, nb.int32, nb.int32, nb.int32))
    def qc_hamil_partition(i, j, k, l, mrank, msize):
        if k == -1:
            return mrank == i % msize
        else:
            x = np.array([i, j, k, l], dtype=np.int32)
            x.sort()
            if x[1] == x[2]:
                return mrank == x[1] % msize
            else:
                i, j = x[2], x[3]
                if i <= j:
                    return mrank == (j * (j + 1) // 2 + i) % msize
                else:
                    return mrank == (i * (i + 1) // 2 + j) % msize
elif method == "i":
    @nb.njit(nb.boolean(nb.int32, nb.int32, nb.int32, nb.int32, nb.int32, nb.int32))
    def qc_hamil_partition(i, j, k, l, mrank, msize):
        return mrank == i % msize
elif method == "j":
    @nb.njit(nb.boolean(nb.int32, nb.int32, nb.int32, nb.int32, nb.int32, nb.int32))
    def qc_hamil_partition(i, j, k, l, mrank, msize):
        return mrank == j % msize
elif method == "l":
    @nb.njit(nb.boolean(nb.int32, nb.int32, nb.int32, nb.int32, nb.int32, nb.int32))
    def qc_hamil_partition(i, j, k, l, mrank, msize):
        if k == -1:
            return mrank == i % msize
        else:
            return mrank == l % msize
elif method == "si":
    @nb.njit(nb.boolean(nb.int32, nb.int32, nb.int32, nb.int32, nb.int32, nb.int32))
    def qc_hamil_partition(i, j, k, l, mrank, msize):
        if k == -1:
            x = np.array([i, j], dtype=np.int32)
            x.sort()
            return mrank == x[0] % msize
        else:
            x = np.array([i, j, k, l], dtype=np.int32)
            x.sort()
            return mrank == x[0] % msize
elif method == "sj":
    @nb.njit(nb.boolean(nb.int32, nb.int32, nb.int32, nb.int32, nb.int32, nb.int32))
    def qc_hamil_partition(i, j, k, l, mrank, msize):
        if k == -1:
            x = np.array([i, j], dtype=np.int32)
            x.sort()
            return mrank == x[0] % msize
        else:
            x = np.array([i, j, k, l], dtype=np.int32)
            x.sort()
            return mrank == x[1] % msize
elif method == "sk":
    @nb.njit(nb.boolean(nb.int32, nb.int32, nb.int32, nb.int32, nb.int32, nb.int32))
    def qc_hamil_partition(i, j, k, l, mrank, msize):
        if k == -1:
            x = np.array([i, j], dtype=np.int32)
            x.sort()
            return mrank == x[1] % msize
        else:
            x = np.array([i, j, k, l], dtype=np.int32)
            x.sort()
            return mrank == x[2] % msize
elif method == "sl":
    @nb.njit(nb.boolean(nb.int32, nb.int32, nb.int32, nb.int32, nb.int32, nb.int32))
    def qc_hamil_partition(i, j, k, l, mrank, msize):
        if k == -1:
            x = np.array([i, j], dtype=np.int32)
            x.sort()
            return mrank == x[1] % msize
        else:
            x = np.array([i, j, k, l], dtype=np.int32)
            x.sort()
            return mrank == x[3] % msize

@nb.njit(nb.types.Tuple((nb.float64[:], nb.int32[:, :]))
         (nb.int32, nb.int32, nb.int32, nb.float64[:, :], nb.float64[:, :, :, :], nb.float64))
def generate_qc_terms(mrank, msize, n_sites, h1e, g2e, cutoff=1E-9):
    OP_C, OP_D = 0 * OP, 1 * OP
    h_values = []
    h_terms = []
    for i in range(0, n_sites):
        for j in range(0, n_sites):
            t = h1e[i, j]
            if abs(t) > cutoff and qc_hamil_partition(i, j, -1, -1, mrank, msize):
                for s in [0, 1]:
                    h_values.append(t)
                    h_terms.append([OP_C + i * SITE + s * SPIN,
                                    OP_D + j * SITE + s * SPIN, -1, -1])
    for i in range(0, n_sites):
        for j in range(0, n_sites):
            for k in range(0, n_sites):
                for l in range(0, n_sites):
                    v = g2e[i, j, k, l]
                    if abs(v) > cutoff and qc_hamil_partition(i, j, k, l, mrank, msize):
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

def build_qc(filename, mrank, msize, pg='d2h', cutoff=1E-9, max_bond_dim=-1):
    fcidump = FCIDUMP(pg=pg).read(filename)
    if mrank != 0:
        fcidump.const_e = 0
    hamil = Hamiltonian(fcidump, flat=flat)

    tx = time.perf_counter()
    if fcidump.uhf:
        terms = generate_qc_terms(mrank, msize,
            fcidump.n_sites, fcidump.h1e[0], fcidump.g2e[0], cutoff)
    else:
        terms = generate_qc_terms(mrank, msize,
            fcidump.n_sites, fcidump.h1e, fcidump.g2e, cutoff)
    print('hamil term time = ', time.perf_counter() - tx, len(terms[0]))
    tx = time.perf_counter()
    mm = hamil.build_mpo(terms, cutoff=cutoff, max_bond_dim=max_bond_dim).to_sparse()
    return hamil, mm, time.perf_counter() - tx

tx = time.perf_counter()
# fd = '../data/HUBBARD-L8.FCIDUMP'
# fd = '../data/N2.STO3G.FCIDUMP'
# fd = '../data/H8.STO6G.R1.8.FCIDUMP'
# fd = '../my_test/n2/N2.FCIDUMP'
# fd = '../my_test/n2/N2.CCPVDZ.FCIDUMP'
# fd = '../data/CR2.SVP.FCIDUMP'
fd = '/home/hczhai/work/block2-old/my_test/cuprate3/prepare/REV.FCIDUMP'
# hamil, mpo = build_hubbard(n=32, cutoff=cutoff)

msize = 16
mmz = 0
mtz = 0
for mrank in range(0, msize):
    hamil, mpo, tmpo = build_qc(fd, mrank, msize, cutoff=cutoff, max_bond_dim=-5)
    print(mrank, mpo.bond_dim, tmpo, mpo.show_bond_dims())
    mmz = max(mmz, mpo.bond_dim)
    mtz = max(mtz, tmpo)

print(mmz, mtz)
