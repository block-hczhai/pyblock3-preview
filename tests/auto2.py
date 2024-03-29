
import sys
sys.path[:0] = ['..']

from functools import reduce
import time
import numpy as np
import numba as nb
import pyblock3.algebra.funcs as pbalg
from pyblock3.algebra.mpe import MPE
from pyblock3.hamiltonian import Hamiltonian
from pyblock3.fcidump import FCIDUMP
from pyblock3.symbolic.expr import OpNames

from pyblock3.algebra.mps import MPSInfo, MPS

bdims = 200
flat = True
contract = True
fast = flat
iprint = False
dot = 2
cutoff = 1E-9

np.random.seed(1000)

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
                h_terms.append([OP_C + i * SITE + s * SPIN, OP_D + (i - 1) * SITE + s * SPIN, -1, -1])
            if i + 1 < n_sites:
                h_values.append(t)
                h_terms.append([OP_C + i * SITE + s * SPIN, OP_D + (i + 1) * SITE + s * SPIN, -1, -1])
            h_values.append(0.5 * u)
            h_terms.append([OP_C + i * SITE + s * SPIN, OP_C + i * SITE + (1 - s) * SPIN,
                            OP_D + i * SITE + (1 - s) * SPIN, OP_D + i * SITE + s * SPIN])
    return np.array(h_values, dtype=np.float64), np.array(h_terms, dtype=np.int32)


@nb.njit(nb.types.Tuple((nb.float64[:], nb.int32[:, :]))
         (nb.int32, nb.float64[:, :], nb.float64[:, :, :, :], nb.float64))
def generate_qc_terms(n_sites, h1e, g2e, cutoff=1E-9):
    OP_C, OP_D = 0 * OP, 1 * OP
    h_values = []
    h_terms = []
    for i in range(0, n_sites):
        for j in range(0, n_sites):
            t = h1e[i, j]
            if abs(t) > cutoff:
                for s in [0, 1]:
                    h_values.append(t)
                    h_terms.append([OP_C + i * SITE + s * SPIN,
                                    OP_D + j * SITE + s * SPIN, -1, -1])
    for i in range(0, n_sites):
        for j in range(0, n_sites):
            for k in range(0, n_sites):
                for l in range(0, n_sites):
                    v = g2e[i, j, k, l]
                    if abs(v) > cutoff:
                        for sij in [0, 1]:
                            for skl in [0, 1]:
                                h_values.append(0.5 * v)
                                h_terms.append([OP_C + i * SITE + sij * SPIN,
                                                OP_C + k * SITE + skl * SPIN,
                                                OP_D + l * SITE + skl * SPIN,
                                                OP_D + j * SITE + sij * SPIN])
    return np.array(h_values, dtype=np.float64), np.array(h_terms, dtype=np.int32)


def build_hubbard(u=2, t=1, n=8, cutoff=1E-9):
    fcidump = FCIDUMP(pg='c1', n_sites=n, n_elec=n,
                      twos=0, ipg=0, orb_sym=[0] * n)
    hamil = Hamiltonian(fcidump, flat=flat)

    terms = generate_hubbard_terms(n, u, t)
    return hamil, hamil.build_mpo(terms, cutoff=cutoff).to_sparse()


def build_qc(filename, pg='d2h', cutoff=1E-9, max_bond_dim=-1):
    fcidump = FCIDUMP(pg=pg).read(filename)
    hamil = Hamiltonian(fcidump, flat=flat)

    tx = time.perf_counter()
    terms = generate_qc_terms(
        fcidump.n_sites, fcidump.h1e, fcidump.g2e, cutoff)
    print('hamil term time = ', time.perf_counter() - tx, len(terms[0]))
    return hamil, hamil.build_mpo(terms, cutoff=cutoff, max_bond_dim=max_bond_dim,
        const=hamil.fcidump.const_e).to_sparse()


tx = time.perf_counter()
# fd = '../data/N2.STO3G.FCIDUMP'
# fd = '../data/H8.STO6G.R1.8.FCIDUMP'
fd = '../my_test/n2/N2.FCIDUMP'
# fd = '../data/CR2.SVP.FCIDUMP'
# hamil, mpo = build_hubbard(n=16, cutoff=cutoff)
hamil, mpo = build_qc(fd, cutoff=cutoff, max_bond_dim=-1)
print('mpo build time = ', time.perf_counter() - tx)

tx = time.perf_counter()
mps_info = MPSInfo(hamil.n_sites, hamil.vacuum, hamil.target, hamil.basis)
mps_info.set_bond_dimension(bdims)
mps = MPS.random(mps_info)
print('mps build time = ', time.perf_counter() - tx)

mps.opts = dict(cutoff=1E-12, norm_cutoff=1E-12, max_bond_dim=bdims)
if flat:
    mps = mps.to_flat()
    mpo = mpo.to_flat()

print('MPS = ', mps.show_bond_dims())
print('MPO (build) =      ', mpo.show_bond_dims())
mpo, _ = mpo.compress(left=True, cutoff=cutoff, norm_cutoff=cutoff)
print('MPO (compressed) = ', mpo.show_bond_dims())

mpe = MPE(mps, mpo, mps)


def dmrg(n_sweeps=10, tol=1E-6, dot=2):
    eners = np.zeros((n_sweeps, ))
    for iw in range(n_sweeps):
        eners[iw] = 1E10
        print("Sweep = %4d | Direction = %8s | Bond dimension = %4d" % (
            iw, "backward" if iw % 2 else "forward", bdims))
        for i in range(0, mpe.n_sites - dot + 1)[::(-1) ** iw]:
            tt = time.perf_counter()
            eff = mpe[i:i + dot]
            if contract:
                eff.ket[:] = [reduce(pbalg.hdot, eff.ket[:])]
                ener, eff, ndav, _ = eff.eigs(iprint=iprint, fast=fast)
                if dot == 2:
                    lsr = eff.ket[0].tensor_svd(idx=3, pattern='++++-+')
                    l, s, r, error = pbalg.truncate_svd(
                        *lsr, cutoff=1E-12, max_bond_dim=bdims)
                    eff.ket[:] = [np.tensordot(l, s.diag(), axes=1), r]
                else:
                    error = 0
            else:
                ener, eff, ndav, _ = eff.eigs(iprint=iprint, fast=fast)
                cket, error = eff.ket.compress(
                    cutoff=1E-12, max_bond_dim=bdims)
                eff.ket[:] = cket[:]
            mpe[i:i + dot] = eff
            eners[iw] = min(eners[iw], ener)
            print(" %3s Site = %4d-%4d .. Ndav = %4d E = %20.12f Error = %10.5g T = %8.3f" % (
                "<--" if iw % 2 else "-->", i, i + dot - 1, ndav, ener, error, time.perf_counter() - tt))
        if abs(reduce(np.subtract, eners[:iw + 1][-2:])) < tol:
            break
    return eners[iw]


tx = time.perf_counter()
print("GS Energy = %20.12f" % dmrg(dot=dot))
print('time = ', time.perf_counter() - tx)
