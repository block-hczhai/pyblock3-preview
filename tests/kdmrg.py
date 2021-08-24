
import sys
sys.path[:0] = ['..']

import time
import numpy as np
from pyblock3.algebra.mpe import CachedMPE
from pyblock3.hamiltonian import Hamiltonian
from pyblock3.fcidump import FCIDUMP

np.random.seed(1000)

scratch = './tmp'
n_threads = 16

import os
if not os.path.isdir(scratch):
    os.mkdir(scratch)
os.environ['TMPDIR'] = scratch
os.environ['OMP_NUM_THREADS'] = str(n_threads)

def build_rspace_hubbard(u=4, t=1, n=16, cutoff=1E-9, n_elec=None, filename=None):
    n_elec = n if n_elec is None else n_elec
    fcidump = FCIDUMP(pg='c1', n_sites=n, n_elec=n_elec, twos=n_elec % 2, ipg=0, orb_sym=[0] * n)
    hamil = Hamiltonian(fcidump, flat=True)

    def generate_terms(n_sites, c, d):
        for i in range(0, n_sites):
            for s in [0, 1]:
                yield t * c[i, s] * d[(i - 1 + n_sites) % n_sites, s]
                yield t * c[i, s] * d[(i + 1) % n_sites, s]
            yield u * (c[i, 0] * c[i, 1] * d[i, 1] * d[i, 0])
    if filename is not None:
        FCIDUMP(n_sites=n, n_elec=n_elec, twos=n_elec % 2).build(generate_terms).write(filename)
    return hamil, hamil.build_mpo(generate_terms, cutoff=cutoff).to_sparse()

def build_kspace_hubbard(u=4, t=1, n=16, cutoff=1E-9, n_elec=None, filename=None):
    n_elec = n if n_elec is None else n_elec
    fcidump = FCIDUMP(pg='c1', n_sites=n, n_elec=n_elec, twos=n_elec % 2, ipg=0, orb_sym=[0] * n)
    hamil = Hamiltonian(fcidump, flat=True)

    def generate_terms(n_sites, c, d):
        for k in range(0, n_sites):
            for s in [0, 1]:
                yield (-2 * t * np.cos(2 * k * np.pi / n_sites)) * c[k, s] * d[k, s]
        for k in range(0, n_sites):
            for k2 in range(0, n_sites):
                for k3 in range(0, n_sites):
                    for k4 in range(0, n_sites):
                        if (k - k2 + k3 - k4) % n_sites == 0:
                            for sij in [0, 1]:
                                for skl in [0, 1]:
                                    yield (u / n_sites / 2) * (c[k, sij] * c[k3, skl] * d[k4, skl] * d[k2, sij])
    if filename is not None:
        FCIDUMP(n_sites=n, n_elec=n_elec, twos=n_elec % 2, general=True).build(generate_terms).write(filename)

    return hamil, hamil.build_mpo(generate_terms, cutoff=cutoff).to_sparse()

# hamil = Hamiltonian(FCIDUMP(pg='d2h').read('FCIDUMP'), flat=True)
# mpo = hamil.build_qc_mpo().to_sparse()
hamil, mpo = build_rspace_hubbard(n=8, t=1, u=2, n_elec=8, filename=None)

bond_dim = 500
mrank = 0

tx = time.perf_counter()
print('rank = %2d MPO (NC) =         ' % mrank, mpo.show_bond_dims())
print('rank = %2d build mpo time = ' % mrank, time.perf_counter() - tx)

tx = time.perf_counter()
mpo, _ = mpo.compress(left=True, cutoff=1E-9, norm_cutoff=1E-9)
print('rank = %2d MPO (compressed) = ' % mrank, mpo.show_bond_dims())

mps = hamil.build_mps(bond_dim)
print('MPS = ', mps.show_bond_dims())

bdims = [500] * 5 + [1000] * 5
noises = [1E-4] * 2 + [1E-5] * 4 + [1E-7] * 3 + [0]
davthrds = [5E-3] * 4 + [1E-3] * 4 + [1E-4]

dmrg = CachedMPE(mps, mpo, mps).dmrg(bdims=bdims, noises=noises,
                                     dav_thrds=davthrds, iprint=2, n_sweeps=10)
ener = dmrg.energies[-1]
print("Energy = %20.12f" % ener)
