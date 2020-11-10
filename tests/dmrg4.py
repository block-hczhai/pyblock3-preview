
import sys
sys.path[:0] = ['..']

import time
import numpy as np
from pyblock3.algebra.mpe import MPE, CachedMPE
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

# fd = '../data/HUBBARD-L8.FCIDUMP'
# fd = '../data/HUBBARD-L16.FCIDUMP'
# fd = '../data/N2.STO3G.FCIDUMP'
# fd = '../data/H8.STO6G.R1.8.FCIDUMP'
# fd = '../data/H10.STO6G.R1.8.FCIDUMP'
fd = '../my_test/n2/N2.FCIDUMP'
# fd = '../data/CR2.SVP.FCIDUMP'
occ = None
bond_dim = 250

# occf = '../data/CR2.SVP.OCC'
# occ = [float(x) for x in open(occf, 'r').readline().split()]

hamil = Hamiltonian(FCIDUMP(pg='d2h').read(fd), flat=True)

tx = time.perf_counter()
mpo = hamil.build_qc_mpo()
print('MPO (NC) =         ', mpo.show_bond_dims())
print('build mpo time = ', time.perf_counter() - tx)

tx = time.perf_counter()
mpo, _ = mpo.compress(left=True, cutoff=1E-9, norm_cutoff=1E-9)
print('MPO (compressed) = ', mpo.show_bond_dims())
print('compress mpo time = ', time.perf_counter() - tx)

mps = hamil.build_mps(bond_dim, occ=occ)
print('MPS = ', mps.show_bond_dims())

bdims = [250] * 5 + [500] * 5
noises = [1E-5] * 2 + [1E-6] * 4 + [1E-7] * 3 + [0]
davthrds = [5E-3] * 4 + [1E-3] * 4 + [1E-4]

dmrg = CachedMPE(mps, mpo, mps).dmrg(bdims=bdims, noises=noises,
                              dav_thrds=davthrds, iprint=2, n_sweeps=20)
ener = dmrg.energies[-1]
print("Energy = %20.12f" % ener)
