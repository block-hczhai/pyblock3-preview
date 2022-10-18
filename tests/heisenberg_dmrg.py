
# twos = 2 ===> S=1 Heisenberg model

import sys
sys.path[:0] = ['..']

from pyblock3.algebra.mpe import CachedMPE
from pyblock3.heisenberg import Heisenberg
import numpy as np
import os

np.random.seed(1000)

scratch = './tmp'
n_threads = 4

if not os.path.isdir(scratch):
    os.mkdir(scratch)

os.environ['TMPDIR'] = scratch
os.environ['OMP_NUM_THREADS'] = str(n_threads)
os.environ['MKL_NUM_THREADS'] = str(n_threads)

n = 100
topology = [(i, i + 1, 1.0) for i in range(n - 1)]
hamil = Heisenberg(twos=2, n_sites=n, topology=topology, flat=False)

mpo = hamil.build_mpo().to_sparse().to_flat()
mps = hamil.build_mps(100).to_flat()

print('MPS = ', mps.show_bond_dims())
print('MPO = ', mpo.show_bond_dims())

bdims = [200] * 5
noises = [1E-5] * 2 + [1E-6] * 4 + [1E-7] * 3 + [0]
davthrds = [5E-3] * 4 + [1E-3] * 4 + [1E-4]

dmrg = CachedMPE(mps, mpo, mps).dmrg(bdims=bdims, noises=noises,
                                     dav_thrds=davthrds, iprint=2, n_sweeps=10)
ener = dmrg.energies[-1]
print("Energy = %20.12f" % ener)

# itensor  -138.94008614267705 (M=200, S=1)
# pyblock3 -138.940086143382   (M=200, S=1)

# itensor  -44.12773603118644  (M=200, S=1/2)
# pyblock3 -44.127739893217    (M=200, S=1/2)
