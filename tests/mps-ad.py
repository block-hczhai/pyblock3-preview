

import sys

from pyblock3.algebra import ad
sys.path[:0] = ['..', "../build"]

import time
import numpy as np
from pyblock3.algebra.mpe import MPE
from pyblock3.hamiltonian import Hamiltonian
from pyblock3.fcidump import FCIDUMP

fd = '../data/HUBBARD-L4.FCIDUMP'
bond_dim = 250

hamil = Hamiltonian(FCIDUMP(pg='d2h').read(fd), flat=True)

tx = time.perf_counter()
mpo = hamil.build_qc_mpo()
print('MPO (NC) =         ', mpo.show_bond_dims())
print('build mpo time = ', time.perf_counter() - tx)

tx = time.perf_counter()
mpo, _ = mpo.compress(left=True, cutoff=1E-9, norm_cutoff=1E-9)
print('MPO (compressed) = ', mpo.show_bond_dims())

mps = hamil.build_mps(bond_dim)
print('MPS = ', mps.show_bond_dims())

bdims = [250] * 5 + [500] * 5
noises = [1E-5] * 2 + [1E-6] * 4 + [1E-7] * 3 + [0]
davthrds = [5E-3] * 4 + [1E-3] * 4 + [1E-4]

dmrg = MPE(mps, mpo, mps).dmrg(bdims=bdims, noises=noises, dav_thrds=davthrds, iprint=2, n_sweeps=10)
ener = dmrg.energies[-1]
print("Energy = %20.12f" % ener)

print(np.dot(mps, mpo @ mps))

from pyblock3.algebra.ad.mps import MPS
import jax

admps = MPS.from_flat(mps)
admpo = MPS.from_flat(mpo)

key = jax.random.PRNGKey(0)

for ts in admps.tensors:
    key, subkey = jax.random.split(key)
    ts.data = jax.random.uniform(subkey, ts.data.shape)

print('init energy = ', np.dot(admps, admpo @ admps) / np.dot(admps, admps))

def get_energy(admps):
    return np.dot(admps, admpo @ admps) / np.dot(admps, admps)

get_grad = jax.grad(get_energy)

def update(admps, dstep=0.5):
    grads = get_grad(admps)
    return jax.tree_util.tree_map(lambda p, g: p - dstep * g, admps, grads)

for i in range(100):
    admps = update(admps)
    print("step %3d / 100 :: energy = %20.15f" % (i, get_energy(admps)))
