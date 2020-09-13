
import sys
sys.path[:0] = ['..', "../../block2-old/build"]

from pyblock3.aux.hamil import HamilTools
from pyblock3.algebra.integrate import rk4_apply
import numpy as np
import time
from functools import reduce

# ancilla

with HamilTools.from_fcidump('../data/H8.STO6G.R1.8.FCIDUMP') as hamil:
    mps = hamil.get_thermal_limit_mps()
    mpo = hamil.get_mpo(mu=-1.0, ancilla=True)
    mpo.const = 0.0

print('MPS = ', mps.show_bond_dims())
print('MPO = ', mpo.show_bond_dims())
mpo, error = mpo.compress(cutoff=1E-8)
print('MPO = ', mpo.show_bond_dims(), error)

init_e = np.dot(mps, mpo @ mps) / np.dot(mps, mps)
print('Initial Energy = ', init_e)
print('Error          = ', init_e - 0.3124038410492045)

mps.opts = dict(max_bond_dim=200, cutoff=1E-8)
beta = 0.01
tt = time.perf_counter()
fmps = rk4_apply((-beta / 2) * mpo, mps)
ener = np.dot(fmps, mpo @ fmps) / np.dot(fmps, fmps)
print('time = ', time.perf_counter() - tt)
print('Energy = ', ener)
print('Error  = ', ener - 0.2408363230374028)

# mpdo

with HamilTools.from_fcidump('../data/H8.STO6G.R1.8.FCIDUMP') as hamil:
    mps = hamil.get_thermal_limit_mps()
    mpo = hamil.get_mpo(mu=-1.0, ancilla=False)
    mpo.const = 0.0

mps.tensors = [a.hdot(b) for a, b in zip(mps.tensors[0::2], mps.tensors[1::2])]

print('MPS = ', mps.show_bond_dims())
print('MPO = ', mpo.show_bond_dims())
mpo, _ = mpo.compress(cutoff=1E-8)
print('MPO = ', mpo.show_bond_dims())

init_e = np.dot(mps, mpo @ mps) / np.dot(mps, mps)
print('Initial Energy = ', init_e)
print('Error          = ', init_e - 0.3124038410492045)

mps.opts = dict(max_bond_dim=200, cutoff=1E-8)
beta = 0.01
tt = time.perf_counter()
fmps = rk4_apply((-beta / 2) * mpo, mps)
ener = np.dot(fmps, mpo @ fmps) / np.dot(fmps, fmps)
print('time = ', time.perf_counter() - tt)
print('Energy = ', ener)
print('Error  = ', ener - 0.2408363230374028)
