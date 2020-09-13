
import sys
sys.path[:0] = ['..', "../../block2/build"]

from pyblock3.aux.hamil import HamilTools
from pyblock3.algebra.mpe import MPE
import numpy as np
import time
from functools import reduce

with HamilTools.from_fcidump('../data/N2.STO3G.FCIDUMP') as hamil:
    mps = hamil.get_ground_state_mps(bond_dim=250)
    mpo = hamil.get_mpo()

print('MPO = ', mpo.show_bond_dims())
mpo, _ = mpo.compress(cutoff=1E-12)
print('MPO = ', mpo.show_bond_dims())

print('MPS = ', mps.show_bond_dims())
print(mps.norm())

print(mps @ (mpo @ mps))

hmps = mpo @ mps
print(hmps.show_bond_dims())
print(hmps.norm())

hmps, _ = hmps.compress(cutoff=1E-12)
print(hmps.show_bond_dims())
print(hmps.norm())

cmpo, cps_error = mpo.compress(max_bond_dim=50, cutoff=1E-12)
print('error = ', cps_error)
print(cmpo.show_bond_dims())

hmps = cmpo @ mps
print(hmps.show_bond_dims())
print(hmps.norm())

print(sorted(hmps[0].infos[1]))

smps = mps.to_sliceable()
print(smps[0])
print(smps[0][:, 2:, 2])
print(smps[0][:, :2, 2].infos)
print(smps[0])
print(smps.amplitude([3, 3, 0, 3, 0, 3, 3, 3, 3, 0]))
print(smps.amplitude([3, 3, 0, 0, 0, 3, 3, 3, 3, 0]))

import itertools, time

ts = 0
coeffs = []
for ocp in itertools.combinations(range(10), 7):
    det = [0] * mps.n_sites
    for t in ocp:
        det[t] = 3
    tt = time.perf_counter()
    coeffs.append(smps.amplitude(det))
    ts += time.perf_counter() - tt
    print(np.array(det), "%10.5f" % coeffs[-1])
print('time = ', ts)

print((np.array(coeffs) ** 2).sum())
