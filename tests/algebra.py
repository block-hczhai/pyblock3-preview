
import sys
sys.path[:0] = ['..', "../../block2-old/build_seq"]

from pyblock3.block2.hamil import HamilTools
import numpy as np
import time
from functools import reduce

flat = True

# hubbard

with HamilTools.hubbard(n_sites=8, u=2, t=1) as hamil:
    mps = hamil.get_ground_state_mps(bond_dim=100)
    mpo = hamil.get_mpo()

print('MPO (NC) =         ', mpo.show_bond_dims())
mpo, _ = mpo.compress(left=True, cutoff=1E-12)
print('MPO (compressed) = ', mpo.show_bond_dims())

if flat:
    mps = mps.to_flat()
    mpo = mpo.to_flat()

print('MPS energy = ', mps @ mpo.T @ mps)

print('MPS = ', mps.show_bond_dims())
print('MPS norm = ', mps.norm())

print('2 MPS = ', (2 * mps).show_bond_dims())
print((2 * mps).norm())

print(mps[0])

print((2 * mps)[0])

mps_add = mps + mps
print('MPS + MPS = ', mps_add.show_bond_dims())
print(mps_add.norm())

print(mps_add @ (2 * mps))

lmps = mps_add.canonicalize(center=mps_add.n_sites - 1)
print('L-MPS = ', lmps.show_bond_dims())

rmps = mps_add.canonicalize(center=0)
print('R-MPS = ', rmps.show_bond_dims())

print(lmps @ rmps)

print('MPS + MPS = ', mps_add.show_bond_dims())
mps_add, _ = mps_add.compress(cutoff=1E-12)
print('MPS + MPS = ', mps_add.show_bond_dims())
print(mps_add.norm())

mps_minus = mps - mps
print(mps_minus.norm())
print('MPS - MPS = ', mps_minus.show_bond_dims())

mps_minus, _ = mps_minus.compress(cutoff=1E-12)
print('MPS - MPS = ', mps_minus.show_bond_dims())
print(mps_minus.norm())

hhmps = mpo @ (mpo @ mps)
print(hhmps.show_bond_dims())
print(np.sqrt(hhmps @ mps))

hhmps, cps_error = hhmps.compress(cutoff=1E-12)
print('error = ', cps_error)
print(hhmps.show_bond_dims())
print(np.sqrt(hhmps @ mps))

hhmps, cps_error = hhmps.compress(max_bond_dim=100, cutoff=1E-12)
print('error = ', cps_error)
print(hhmps.show_bond_dims())
print(np.sqrt(hhmps @ mps))

hhmps, cps_error = hhmps.compress(max_bond_dim=30, cutoff=1E-12)
print('error = ', cps_error)
print(hhmps.show_bond_dims())
print(np.sqrt(hhmps @ mps))

h2 = mpo @ mpo
print(h2.show_bond_dims())

print(np.sqrt(mps @ (h2 @ mps)))

h2, cps_error = h2.compress(cutoff=1E-16)
print('error = ', cps_error)
print(h2.show_bond_dims())
print(np.sqrt((h2 @ mps) @ mps))

h2, cps_error = h2.compress(max_bond_dim=15, cutoff=1E-16)
print('error = ', cps_error)
print(h2.show_bond_dims())
print(np.sqrt((h2 @ mps) @ mps))

h2, cps_error = h2.compress(max_bond_dim=12, cutoff=1E-16)
print('error = ', cps_error)
print(h2.show_bond_dims())
print(np.sqrt((h2 @ mps) @ mps))
