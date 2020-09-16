
import sys
sys.path[:0] = ['..', "../../block2-old/build"]

from pyblock3.aux.hamil import HamilTools
from pyblock3.algebra.mpe import MPE
from pyblock3.algebra.symmetry import BondFusingInfo
import pyblock3.algebra.funcs as pbalg
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

print('MPS energy = ', np.dot(mps, mpo @ mps))

mps[-3:] = [reduce(pbalg.hdot, mps[-3:])]
mpo[-3:] = [reduce(pbalg.hdot, mpo[-3:])]

print('MPO (hdot) = ', mpo.show_bond_dims())
print('MPS (hdot) = ', mps.show_bond_dims())

print('MPS energy = ', mps @ (mpo @ mps))

info = mps[-1].kron_product_info(1, 2, 3)
mps[-1] = mps[-1].fuse(1, 2, 3, info=info)
mpo[-1] = mpo[-1].fuse(4, 5, 6, info=info).fuse(1, 2, 3, info=info)

print('MPS energy = ', mps @ (mpo @ mps))

me = MPE(mps, mpo, mps)
mex = me[0:2]
# print(mex.ket[-2])
# print(mex.ket[-1])
mex.ket[:] = [reduce(pbalg.hdot, mex.ket[:])]
print(mex.expectation)
# l, s, r = mex.ket[0].tensor_svd(idx=3, pattern='+++-++', full_matrices=False)
# ls = np.tensordot(l, s.diag(), axes=1)
# mex.ket[:] = [ls, r]
# # print(mex.ket[-2])
# # print(mex.ket[-1])
# print(mex.expectation)
