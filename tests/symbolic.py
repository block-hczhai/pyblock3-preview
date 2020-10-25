
import sys
sys.path[:0] = ['..', "../../block2/build"]

from pyblock3.aux.hamil import HamilTools
from pyblock3.aux.io import SymbolicMPOTools
from pyblock3.algebra.mpe import MPE
from pyblock3.fcidump import FCIDUMP
from pyblock3.hamiltonian import Hamiltonian
from pyblock3.symbolic.symbolic_mpo import QCSymbolicMPO
from pyblock3.algebra.symmetry import BondFusingInfo
import pyblock3.algebra.funcs as pbalg
import numpy as np
import time
from functools import reduce

from block2 import QCTypes
from block2.sz import MPOQC

fd = '../data/N2.STO3G.FCIDUMP'
fcidump = FCIDUMP(pg='d2h').read(fd)
qchamil = Hamiltonian(fcidump)
pympo = QCSymbolicMPO(qchamil)

with HamilTools.from_fcidump(fd) as hamil:
    with hamil.get_mpo_block2() as bmpo:
        ppmpo = SymbolicMPOTools.from_block2(bmpo)

print(pympo.show_bond_dims())

with HamilTools.from_fcidump(fd) as hamil:
    mps = hamil.get_ground_state_mps(bond_dim=100)
    mpo = hamil.get_mpo()

print('MPO (block2 -> pyblock3) = ', mpo.show_bond_dims())

print('MPS energy (block2 -> pyblock3) = ', mps @ (mpo @ mps))

print(ppmpo.show_bond_dims())

gmpo = ppmpo.to_sparse()

print('MPS energy (block2 -> pyblock3 - sym) = ', mps @ (gmpo @ mps))

print('MPO (pyblock3 - sym) = ', pympo.show_bond_dims())

pympo = pympo.simplify()

print('MPO (pyblock3 - sym - simplified) = ', pympo.show_bond_dims())

xmpo = pympo.to_sparse()
xmps = mps.copy()

xmpo, _ = xmpo.compress(left=True, cutoff=1E-12)

print('MPO (pyblock3 - sym - simplified - cpsd) = ', xmpo.show_bond_dims())

print('MPS energy (pyblock3 - sym - simplified - cpsd) = ', np.dot(mps, xmpo @ mps))

# contract last three sites
mps[-2:] = [reduce(pbalg.hdot, mps[-2:])]
xmpo[-2:] = [reduce(pbalg.hdot, xmpo[-2:])]

mps[-2:] = [reduce(pbalg.hdot, mps[-2:])]
xmpo[-2:] = [reduce(pbalg.hdot, xmpo[-2:])]

print('MPO (3-ctrd tensor) = ', xmpo.show_bond_dims())

print('MPS energy (3-ctrd tensor) = ', np.dot(mps, xmpo @ mps))

xmpo = pympo.to_sparse()
mps = xmps.copy()

print('MPO (pyblock3 - sym - simplified) = ', xmpo.show_bond_dims())

print('MPS energy (pyblock3 - sym - simplified) = ', np.dot(xmps, xmpo @ xmps))

mps[-2:] = [reduce(pbalg.hdot, mps[-2:])]
pympo[-2:] = [reduce(pbalg.hdot, pympo[-2:])]

xmpo = pympo.to_sparse()

print('MPO (2-ctrd sym -> tensor) = ', xmpo.show_bond_dims())

print('MPS energy (2-ctrd sym -> tensor) = ', np.dot(mps, xmpo @ mps))

mps[-2:] = [reduce(pbalg.hdot, mps[-2:])]
pympo[-2:] = [reduce(pbalg.hdot, pympo[-2:])]

xmpo = pympo.to_sparse()

print('MPO (3-ctrd sym -> tensor) = ', xmpo.show_bond_dims())

print('MPS energy (3-ctrd sym -> tensor) = ', np.dot(mps, xmpo @ mps))

info = BondFusingInfo.tensor_product(*mps[-1].infos[1:4])
mps[-1] = mps[-1].fuse(1, 2, 3, info=info)
pympo[-1] = pympo[-1].fuse(4, 5, 6, info=info).fuse(1, 2, 3, info=info)

xmpo = pympo.to_sparse()

print('MPO (3-ctrd sym fused -> tensor) =', xmpo.show_bond_dims())

print('MPS energy (3-ctrd sym fused -> tensor) = ', np.dot(mps, xmpo @ mps))
