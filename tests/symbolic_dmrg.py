
import sys
sys.path[:0] = ['..', "../../block2/build"]

from block2.sz import MPOQC
from block2 import QCTypes
from functools import reduce
import time
import numpy as np
import pyblock3.algebra.funcs as pbalg
from pyblock3.algebra.symmetry import BondFusingInfo
from pyblock3.symbolic.symbolic_mpo import QCSymbolicMPO
from pyblock3.hamiltonian import QCHamiltonian
from pyblock3.fcidump import FCIDUMP
from pyblock3.algebra.mpe import MPE
from pyblock3.aux.io import SymbolicMPOTools
from pyblock3.aux.hamil import HamilTools


fd = '../data/N2.STO3G.FCIDUMP'
fcidump = FCIDUMP(pg='d2h').read(fd)
qchamil = QCHamiltonian(fcidump)
mpo = QCSymbolicMPO(qchamil)

print('MPO (original)  = ', mpo.show_bond_dims())

mpo = mpo.simplify()

print('MPO (simplified) = ', mpo.show_bond_dims())

with HamilTools.from_fcidump(fd) as hamil:
    # mps = hamil.get_ground_state_mps(bond_dim=100)
    mps = hamil.get_init_mps(bond_dim=100)
    # pxmpo = hamil.get_mpo()
    # with hamil.get_mpo_block2() as bmpo:
    #     ppmpo = SymbolicMPOTools.from_block2(bmpo)

# mpo = mpo.to_sparse()
# mpo, _ = mpo.compress(left=True, cutoff=1E-12)
# print('MPO (compressed) = ', mpo.show_bond_dims())

mps.opts = dict(cutoff=1E-12, norm_cutoff=1E-12, max_bond_dim=200)
me = MPE(mps, mpo, mps)


def dmrg(n_sweeps=10, tol=1E-6, dot=2):
    eners = np.zeros((n_sweeps, ))
    for iw in range(n_sweeps):
        print("Sweep = %4d | Direction = %8s | Bond dimension = %4d" % (
            iw, "backward" if iw % 2 else "forward", me.ket.opts["max_bond_dim"]))
        for i in range(0, me.n_sites - dot + 1)[::(-1) ** iw]:
            tt = time.perf_counter()
            eff = me[i:i + dot]
            eff.ket[:] = [reduce(pbalg.hdot, eff.ket[:])]
            eners[iw], eff, ndav = eff.gs_optimize(iprint=True)
            if dot == 2:
                l, s, r = eff.ket[0].tensor_svd(
                    idx=3, pattern='+++-++', full_matrices=False)
                eff.ket[:] = [np.tensordot(l, s.diag(), axes=1), r]
            me[i:i + dot] = eff
            print(" %3s Site = %4d-%4d .. Ndav = %4d E = %20.12f T = %8.3f" % (
                "<--" if iw % 2 else "-->", i, i + dot - 1, ndav, eners[iw], time.perf_counter() - tt))
        if abs(reduce(np.subtract, eners[:iw + 1][-2:])) < tol:
            break
    return eners[iw]


tx = time.perf_counter()
print("GS Energy = %20.12f" % dmrg())
print('time = ', time.perf_counter() - tx)
