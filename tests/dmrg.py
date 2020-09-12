
import sys
sys.path[:0] = ['..', "../../block2-old/build"]

from functools import reduce
import time
import numpy as np
import pyblock3.algebra.funcs as pbalg
from pyblock3.moving_environment import MovingEnvironment
from pyblock3.aux.hamil import HamilTools
from pyblock3.hamiltonian import QCHamiltonian
from pyblock3.fcidump import FCIDUMP
from pyblock3.symbolic.symbolic_mpo import QCSymbolicMPO
from pyblock3.algebra.mps import MPSInfo, MPS

flat = True
fast = True
dot = 2

fd = '../data/H8.STO6G.R1.8.FCIDUMP'
bdims = 200

# with HamilTools.hubbard(n_sites=8, u=2, t=1) as hamil:
# with HamilTools.hubbard(n_sites=16, u=2, t=1) as hamil:
# with HamilTools.from_fcidump(fd) as hamil:
#     mps = hamil.get_init_mps(bond_dim=bdims)
#     # mps = hamil.get_ground_state_mps(bond_dim=100)
#     mpo = hamil.get_mpo()

fcidump = FCIDUMP(pg='d2h').read(fd)
hamil = QCHamiltonian(fcidump)
mpo = QCSymbolicMPO(hamil).to_sparse()

mps_info = MPSInfo(hamil.n_sites, hamil.vacuum, hamil.target, hamil.basis)
mps_info.set_bond_dimension(bdims)
mps = MPS.random(mps_info)

print('MPS = ', mps.show_bond_dims())
print('MPO (NC) =         ', mpo.show_bond_dims())
mpo, _ = mpo.compress(left=True, cutoff=1E-9, norm_cutoff=1E-9)
print('MPO (compressed) = ', mpo.show_bond_dims())

mps.opts = dict(cutoff=1E-12, norm_cutoff=1E-12, max_bond_dim=bdims)
if flat:
    mps = mps.to_flat()
    mpo = mpo.to_flat()
me = MovingEnvironment(mps, mpo, mps)


def dmrg(n_sweeps=10, tol=1E-6, dot=2):
    eners = np.zeros((n_sweeps, ))
    for iw in range(n_sweeps):
        print("Sweep = %4d | Direction = %8s | Bond dimension = %4d" % (
            iw, "backward" if iw % 2 else "forward", me.ket.opts["max_bond_dim"]))
        for i in range(0, me.n_sites - dot + 1)[::(-1) ** iw]:
            tt = time.perf_counter()
            eff = me[i:i + dot]
            eff.ket[:] = [reduce(pbalg.hdot, eff.ket[:])]
            eners[iw], eff, ndav = eff.eigh(iprint=False, fast=fast)
            if dot == 2:
                wfn = eff.ket[0]
                l, s, r = wfn.tensor_svd(idx=3, pattern='+++-+-')
                eff.ket[:] = [np.tensordot(l, s.diag(), axes=1), r]
                cket, error = eff.ket.compress(left=True, cutoff=1E-12, max_bond_dim=bdims)
                eff.ket[:] = cket[:]
            else:
                error = 0
            me[i:i + dot] = eff
            print(" %3s Site = %4d-%4d .. Ndav = %4d E = %20.12f Error = %10.5g T = %8.3f" % (
                "<--" if iw % 2 else "-->", i, i + dot - 1, ndav, eners[iw], error, time.perf_counter() - tt))
        if abs(reduce(np.subtract, eners[:iw + 1][-2:])) < tol:
            break
    return eners[iw]


tx = time.perf_counter()
print("GS Energy = %20.12f" % dmrg(dot=dot))
print('time = ', time.perf_counter() - tx)
