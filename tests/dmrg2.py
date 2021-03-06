
import sys
sys.path[:0] = ['..']

from functools import reduce
import time
import numpy as np
import pyblock3.algebra.funcs as pbalg
from pyblock3.algebra.mpe import MPE
from pyblock3.hamiltonian import Hamiltonian
from pyblock3.fcidump import FCIDUMP
from pyblock3.symbolic.symbolic_mpo import QCSymbolicMPO
from pyblock3.algebra.mps import MPSInfo, MPS

flat = True
fast = True
iprint = False
contract = True
dot = 1

fd = '../data/HUBBARD-L8.FCIDUMP'
# fd = '../my_test/n2/N2.FCIDUMP'
# fd = '../data/CR2.SVP.FCIDUMP'
bdims = 200

fcidump = FCIDUMP(pg='d2h').read(fd)
hamil = Hamiltonian(fcidump)
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
mpe = MPE(mps, mpo, mps)

def dmrg(n_sweeps=10, tol=1E-6, dot=2):
    eners = np.zeros((n_sweeps, ))
    for iw in range(n_sweeps):
        print("Sweep = %4d | Direction = %8s | Bond dimension = %4d" % (
            iw, "backward" if iw % 2 else "forward", bdims))
        for i in range(0, mpe.n_sites - dot + 1)[::(-1) ** iw]:
            tt = time.perf_counter()
            eff = mpe[i:i + dot]
            if contract:
                eff.ket[:] = [reduce(pbalg.hdot, eff.ket[:])]
                ener, eff, ndav, _ = eff.eigs(iprint=iprint, fast=fast)
                if dot == 2:
                    lsr = eff.ket[0].tensor_svd(idx=3, pattern='++++-+')
                    l, s, r, error = pbalg.truncate_svd(*lsr, cutoff=1E-12, max_bond_dim=bdims)
                    eff.ket[:] = [np.tensordot(l, s.diag(), axes=1), r]
                else:
                    error = 0
            else:
                ener, eff, ndav, _ = eff.eigs(iprint=iprint, fast=fast)
                cket, error = eff.ket.compress(cutoff=1E-12, max_bond_dim=bdims)
                eff.ket[:] = cket[:]
            mpe[i:i + dot] = eff
            eners[iw] = min(eners[iw], ener)
            print(" %3s Site = %4d-%4d .. Ndav = %4d E = %20.12f Error = %10.5g T = %8.3f" % (
                "<--" if iw % 2 else "-->", i, i + dot - 1, ndav, eners[iw], error, time.perf_counter() - tt))
        if abs(reduce(np.subtract, eners[:iw + 1][-2:])) < tol:
            break
    return eners[iw]
  
print("GS Energy = %20.12f" % dmrg(dot=dot))
print('MPS energy = ', np.dot(mps, mpo @ mps))
