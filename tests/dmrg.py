
import sys
sys.path[:0] = ['..']

from pyblock3.aux.hamil import HamilTools
from pyblock3.moving_environment import MovingEnvironment
import numpy as np
import time
from functools import reduce

with HamilTools.hubbard(n_sites=6, u=2, t=1) as hamil:
    mps = hamil.get_init_mps(bond_dim=50)
    mpo = hamil.get_mpo()

print('MPS = ', mps.show_bond_dims())
print('MPO (NC) =         ', mpo.show_bond_dims())
mpo, _ = mpo.compress(left=True, cutoff=1E-12, norm_cutoff=1E-12)
print('MPO (compressed) = ', mpo.show_bond_dims())

mps.opts = dict(cutoff=1E-12, norm_cutoff=1E-12, max_bond_dim=200)
me = MovingEnvironment(mps, mpo, mps)

def dmrg(n_sweeps=10, tol=1E-6, dot=2):
    eners = np.zeros((n_sweeps, ))
    for iw in range(n_sweeps):
        print("Sweep = %4d | Direction = %8s | Bond dimension = %4d" % (
            iw, "backward" if iw % 2 else "forward", me.ket.opts["max_bond_dim"]))
        for i in range(0, me.n_sites - dot + 1)[::(-1) ** iw]:
            tt = time.perf_counter()
            eners[iw], me[i:i+dot], ndav = np.linalg.eigh(me[i:i+dot])
            print(" %3s Site = %4d-%4d .. Ndav = %4d E = %20.12f T = %8.3f" %(
                "<--" if iw % 2 else "-->", i, i + dot - 1, ndav, eners[iw], time.perf_counter() - tt))
        if abs(reduce(np.subtract, eners[:iw + 1][-2:])) < tol:
            break
    return eners[iw]

print("GS Energy = %20.12f" % dmrg())
