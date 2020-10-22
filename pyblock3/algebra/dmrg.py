
import time
import numpy as np
from functools import reduce
import pyblock3.algebra.funcs as pbalg

class DMRG:
    def __init__(self, mpe, bdims, noises=[], iprint=2):
        self.mpe = mpe
        self.bdims = bdims
        self.noises = noises
        self.contract = True
        self.fast = True
        self.iprint = iprint
        self.cutoff = 1E-12
        self.energies = []

    def solve(self, n_sweeps=10, tol=1E-6, dot=2):
        mpe = self.mpe
        if len(self.bdims) < n_sweeps:
            self.bdims += [self.bdims[-1]] * (n_sweeps - len(self.bdims))
        if len(self.noises) == 0:
            self.noises = [0.0] * n_sweeps
        else:
            self.noises += [self.noises[-1]] * (n_sweeps - len(self.noises))
        self.energies = []
        for iw in range(n_sweeps):
            self.energies.append(1E10)
            if self.iprint >= 1:
                print("Sweep = %4d | Direction = %8s | Bond dimension = %4d | Noise = %5.2E" % (
                    iw, "backward" if iw % 2 else "forward", self.bdims[iw], self.noises[iw]))
            for i in range(0, mpe.n_sites - dot + 1)[::(-1) ** iw]:
                tt = time.perf_counter()
                eff = mpe[i:i + dot]
                if self.contract:
                    eff.ket[:] = [reduce(pbalg.hdot, eff.ket[:])]
                    ener, eff, ndav = eff.gs_optimize(iprint=self.iprint >= 3, fast=self.fast)
                    if dot == 2:
                        lsr = eff.ket[0].tensor_svd(idx=3, pattern='+++-+-')
                        l, s, r, error = pbalg.truncate_svd(*lsr, cutoff=self.cutoff, max_bond_dim=self.bdims[iw])
                        eff.ket[:] = [np.tensordot(l, s.diag(), axes=1), r]
                    else:
                        error = 0
                else:
                    ener, eff, ndav = eff.gs_optimize(iprint=self.iprint >= 3, fast=self.fast)
                    cket, error = eff.ket.compress(left=True, cutoff=self.cutoff, max_bond_dim=self.bdims[iw])
                    eff.ket[:] = cket[:]
                mmps = eff.ket[0].infos[-1].n_bonds
                mpe[i:i + dot] = eff
                self.energies[iw] = min(self.energies[iw], ener)
                if self.iprint >= 2:
                    print(" %3s Site = %4d-%4d .. Mmps = %4d Ndav = %4d E = %20.12f Error = %5.2E T = %8.3f" % (
                        "<--" if iw % 2 else "-->", i, i + dot - 1, mmps, ndav, ener, error, time.perf_counter() - tt))
            if iw > 0 and abs(self.energies[iw] - self.energies[iw - 1]) < tol:
                break
        return self

    def __repr__(self):
        return "DMRG Energy = %20.15f" % self.energies[-1]
