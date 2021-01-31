
#  pyblock3: An Efficient python MPS/DMRG Library
#  Copyright (C) 2020 The pyblock3 developers. All Rights Reserved.
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program. If not, see <https://www.gnu.org/licenses/>.
#
#

import time
import numpy as np
from functools import reduce
import pyblock3.algebra.funcs as pbalg
from enum import Enum, auto
from .core import SweepAlgorithm


class Linear(SweepAlgorithm):
    """Solving linear equation or compression in sweeps."""
    def __init__(self, mpe, bdims, noises=None, cg_thrds=None, iprint=2):
        self.mpe = mpe
        assert mpe.bra is not mpe.ket
        self.bdims = bdims
        self.noises = noises
        self.cg_thrds = cg_thrds
        if self.noises is None:
            self.noises = [1E-6, 1E-7, 0.0]
        if self.cg_thrds is None:
            self.cg_thrds = []
        self.contract = True
        self.fast = True
        self.iprint = iprint
        self.targets = []
        super().__init__()

    def solve(self, n_sweeps=10, tol=1E-6, dot=2):
        mpe = self.mpe
        if len(self.bdims) < n_sweeps:
            self.bdims += [self.bdims[-1]] * (n_sweeps - len(self.bdims))
        if len(self.noises) == 0:
            self.noises = [0.0] * n_sweeps
        else:
            self.noises += [self.noises[-1]] * (n_sweeps - len(self.noises))
        if len(self.cg_thrds) < n_sweeps:
            for i in range(len(self.cg_thrds), n_sweeps):
                if self.noises[i] == 0:
                    self.cg_thrds.append(1E-10 if tol == 0 else max(tol * 0.1, 1E-10))
                else:
                    self.cg_thrds.append(max(self.noises[i], 1E-10))
        self.targets = []
        telp = time.perf_counter()
        for iw in range(n_sweeps):
            forward = iw % 2 == 0
            self.targets.append(1E10)
            if self.iprint >= 1:
                print("Sweep = %4d | Direction = %8s | BondDim = %4d | Noise = %5.2E | CGThrd = %5.2E" % (
                    iw, "forward" if forward else "backward", self.bdims[iw], self.noises[iw], self.cg_thrds[iw]))
            for i in range(0, mpe.n_sites - dot + 1)[::(-1) ** iw]:
                tt = time.perf_counter()
                eff = mpe[i:i + dot]
                if self.contract:
                    prev_ket = eff.ket[:]
                    eff.ket[:] = [reduce(pbalg.hdot, eff.ket[:])]
                    tx = time.perf_counter()
                    func, eff, ncg = eff.multiply(fast=self.fast)
                    tcg = time.perf_counter() - tx
                    if dot == 2:
                        error = self.decomp_two_site(
                            eff.mpo, eff.bra, forward, self.noises[iw], self.bdims[iw])
                    else:
                        error = 0
                    eff.ket[:] = prev_ket
                else:
                    assert False
                mmps = eff.bra[0].infos[-1].n_bonds
                mpe[i:i + dot] = eff
                self.targets[iw] = min(self.targets[iw], func)
                if self.iprint >= 2:
                    print(" %3s Site = %4d-%4d .. Mmps = %4d Ncg = %4d E = %20.12f DW = %5.2E Tcg = %8.3f T = %8.3f" % (
                        "<--" if iw % 2 else "-->", i, i + dot - 1, mmps, ncg, func, error, tcg, time.perf_counter() - tt))
            df = 0 if iw == 0 else abs(
                self.targets[iw] - self.targets[iw - 1])
            print("Time elapsed = %10.3f | F = %20.12f | DF = %5.2E" %
                  (time.perf_counter() - telp, self.targets[iw], df))
            if iw > 0 and df < tol and self.noises[iw] == self.noises[-1]:
                break
        return self

    def __repr__(self):
        return "Norm = %20.15f" % self.targets[-1]
