
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
from .core import SweepAlgorithm, fmt_size
import os
import psutil

def _complex_repr(form, x):
    if x.imag == 0:
        return "%s" % form % x.real
    elif x.real == 0:
        return "%s i" % form % x.imag
    else:
        return "%s + %s i" % (form, form) % (x.real, x.imag)

class TDDMRG(SweepAlgorithm):
    """Time-step targetting td-DMRG approach."""
    def __init__(self, mpe, bdims, iprint=2, **kwargs):
        self.mpe = mpe
        self.bdims = bdims
        self.contract = True
        self.fast = True
        self.iprint = iprint
        self.energies = []
        self.normsqs = []
        super().__init__(**kwargs)

    def solve(self, dt, n_sweeps=10, n_sub_sweeps=2, dot=2, forward=True, normalize=True):
        mpe = self.mpe
        if len(self.bdims) < n_sweeps:
            self.bdims += [self.bdims[-1]] * (n_sweeps - len(self.bdims))
        self.energies = []
        self.normsqs = []
        telp = time.perf_counter()
        for iw, isw in [(a, b) for a in range(n_sweeps) for b in range(n_sub_sweeps)]:
            if isw == 0:
                self.energies.append(0)
                self.normsqs.append(0)
            if self.iprint >= 1:
                if n_sub_sweeps == 1:
                    print("Sweep = %4d | Direction = %8s | DT = %s | BondDim = %4d" % (
                        iw, "forward" if forward else "backward", _complex_repr("%9.2g", dt), self.bdims[iw]))
                else:
                    print("Sweep = %4d (%2d/%2d) | Direction = %8s | DT = %s | BondDim = %4d" % (
                        iw, isw, n_sub_sweeps, "forward" if forward else "backward",
                        _complex_repr("%9.2g", dt), self.bdims[iw]))
            peak_mem = 0
            dw = 0
            for i in range(0, mpe.n_sites - dot + 1)[::1 if forward else -1]:
                eval_ener = i == mpe.n_sites - dot if forward else i == 0
                advance = isw == n_sub_sweeps - 1 and eval_ener
                tt = time.perf_counter()
                mpe.build_envs(i, i + dot)
                eff = mpe[i:i + dot]
                mem = psutil.Process(os.getpid()).memory_info().rss
                peak_mem = max(mem, peak_mem)
                if self.contract:
                    eff.ket[:] = [reduce(pbalg.hdot, eff.ket[:])]
                    tx = time.perf_counter()
                    if advance:
                        nmult, nflop = 0, 0
                        for ii in range(10):
                            ener, norm, kets, eff, xnmult, xnflop = eff.rk4(dt=dt / 10, fast=self.fast, eval_ener=ii == 9)
                            nmult += xnmult
                            nflop += xnflop
                            eff.ket = eff.bra = kets[-1]
                        if normalize:
                            eff.ket /= np.linalg.norm(eff.ket)
                    else:
                        ener, norm, kets, eff, nmult, nflop = eff.rk4(dt=dt, fast=self.fast, eval_ener=eval_ener)
                    tmult = time.perf_counter() - tx
                    if dot == 2:
                        if advance:
                            error = self.decomp_two_site(
                                eff.mpo, eff.ket, forward, 0, self.bdims[iw])
                            eff.bra = eff.ket
                        else:
                            weights = [1.0 / 3, 1.0 / 6, 1.0 / 6, 1.0 / 3]
                            error = self.decomp_two_site(
                                eff.mpo, kets, forward, 0, self.bdims[iw], weights=weights)
                            eff.ket = eff.bra = kets[0]
                    else:
                        error = 0
                else:
                    raise NotImplementedError
                mmps = eff.ket[0].infos[-1].n_bonds
                mpe[i:i + dot] = eff
                self.energies[iw] = ener
                self.normsqs[iw] = norm ** 2
                if self.iprint >= 2:
                    print(" %3s Site = %4d-%4d .. Mmps = %4d Nmult = %4d E = %s DW = %5.2E FLOPS = %5.2E Tmult = %8.3f T = %8.3f MEM = %7s" % (
                        "<--" if iw % 2 else "-->", i, i + dot - 1, mmps, nmult, _complex_repr("%20.12f", ener), error, nflop / tmult, tmult, time.perf_counter() - tt, fmt_size(mem)))
                dw = max(dw, abs(error))
            if self.iprint > 0:
                print("Time elapsed = %10.3f | E = %s | MDW = %5.2E | Norm^2 = %20.12f | MEM = %7s" %
                    (time.perf_counter() - telp, _complex_repr("%20.12f", self.energies[iw]), dw, self.normsqs[iw], fmt_size(peak_mem)))
            forward = not forward
        return self

    def __repr__(self):
        return "TDDMRG Energy = %20.15f Norm^2 = %20.15f" % (self.energies[-1], self.normsqs[-1])
