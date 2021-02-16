
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


class GreensFunction(SweepAlgorithm):
    """DDMRG++ for solving Green's function."""
    def __init__(self, mpe, mpo, omega, eta, bdims, noises=None, cg_thrds=None, iprint=2):
        self.mpe = mpe
        assert mpe.bra is not mpe.ket
        if hasattr(mpe, "tag"):
            self.impe = mpe.__class__(mpe.bra, mpo, mpe.bra, do_canon=False,
                tag=mpe.tag + "@GF", scratch=mpe.scratch, maxsize=mpe.maxsize)
        else:
            self.impe = mpe.__class__(mpe.bra, mpo, mpe.bra, do_canon=False)
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
        self.omega = omega
        self.eta = eta
        super().__init__()

    def solve(self, n_sweeps=10, tol=1E-6, dot=2):
        mpe = self.mpe
        impe = self.impe
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
                mpe.build_envs(i, i + dot)
                impe.build_envs(i, i + dot)
                eff = mpe[i:i + dot]
                ieff = impe[i:i + dot]
                if self.contract:
                    prev_ket = eff.ket[:]
                    eff.ket[:] = [reduce(pbalg.hdot, eff.ket[:])]
                    tx = time.perf_counter()
                    _, eff, _ = eff.multiply(fast=self.fast)
                    ieff.ket[:] = [reduce(pbalg.hdot, ieff.ket[:])]
                    func, ieff, ncg = ieff.solve_gf(
                        eff.bra[0], self.omega, self.eta,
                        iprint=self.iprint >= 3, fast=self.fast, conv_thrd=self.cg_thrds[iw])
                    tcg = time.perf_counter() - tx
                    if dot == 2:
                        error = self.decomp_two_site(
                            ieff.mpo, [ieff.ket, ieff.bra],
                            forward, self.noises[iw], self.bdims[iw], weights=[0.5, 0.5])
                    else:
                        error = 0
                    ieff.bra = ieff.ket
                    eff.bra[:] = ieff.bra[:]
                    eff.ket[:] = prev_ket
                else:
                    assert False
                mmps = ieff.bra[0].infos[-1].n_bonds
                impe[i:i + dot] = ieff
                mpe[i:i + dot] = eff
                if abs(self.targets[iw]) > abs(func):
                    self.targets[iw] = func
                if self.iprint >= 2:
                    print(" %3s Site = %4d-%4d .. Mmps = %4d Ncg = %4d Re = %20.12f Im = %20.12f DW = %5.2E Tcg = %8.3f T = %8.3f" % (
                        "<--" if iw % 2 else "-->", i, i + dot - 1, mmps, ncg, func.real, func.imag, error, tcg, time.perf_counter() - tt))
            df = 0 if iw == 0 else abs(
                abs(self.targets[iw]) - abs(self.targets[iw - 1]))
            print("Time elapsed = %10.3f | GF = Re %20.12f Im = %20.12f | DGF = %5.2E" %
                  (time.perf_counter() - telp, self.targets[iw].real, self.targets[iw].imag, df))
            if iw > 0 and df < tol and self.noises[iw] == self.noises[-1]:
                break
        return self

    def __repr__(self):
        return "GF = Re %20.15f Im = %20.15f" % (self.targets[-1].real, self.targets[-1].imag)
