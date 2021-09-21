
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


class DMRG(SweepAlgorithm):
    """Density Matrix Renormalization Group (DMRG)."""

    def __init__(self, mpe, bdims, noises=None, dav_thrds=None, max_iter=500, iprint=2, cutoff=1E-14,
                 extra_mpes=None, weights=None, init_site=None):
        self.mpe = mpe
        self.extra_mpes = [] if extra_mpes is None else extra_mpes
        self.weights = weights
        self.bdims = bdims
        self.noises = noises
        self.dav_thrds = dav_thrds
        self.max_iter = max_iter
        self.init_site = init_site
        if self.noises is None:
            self.noises = [1E-6, 1E-7, 0.0]
        if self.dav_thrds is None:
            self.dav_thrds = []
        self.contract = True
        self.fast = hasattr(mpe.ket.tensors[0], "q_labels")
        self.energies = []
        super().__init__(cutoff=cutoff, mpi=self.mpe.mpi)
        if self.mpi:
            self.iprint = iprint if self.mrank == 0 else -1
        else:
            self.iprint = iprint

    def solve(self, n_sweeps=10, tol=1E-6, dot=2, forward=True):
        mpe = self.mpe
        extra_mpes = self.extra_mpes
        nroots = len(extra_mpes) + 1
        if len(self.bdims) < n_sweeps:
            self.bdims += [self.bdims[-1]] * (n_sweeps - len(self.bdims))
        if len(self.noises) == 0:
            self.noises = [0.0] * n_sweeps
        else:
            self.noises += [self.noises[-1]] * (n_sweeps - len(self.noises))
        if len(self.dav_thrds) < n_sweeps:
            for i in range(len(self.dav_thrds), n_sweeps):
                if self.noises[i] == 0:
                    self.dav_thrds.append(
                        1E-10 if tol == 0 else max(tol * 0.1, 1E-10))
                else:
                    self.dav_thrds.append(max(self.noises[i], 1E-10))
        self.energies = []
        telp = time.perf_counter()
        for iw in range(n_sweeps):
            tswp = time.perf_counter()
            tdav_tot = 0
            tdec_tot = 0
            self.energies.append(1E10 if nroots == 1 else [1E10] * nroots)
            if self.iprint >= 1:
                print("Sweep = %4d | Direction = %8s | BondDim = %4d | Noise = %5.2E | DavThrd = %5.2E" % (
                    iw, "forward" if forward else "backward", self.bdims[iw], self.noises[iw], self.dav_thrds[iw]))
            peak_mem = 0
            dw = 0
            for i in range(0, mpe.n_sites - dot + 1)[::1 if forward else -1]:
                if self.init_site is not None and forward and i < self.init_site:
                    continue
                tt = time.perf_counter()
                mpe.build_envs(i, i + dot)
                eff = mpe[i:i + dot]
                extra_effs = [None] * len(extra_mpes)
                for ix, ex in enumerate(extra_mpes):
                    ex.build_envs_no_contract(i, i + dot)
                    extra_effs[ix] = ex[i:i + dot]
                if self.mpe.mpi:
                    eff.ket = self.comm.bcast(eff.ket, root=0)
                    for xe in extra_effs:
                        xe.ket = self.comm.bcast(xe.ket, root=0)
                mem = psutil.Process(os.getpid()).memory_info().rss
                peak_mem = max(mem, peak_mem)
                if self.contract:
                    eff.ket[:] = [reduce(pbalg.hdot, eff.ket[:])]
                    for xe in extra_effs:
                        xe.ket[:] = [reduce(pbalg.hdot, xe.ket[:])]
                    tx = time.perf_counter()
                    ener, eff, ndav, nflop = eff.eigs(
                        iprint=self.iprint >= 3, fast=self.fast, conv_thrd=self.dav_thrds[iw],
                        max_iter=self.max_iter, extra_mpes=extra_effs)
                    tdav = time.perf_counter() - tx
                    if dot == 2:
                        tx = time.perf_counter()
                        wfns = [eff.ket]
                        for xe in extra_effs:
                            wfns.append(xe.ket)
                        error = self.decomp_two_site(
                            eff.mpo, wfns, forward, self.noises[iw], self.bdims[iw], weights=self.weights)
                        tdec_tot += time.perf_counter() - tx
                    else:
                        error = 0
                else:
                    assert nroots == 1
                    tx = time.perf_counter()
                    ener, eff, ndav, nflop = eff.eigs(
                        iprint=self.iprint >= 3, fast=self.fast, conv_thrd=self.dav_thrds[iw], max_iter=self.max_iter)
                    tdav = time.perf_counter() - tx
                    cket, error = eff.ket.compress(
                        left=True, cutoff=self.cutoff, max_bond_dim=self.bdims[iw])
                    eff.ket[:] = cket[:]
                if forward:
                    mmps = eff.ket[0].infos[-1].n_bonds
                else:
                    mmps = eff.ket[-1].infos[0].n_bonds
                mpe[i:i + dot] = eff
                for ix, ex in enumerate(extra_mpes):
                    ex[i:i + dot] = extra_effs[ix]
                if nroots == 1:
                    self.energies[iw] = min(self.energies[iw], ener)
                else:
                    self.energies[iw] = min(tuple(self.energies[iw]), tuple(ener))
                if self.iprint >= 2:
                    eners = ener if nroots > 1 else [ener]
                    print((" %3s Site = %4d-%4d .. Mmps = %4d Ndav = %4d E =" + " %20.12f" * nroots + " DW = %5.2E FLOPS = %5.2E Tdav = %8.3f T = %8.3f MEM = %7s") % (
                        "<--" if iw % 2 else "-->", i, i + dot - 1, mmps, ndav, *eners, error, nflop / tdav, tdav, time.perf_counter() - tt, fmt_size(mem)))
                    tdav_tot += tdav
                dw = max(dw, error)
            if iw == 0:
                de = 0
            elif nroots > 1:
                de = abs(self.energies[iw][0] - self.energies[iw - 1][0])
            else:
                de = abs(self.energies[iw] - self.energies[iw - 1])
            if self.iprint >= 0:
                eners = self.energies[iw] if nroots > 1 else [self.energies[iw]]
                print(("Time elapsed = %10.3f | E =" + " %20.15f" * nroots +
                    " | DE = %5.2E | MDW = %5.2E | MEM = %7s") %
                    (time.perf_counter() - telp, *eners, de, dw, fmt_size(peak_mem)))
                print("Time sweep = %10.3f | Time davidson = %10.3f | Time decomp = %10.3f" %
                    (time.perf_counter() - tswp, tdav_tot, tdec_tot))
            if iw > 0 and de < tol and self.noises[iw] == self.noises[-1]:
                break
            forward = not forward
        return self

    def __repr__(self):
        if isinstance(self.energies[-1], list):
            return ("DMRG Energies =" + " %20.15f" * len(self.energies[-1])) % tuple(self.energies[-1])
        else:
            return "DMRG Energy = %20.15f" % self.energies[-1]
