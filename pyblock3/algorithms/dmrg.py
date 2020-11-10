
import time
import numpy as np
from functools import reduce
import pyblock3.algebra.funcs as pbalg
from enum import Enum, auto
from .core import SweepAlgorithm, fmt_size
import os
import psutil


class DMRG(SweepAlgorithm):
    def __init__(self, mpe, bdims, noises=None, dav_thrds=None, iprint=2):
        self.mpe = mpe
        self.bdims = bdims
        self.noises = noises
        self.dav_thrds = dav_thrds
        if self.noises is None:
            self.noises = [1E-6, 1E-7, 0.0]
        if self.dav_thrds is None:
            self.dav_thrds = []
        self.contract = True
        self.fast = True
        self.iprint = iprint
        self.energies = []
        super().__init__()

    def solve(self, n_sweeps=10, tol=1E-6, dot=2):
        mpe = self.mpe
        if len(self.bdims) < n_sweeps:
            self.bdims += [self.bdims[-1]] * (n_sweeps - len(self.bdims))
        if len(self.noises) == 0:
            self.noises = [0.0] * n_sweeps
        else:
            self.noises += [self.noises[-1]] * (n_sweeps - len(self.noises))
        if len(self.dav_thrds) < n_sweeps:
            for i in range(len(self.dav_thrds), n_sweeps):
                if self.noises[i] == 0:
                    self.dav_thrds.append(1E-10 if tol == 0 else max(tol * 0.1, 1E-10))
                else:
                    self.dav_thrds.append(max(self.noises[i], 1E-10))
        self.energies = []
        telp = time.perf_counter()
        for iw in range(n_sweeps):
            forward = iw % 2 == 0
            self.energies.append(1E10)
            if self.iprint >= 1:
                print("Sweep = %4d | Direction = %8s | BondDim = %4d | Noise = %5.2E | DavThrd = %5.2E" % (
                    iw, "forward" if forward else "backward", self.bdims[iw], self.noises[iw], self.dav_thrds[iw]))
            peak_mem = 0
            for i in range(0, mpe.n_sites - dot + 1)[::(-1) ** iw]:
                tt = time.perf_counter()
                eff = mpe[i:i + dot]
                # mem = mpe.nbytes
                mem = psutil.Process(os.getpid()).memory_info().rss
                peak_mem = max(mem, peak_mem)
                if self.contract:
                    eff.ket[:] = [reduce(pbalg.hdot, eff.ket[:])]
                    tx = time.perf_counter()
                    ener, eff, ndav = eff.eigs(
                        iprint=self.iprint >= 3, fast=self.fast, conv_thrd=self.dav_thrds[iw])
                    tdav = time.perf_counter() - tx
                    if dot == 2:
                        error = self.decomp_two_site(
                            eff.mpo, eff.ket, forward, self.noises[iw], self.bdims[iw])
                    else:
                        error = 0
                else:
                    tx = time.perf_counter()
                    ener, eff, ndav = eff.eigs(
                        iprint=self.iprint >= 3, fast=self.fast, conv_thrd=self.dav_thrds[iw])
                    tdav = time.perf_counter() - tx
                    cket, error = eff.ket.compress(
                        left=True, cutoff=self.cutoff, max_bond_dim=self.bdims[iw])
                    eff.ket[:] = cket[:]
                mmps = eff.ket[0].infos[-1].n_bonds
                mpe[i:i + dot] = eff
                self.energies[iw] = min(self.energies[iw], ener)
                if self.iprint >= 2:
                    print(" %3s Site = %4d-%4d .. Mmps = %4d Ndav = %4d E = %20.12f MaxDW = %5.2E Tdav = %8.3f T = %8.3f MEM = %7s" % (
                        "<--" if iw % 2 else "-->", i, i + dot - 1, mmps, ndav, ener, error, tdav, time.perf_counter() - tt, fmt_size(mem)))
            de = 0 if iw == 0 else abs(
                self.energies[iw] - self.energies[iw - 1])
            print("Time elapsed = %10.3f | E = %20.12f | DE = %5.2E | MEM = %7s" %
                  (time.perf_counter() - telp, self.energies[iw], de, fmt_size(peak_mem)))
            if iw > 0 and de < tol and self.noises[iw] == self.noises[-1]:
                break
        return self

    def __repr__(self):
        return "DMRG Energy = %20.15f" % self.energies[-1]
