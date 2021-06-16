
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

"""One-body/two-body integral object."""

import numpy as np
import numba as nb

PointGroup = {
    "d2h": [8, 0, 7, 6, 1, 5, 2, 3, 4],
    "c2v": [8, 0, 2, 3, 1],
    "c2h": [8, 0, 2, 3, 1],
    "d2": [8, 0, 3, 2, 1],
    "cs": [8, 0, 1],
    "c2": [8, 0, 1],
    "ci": [8, 0, 1],
    "c1": [8, 0]
}


@nb.njit(nb.void(nb.int32, nb.int32, nb.int32, nb.float64[:, :]))
def parallelize_h1e(n_sites, mrank, msize, h1e):
    for i in range(0, n_sites):
        if mrank != i % msize:
            h1e[i, :] = 0.0


@nb.njit(nb.void(nb.int32, nb.int32, nb.int32, nb.float64[:, :, :, :]))
def parallelize_g2e(n_sites, mrank, msize, g2e):
    for i in range(0, n_sites):
        for j in range(0, n_sites):
            for k in range(0, n_sites):
                for l in range(0, n_sites):
                    if g2e[i, j, k, l] == 0:
                        continue
                    x = np.array([i, j, k, l], dtype=np.int32)
                    x.sort()
                    if x[1] == x[2]:
                        if mrank != x[1] % msize:
                            g2e[i, j, k, l] = 0
                    else:
                        ii, jj = x[0], x[1]
                        if ii <= jj:
                            if mrank != (jj * (jj + 1) // 2 + ii) % msize:
                                g2e[i, j, k, l] = 0
                        else:
                            if mrank != (ii * (ii + 1) // 2 + jj) % msize:
                                g2e[i, j, k, l] = 0


class FCIDUMP:

    def __init__(self, pg='c1', n_sites=0, n_elec=0, twos=0, ipg=0, uhf=False,
                 h1e=None, g2e=None, orb_sym=None, const_e=0, mu=0):
        self.pg = pg
        self.n_sites = n_sites
        self.n_elec = n_elec
        self.twos = twos
        self.ipg = ipg
        self.h1e = h1e
        self.g2e = g2e  # aa, ab, bb
        self.const_e = const_e
        self.uhf = uhf
        self.general = False
        self.tgeneral = False
        self.orb_sym = [0] * self.n_sites if orb_sym is None else orb_sym
        self.mu = mu
        if self.mu != 0 and self.h1e is not None:
            if not self.uhf:
                self.h1e = self.h1e - self.mu * np.identity(self.n_sites)
            else:
                self.h1e[0] = self.h1e[0] - self.mu * np.identity(self.n_sites)
                self.h1e[1] = self.h1e[1] - self.mu * np.identity(self.n_sites)

    def t(self, s, i, j):
        return self.h1e[s][i, j] if self.uhf else self.h1e[i, j]

    def v(self, sij, skl, i, j, k, l):
        if self.uhf:
            if sij == skl:
                return self.g2e[sij + sij][i, j, k, l]
            elif sij == 0 and skl == 1:
                return self.g2e[1][i, j, k, l]
            else:
                return self.g2e[1][k, l, i, j]
        else:
            return self.g2e[i, j, k, l]

    def parallelize(self, mpi=True):
        if not mpi:
            return self
        from mpi4py import MPI
        mrank = MPI.COMM_WORLD.Get_rank()
        msize = MPI.COMM_WORLD.Get_size()
        if mrank != 0:
            self.const_e = 0
        if isinstance(self.h1e, tuple):
            for h in self.h1e:
                parallelize_h1e(self.n_sites, mrank, msize, h)
        else:
            parallelize_h1e(self.n_sites, mrank, msize, self.h1e)
        if isinstance(self.g2e, tuple):
            for g in self.g2e:
                parallelize_g2e(self.n_sites, mrank, msize, g)
        else:
            parallelize_g2e(self.n_sites, mrank, msize, self.g2e)
        return self

    def read(self, filename):
        """
        Read FCI options and integrals from FCIDUMP file.

        Args:
            filename : str
        """
        with open(filename, 'r') as f:
            ff = f.read().lower()
            if '/' in ff:
                pars, ints = ff.split('/')
            elif '&end' in ff:
                pars, ints = ff.split('&end')

        cont = ','.join(pars.split()[1:])
        cont = cont.split(',')
        cont_dict = {}
        p_key = None
        for c in cont:
            if '=' in c or p_key is None:
                p_key, b = c.split('=')
                cont_dict[p_key.strip().lower()] = b.strip()
            elif len(c.strip()) != 0:
                if len(cont_dict[p_key.strip().lower()]) != 0:
                    cont_dict[p_key.strip().lower()] += ',' + c.strip()
                else:
                    cont_dict[p_key.strip().lower()] = c.strip()

        for k, v in cont_dict.items():
            if ',' in v:
                v = cont_dict[k] = v.split(',')

        self.n_sites = int(cont_dict.get('norb'))
        self.twos = int(cont_dict.get('ms2', 0))
        self.ipg = PointGroup[self.pg][int(cont_dict.get('isym', 0))]
        self.n_elec = int(cont_dict.get('nelec', 0))
        self.uhf = int(cont_dict.get('iuhf', 0)) != 0
        self.general = int(cont_dict.get('igeneral', 0)) != 0
        self.tgeneral = int(cont_dict.get('itgeneral', 0)) != 0
        self.orb_sym = [PointGroup[self.pg]
                        [int(i)] for i in cont_dict.get('orbsym')]

        data = []
        for l in ints.split('\n'):
            ll = l.strip()
            if len(ll) == 0 or ll.strip()[0] == '!':
                continue
            ll = ll.split()
            d = float(ll[0])
            i, j, k, l = [int(x) for x in ll[1:]]
            data.append((i, j, k, l, d))
        if not self.uhf:
            self.h1e = np.zeros((self.n_sites, self.n_sites))
            self.g2e = np.zeros((self.n_sites, self.n_sites,
                                 self.n_sites, self.n_sites))
            for i, j, k, l, d in data:
                if i + j + k + l == 0:
                    self.const_e = d
                elif k + l == 0:
                    self.h1e[i - 1, j - 1] = self.h1e[j - 1, i - 1] = d
                else:
                    self.g2e[i - 1, j - 1, k - 1, l - 1] = d
                    if not self.general:
                        self.g2e[j - 1, i - 1, k - 1, l - 1] = d
                        self.g2e[j - 1, i - 1, l - 1, k - 1] = d
                        self.g2e[i - 1, j - 1, l - 1, k - 1] = d
                        self.g2e[k - 1, l - 1, i - 1, j - 1] = d
                        self.g2e[k - 1, l - 1, j - 1, i - 1] = d
                        self.g2e[l - 1, k - 1, j - 1, i - 1] = d
                        self.g2e[l - 1, k - 1, i - 1, j - 1] = d
            if self.mu != 0:
                self.h1e -= self.mu * np.identity(self.n_sites)
        else:
            self.h1e = (
                np.zeros((self.n_sites, self.n_sites)),
                np.zeros((self.n_sites, self.n_sites)))
            self.g2e = (
                np.zeros((self.n_sites, self.n_sites,
                          self.n_sites, self.n_sites)),
                np.zeros((self.n_sites, self.n_sites,
                          self.n_sites, self.n_sites)),
                np.zeros((self.n_sites, self.n_sites, self.n_sites, self.n_sites)))
            ip = 0
            for i, j, k, l, d in data:
                if i + j + k + l == 0:
                    ip += 1
                    if ip == 6:
                        self.const_e = d
                elif k + l == 0:
                    assert ip == 3 or ip == 4
                    self.h1e[ip - 3][i - 1, j - 1] = d
                    if not self.tgeneral:
                        self.h1e[ip - 3][j - 1, i - 1] = d
                else:
                    ig = [0, 2, 1][ip]
                    self.g2e[ig][i - 1, j - 1, k - 1, l - 1] = d
                    if not self.general:
                        self.g2e[ig][j - 1, i - 1, k - 1, l - 1] = d
                        self.g2e[ig][j - 1, i - 1, l - 1, k - 1] = d
                        self.g2e[ig][i - 1, j - 1, l - 1, k - 1] = d
                    if not self.general and (ip == 0 or ip == 1):
                        self.g2e[ig][k - 1, l - 1, i - 1, j - 1] = d
                        self.g2e[ig][k - 1, l - 1, j - 1, i - 1] = d
                        self.g2e[ig][l - 1, k - 1, j - 1, i - 1] = d
                        self.g2e[ig][l - 1, k - 1, i - 1, j - 1] = d
            if self.mu != 0:
                self.h1e[0] -= self.mu * np.identity(self.n_sites)
                self.h1e[1] -= self.mu * np.identity(self.n_sites)
        return self

    def write(self, filename, tol=1E-13):
        """
        Write FCI options and integrals to FCIDUMP file.

        Args:
            filename : str
            tol : threshold for terms written into file
        """
        with open(filename, 'w') as fout:
            fout.write(' &FCI NORB=%4d,NELEC=%2d,MS2=%d,\n' %
                       (self.n_sites, self.n_elec, self.twos))
            if self.orb_sym is not None and len(self.orb_sym) > 0:
                fout.write('  ORBSYM=%s,\n' % ','.join(
                    [str(PointGroup[self.pg].index(x)) for x in self.orb_sym]))
            else:
                fout.write('  ORBSYM=%s\n' % ('1,' * self.n_sites))
            fout.write('  ISYM=%d,\n' % PointGroup[self.pg].index(self.ipg))
            if isinstance(self.h1e, tuple) and len(self.h1e) == 2:
                fout.write('  IUHF=1,\n')
                assert isinstance(self.g2e, tuple)
            if self.general:
                fout.write('  IGENERAL=1,\n')
            if self.tgeneral:
                fout.write('  ITGENERAL=1,\n')
            fout.write(' &END\n')
            output_format = '%20.16f%4d%4d%4d%4d\n'
            npair = self.n_sites * (self.n_sites + 1) // 2
            nmo = self.n_sites

            def write_eri(fout, eri):
                assert eri.ndim in [1, 2, 4]
                if eri.ndim == 4:
                    # general
                    assert(eri.size == nmo ** 4)
                    for i in range(nmo):
                        for j in range(nmo):
                            for k in range(nmo):
                                for l in range(nmo):
                                    if abs(eri[i, j, k, l]) > tol:
                                        fout.write(output_format % (
                                            eri[i, j, k, l], i + 1, j + 1, k + 1, l + 1))
                elif eri.ndim == 2:
                    # 4-fold symmetry
                    assert eri.size == npair ** 2
                    ij = 0
                    for i in range(nmo):
                        for j in range(0, i + 1):
                            kl = 0
                            for k in range(0, nmo):
                                for l in range(0, k + 1):
                                    if abs(eri[ij, kl]) > tol:
                                        fout.write(output_format % (
                                            eri[ij, kl], i + 1, j + 1, k + 1, l + 1))
                                    kl += 1
                            ij += 1
                else:
                    # 8-fold symmetry
                    assert eri.size == npair * (npair + 1) // 2
                    ij = 0
                    ijkl = 0
                    for i in range(nmo):
                        for j in range(0, i + 1):
                            kl = 0
                            for k in range(0, i + 1):
                                for l in range(0, k + 1):
                                    if ij >= kl:
                                        if abs(eri[ijkl]) > tol:
                                            fout.write(output_format % (
                                                eri[ijkl], i + 1, j + 1, k + 1, l + 1))
                                        ijkl += 1
                                    kl += 1
                            ij += 1

            def write_h1e(fout, hx, tgen=False):
                h = hx.reshape(nmo, nmo)
                for i in range(nmo):
                    for j in range(0, nmo if tgen else i + 1):
                        if abs(h[i, j]) > tol:
                            fout.write(output_format %
                                       (h[i, j], i + 1, j + 1, 0, 0))
            
            def fold_eri(eri, fold=1):
                if fold == 1:
                    return eri
                elif fold == 8:
                    xeri = np.zeros((npair * (npair + 1) // 2, ))
                    ij = 0
                    ijkl = 0
                    for i in range(nmo):
                        for j in range(0, i + 1):
                            kl = 0
                            for k in range(0, i + 1):
                                for l in range(0, k + 1):
                                    if ij >= kl:
                                        xeri[ijkl] = eri[i, j, k, l]
                                        ijkl += 1
                                    kl += 1
                            ij += 1
                    return xeri
                elif fold == 4:
                    xeri = np.zeros((npair, npair))
                    ij = 0
                    for i in range(nmo):
                        for j in range(0, i + 1):
                            kl = 0
                            for k in range(0, nmo):
                                for l in range(0, k + 1):
                                    xeri[ij, kl] = eri[i, j, k, l]
                                    kl += 1
                            ij += 1

            if isinstance(self.g2e, tuple):
                assert len(self.g2e) == 3
                vaa, vab, vbb = self.g2e
                if not self.general:
                    vaa = fold_eri(vaa, 8)
                    vab = fold_eri(vab, 4)
                    vbb = fold_eri(vbb, 8)
                    assert vaa.ndim == vbb.ndim == 1 and vab.ndim == 2
                else:
                    assert vaa.ndim == 4 and vbb.ndim == 4 and vab.ndim == 4
                for eri in [vaa, vbb, vab]:
                    write_eri(fout, eri)
                    fout.write(output_format % (0, 0, 0, 0, 0))
                assert len(self.h1e) == 2
                for hx in self.h1e:
                    write_h1e(fout, hx, self.tgeneral)
                    fout.write(output_format % (0, 0, 0, 0, 0))
                fout.write(output_format % (self.const_e, 0, 0, 0, 0))
            else:
                if not self.general:
                    eri = fold_eri(self.g2e, 8)
                else:
                    eri = self.g2e
                write_eri(fout, eri)
                write_h1e(fout, self.h1e, self.tgeneral)
                fout.write(output_format % (self.const_e, 0, 0, 0, 0))
