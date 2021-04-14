
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

"""
Partially contracted tensor network (MPE) with
methods for DMRG-like sweep algorithm.

CachedMPE enables swapping with disk storage
to reduce memory requirement.
"""

import numpy as np
import os
import time
import pickle
from functools import reduce
from .linalg import davidson, conjugate_gradient
from ..algorithms.dmrg import DMRG
from ..algorithms.tddmrg import TDDMRG
from ..algorithms.linear import Linear
from ..algorithms.green import GreensFunction


def implements(np_func):
    global _numpy_func_impls
    return lambda f: (_numpy_func_impls.update({np_func: f})
                      if np_func not in _numpy_func_impls else None,
                      _numpy_func_impls[np_func])[1]


_me_numpy_func_impls = {}
_numpy_func_impls = _me_numpy_func_impls


class MPE:
    """Matrix Product Expectation (MPE).
    Original and partially contracted tensor network <bra|mpo|ket>."""

    def __init__(self, bra, mpo, ket, opts=None, do_canon=True, idents=None, mpi=False):
        self.bra = bra
        self.mpo = mpo
        self.ket = ket
        assert self.bra.n_sites == self.ket.n_sites
        self.left_envs = {}
        self.right_envs = {}
        self.do_canon = do_canon
        if opts is not None:
            self.bra.opts = self.mpo.opts = self.ket.opts = opts
        if idents is None:
            self.idents = self._init_identities()
        else:
            self.idents = idents
        self.t_ctr = 0
        self.t_rot = 0
        self.mpi = mpi
        if self.mpi:
            from mpi4py import MPI
            self.rank = MPI.COMM_WORLD.Get_rank()
            self.comm = MPI.COMM_WORLD

    @property
    def nbytes(self):
        nls = [v.nbytes for v in self.left_envs.values() if v is not None]
        nrs = [v.nbytes for v in self.right_envs.values() if v is not None]
        return sum(nls) + sum(nrs)

    def copy_shell(self, bra, mpo, ket):
        return MPE(bra=bra, mpo=mpo, ket=ket, do_canon=self.do_canon, idents=self.idents, mpi=self.mpi)

    def copy(self):
        bra = self.bra.__class__(
            tensors=self.bra.tensors[:], opts=self.bra.opts)
        ket = self.ket.__class__(
            tensors=self.ket.tensors[:], opts=self.ket.opts)
        mpo = self.mpo.__class__(
            tensors=self.mpo.tensors[:], const=self.mpo.const, opts=self.mpo.opts)
        mpe = self.__class__(bra=bra, mpo=mpo, ket=ket,
                             do_canon=self.do_canon, idents=self.idents, mpi=self.mpi)
        mpe.left_envs = self.left_envs.copy()
        mpe.right_envs = self.right_envs.copy()
        return mpe

    def __array_function__(self, func, types, args, kwargs):
        if func not in _me_numpy_func_impls:
            return NotImplemented
        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented
        return _me_numpy_func_impls[func](*args, **kwargs)

    def _init_identities(self):
        qbl, qkl, qml = self.bra[0].infos[0], self.ket[0].infos[0], self.mpo[0].infos[0]
        qbr, qkr, qmr = self.bra[-1].infos[-1], self.ket[-1].infos[-1], self.mpo[-1].infos[-1]
        l_mpo_id = self.mpo[0].ones(
            bond_infos=(qml, qbl, qkl, qml), pattern="++--")
        r_mpo_id = self.mpo[-1].ones(bond_infos=(qmr,
                                                 qbr, qkr, qmr), pattern="++--", dq=self.mpo.dq)
        l_bra_id = self.bra[0].ones(bond_infos=(qbl, ))
        r_bra_id = self.bra[-1].ones(bond_infos=(qbl, ))
        l_ket_id = self.ket[0].ones(bond_infos=(qkl, ))
        r_ket_id = self.ket[-1].ones(bond_infos=(qkl, ))
        return l_mpo_id, r_mpo_id, l_bra_id, r_bra_id, l_ket_id, r_ket_id

    def _left_canonicalize_site(self, mps, i):
        """Left canonicalize mps at site i."""
        mps[i], r = mps[i].left_canonicalize()
        mps[i + 1] = np.tensordot(r, mps[i + 1], axes=1)

    def _right_canonicalize_site(self, mps, i):
        """Right canonicalize mps at site i."""
        l, mps[i] = mps[i].right_canonicalize()
        mps[i - 1] = np.tensordot(mps[i - 1], l, axes=1)

    def _left_contract_rotate(self, i, prev=None):
        """Left canonicalize bra and ket at site i.
        Contract left-side contracted mpo with mpo at site i."""
        if i == -1:
            return self.idents[0]
        # contract
        t = time.perf_counter()
        r = prev.hdot(self.mpo[i])
        self.t_ctr += time.perf_counter() - t
        # left canonicalize
        if self.do_canon:
            self._left_canonicalize_site(self.ket, i)
            if self.ket is not self.bra:
                self._left_canonicalize_site(self.bra, i)
            if self.mpi:
                self.ket[i] = self.comm.bcast(self.ket[i], root=0)
                if self.ket is not self.bra:
                    self.bra[i] = self.comm.bcast(self.bra[i], root=0)
        # rotate
        du, dd = self.bra[i].ndim - 2, self.ket[i].ndim - 2
        t = time.perf_counter()
        r = np.tensordot(r, self.ket[i], axes=(
            [*range(du + 2, du + dd + 3)], [*range(0, dd + 1)]))
        r = np.tensordot(self.bra[i].conj(), r, axes=(
            [*range(0, du + 1)], [*range(1, du + 2)]))
        r = r.transpose((1, 0, 3, 2))
        self.t_rot += time.perf_counter() - t
        return r

    def _right_contract_rotate(self, i, prev=None):
        """Right canonicalize bra and ket at site i.
        Contract right-side contracted mpo with mpo at site i."""
        if i == self.n_sites:
            return self.idents[1]
        # contract
        t = time.perf_counter()
        r = self.mpo[i].hdot(prev)
        self.t_ctr += time.perf_counter() - t
        # right canonicalize
        if self.do_canon:
            self._right_canonicalize_site(self.ket, i)
            if self.ket is not self.bra:
                self._right_canonicalize_site(self.bra, i)
            if self.mpi:
                self.ket[i] = self.comm.bcast(self.ket[i], root=0)
                if self.ket is not self.bra:
                    self.bra[i] = self.comm.bcast(self.bra[i], root=0)
        # rotate
        du, dd = self.bra[i].ndim - 2, self.ket[i].ndim - 2
        t = time.perf_counter()
        r = np.tensordot(r, self.ket[i], axes=(
            [*range(du + 2, du + dd + 3)], [*range(1, dd + 2)]))
        r = np.tensordot(self.bra[i].conj(), r, axes=(
            [*range(1, du + 2)], [*range(1, du + 2)]))
        r = r.transpose((1, 0, 3, 2))
        self.t_rot += time.perf_counter() - t
        return r

    def build_envs(self, l=0, r=2):
        """Canonicalize bra and ket around sites [l, r). Contract mpo around sites [l, r)."""
        self.left_envs[-2] = None
        self.right_envs[self.n_sites + 1] = None
        for i in range(-1, l):
            if i not in self.left_envs or self.left_envs[i] is None:
                self.left_envs[i] = self._left_contract_rotate(
                    i, prev=self.left_envs[i - 1])
        for i in range(self.n_sites, r - 1, -1):
            if i not in self.right_envs or self.right_envs[i] is None:
                self.right_envs[i] = self._right_contract_rotate(
                    i, prev=self.right_envs[i + 1])

    def _effective_mpo(self, l=0, r=2):
        """Get mpo in sub-system with sites [l, r)"""
        tensors = [self.left_envs[l - 1], *self.mpo[l:r], self.right_envs[r]]
        tensors[:2] = [tensors[0].hdot(tensors[1])]
        if len(tensors) > 2:
            tensors[-2:] = [tensors[-2].hdot(tensors[-1])]
        return self.mpo.__class__(tensors=tensors, const=self.mpo.const, opts=self.mpo.opts)

    def _effective_mps(self, mps, lid, rid, l=0, r=2):
        """Get mps in sub-system with sites [l, r)"""
        tensors = [lid, *self.ket[l:r], rid]
        tensors[:2] = [
            reduce(lambda a, b: np.tensordot(a, b, axes=0), tensors[:2])]
        tensors[-2:] = [reduce(lambda a, b: np.tensordot(a,
                                                         b, axes=0), tensors[-2:])]
        return self.ket.__class__(tensors=tensors, opts=mps.opts)

    def _effective_bra(self, l=0, r=2):
        """Get bra in sub-system with sites [l, r)"""
        return self._effective_mps(self.bra, self.idents[2], self.idents[3], l=l, r=r)

    def _effective_ket(self, l=0, r=2):
        """Get ket in sub-system with sites [l, r)"""
        return self._effective_mps(self.ket, self.idents[4], self.idents[5], l=l, r=r)

    def _embedded_mps(self, mps, lid, rid):
        """Change mps format for embedding into larger system."""
        tensors = mps.tensors
        tensors[0] = np.tensordot(lid, tensors[0], axes=1)
        tensors[-1] = np.tensordot(tensors[-1], rid, axes=1)
        return mps.__class__(tensors=tensors, opts=mps.opts)

    def _embedded_bra(self):
        """Change bra format for embedding into larger system."""
        return self._embedded_mps(self.bra, self.idents[2], self.idents[3])

    def _embedded_ket(self):
        """Change ket format for embedding into larger system."""
        return self._embedded_mps(self.ket, self.idents[4], self.idents[5])

    def _effective(self, l=0, r=2):
        """Get sub-system with sites [l, r)"""
        mpe = self.copy()
        mpe.build_envs(l=l, r=r)
        eff_ket = mpe._effective_ket(l=l, r=r)
        eff_bra = eff_ket if mpe.bra is mpe.ket else mpe._effective_bra(
            l=l, r=r)
        eff_mpo = mpe._effective_mpo(l=l, r=r)
        return mpe.copy_shell(bra=eff_bra, mpo=eff_mpo, ket=eff_ket)

    def _embedded(self, me, l=0, r=2):
        """Modify sub-system with sites [l, r)"""
        self.build_envs(l=l, r=r)
        self.ket[l:r] = me._embedded_ket().tensors
        if me.ket is me.bra:
            self.bra[l:r] = self.ket[l:r]
        else:
            self.bra[l:r] = me._embedded_bra().tensors
        for i in range(l, self.n_sites + 1):
            self.left_envs[i] = None
        for i in range(r - 1, -2, -1):
            self.right_envs[i] = None

    def __getitem__(self, idx):
        """Return sub-system including sites specified by ``idx``"""
        if isinstance(idx, int):
            l, r = idx, idx + 1
        elif isinstance(idx, slice):
            l = 0 if idx.start is None else idx.start
            r = self.n_sites if idx.stop is None else idx.stop
            l = l if l >= 0 else self.n_sites + l
            r = r if r >= 0 else self.n_sites + r
            assert r > l
        else:
            raise TypeError("Unknown index %r" % idx)
        return self._effective(l, r)

    def __setitem__(self, idx, me):
        """Modify sub-system including sites specified by ``idx``"""
        if isinstance(idx, int):
            l, r = idx, idx + 1
        elif isinstance(idx, slice):
            l = 0 if idx.start is None else idx.start
            r = self.n_sites if idx.stop is None else idx.stop
            l = l if l >= 0 else self.n_sites + l
            r = r if r >= 0 else self.n_sites + r
            assert r > l
        else:
            raise TypeError("Unknown index %r" % idx)
        self._embedded(me, l, r)

    @property
    def expectation(self):
        """<bra|mpo|ket> for the whole system."""
        r = np.dot(self.bra.conj(), self.mpo @ self.ket)
        if self.mpi:
            from mpi4py import MPI
            r = self.comm.allreduce(r, op=MPI.SUM)
        return r

    @staticmethod
    def _eigs(x, iprint=False, fast=False, conv_thrd=1E-7, max_iter=500):
        """Return ground-state energy and ground-state effective MPE."""
        if fast and x.ket.n_sites == 1 and x.mpo.n_sites == 2:
            from .flat_functor import FlatSparseFunctor
            pattern = '++' + '+' * (x.ket[0].ndim - 4) + '-+'
            fst = FlatSparseFunctor(x.mpo, pattern=pattern, mpi=x.mpi)
            w, v, ndav = davidson(
                fst, [fst.prepare_vector(x.ket[0])], k=1, max_iter=max_iter, iprint=iprint, conv_thrd=conv_thrd, mpi=x.mpi)
            v = [x.ket.__class__(
                tensors=[fst.finalize_vector(v[0])], opts=x.ket.opts)]
            nflop = fst.nflop
        else:
            w, v, ndav = davidson(
                x.mpo, [x.ket], k=1, max_iter=max_iter, iprint=iprint, conv_thrd=conv_thrd, mpi=x.mpi)
            nflop = 0
        mpe = x.copy_shell(bra=v[0], mpo=x.mpo, ket=v[0])
        return w[0], mpe, ndav, nflop

    def eigs(self, iprint=False, fast=False, conv_thrd=1E-7, max_iter=500):
        """Return ground-state energy and ground-state effective MPE."""
        return self._eigs(self, iprint=iprint, fast=fast, conv_thrd=conv_thrd, max_iter=max_iter)

    def multiply(self, fast=False):
        if fast and self.ket.n_sites == 1 and self.mpo.n_sites == 2:
            from .flat_functor import FlatSparseFunctor
            pattern = '++' + '+' * (self.ket[0].ndim - 4) + '-+'
            fst = FlatSparseFunctor(
                self.mpo, pattern=pattern, symmetric=False, mpi=self.mpi)
            v = fst.finalize_vector(fst @ fst.prepare_vector(self.ket[0]))
            v = self.ket.__class__(tensors=[v], opts=self.ket.opts)
        else:
            v = self.mpo @ self.ket
        w = np.linalg.norm(v)
        mpe = self.copy_shell(bra=v, mpo=self.mpo, ket=self.ket)
        return w, mpe, 1

    def rk4(self, dt, fast=False, eval_ener=True):
        if fast and self.ket.n_sites == 1 and self.mpo.n_sites == 2:
            from .flat_functor import FlatSparseFunctor
            pattern = '++' + '+' * (self.ket[0].ndim - 4) + '-+'
            fst = FlatSparseFunctor(
                self.mpo, pattern=pattern, symmetric=False, mpi=self.mpi)
            b = fst.prepare_vector(self.ket[0])
            k1 = dt * (fst @ b)
            k2 = dt * (fst @ (0.5 * k1 + b))
            k3 = dt * (fst @ (0.5 * k2 + b))
            k4 = dt * (fst @ (k3 + b))
            r1 = b + (31.0 / 162) * k1 + (14.0 / 162) * k2 + \
                (14.0 / 162) * k3 + (-5.0 / 162) * k4
            r2 = b + (16.0 / 81) * k1 + (20.0 / 81) * k2 + \
                (20.0 / 81) * k3 + (-2.0 / 81) * k4
            r3 = b + k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6
            g = np.linalg.norm(r3)
            w = np.dot(r3.conj(), fst @ r3) / g ** 2 if eval_ener else 0
            kets = [self.ket.__class__(tensors=[fst.finalize_vector(x)],
                                       opts=self.ket.opts) for x in [b, r1, r2, r3]]
            nflop = fst.nflop
        else:
            raise NotImplementedError
        mpe = self.copy_shell(bra=kets[0], mpo=self.mpo, ket=kets[0])
        return w, g, kets, mpe, 4 + eval_ener, nflop

    def solve_gf(self, ket, omega, eta, iprint=False, fast=False, conv_thrd=1E-7):
        if fast and self.ket.n_sites == 1 and self.mpo.n_sites == 2:
            from .flat_functor import GreensFunctionFunctor
            pattern = '++' + '+' * (self.ket[0].ndim - 4) + '-+'
            fst = GreensFunctionFunctor(
                self.mpo, pattern=pattern, omega=omega, eta=eta, mpi=self.mpi)
            b = (-eta) * fst.prepare_vector(ket)
            x = fst.prepare_vector(self.ket[0])
            iw, x, ncg = conjugate_gradient(
                fst, x, b, iprint=iprint, conv_thrd=conv_thrd)
            r = fst.real_part(x)
            rw = np.dot(r, b) / (-eta)
            iw /= -eta
            x = self.ket.__class__(
                tensors=[fst.finalize_vector(x)], opts=self.ket.opts)
            r = self.ket.__class__(
                tensors=[fst.finalize_vector(r)], opts=self.ket.opts)
        else:
            raise NotImplementedError
        mpe = self.copy_shell(bra=r, mpo=self.mpo, ket=x)
        return rw + iw * 1j, mpe, ncg

    def dmrg(self, bdims, noises=None, dav_thrds=None, n_sweeps=10, tol=1E-6, max_iter=500, dot=2, iprint=2, forward=True):
        return DMRG(self, bdims, noises, dav_thrds, max_iter=max_iter, iprint=iprint).solve(n_sweeps, tol, dot, forward=forward)

    def tddmrg(self, bdims, dt, n_sweeps=10, n_sub_sweeps=2, dot=2, iprint=2, forward=True, normalize=True, **kwargs):
        return TDDMRG(self, bdims, iprint=iprint, **kwargs).solve(dt, n_sweeps, n_sub_sweeps, dot, forward=forward, normalize=normalize)

    def linear(self, bdims, noises=None, cg_thrds=None, n_sweeps=10, tol=1E-6, dot=2, iprint=2):
        return Linear(self, bdims, noises, cg_thrds, iprint=iprint).solve(n_sweeps, tol, dot)

    def greens_function(self, mpo, omega, eta, bdims, noises=None, cg_thrds=None, n_sweeps=10, tol=1E-6, dot=2, iprint=2):
        return GreensFunction(self, mpo, omega, eta, bdims, noises, cg_thrds, iprint=iprint).solve(n_sweeps, tol, dot)

    @property
    def n_sites(self):
        return self.ket.n_sites


class CachedMPE(MPE):
    """MPE for large system. Using disk storage to reduce memory usage."""

    def __init__(self, bra, mpo, ket, opts=None, do_canon=True, idents=None, tag='MPE', scratch=None, maxsize=3, mpi=False):
        super().__init__(bra, mpo, ket, opts=opts, do_canon=do_canon, idents=idents, mpi=mpi)
        self.tag = tag
        self.scratch = scratch if scratch is not None else os.environ['TMPDIR']
        self.cached = []
        self.maxsize = maxsize

    @property
    def nbytes(self):
        return sum([v[1].nbytes for v in self.cached])

    def copy(self):
        bra = self.bra.__class__(
            tensors=self.bra.tensors[:], opts=self.bra.opts)
        ket = self.ket.__class__(
            tensors=self.ket.tensors[:], opts=self.ket.opts)
        mpo = self.mpo.__class__(
            tensors=self.mpo.tensors[:], const=self.mpo.const, opts=self.mpo.opts)
        mpe = self.__class__(bra=bra, mpo=mpo, ket=ket, do_canon=self.do_canon,
                             idents=self.idents, tag=self.tag + '@TMP',
                             scratch=self.scratch, maxsize=self.maxsize, mpi=self.mpi)
        mpe.cached = self.cached.copy()
        mpe.left_envs = self.left_envs.copy()
        mpe.right_envs = self.right_envs.copy()
        return mpe

    def _get_filename(self, left, i):
        if self.mpi:
            return "%s-%d.%s.%d" % (self.tag, self.rank, "L" if left else "R", i)
        else:
            return "%s.%s.%d" % (self.tag, "L" if left else "R", i)

    def _load_data(self, fn):
        if fn in [c[0] for c in self.cached]:
            return [c[1] for c in self.cached if c[0] == fn][0]
        else:
            x = pickle.load(open(os.path.join(self.scratch, fn), 'rb'))
            self.cached.append([fn, x, True])
            if len(self.cached) > self.maxsize:
                if not self.cached[0][2]:
                    pickle.dump(self.cached[0][1], open(
                        os.path.join(self.scratch, self.cached[0][0]), 'wb'))
                self.cached = self.cached[1:]
            return x

    def _save_data(self, fn, x):
        if fn in [c[0] for c in self.cached]:
            [c for c in self.cached if c[0] == fn][1:] = [x, False]
        else:
            self.cached.append([fn, x, False])
            if len(self.cached) > self.maxsize:
                if not self.cached[0][2]:
                    pickle.dump(self.cached[0][1], open(
                        os.path.join(self.scratch, self.cached[0][0]), 'wb'))
                self.cached = self.cached[1:]

    def _left_contract_rotate(self, i, prev=None):
        """Left canonicalize bra and ket at site i.
        Contract left-side contracted mpo with mpo at site i."""
        if i == -1:
            return self.idents[0]
        if isinstance(prev, str):
            prev = self._load_data(prev)
        x = super()._left_contract_rotate(i, prev)
        fn = self._get_filename(True, i)
        self._save_data(fn, x)
        return fn

    def _right_contract_rotate(self, i, prev=None):
        """Right canonicalize bra and ket at site i.
        Contract right-side contracted mpo with mpo at site i."""
        if i == self.n_sites:
            return self.idents[1]
        if isinstance(prev, str):
            prev = self._load_data(prev)
        x = super()._right_contract_rotate(i, prev)
        fn = self._get_filename(False, i)
        self._save_data(fn, x)
        return fn

    def _effective_mpo(self, l=0, r=2):
        """Get mpo in sub-system with sites [l, r)"""
        fnl = self.left_envs[l - 1]
        fnr = self.right_envs[r]
        if isinstance(fnl, str):
            self.left_envs[l - 1] = self._load_data(fnl)
        if isinstance(fnr, str):
            self.right_envs[r] = self._load_data(fnr)
        mpo = super()._effective_mpo(l=l, r=r)
        self.left_envs[l - 1] = fnl
        self.right_envs[r] = fnr
        return mpo
