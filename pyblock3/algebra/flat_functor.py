
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
:math:`H|\\psi\\rangle` operation with high performance.
"""

import numpy as np
from .flat import FlatSparseTensor, ENABLE_FAST_IMPLS
from .symmetry import SZ
from .mps import MPS


def flat_sparse_matmul_init(spt, pattern, dq, symmetric, mpi):
    tl, tr = [ts.infos for ts in spt.tensors]
    dl, dr = [(len(tl) - 2) // 2, (len(tr) - 2) // 2]
    cinfos = [tl[0], *tl[1 + dl:1 + dl + dl], *tr[1 + dr:1 + dr + dr], tr[-1]]
    vinfos = [tl[0], *tl[1:1 + dl], *tr[1:1 + dr], tr[-1]]
    winfos = [tl[-1] | tr[0], *tr[1: 1 + dr], *tl[1 + dl: 1 + dl + dl]]
    if mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        for cvw in [cinfos, vinfos, winfos]:
            for ic, c in enumerate(cvw):
                cvw[ic] = comm.allreduce(c, op=MPI.BOR)
    if symmetric:
        vinfos = [x | y for x, y in zip(cinfos, vinfos)]
        vmat = FlatSparseTensor.zeros(vinfos, pattern, dq)
        cmat = vmat
    else:
        cmat = FlatSparseTensor.zeros(cinfos, pattern, dq)
        vmat = FlatSparseTensor.zeros(vinfos, pattern, dq)
    work = FlatSparseTensor.zeros(
        winfos, '+' + pattern[1 + dl:1 + dl + dr] + pattern[1:1 + dl], dq)
    return dl, dr, cmat, vmat, work


if ENABLE_FAST_IMPLS:
    import block3.sz as block3

    def flat_sparse_matmul_init_impl(spt, pattern, dq, symmetric, mpi):
        fdq = dq.to_flat() if dq is not None else SZ(0, 0, 0).to_flat()
        l, r = spt.tensors
        lo, le = l.odd, l.even
        ro, re = r.odd, r.even
        dl, dr, cinfos, vinfos = block3.flat_sparse_tensor.matmul_init(
            lo.q_labels, lo.shapes, le.q_labels, le.shapes, ro.q_labels,
            ro.shapes, re.q_labels, re.shapes)
        if mpi:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            for cvw in [cinfos, vinfos]:
                for ic, c in enumerate(cvw):
                    cvw[ic] = comm.allreduce(c, op=MPI.BOR)
        if symmetric:
            vinfos = vinfos.__class__([x | y for x, y in zip(cinfos, vinfos)])
            vdqs, vshs, vidxs = block3.flat_sparse_tensor.skeleton(
                vinfos, pattern, fdq)
            cdqs, cshs, cidxs = vdqs, vshs, vidxs
        else:
            cdqs, cshs, cidxs = block3.flat_sparse_tensor.skeleton(
                cinfos, pattern, fdq)
            vdqs, vshs, vidxs = block3.flat_sparse_tensor.skeleton(
                vinfos, pattern, fdq)
        axru = np.arange(1 + dr, 1 + dr + dr, dtype=np.int32)
        axrd = np.arange(dl, dl + dr, dtype=np.int32)
        if ro.n_blocks != 0 and re.n_blocks != 0:
            qxr = np.concatenate(
                [ro.q_labels[:, :-1], re.q_labels[:, :-1]], axis=0)
            sxr = np.concatenate(
                [ro.shapes[:, :-1], re.shapes[:, :-1]], axis=0)
        elif ro.n_blocks == 0:
            qxr, sxr = re.q_labels[:, :-1], re.shapes[:, :-1]
        else:
            qxr, sxr = ro.q_labels[:, :-1], ro.shapes[:, :-1]
        wdqs, wshs, widxs = block3.flat_sparse_tensor.tensordot_skeleton(
            qxr, sxr, cdqs[:, 1:-1], cshs[:, 1:-1], axru, axrd)
        vdata = np.zeros((vidxs[-1], ), dtype=float)
        cdata = vdata if symmetric else np.zeros((cidxs[-1], ), dtype=float)
        wdata = np.zeros((widxs[-1], ), dtype=float)
        return dl, dr, FlatSparseTensor(cdqs, cshs, cdata, cidxs), \
            FlatSparseTensor(vdqs, vshs, vdata, vidxs), FlatSparseTensor(
                wdqs, wshs, wdata, widxs)
    flat_sparse_matmul_init = flat_sparse_matmul_init_impl
    flat_sparse_matmul_plan = block3.flat_sparse_tensor.matmul_plan
    flat_sparse_matmul = block3.flat_sparse_tensor.matmul
    flat_sparse_transpose = block3.flat_sparse_tensor.transpose
    flat_sparse_diag = block3.flat_sparse_tensor.diag


class FlatSparseFunctor:
    def __init__(self, spt, pattern, dq=None, symmetric=True, mpi=False):
        assert ENABLE_FAST_IMPLS
        assert isinstance(spt, MPS)
        self.op = spt
        dl, dr, self.cmat, self.vmat, self.work = flat_sparse_matmul_init(
            self.op, pattern, dq, symmetric, mpi=mpi)
        self.axtr = np.array(
            (*range(dr + 1, dr + dl + 1), 0, *range(1, dr + 1)), dtype=np.int32)
        self.work2 = FlatSparseTensor(
            self.work.q_labels[:, self.axtr], self.work.shapes[:, self.axtr], np.zeros_like(self.work.data), self.work.idxs)
        self.dmat = None
        self.dlr = dl, dr
        self.plan = self._matmul_plan(dl, dr)
        self.nflop = 0
        self.mpi = mpi
        if self.mpi:
            from mpi4py import MPI
            self.rank = MPI.COMM_WORLD.Get_rank()
            self.comm = MPI.COMM_WORLD

    def diag(self):
        if self.dmat is None:
            self.dmat = self._prepare_diag(*self.dlr)
            if self.op.const != 0:
                self.dmat += self.op.const
            if self.mpi:
                from mpi4py import MPI
                if self.rank == 0:
                    self.comm.Reduce(
                        MPI.IN_PLACE, [self.dmat, MPI.DOUBLE], op=MPI.SUM, root=0)
                else:
                    self.comm.Reduce([self.dmat, MPI.DOUBLE], None, op=MPI.SUM, root=0)
        return self.dmat

    def _prepare_diag(self, dl, dr):
        l, r = self.op.tensors
        le, re = l.even, r.even
        axlu = np.arange(1, 1 + dl, dtype=np.int32)
        axld = np.arange(1 + dl, 1 + dl + dl, dtype=np.int32)
        axru = np.arange(1, 1 + dr, dtype=np.int32)
        axrd = np.arange(1 + dr, 1 + dr + dr, dtype=np.int32)
        led = FlatSparseTensor(
            *flat_sparse_diag(le.q_labels, le.shapes, le.data, le.idxs, axlu, axld))
        red = FlatSparseTensor(
            *flat_sparse_diag(re.q_labels, re.shapes, re.data, re.idxs, axru, axrd))
        diag = np.tensordot(led, red, axes=1)
        return FlatSparseTensor.zeros_like(self.vmat).cast_assign(diag).data

    def _matmul_plan(self, dl, dr):
        axru = np.arange(1 + dr, 1 + dr + dr, dtype=np.int32)
        axrd = np.arange(dl, dl + dr, dtype=np.int32)
        axlu = np.arange(dl, dl + dl + 1, dtype=np.int32)
        axld = np.arange(0, dl + 1, dtype=np.int32)
        l, r = self.op.tensors
        c, v, w, w2 = self.cmat, self.vmat, self.work, self.work2
        vqs, vshs = v.q_labels[:, 1:-1], v.shapes[:, 1:-1]
        cqs, cshs = c.q_labels[:, 1:-1], c.shapes[:, 1:-1]
        lo, le = l.odd, l.even
        ro, re = r.odd, r.even
        if ro.n_blocks != 0:
            plan_ro = flat_sparse_matmul_plan(
                ro.q_labels[:, :-1], ro.shapes[:, :-1], ro.idxs, cqs, cshs, c.idxs, axru, axrd, w.q_labels, w.idxs, True)
        else:
            plan_ro = np.zeros((0, 9), dtype=np.int32)
        if re.n_blocks != 0:
            plan_re = flat_sparse_matmul_plan(
                re.q_labels[:, :-1], re.shapes[:, :-1], re.idxs, cqs, cshs, c.idxs, axru, axrd, w.q_labels, w.idxs, False)
        else:
            plan_re = np.zeros((0, 9), dtype=np.int32)
        if lo.n_blocks != 0:
            plan_lo = flat_sparse_matmul_plan(
                lo.q_labels[:, 1:], lo.shapes[:, 1:], lo.idxs, w2.q_labels, w2.shapes, w2.idxs, axlu, axld, vqs, v.idxs, True)
        else:
            plan_lo = np.zeros((0, 9), dtype=np.int32)
        if le.n_blocks != 0:
            plan_le = flat_sparse_matmul_plan(
                le.q_labels[:, 1:], le.shapes[:, 1:], le.idxs, w2.q_labels, w2.shapes, w2.idxs, axlu, axld, vqs, v.idxs, False)
        else:
            plan_le = np.zeros((0, 9), dtype=np.int32)
        return plan_ro, plan_re, plan_lo, plan_le

    def __matmul__(self, other):
        assert other.size == self.cmat.idxs[-1]
        l, r = self.op.tensors
        pro, pre, plo, ple = self.plan
        self.work.data = np.zeros_like(self.work.data, dtype=other.dtype)
        self.work2.data = np.zeros_like(self.work2.data, dtype=other.dtype)
        self.vmat.data = np.zeros_like(self.vmat.data, dtype=other.dtype)
        self.nflop += flat_sparse_matmul(pro,
                                         r.odd.data, other, self.work.data)
        self.nflop += flat_sparse_matmul(pre,
                                         r.even.data, other, self.work.data)
        flat_sparse_transpose(self.work.shapes, self.work.data,
                              self.work.idxs, self.axtr, self.work2.data)
        self.nflop += flat_sparse_matmul(plo, l.odd.data,
                                         self.work2.data, self.vmat.data)
        self.nflop += flat_sparse_matmul(ple, l.even.data,
                                         self.work2.data, self.vmat.data)
        if self.op.const != 0:
            self.vmat.data += self.op.const * other
        if self.mpi:
            from mpi4py import MPI
            if self.rank == 0:
                self.comm.Reduce(
                    MPI.IN_PLACE, [self.vmat.data, MPI.DOUBLE], op=MPI.SUM, root=0)
            else:
                self.comm.Reduce([self.vmat.data, MPI.DOUBLE], None, op=MPI.SUM, root=0)
        return self.vmat.data

    def prepare_vector(self, spt):
        return FlatSparseTensor.zeros_like(self.cmat, dtype=spt.dtype).cast_assign(spt).data

    def finalize_vector(self, data):
        self.vmat.data = data
        return self.vmat.copy()


class GreensFunctionFunctor(FlatSparseFunctor):
    def __init__(self, spt, pattern, omega, eta, dq=None, symmetric=True, mpi=False):
        super().__init__(spt, pattern, dq=dq, symmetric=symmetric, mpi=mpi)
        self.omega = omega
        self.eta = eta

    def diag(self):
        if self.dmat is None:
            self.dmat = super().diag()
            self.dmat = (self.dmat + self.omega) ** 2 + self.eta ** 2
        return self.dmat

    def __matmul__(self, other):
        btmp = super().__matmul__(other)
        if self.omega != 0:
            btmp += self.omega * other
        c = super().__matmul__(btmp)
        if self.omega != 0:
            c += self.omega * btmp
        if self.eta != 0:
            c += (self.eta * self.eta) * other
        return c

    def real_part(self, other):
        r = super().__matmul__(other)
        if self.omega != 0:
            r += self.omega * other
        r /= -self.eta
        return r
