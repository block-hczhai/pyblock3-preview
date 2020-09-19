
import numpy as np
from .flat import FlatSparseTensor, ENABLE_FAST_IMPLS
from .symmetry import SZ
from .mps import MPS

def flat_sparse_matmul_init(spt, pattern, dq):
    tl, tr = [ts.infos for ts in spt.tensors]
    dl, dr = [(len(tl) - 2) // 2, (len(tr) - 2) // 2]
    vinfos = [tl[0], *[tl[i + 1] | tl[i + 1 + dl] for i in range(dl)], *[tr[i + 1] | tr[i + 1 + dr] for i in range(dr)], tr[-1]]
    winfos = [tl[-1] | tr[0], *tr[1: 1 + dr], *tl[1 + dl: 1 + dl + dl]]
    vmat = FlatSparseTensor.zeros(vinfos, pattern, dq)
    work = FlatSparseTensor.zeros(winfos, '+' + pattern[1 + dl:1 + dl + dr] + pattern[1:1 + dl], dq)
    return dl, dr, vmat, work

if ENABLE_FAST_IMPLS:
    import block3.flat_sparse_tensor
    def flat_sparse_matmul_init_impl(spt, pattern, dq):
        fdq = dq.to_flat() if dq is not None else SZ(0, 0, 0).to_flat()
        l, r = spt.tensors
        lo, le = l.odd, l.even
        ro, re = r.odd, r.even
        dl, dr, vinfos, winfos = block3.flat_sparse_tensor.matmul_init(
            lo.q_labels, lo.shapes, le.q_labels, le.shapes, ro.q_labels,
            ro.shapes, re.q_labels, re.shapes)
        vdqs, vshs, vidxs = block3.flat_sparse_tensor.skeleton(vinfos, pattern, fdq)
        wpattern = '+' + pattern[1 + dl:1 + dl + dr] + pattern[1:1 + dl]
        wdqs, wshs, widxs = block3.flat_sparse_tensor.skeleton(winfos, wpattern, fdq)
        vdata = np.zeros((vidxs[-1], ), dtype=float)
        wdata = np.zeros((widxs[-1], ), dtype=float)
        return dl, dr, FlatSparseTensor(vdqs, vshs, vdata, vidxs), FlatSparseTensor(wdqs, wshs, wdata, widxs)
    flat_sparse_matmul_init = flat_sparse_matmul_init_impl
    flat_sparse_matmul_plan = block3.flat_sparse_tensor.matmul_plan
    flat_sparse_matmul = block3.flat_sparse_tensor.matmul
    flat_sparse_transpose = block3.flat_sparse_tensor.transpose
    flat_sparse_diag = block3.flat_sparse_tensor.diag


class FlatSparseFunctor:
    def __init__(self, spt, pattern, dq=None):
        assert ENABLE_FAST_IMPLS
        assert isinstance(spt, MPS)
        self.op = spt
        dl, dr, self.vmat, self.work = flat_sparse_matmul_init(self.op, pattern, dq)
        self.axtr = np.array(
            (*range(dr + 1, dr + dl + 1), 0, *range(1, dr + 1)), dtype=np.int32)
        self.work2 = FlatSparseTensor(self.work.q_labels[:, self.axtr], self.work.shapes[:, self.axtr], np.zeros_like(self.work.data), self.work.idxs)
        self.dmat = self._prepare_diag(dl, dr)
        if self.op.const != 0:
            self.dmat += self.op.const
        self.plan = self._matmul_plan(dl, dr)

    def diag(self):
        return self.dmat

    def _prepare_diag(self, dl, dr):
        l, r = self.op.tensors
        le, re = l.even, r.even
        axlu = np.arange(1, 1 + dl, dtype=np.int32)
        axld = np.arange(1 + dl, 1 + dl + dl, dtype=np.int32)
        axru = np.arange(1, 1 + dr, dtype=np.int32)
        axrd = np.arange(1 + dr, 1 + dr + dr, dtype=np.int32)
        led = FlatSparseTensor(*flat_sparse_diag(le.q_labels, le.shapes, le.data, le.idxs, axlu, axld))
        red = FlatSparseTensor(*flat_sparse_diag(re.q_labels, re.shapes, re.data, re.idxs, axru, axrd))
        diag = np.tensordot(led, red, axes=1)
        return FlatSparseTensor.zeros_like(self.vmat).cast_assign(diag).data

    def _matmul_plan(self, dl, dr):
        axru = np.arange(1 + dr, 1 + dr + dr, dtype=np.int32)
        axrd = np.arange(dl, dl + dr, dtype=np.int32)
        axlu = np.arange(dl, dl + dl + 1, dtype=np.int32)
        axld = np.arange(0, dl + 1, dtype=np.int32)
        l, r = self.op.tensors
        v, w, w2 = self.vmat, self.work, self.work2
        vqs, vshs = v.q_labels[:, 1:-1], v.shapes[:, 1:-1]
        lo, le = l.odd, l.even
        ro, re = r.odd, r.even
        if ro.n_blocks != 0:
            plan_ro = flat_sparse_matmul_plan(
                ro.q_labels[:, :-1], ro.shapes[:, :-1], ro.idxs, vqs, vshs, v.idxs, axru, axrd, w.q_labels, w.idxs, True)
        else:
            plan_ro = np.zeros((0, 9), dtype=np.int32)
        if re.n_blocks != 0:
            plan_re = flat_sparse_matmul_plan(
                re.q_labels[:, :-1], re.shapes[:, :-1], re.idxs, vqs, vshs, v.idxs, axru, axrd, w.q_labels, w.idxs, False)
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
        assert other.dtype == np.float64
        assert other.size == self.vmat.idxs[-1]
        l, r = self.op.tensors
        pro, pre, plo, ple = self.plan
        self.work.data[:] = 0
        self.work2.data[:] = 0
        self.vmat.data = np.zeros_like(self.vmat.data)
        flat_sparse_matmul(pro, r.odd.data, other, self.work.data)
        flat_sparse_matmul(pre, r.even.data, other, self.work.data)
        flat_sparse_transpose(self.work.shapes, self.work.data,
                              self.work.idxs, self.axtr, self.work2.data)
        flat_sparse_matmul(plo, l.odd.data, self.work2.data, self.vmat.data)
        flat_sparse_matmul(ple, l.even.data, self.work2.data, self.vmat.data)
        if self.op.const != 0:
            self.vmat.data += self.op.const * other
        return self.vmat.data

    def prepare_vector(self, spt):
        self.ket = spt
        return FlatSparseTensor.zeros_like(self.vmat).cast_assign(spt).data

    def finalize_vector(self, data):
        self.vmat.data = data
        return self.vmat
