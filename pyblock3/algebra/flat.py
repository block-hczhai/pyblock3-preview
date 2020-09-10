
import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin
import numbers
from collections import Counter
from itertools import accumulate, groupby

from .symmetry import BondInfo, BondFusingInfo, SZ
from .core import SparseTensor, SubTensor, FermionTensor

import numba as nb

ENABLE_FAST_IMPLS = False


def method_alias(name):
    def ff(f):
        def fff(obj, *args, **kwargs):
            if hasattr(obj, name):
                if isinstance(getattr(obj, name), staticmethod):
                    return getattr(obj, name)(obj, *args, **kwargs)
                else:
                    return getattr(obj, name)(*args, **kwargs)
            else:
                return f(obj, *args, **kwargs)
        return fff
    return ff


def implements(np_func):
    global _numpy_func_impls
    return lambda f: (_numpy_func_impls.update({np_func: f})
                      if np_func not in _numpy_func_impls else None,
                      _numpy_func_impls[np_func])[1]


_flat_sparse_tensor_numpy_func_impls = {}
_numpy_func_impls = _flat_sparse_tensor_numpy_func_impls


class FlatSZ:
    @staticmethod
    def from_sz(x):
        return ((x.n + 8192) * 16384 + (x.twos + 8192)) * 8 + x.pg

    @staticmethod
    def to_sz(x):
        return SZ((x // 131072) % 16384 - 8192, (x // 8) % 16384 - 8192, x % 8)


# @nb.jit(nb.boolean(nb.uint32), nopython=True)
def is_fermion(x):
    return (x & 8) != 0


# @nb.jit(nb.types.Tuple((nb.uint32[:, :], nb.uint32[:, :], nb.float64[:], nb.uint32[:]))(
#         nb.uint32[:, :], nb.uint32[:,
#                                    :], nb.float64[:], nb.uint32[:], nb.uint32[:, :],
#         nb.uint32[:, :], nb.float64[:], nb.uint32[:], nb.int32[:], nb.int32[:]
#         ), forceobj=True)
def flat_sparse_tensordot(aqs, ashs, adata, aidxs, bqs, bshs, bdata, bidxs, idxa, idxb):
    if len(aqs) == 0:
        return aqs, ashs, adata, aidxs
    elif len(bqs) == 0:
        return bqs, bshs, bdata, bidxs
    la, ndima = aqs.shape
    lb, ndimb = bqs.shape
    out_idx_a = np.delete(np.arange(0, ndima), idxa)
    out_idx_b = np.delete(np.arange(0, ndimb), idxb)
    ctrqas, outqas = aqs[:, idxa], aqs[:, out_idx_a]
    ctrqbs, outqbs = bqs[:, idxb], bqs[:, out_idx_b]

    map_idx_b = {}
    for ib in range(lb):
        ctrq = ctrqbs[ib].tobytes()
        if ctrq not in map_idx_b:
            map_idx_b[ctrq] = []
        map_idx_b[ctrq].append(ib)

    blocks_map = {}
    idxs = [0]
    qs = []
    shapes = []
    mats = []
    for ia in range(la):
        ctrq = ctrqas[ia].tobytes()
        mata = adata[aidxs[ia]: aidxs[ia + 1]].reshape(ashs[ia])
        if ctrq in map_idx_b:
            for ib in map_idx_b[ctrq]:
                outq = np.concatenate((outqas[ia], outqbs[ib]))
                matb = bdata[bidxs[ib]: bidxs[ib + 1]].reshape(bshs[ib])
                mat = np.tensordot(mata, matb, axes=(idxa, idxb))
                outqk = outq.tobytes()
                if outqk not in blocks_map:
                    blocks_map[outqk] = len(qs)
                    idxs.append(idxs[-1] + mat.size)
                    qs.append(outq)
                    shapes.append(mat.shape)
                    mats.append(mat.flatten())
                else:
                    mats[blocks_map[outqk]] += mat.flatten()

    return (np.array(qs, dtype=np.uint32),
            np.array(shapes, dtype=np.uint32),
            np.concatenate(mats) if len(mats) != 0 else np.array(
                [], dtype=np.float64),
            np.array(idxs, dtype=np.uint32))

# einsum version, much slower
# def flat_sparse_tensordot(aqs, ashs, adata, aidxs, bqs, bshs, bdata, bidxs, idxa, idxb):
#     if len(aqs) == 0:
#         return aqs, ashs, adata, aidxs
#     elif len(bqs) == 0:
#         return bqs, bshs, bdata, bidxs
#     la, ndima = aqs.shape
#     lb, ndimb = bqs.shape
#     out_idx_a = np.delete(np.arange(0, ndima), idxa)
#     out_idx_b = np.delete(np.arange(0, ndimb), idxb)
#     ctrqas, outqas = aqs[:, idxa], aqs[:, out_idx_a]
#     ctrqbs, outqbs = bqs[:, idxb], bqs[:, out_idx_b]
#     outsas, outsbs = ashs[:, out_idx_a], bshs[:, out_idx_b]
#     eidxa = np.arange(0, ndima)
#     eidxb = np.arange(ndima, ndima + ndimb)
#     eidxb[idxb] = idxa
#     eidxout = np.concatenate((eidxa[out_idx_a], eidxb[out_idx_b])).tolist()
#     eidxa, eidxb = eidxa.tolist(), eidxb.tolist()

#     map_idx_b = {}
#     for ib in range(lb):
#         ctrq = ctrqbs[ib].tobytes()
#         if ctrq not in map_idx_b:
#             map_idx_b[ctrq] = []
#         map_idx_b[ctrq].append(ib)

#     blocks_map = {}
#     idxs = [0]
#     qs = []
#     shapes = []
#     mats = []
#     for ia in range(la):
#         ctrq = ctrqas[ia].tobytes()
#         mata = adata[aidxs[ia]: aidxs[ia + 1]].reshape(ashs[ia])
#         if ctrq in map_idx_b:
#             for ib in map_idx_b[ctrq]:
#                 outq = np.concatenate((outqas[ia], outqbs[ib]))
#                 matb = bdata[bidxs[ib]: bidxs[ib + 1]].reshape(bshs[ib])
#                 shape = np.concatenate((outsas[ia], outsbs[ib]))
#                 # mat = np.tensordot(mata, matb, axes=(idxa, idxb))
#                 outqk = outq.tobytes()
#                 if outqk not in blocks_map:
#                     blocks_map[outqk] = len(qs)
#                     idxs.append(idxs[-1] + int(np.prod(shape)))
#                     qs.append(outq)
#                     shapes.append(shape)
#                     mats.append((mata, matb, blocks_map[outqk]))
#                 else:
#                     mats.append((mata, matb, blocks_map[outqk]))
#     data = np.zeros((idxs[-1], ), dtype=np.float64)
#     for ma, mb, im in mats:
#         # np.einsum(ma, eidxa, mb, eidxb, eidxout, out=data[idxs[im]:idxs[im + 1]].reshape(shapes[im]))
#         data[idxs[im]:idxs[im + 1]] += np.einsum(ma, eidxa, mb, eidxb, eidxout).flatten()
#         # data[idxs[im]:idxs[im + 1]] += np.tensordot(ma, mb, axes=(idxa, idxb)).flatten()

#     return (np.array(qs, dtype=np.uint32),
#             np.array(shapes, dtype=np.uint32),
#             data,
#             np.array(idxs, dtype=np.uint32))


def flat_sparse_add(aqs, ashs, adata, aidxs, bqs, bshs, bdata, bidxs):
    blocks_map = {q.tobytes(): iq for iq, q in enumerate(aqs)}
    data = adata.copy()
    dd = []
    bmats = []
    for ib in range(bqs.shape[0]):
        q = bqs[ib].tobytes()
        if q in blocks_map:
            ia = blocks_map[q]
            data[aidxs[ia]:aidxs[ia + 1]
                 ] += bdata[bidxs[ib]:bidxs[ib + 1]]
            dd.append(ib)
        else:
            bmats.append(bdata[bidxs[ib]:bidxs[ib + 1]])

    return (np.concatenate((aqs, np.delete(bqs, dd, axis=0))),
            np.concatenate((ashs, np.delete(bshs, dd, axis=0))),
            np.concatenate((data, *bmats)),
            None)

def flat_sparse_left_canonicalize(aqs, ashs, adata, aidxs):
    collected_rows = {}
    for i, q in enumerate(aqs[:, -1]):
        if q not in collected_rows:
            collected_rows[q] = [i]
        else:
            collected_rows[q].append(i)
    nblocks_r = len(collected_rows)
    qmats = [None] * aqs.shape[0]
    rmats = [None] * nblocks_r
    qqs = aqs
    qshs = ashs.copy()
    rqs = np.zeros((nblocks_r, 2), aqs.dtype)
    rshs = np.zeros((nblocks_r, 2), ashs.dtype)
    ridxs = np.zeros((nblocks_r + 1), aidxs.dtype)
    for ir, (qq, v) in enumerate(collected_rows.items()):
        pashs = ashs[v, :-1]
        l_shapes = np.prod(pashs, axis=1)
        mat = np.concatenate([adata[aidxs[ia]:aidxs[ia + 1]].reshape((sh, -1)) for sh, ia in zip(l_shapes, v)], axis=0)
        q, r = np.linalg.qr(mat, mode='reduced')
        rqs[ir, :] = qq
        rshs[ir] = r.shape
        ridxs[ir + 1] = ridxs[ir] + r.size
        rmats[ir] = r.flatten()
        qs = np.split(q, list(accumulate(l_shapes[:-1])), axis=0)
        assert len(qs) == len(v)
        qshs[v, -1] = r.shape[0]
        for q, ia in zip(qs, v):
            qmats[ia] = q.flatten()
    return (qqs, qshs, np.concatenate(qmats), None, rqs, rshs, np.concatenate(rmats), ridxs)

def flat_sparse_right_canonicalize(aqs, ashs, adata, aidxs):
    collected_cols = {}
    for i, q in enumerate(aqs[:, 0]):
        if q not in collected_cols:
            collected_cols[q] = [i]
        else:
            collected_cols[q].append(i)
    nblocks_l = len(collected_cols)
    lmats = [None] * nblocks_l
    qmats = [None] * aqs.shape[0]
    lqs = np.zeros((nblocks_l, 2), aqs.dtype)
    lshs = np.zeros((nblocks_l, 2), ashs.dtype)
    lidxs = np.zeros((nblocks_l + 1), aidxs.dtype)
    qqs = aqs
    qshs = ashs.copy()
    for il, (qq, v) in enumerate(collected_cols.items()):
        pashs = ashs[v, 1:]
        r_shapes = np.prod(pashs, axis=1)
        mat = np.concatenate([adata[aidxs[ia]:aidxs[ia + 1]].reshape((-1, sh)).T for sh, ia in zip(r_shapes, v)], axis=0)
        q, r = np.linalg.qr(mat, mode='reduced')
        lqs[il, :] = qq
        lshs[il] = r.shape[::-1]
        lidxs[il + 1] = lidxs[il] + r.size
        lmats[il] = r.T.flatten()
        qs = np.split(q, list(accumulate(r_shapes[:-1])), axis=0)
        assert len(qs) == len(v)
        qshs[v, 0] = r.shape[0]
        for q, ia in zip(qs, v):
            qmats[ia] = q.T.flatten()
    return (lqs, lshs, np.concatenate(lmats), lidxs, qqs, qshs, np.concatenate(qmats), None)

if ENABLE_FAST_IMPLS:
    import block3.flat_sparse_tensor
    flat_sparse_tensordot = block3.flat_sparse_tensor.tensordot
    # xadd = flat_sparse_add
    # def flat_sparse_add(*args, **kwargs):
    #     r = block3.flat_sparse_tensor.add(*args, **kwargs)
    #     x = xadd(*args, **kwargs)
    #     if abs(np.sum(r[2]) - np.sum(x[2])) > 1E-10 or len(r[0]) != len(x[0]):
    #         print(np.sum(r[2]), np.sum(x[2]))
    #         exit(0)
    #     return r
    flat_sparse_add = block3.flat_sparse_tensor.add

class FlatSparseTensor(NDArrayOperatorsMixin):
    """
    block-sparse tensor with efficient flat storage
    """

    def __init__(self, q_labels, shapes, data, idxs=None):
        self.n_blocks = len(q_labels)
        self.ndim = q_labels.shape[1] if self.n_blocks != 0 else 0
        self.shapes = shapes
        self.q_labels = q_labels
        self.data = data
        if idxs is None:
            self.idxs = np.zeros((self.n_blocks + 1, ), dtype=shapes.dtype)
            self.idxs[1:] = np.cumsum(shapes.prod(axis=1))
        else:
            self.idxs = idxs
        if self.n_blocks != 0:
            assert shapes.shape == (self.n_blocks, self.ndim)
            assert q_labels.shape == (self.n_blocks, self.ndim)

    @property
    def dtype(self):
        return self.data.dtype

    def item(self):
        return self.data.item()

    def to_sparse(self):
        blocks = [None] * self.n_blocks
        for i in range(self.n_blocks):
            qs = tuple(map(FlatSZ.to_sz, self.q_labels[i]))
            blocks[i] = SubTensor(
                self.data[self.idxs[i]:self.idxs[i + 1]].reshape(self.shapes[i]), q_labels=qs)
        return SparseTensor(blocks=blocks)

    @staticmethod
    def from_sparse(spt):
        ndim = spt.ndim
        n_blocks = spt.n_blocks
        shapes = np.zeros((n_blocks, ndim), dtype=np.uint32)
        q_labels = np.zeros((n_blocks, ndim), dtype=np.uint32)
        for i in range(n_blocks):
            shapes[i] = spt.blocks[i].shape
            q_labels[i] = list(map(FlatSZ.from_sz, spt.blocks[i].q_labels))
        idxs = np.zeros((n_blocks + 1, ), dtype=np.uint32)
        idxs[1:] = np.cumsum(shapes.prod(axis=1))
        data = np.zeros((idxs[-1], ), dtype=np.float64)
        for i in range(n_blocks):
            data[idxs[i]:idxs[i + 1]] = spt.blocks[i].flatten()
        return FlatSparseTensor(q_labels, shapes, data, idxs)

    def __str__(self):
        return str(self.to_sparse())

    def __repr__(self):
        return repr(self.to_sparse())

    @staticmethod
    def zeros(*args, **kwargs):
        """Create tensor from tuple of BondInfo with zero elements."""
        return FlatSparseTensor.from_sparse(SparseTensor.zeros(*args, **kwargs))

    @staticmethod
    def ones(*args, **kwargs):
        """Create tensor from tuple of BondInfo with ones."""
        return FlatSparseTensor.from_sparse(SparseTensor.ones(*args, **kwargs))

    @staticmethod
    def random(*args, **kwargs):
        """Create tensor from tuple of BondInfo with random elements."""
        return FlatSparseTensor.from_sparse(SparseTensor.random(*args, **kwargs))

    @property
    def infos(self):
        bond_infos = tuple(BondInfo() for _ in range(self.ndim))
        for i in range(self.n_blocks):
            for j in range(self.ndim):
                q = FlatSZ.to_sz(self.q_labels[i][j])
                if q in bond_infos[j]:
                    assert self.shapes[i, j] == bond_infos[j][q]
                else:
                    bond_infos[j][q] = self.shapes[i, j]
        return bond_infos

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if ufunc in _flat_sparse_tensor_numpy_func_impls:
            types = tuple(
                x.__class__ for x in inputs if not isinstance(x, numbers.Number))
            return self.__array_function__(ufunc, types, inputs, kwargs)
        out = kwargs.pop("out", None)
        if out is not None and not all(isinstance(x, FlatSparseTensor) for x in out):
            return NotImplemented
        if any(isinstance(x, FlatFermionTensor) for x in inputs):
            return NotImplemented
        if method == "__call__":
            if ufunc.__name__ in ["matmul"]:
                a, b = inputs
                if isinstance(a, numbers.Number):
                    shs, qs, data, idxs = b.shapes, b.q_labels, a * b.data, b.idxs
                elif isinstance(b, numbers.Number):
                    shs, qs, data, idxs = a.shapes, a.q_labels, a.data * b, a.idxs
                else:
                    c = self._tensordot(a, b, axes=([-1], [0]))
                    shs, qs, data, idxs = c.shapes, c.q_labels, c.data, c.idxs
            elif ufunc.__name__ in ["multiply", "divide", "true_divide"]:
                a, b = inputs
                if isinstance(a, numbers.Number):
                    shs, qs, data, idxs = b.shapes, b.q_labels, getattr(
                        ufunc, method)(a, b.data), b.idxs
                elif isinstance(b, numbers.Number):
                    shs, qs, data, idxs = a.shapes, a.q_labels, getattr(
                        ufunc, method)(a.data, b), a.idxs
                else:
                    return NotImplemented
            elif len(inputs) == 1:
                a = inputs[0]
                shs, qs, data, idxs = a.shapes, a.q_labels, getattr(
                    ufunc, method)(a.data), a.idxs
            else:
                return NotImplemented
        else:
            return NotImplemented
        if out is not None:
            out.shapes[...] = shs
            out.q_labels[...] = qs
            out.data[...] = data
            out.idxs[...] = idxs
        return FlatSparseTensor(q_labels=qs, shapes=shs, data=data, idxs=idxs)

    def __array_function__(self, func, types, args, kwargs):
        if func not in _flat_sparse_tensor_numpy_func_impls:
            return NotImplemented
        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented
        return _flat_sparse_tensor_numpy_func_impls[func](*args, **kwargs)

    @staticmethod
    @implements(np.copy)
    def _copy(x):
        return FlatSparseTensor(q_labels=x.q_labels.copy(), shapes=x.shapes.copy(), data=x.data.copy(), idxs=x.idxs.copy())

    def copy(self):
        return np.copy(self)

    @staticmethod
    @implements(np.linalg.norm)
    def _norm(x):
        return np.linalg.norm(x.data)

    def norm(self):
        return np.linalg.norm(self.data)

    @staticmethod
    @implements(np.tensordot)
    def _tensordot(a, b, axes=2):
        """
        Contract two FlatSparseTensor to form a new FlatSparseTensor.

        Args:
            a : FlatSparseTensor
                FlatSparseTensor a, as left operand.
            b : FlatSparseTensor
                FlatSparseTensor b, as right operand.
            axes : int or (2,) array_like
                If an int N, sum over the last N axes of a and the first N axes of b in order.
                Or, a list of axes to be summed over, first sequence applying to a, second to b.

        Returns:
            FlatSparseTensor : FlatSparseTensor
                The contracted FlatSparseTensor.
        """
        if isinstance(axes, int):
            idxa = np.arange(-axes, 0, dtype=np.int32)
            idxb = np.arange(0, axes, dtype=np.int32)
        else:
            idxa = np.array(axes[0], dtype=np.int32)
            idxb = np.array(axes[1], dtype=np.int32)
        idxa[idxa < 0] += a.ndim
        idxb[idxb < 0] += b.ndim

        return FlatSparseTensor(*flat_sparse_tensordot(
            a.q_labels, a.shapes, a.data, a.idxs,
            b.q_labels, b.shapes, b.data, b.idxs,
            idxa, idxb))

    def tensordot(self, b, axes=2):
        return np.tensordot(self, b, axes)

    @staticmethod
    def _hdot(a, b, out=None):
        """Horizontal contraction (contracting connected virtual dimensions)."""
        if isinstance(a, numbers.Number) or isinstance(b, numbers.Number):
            return np.multiply(a, b, out=out)

        assert isinstance(a, FlatSparseTensor) and isinstance(
            b, FlatSparseTensor)
        r = np.tensordot(a, b, axes=1)

        if out is not None:
            out.q_labels[...] = r.q_labels
            out.shapes[...] = r.shapes
            out.data[...] = r.data
            out.idxs[...] = r.idxs

        return r

    def hdot(self, b, out=None):
        """Horizontal contraction (contracting connected virtual dimensions)."""
        if b.__class__ != self.__class__:
            return b._hdot(self, b, out=out)
        else:
            return self._hdot(self, b, out=out)

    @staticmethod
    def _pdot(a, b, out=None):
        """Vertical contraction (all middle/physical dims)."""
        if isinstance(a, numbers.Number) or isinstance(b, numbers.Number):
            return np.multiply(a, b, out=out)

        assert isinstance(a, FlatSparseTensor) and isinstance(
            b, FlatSparseTensor)

        assert a.ndim == b.ndim
        d = a.ndim - 2
        aidx = np.arange(1, d + 1, dtype=np.uint32)
        bidx = np.arange(1, d + 1, dtype=np.uint32)
        tr = (0, 2, 1, 3)

        r = np.tensordot(a, b, axes=(aidx, bidx)).transpose(axes=tr)

        if out is not None:
            out.q_labels[...] = r.q_labels
            out.shapes[...] = r.shapes
            out.data[...] = r.data
            out.idxs[...] = r.idxs

        return r

    def pdot(self, b, out=None):
        """Vertical contraction (all middle dims)."""
        if b.__class__ != self.__class__:
            return b._pdot(self, b, out=out)
        else:
            return self._pdot(self, b, out=out)

    @staticmethod
    @implements(np.add)
    def _add(a, b):
        if isinstance(a, numbers.Number):
            data = a + b.data
            return FlatSparseTensor(b.q_labels, b.shapes, data, b.idxs)
        elif isinstance(b, numbers.Number):
            data = a.data + b
            return FlatSparseTensor(a.q_labels, a.shapes, data, a.idxs)
        elif b.n_blocks == 0:
            return a
        elif a.n_blocks == 0:
            return b
        else:
            return FlatSparseTensor(*flat_sparse_add(a.q_labels, a.shapes, a.data,
                                                     a.idxs, b.q_labels, b.shapes, b.data, b.idxs))

    def add(self, b):
        return self._add(self, b)

    @staticmethod
    @implements(np.subtract)
    def _subtract(a, b):
        if isinstance(a, numbers.Number):
            data = a - b.data
            return FlatSparseTensor(b.q_labels, b.shapes, data, b.idxs)
        elif isinstance(b, numbers.Number):
            data = a.data - b
            return FlatSparseTensor(a.q_labels, a.shapes, data, a.idxs)
        elif b.n_blocks == 0:
            return a
        elif a.n_blocks == 0:
            return -b
        else:
            blocks_map = {q.tobytes(): iq for iq, q in enumerate(a.q_labels)}
            data = a.data.copy()
            dd = []
            bmats = []
            for ib in range(b.n_blocks):
                q = b.q_labels[ib].tobytes()
                if q in blocks_map:
                    ia = blocks_map[q]
                    data[a.idxs[ia]:a.idxs[ia + 1]
                         ] -= b.data[b.idxs[ib]:b.idxs[ib + 1]]
                    dd.append(ib)
                else:
                    bmats.append(b.data[b.idxs[ib]:b.idxs[ib + 1]])
            return FlatSparseTensor(
                q_labels=np.concatenate(
                    (a.q_labels, np.delete(b.q_labels, dd, axis=0))),
                shapes=np.concatenate(
                    (a.shapes, np.delete(b.shapes, dd, axis=0))),
                data=np.concatenate((data, *bmats))
            )

    def subtract(self, b):
        return self._subtract(self, b)

    def left_canonicalize(self, mode='reduced'):
        """
        Left canonicalization (using QR factorization).
        Left canonicalization needs to collect all left indices for each specific right index.
        So that we will only have one R, but left dim of q is unchanged.

        Returns:
            q, r : tuple(FlatSparseTensor)
        """
        return tuple(FlatSparseTensor.from_sparse(x) for x in self.to_sparse().left_canonicalize())
        assert mode == 'reduced'
        r = flat_sparse_left_canonicalize(self.q_labels, self.shapes, self.data, self.idxs)
        return FlatSparseTensor(*r[:4]), FlatSparseTensor(*r[4:])

    def right_canonicalize(self, mode='reduced'):
        """
        Right canonicalization (using QR factorization).

        Returns:
            l, q : tuple(FlatSparseTensor)
        """
        return tuple(FlatSparseTensor.from_sparse(x) for x in self.to_sparse().right_canonicalize())
        assert mode == 'reduced'
        r = flat_sparse_right_canonicalize(self.q_labels, self.shapes, self.data, self.idxs)
        return FlatSparseTensor(*r[:4]), FlatSparseTensor(*r[4:])
    
    def tensor_svd(self, *args, **kwargs):
        """
        Separate tensor in the middle, collecting legs as [0, idx) and [idx, ndim), then perform SVD.

        Returns:
            l, s, r : tuple(FlatSparseTensor)
        """
        lsr = self.to_sparse().tensor_svd(*args, **kwargs)
        return tuple(FlatSparseTensor.from_sparse(x) for x in lsr)

    @staticmethod
    @implements(np.diag)
    def _diag(v):
        """
        Extract a diagonal or construct a diagonal array.

        Args:
            v : FlatSparseTensor
                If v is a 2-D array, return a copy of its 0-th diagonal.
                If v is a 1-D array, return a 2-D array with v on the 0-th diagonal.
        """
        if v.ndim == 2:
            mask = v.q_labels[:, 0] == v.q_labels[:, 1]
            shapes = v.shapes[mask, 0]
            return FlatSparseTensor(
                q_labels=v.q_labels[mask, 0], shapes=shapes,
                data=np.concatenate([np.diag(v.data[i:j].reshape(sh, sh)) for i, j, sh in zip(v.idxs[:-1][mask], v.idxs[1:][mask], shapes)]))
        elif v.ndim == 1:
            return FlatSparseTensor(
                q_labels=np.repeat(v.q_labels, 2, axis=1),
                shapes=np.repeat(v.shapes, 2, axis=1),
                data=np.concatenate([np.diag(v.data[i:j]).flatten() for i, j in zip(v.idxs, v.idxs[1:])]))
        elif len(v.blocks) != 0:
            raise RuntimeError("ndim for np.diag must be 1 or 2.")

    def diag(self):
        return np.diag(self)

    @staticmethod
    @implements(np.transpose)
    def _transpose(a, axes=None):
        """
        Reverse or permute the axes of an array; returns the modified array.

        Args:
            a : array_like
                Input array.
            axes : tuple or list of ints, optional
                If specified, it must be a tuple or list which contains a permutation of [0,1,..,N-1]
                where N is the number of axes of a.
                The i’th axis of the returned array will correspond to the axis numbered axes[i] of the input.
                If not specified, defaults to ``range(a.ndim)[::-1]``, which reverses the order of the axes.

        Returns
            p : FlatSparseTensor
                a with its axes permuted. A view is returned whenever possible.
        """
        if axes is None:
            axes = np.arange(a.ndim)[::-1]
        if a.n_blocks == 0:
            return a
        else:
            return FlatSparseTensor(
                q_labels=a.q_labels[:, axes],
                shapes=a.shapes[:, axes],
                data=np.concatenate([np.transpose(a.data[i:j].reshape(sh), axes=axes).flatten(
                ) for i, j, sh in zip(a.idxs, a.idxs[1:], a.shapes)]),
                idxs=a.idxs)

    def transpose(self, axes=None):
        return np.transpose(self, axes=axes)

    @staticmethod
    def _to_dense(a, infos=None):
        return a.to_dense(infos=infos)

    def to_dense(self, infos=None):
        return self.to_sparse().to_dense(infos=infos)


_flat_fermion_tensor_numpy_func_impls = {}
_numpy_func_impls = _flat_fermion_tensor_numpy_func_impls


class FlatFermionTensor(FermionTensor):
    """
    flat block-sparse tensor with fermion factors.

    Attributes:
        odd : FlatSparseTensor
            Including blocks with odd fermion parity.
        even : FlatSparseTensor
            Including blocks with even fermion parity.
    """

    def __init__(self, odd=None, even=None):
        self.odd = odd
        self.even = even

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if ufunc in _flat_fermion_tensor_numpy_func_impls:
            types = tuple(
                x.__class__ for x in inputs if not isinstance(x, numbers.Number))
            return self.__array_function__(ufunc, types, inputs, kwargs)
        out = kwargs.pop("out", None)
        if out is not None and not all(isinstance(x, FlatFermionTensor) for x in out):
            return NotImplemented
        if method == "__call__":
            if ufunc.__name__ in ["matmul"]:
                a, b = inputs
                if isinstance(a, numbers.Number):
                    blocks = a * b.odd, a * b.even
                elif isinstance(b, numbers.Number):
                    blocks = a.odd * b, a.even * b
                else:
                    r = self._tensordot(a, b, axes=([-1], [0]))
                    blocks = r.odd, r.even
            elif ufunc.__name__ in ["add", "subtract"]:
                a, b = inputs
                if isinstance(a, numbers.Number):
                    blocks = getattr(ufunc, method)(
                        a, b.odd), getattr(ufunc, method)(a, b.even)
                elif isinstance(b, numbers.Number):
                    blocks = getattr(ufunc, method)(
                        a.odd, b), getattr(ufunc, method)(a.even, b)
                else:
                    blocks = getattr(ufunc, method)(a.odd, b.odd), getattr(
                        ufunc, method)(a.even, b.even)
            elif ufunc.__name__ in ["multiply", "divide", "true_divide"]:
                a, b = inputs
                if isinstance(a, numbers.Number):
                    blocks = getattr(ufunc, method)(
                        a, b.odd), getattr(ufunc, method)(a, b.even)
                elif isinstance(b, numbers.Number):
                    blocks = getattr(ufunc, method)(
                        a.odd, b), getattr(ufunc, method)(a.even, b)
                else:
                    return NotImplemented
            elif len(inputs) == 1:
                a = inputs[0]
                blocks = getattr(ufunc, method)(
                    a.odd), getattr(ufunc, method)(a.even)
            else:
                return NotImplemented
        else:
            return NotImplemented
        if out is not None:
            out.odd, out.even = blocks
        return FlatFermionTensor(odd=blocks[0], even=blocks[1])

    def __array_function__(self, func, types, args, kwargs):
        if func not in _flat_fermion_tensor_numpy_func_impls:
            return NotImplemented
        if not all(issubclass(t, self.__class__) or issubclass(t, self.odd.__class__) for t in types):
            return NotImplemented
        return _flat_fermion_tensor_numpy_func_impls[func](*args, **kwargs)

    @staticmethod
    def from_fermion(x):
        return FlatFermionTensor(
            odd=FlatSparseTensor.from_sparse(x.odd),
            even=FlatSparseTensor.from_sparse(x.even)
        )

    @staticmethod
    def zeros(*args, **kwargs):
        return FlatFermionTensor.from_fermion(FermionTensor.zeros(*args, **kwargs))

    @staticmethod
    def ones(*args, **kwargs):
        return FlatFermionTensor.from_fermion(FermionTensor.ones(*args, **kwargs))

    @staticmethod
    def random(*args, **kwargs):
        return FlatFermionTensor.from_fermion(FermionTensor.random(*args, **kwargs))

    def deflate(self, cutoff=0):
        return FlatFermionTensor(odd=self.odd.deflate(cutoff), even=self.even.deflate(cutoff))

    @staticmethod
    @implements(np.copy)
    def _copy(x):
        return FlatFermionTensor(odd=x.odd.copy(), even=x.even.copy())

    def copy(self):
        return np.copy(self)

    @staticmethod
    def _unfuse(a, i, info):
        return a.unfuse(i, info)

    def unfuse(self, i, info):
        """Unfuse one leg to several legs.
        May introduce some additional zero blocks.

        Args:
            i : int
                index of the leg to be unfused. The new unfused indices will be i, i + 1, ...
            info : BondFusingInfo
                Indicating how quantum numbers are collected.
        """
        odd = self.odd.unfuse(i, info=info)
        even = self.even.unfuse(i, info=info)
        return FlatFermionTensor(odd=odd, even=even)

    @staticmethod
    def _fuse(a, *idxs, info=None, pattern=None):
        return a.fuse(*idxs, info=info, pattern=pattern)

    def fuse(self, *idxs, info=None, pattern=None):
        """Fuse several legs to one leg.

        Args:
            idxs : tuple(int)
                Leg indices to be fused. The new fused index will be idxs[0].
            info : BondFusingInfo (optional)
                Indicating how quantum numbers are collected.
                If not specified, the direct sum of quantum numbers will be used.
                This will generate minimal and (often) incomplete fused shape.
            pattern : str (optional)
                A str of '+'/'-'. Only required when info is not specified.
                Indicating how quantum numbers are linearly combined.
        """
        idxs = [i if i >= 0 else self.ndim + i for i in idxs]
        if info is None:
            items = []
            for block in self.odd.blocks + self.even.blocks:
                qs = tuple(block.q_labels[i] for i in idxs)
                shs = tuple(block.shape[i] for i in idxs)
                items.append((qs, shs))
            # using minimal fused dimension
            info = BondFusingInfo.kron_sum(items, pattern=pattern)
        odd = self.odd.fuse(*idxs, info=info, pattern=pattern)
        even = self.even.fuse(*idxs, info=info, pattern=pattern)
        return FlatFermionTensor(odd=odd, even=even)

    @staticmethod
    @implements(np.tensordot)
    def _tensordot(a, b, axes=2):
        """
        Contract two FlatFermionTensor/FlatSparseTensor to form a new FlatFermionTensor/FlatSparseTensor.

        Args:
            a : FlatFermionTensor/FlatSparseTensor
                FlatFermionTensor/FlatSparseTensor a, as left operand.
            b : FlatFermionTensor/FlatSparseTensor
                FlatFermionTensor/FlatSparseTensor b, as right operand.
            axes : int or (2,) array_like
                If an int N, sum over the last N axes of a and the first N axes of b in order.
                Or, a list of axes to be summed over, first sequence applying to a, second to b.

        Returns:
            c : FlatFermionTensor/FlatSparseTensor
                The contracted FlatFermionTensor/FlatSparseTensor.
        """

        if isinstance(axes, int):
            idxa = np.arange(-axes, 0, dtype=np.int32)
            idxb = np.arange(0, axes, dtype=np.int32)
        else:
            idxa = np.array(axes[0], dtype=np.int32)
            idxb = np.array(axes[1], dtype=np.int32)
        idxa[idxa < 0] += a.ndim
        idxb[idxb < 0] += b.ndim

        blocks = []
        def r(): return None
        # op x op
        if isinstance(a, FlatFermionTensor) and isinstance(b, FlatFermionTensor):
            odd_a = np.tensordot(a.odd, b.even, (idxa, idxb))
            odd_b = np.tensordot(a.even, b.odd, (idxa, idxb))
            even_a = np.tensordot(a.odd, b.odd, (idxa, idxb))
            even_b = np.tensordot(a.even, b.even, (idxa, idxb))
            def r(): return FlatFermionTensor(odd=odd_a + odd_b, even=even_a + even_b)
            # symbolic horizontal
            if idxa == [] and idxb == []:
                assert a.ndim % 2 == 0
                d = a.ndim // 2
                idx = range(d, d + d) if d != 1 else d
                blocks = [odd_b, even_a]
            # horizontal
            elif idxa == [a.ndim - 1] and idxb == [0]:
                assert a.ndim % 2 == 0
                d = (a.ndim - 2) // 2
                idx = range(d + 1, d + d + 1) if d != 1 else d + 1
                blocks = [odd_b, even_a]
            # vertical
            else:
                idx = a.ndim - len(idxa)
                # 1-site op x n-site op
                idx = range(idx, idx + min(idxb))
                blocks = [odd_a, even_a]
        # op x state
        elif isinstance(a, FlatFermionTensor):
            idx = a.ndim - len(idxa)
            even = np.tensordot(a.even, b, (idxa, idxb))
            odd = np.tensordot(a.odd, b, (idxa, idxb))
            # op rotation / op x gauge (right multiply)
            if b.ndim - len(idxb) == 1:
                def r(): return FlatFermionTensor(odd=odd, even=even)
            else:
                # 1-site op x n-site state
                idx = range(idx, idx + min(idxb))
                blocks = [odd]
                def r(): return odd + even
        # state x op
        elif isinstance(b, FlatFermionTensor):
            idx = 0
            even = np.tensordot(a, b.even, (idxa, idxb))
            odd = np.tensordot(a, b.odd, (idxa, idxb))
            # op rotation / gauge x op (left multiply)
            if a.ndim - len(idxa) == 1:
                def r(): return FlatFermionTensor(odd=odd, even=even)
            else:
                # n-site state x 1-site op
                idx = range(idx, idx + min(idxa))
                blocks = [odd]
                def r(): return odd + even
        else:
            raise TypeError('Unsupported tensordot for %r and %r' %
                            (a.__class__, b.__class__))

        # apply fermion factor
        if isinstance(idx, int):
            for x in blocks:
                if x.n_blocks != 0:
                    for i, j, q in zip(x.idxs, x.idxs[1:], x.q_labels[:, idx]):
                        if is_fermion(q):
                            np.negative(x.data[i:j], out=x.data[i:j])
        else:
            for x in blocks:
                if x.n_blocks != 0:
                    for i, j, qs in zip(x.idxs, x.idxs[1:], x.q_labels[:, idx]):
                        if np.logical_xor.reduce([is_fermion(q) for q in qs]):
                            np.negative(x.data[i:j], out=x.data[i:j])

        return r()

    def tensordot(self, b, axes=2):
        return np.tensordot(self, b, axes)

    @staticmethod
    def _shdot(a, b, out=None):
        """Horizontally contract operator tensors (matrices) in symbolic matrix."""
        if isinstance(a, numbers.Number) or isinstance(b, numbers.Number):
            return np.multiply(a, b, out=out)

        assert isinstance(a, FlatFermionTensor) and isinstance(
            b, FlatFermionTensor)
        r = np.tensordot(a, b, axes=0)

        if isinstance(a, FlatFermionTensor) and isinstance(b, FlatFermionTensor) and a.ndim % 2 == 0 and b.ndim % 2 == 0:
            da, db = a.ndim // 2, b.ndim // 2
            r = r.transpose((*range(da), *range(da + da, da + da + db),
                             *range(da, da + da), *range(da + da + db, da + da + db + db)))

        if out is not None:
            out.odd = r.odd
            out.even = r.even

        return r

    def shdot(self, b, out=None):
        """Horizontally contract operator tensors (matrices) in symbolic matrix."""
        return self._shdot(self, b, out=out)

    def __xor__(self, other):
        return self._shdot(self, other)

    @staticmethod
    def _hdot(a, b, out=None):
        """Horizontally contract operator tensors (contracting connected virtual dimensions)."""
        if isinstance(a, numbers.Number) or isinstance(b, numbers.Number):
            return np.multiply(a, b, out=out)

        assert isinstance(a, FlatFermionTensor) or isinstance(
            b, FlatFermionTensor)
        r = np.tensordot(a, b, axes=1)

        if isinstance(a, FlatFermionTensor) and isinstance(b, FlatFermionTensor) and a.ndim % 2 == 0 and b.ndim % 2 == 0:
            da, db, na = a.ndim // 2 - 1, b.ndim // 2 - 1, a.ndim - 1
            r = r.transpose((0, *range(1, da + 1), *range(na, na + db),
                             *range(da + 1, da + da + 1), *range(na + db, na + db + db), na + db + db))

        if out is not None:
            out.odd = r.odd
            out.even = r.even

        return r

    def hdot(self, b, out=None):
        """Horizontally contract operator tensors (contracting connected virtual dimensions)."""
        return self._hdot(self, b, out=out)

    @staticmethod
    def _pdot(a, b, out=None):
        """Vertical contraction (all middle/physical dims)."""
        if isinstance(a, numbers.Number) or isinstance(b, numbers.Number):
            return np.multiply(a, b, out=out)

        if isinstance(a, list):
            p, ps = 1, []
            for i in range(len(a)):
                ps.append(p)
                p += a[i].ndim // 2 - 1
            r = b
            # [MPO] x MPO
            if isinstance(b, FlatFermionTensor):
                for i in range(len(a))[::-1]:
                    d = a[i].ndim // 2 - 1
                    d2 = r.ndim // 2 - 1
                    if i == len(a) - 1:
                        aidx = list(range(d + 1, d + d + 1))
                        bidx = list(range(ps[i], ps[i] + d))
                        tr = (*range(d + 2, ps[i] + d + 2), *range(0, d + 1),
                              *range(ps[i] + d + 2, ps[i] + d + 2 + d2), d + 1, ps[i] + d + 2 + d2)
                    else:
                        aidx = list(range(d + 1, d + d + 2))
                        bidx = list(range(ps[i], ps[i] + d + 1))
                        tr = (*range(d + 1, ps[i] + d + 1), *range(0, d + 1),
                              *range(ps[i] + d + 1, d2 + d2 + 2))
                    r = np.tensordot(a[i], r, axes=(
                        aidx, bidx)).transpose(axes=tr)
                r = r.transpose(axes=(1, 0, *range(2, r.ndim)))
            # [MPO] x MPS
            elif isinstance(b, FlatSparseTensor):
                for i in range(len(a))[::-1]:
                    d = a[i].ndim // 2 - 1
                    d2 = r.ndim - 2
                    if i == len(a) - 1:
                        aidx = list(range(d + 1, d + d + 1))
                        bidx = list(range(ps[i], ps[i] + d))
                        tr = (*range(d + 2, ps[i] + d + 2),
                              *range(0, d + 2), ps[i] + d + 2)
                    else:
                        aidx = list(range(d + 1, d + d + 2))
                        bidx = list(range(ps[i], ps[i] + d + 1))
                        tr = (*range(d + 1, ps[i] + d + 1), *range(0, d + 1),
                              *range(ps[i] + d + 1, d2 + 2))
                    r = np.tensordot(a[i], r, axes=(
                        aidx, bidx)).transpose(axes=tr)
                r = r.transpose(axes=(1, 0, *range(2, r.ndim)))
            else:
                raise RuntimeError(
                    "Cannot matmul tensors with types %r x %r" % (a.__class__, b.__class__))
        elif isinstance(b, list):
            raise NotImplementedError("not implemented.")
        else:

            assert isinstance(a, FlatFermionTensor) or isinstance(
                b, FlatFermionTensor)

            # MPO x MPO [can be FT x FT or FT x SPT (MPDO) or SPT (MPDO) x FT]
            if a.ndim == b.ndim and a.ndim % 2 == 0:
                assert a.ndim == b.ndim and a.ndim % 2 == 0
                d = a.ndim // 2 - 1
                aidx = list(range(d + 1, d + d + 1))
                bidx = list(range(1, d + 1))
                tr = tuple([0, d + 2] + list(range(1, d + 1)) +
                           list(range(d + 3, d + d + 3)) + [d + 1, d + d + 3])
            # MPO x MPS
            elif a.ndim > b.ndim:
                assert isinstance(a, FlatFermionTensor)
                dau, db = a.ndim - b.ndim, b.ndim - 2
                aidx = list(range(dau + 1, dau + db + 1))
                bidx = list(range(1, db + 1))
                tr = tuple([0, dau + 2] + list(range(1, dau + 1)) +
                           [dau + 1, dau + 3])
            # MPS x MPO
            elif a.ndim < b.ndim:
                assert isinstance(b, FlatFermionTensor)
                da, dbd = a.ndim - 2, b.ndim - a.ndim
                aidx = list(range(1, da + 1))
                bidx = list(range(1, da + 1))
                tr = tuple([0, 2] + list(range(3, dbd + 3)) + [1, dbd + 3])
            else:
                raise RuntimeError(
                    "Cannot matmul tensors with ndim: %d x %d" % (a.ndim, b.ndim))

            r = np.tensordot(a, b, axes=(aidx, bidx)).transpose(axes=tr)

        if out is not None:
            if isinstance(r, FlatSparseTensor):
                out.q_labels[...] = r.q_labels
                out.shapes[...] = r.shapes
                out.data[...] = r.data
                out.idxs[...] = r.idxs
            else:
                out.odd = r.odd
                out.even = r.even

        return r

    def pdot(self, b, out=None):
        return FlatFermionTensor._pdot(self, b, out=out)

    @staticmethod
    def _kron_add(a, b, infos=None):
        """
        Direct sum of first and last legs.
        Middle dims are summed.

        Args:
            infos : (BondInfo, BondInfo) or None
                BondInfo of first and last legs for the result.
        """
        if infos is None:
            abi, bbi = a.infos, b.infos
            infos = (abi[0] + bbi[0], abi[-1], bbi[-1])

        odd = a.odd.kron_add(b.odd, infos=infos)
        even = a.even.kron_add(b.even, infos=infos)
        return FlatFermionTensor(odd=odd, even=even)

    def kron_add(self, b, infos=None):
        """Direct sum of first and last legs.
        Middle legs are summed."""
        return self._kron_add(self, b, infos=infos)

    @staticmethod
    @implements(np.diag)
    def _diag(v):
        """
        Extract a diagonal or construct a diagonal array.

        Args:
            v : FlatFermionTensor
                If v is a 2-D array, return a copy of its 0-th diagonal.
                If v is a 1-D array, return a 2-D array with v on the 0-th diagonal.
        """
        return FlatFermionTensor(odd=np.diag(v.odd), even=np.diag(v.even))

    def diag(self):
        return np.diag(self)

    @staticmethod
    @implements(np.transpose)
    def _transpose(a, axes=None):
        """
        Reverse or permute the axes of an array; returns the modified array.

        Args:
            a : array_like
                Input array.
            axes : tuple or list of ints, optional
                If specified, it must be a tuple or list which contains a permutation of [0,1,..,N-1]
                where N is the number of axes of a.
                The i’th axis of the returned array will correspond to the axis numbered axes[i] of the input.
                If not specified, defaults to ``range(a.ndim)[::-1]``, which reverses the order of the axes.

        Returns
            p : FlatFermionTensor
                a with its axes permuted. A view is returned whenever possible.
        """
        odd = np.transpose(a.odd, axes=axes)
        even = np.transpose(a.even, axes=axes)
        return FlatFermionTensor(odd=odd, even=even)

    def transpose(self, axes=None):
        return np.transpose(self, axes=axes)

    @staticmethod
    def _to_fermion(a):
        return a.to_fermion()

    def to_fermion(self):
        return FermionTensor(odd=self.odd.to_sparse(), even=self.even.to_sparse())

    @staticmethod
    def _to_dense(a, infos=None):
        return a.to_fermion().to_dense(infos=infos)

    def to_dense(self, infos=None):
        return self.to_fermion().to_dense(infos=infos)
