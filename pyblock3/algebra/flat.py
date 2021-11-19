
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
High performance version for block-sparse tensors
and block-sparse tensors with fermion factors.

Flat classes have the same interface as their counterparts in
core.py, but with C++ optimized implementations.
"""

import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin
import numbers
from collections import Counter
from itertools import accumulate, groupby

from .symmetry import BondInfo, BondFusingInfo, SZ
from .core import SparseTensor, SubTensor, FermionTensor
from .impl.flat import *

ENABLE_FAST_IMPLS = True

if ENABLE_FAST_IMPLS:
    try:
        import block3.sz as block3
        flat_sparse_tensordot = block3.flat_sparse_tensor.tensordot
        flat_sparse_add = block3.flat_sparse_tensor.add
        flat_sparse_kron_add = block3.flat_sparse_tensor.kron_add
        flat_sparse_fuse = block3.flat_sparse_tensor.fuse
        flat_sparse_get_infos = block3.flat_sparse_tensor.get_infos
        flat_sparse_tensor_svd = block3.flat_sparse_tensor.tensor_svd
        flat_sparse_left_canonicalize = block3.flat_sparse_tensor.left_canonicalize
        flat_sparse_right_canonicalize = block3.flat_sparse_tensor.right_canonicalize
        flat_sparse_left_svd = block3.flat_sparse_tensor.left_svd
        flat_sparse_right_svd = block3.flat_sparse_tensor.right_svd
        flat_sparse_truncate_svd = block3.flat_sparse_tensor.truncate_svd
        flat_sparse_left_canonicalize_indexed = block3.flat_sparse_tensor.left_canonicalize_indexed
        flat_sparse_right_canonicalize_indexed = block3.flat_sparse_tensor.right_canonicalize_indexed
        flat_sparse_left_svd_indexed = block3.flat_sparse_tensor.left_svd_indexed
        flat_sparse_right_svd_indexed = block3.flat_sparse_tensor.right_svd_indexed

        def flat_sparse_transpose_impl(aqs, ashs, adata, aidxs, axes):
            data = np.zeros_like(adata)
            block3.flat_sparse_tensor.transpose(ashs, adata, aidxs, axes, data)
            return (aqs[:, axes], ashs[:, axes], data, aidxs)
        flat_sparse_transpose = flat_sparse_transpose_impl

        def flat_sparse_skeleton_impl(bond_infos, pattern=None, dq=None):
            fdq = dq.to_flat() if dq is not None else SZ(0, 0, 0).to_flat()
            if pattern is None:
                pattern = "+" * (len(bond_infos) - 1) + "-"
            return block3.flat_sparse_tensor.skeleton(block3.VectorMapUIntUInt(bond_infos), pattern, fdq)
        flat_sparse_skeleton = flat_sparse_skeleton_impl

        def flat_sparse_kron_sum_info_impl(aqs, ashs, pattern=None):
            if pattern is None:
                pattern = "+" * (aqs.shape[1] if aqs.size != 0 else 0)
            return block3.flat_sparse_tensor.kron_sum_info(aqs, ashs, pattern)
        flat_sparse_kron_sum_info = flat_sparse_kron_sum_info_impl

        def flat_sparse_kron_product_info_impl(infos, pattern=None):
            if pattern is None:
                pattern = "+" * len(infos)
            return block3.flat_sparse_tensor.kron_product_info(block3.VectorMapUIntUInt(infos), pattern)
        flat_sparse_kron_product_info = flat_sparse_kron_product_info_impl

    except ImportError:
        import warnings
        warnings.warn('Fast flat sparse implementation is not enabled.')


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
            self.idxs = np.zeros((self.n_blocks + 1, ), dtype=np.uint64)
            self.idxs[1:] = np.cumsum(shapes.prod(axis=1), dtype=np.uint64)
        else:
            self.idxs = idxs
        if self.n_blocks != 0:
            assert shapes.shape == (self.n_blocks, self.ndim)
            assert q_labels.shape == (self.n_blocks, self.ndim)

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def size(self):
        return self.data.size

    @property
    def nbytes(self):
        return self.q_labels.nbytes + self.shapes.nbytes + self.data.nbytes + self.idxs.nbytes

    def item(self):
        if len(self.data) == 0:
            return 0
        return self.data.item()

    def to_sparse(self):
        blocks = [None] * self.n_blocks
        for i in range(self.n_blocks):
            qs = tuple(map(SZ.from_flat, self.q_labels[i]))
            blocks[i] = SubTensor(
                self.data[self.idxs[i]:self.idxs[i + 1]].reshape(self.shapes[i]), q_labels=qs)
        return SparseTensor(blocks=blocks)

    @staticmethod
    def get_zero():
        zu = np.zeros((0, 0), dtype=np.uint32)
        zd = np.zeros((0, ), dtype=float)
        iu = np.zeros((1, ), dtype=np.uint64)
        return FlatSparseTensor(zu, zu, zd, iu)

    @staticmethod
    def from_sparse(spt):
        ndim = spt.ndim
        n_blocks = spt.n_blocks
        shapes = np.zeros((n_blocks, ndim), dtype=np.uint32)
        q_labels = np.zeros((n_blocks, ndim), dtype=np.uint32)
        for i in range(n_blocks):
            shapes[i] = spt.blocks[i].shape
            q_labels[i] = list(map(SZ.to_flat, spt.blocks[i].q_labels))
        idxs = np.zeros((n_blocks + 1, ), dtype=np.uint64)
        idxs[1:] = np.cumsum(shapes.prod(axis=1), dtype=np.uint64)
        data = np.zeros((idxs[-1], ), dtype=spt.dtype)
        for i in range(n_blocks):
            data[idxs[i]:idxs[i + 1]] = spt.blocks[i].flatten()
        return FlatSparseTensor(q_labels, shapes, data, idxs)

    def __str__(self):
        return str(self.to_sparse())

    def __repr__(self):
        return repr(self.to_sparse())

    def __getitem__(self, i, idx=-1):
        if isinstance(i, tuple):
            qq = np.array([x.to_flat() for x in i], dtype=self.q_labels.dtype)
            for j in range(self.n_blocks):
                if np.array_equal(self.q_labels[j], qq):
                    mat = self.data[self.idxs[j]:self.idxs[j + 1]]
                    mat.shape = self.shapes[j]
                    return mat
        elif isinstance(i, slice) or isinstance(i, int):
            return NotImplemented
        else:
            q = i.to_flat()
            for j in range(self.n_blocks):
                if self.q_labels[j, idx] == q:
                    mat = self.data[self.idxs[j]:self.idxs[j + 1]]
                    mat.shape = self.shapes[j]
                    return mat

    @staticmethod
    def zeros_like(x, dtype=None):
        if dtype is None:
            dtype = x.dtype
        return x.__class__(q_labels=x.q_labels, shapes=x.shapes, data=np.zeros_like(x.data, dtype=dtype), idxs=x.idxs)

    @staticmethod
    def random_like(x):
        return x.__class__(q_labels=x.q_labels, shapes=x.shapes, data=np.random.random((x.idxs[-1], )), idxs=x.idxs)

    @staticmethod
    def zeros(bond_infos, pattern=None, dq=None, dtype=float):
        """Create tensor from tuple of BondInfo with zero elements."""
        qs, shs, idxs = flat_sparse_skeleton(bond_infos, pattern, dq)
        data = np.zeros((idxs[-1], ), dtype=dtype)
        return FlatSparseTensor(qs, shs, data, idxs)

    @staticmethod
    def ones(bond_infos, pattern=None, dq=None, dtype=float):
        """Create tensor from tuple of BondInfo with ones."""
        qs, shs, idxs = flat_sparse_skeleton(bond_infos, pattern, dq)
        data = np.ones((idxs[-1], ), dtype=dtype)
        return FlatSparseTensor(qs, shs, data, idxs)

    @staticmethod
    def random(bond_infos, pattern=None, dq=None, dtype=float):
        """Create tensor from tuple of BondInfo with random elements."""
        qs, shs, idxs = flat_sparse_skeleton(bond_infos, pattern, dq)
        if dtype == float:
            data = np.random.random((idxs[-1], ))
        elif dtype == complex:
            data = np.random.random(
                (idxs[-1], )) + np.random.random((idxs[-1], )) * 1j
        else:
            return NotImplementedError('dtype %r not supported!' % dtype)
        return FlatSparseTensor(qs, shs, data, idxs)

    @property
    def infos(self):
        return flat_sparse_get_infos(self.q_labels, self.shapes)

    def conj(self):
        """
        Complex conjugate.
        Note that np.conj() is a ufunc (no need to be defined).
        """
        return self.__class__(q_labels=self.q_labels, shapes=self.shapes,
                              data=self.data.conj(), idxs=self.idxs)

    @staticmethod
    @implements(np.real)
    def _real(x):
        return x.__class__(q_labels=x.q_labels, shapes=x.shapes,
                           data=np.ascontiguousarray(x.data.real), idxs=x.idxs)

    @property
    def real(self):
        return np.real(self)

    @staticmethod
    @implements(np.imag)
    def _imag(x):
        return x.__class__(q_labels=x.q_labels, shapes=x.shapes,
                           data=np.ascontiguousarray(x.data.imag), idxs=x.idxs)

    @property
    def imag(self):
        return np.imag(self)

    def cast_assign(self, other):
        """assign other to self, removing blocks not in self."""
        aqs, adata, aidxs = other.q_labels, other.data, other.idxs
        bqs, bdata, bidxs = self.q_labels, self.data, self.idxs
        blocks_map = {q.tobytes(): iq for iq, q in enumerate(aqs)}
        for ib in range(self.n_blocks):
            q = bqs[ib].tobytes()
            if q in blocks_map:
                ia = blocks_map[q]
                bdata[bidxs[ib]:bidxs[ib + 1]] = adata[aidxs[ia]:aidxs[ia + 1]]
        return self

    def normalize_along_axis(self, axis):
        if axis < 0:
            axis += self.ndim
        normsq = {}
        for ib in range(self.n_blocks):
            q = self.q_labels[ib, axis]
            if q not in normsq:
                normsq[q] = 0.0
            normsq[q] += np.linalg.norm(self.data[self.idxs[ib]:self.idxs[ib + 1]]) ** 2
        for ib in range(self.n_blocks):
            q = self.q_labels[ib, axis]
            if normsq[q] > 1E-12:
                self.data[self.idxs[ib]:self.idxs[ib + 1]] /= np.sqrt(normsq[q])

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
            out[0].shapes[...] = shs
            out[0].q_labels[...] = qs
            assert data.dtype == out[0].data.dtype
            out[0].data[...] = data
            out[0].idxs[...] = idxs
        return self.__class__(q_labels=qs, shapes=shs, data=data, idxs=idxs)

    def __array_function__(self, func, types, args, kwargs):
        if func not in _flat_sparse_tensor_numpy_func_impls:
            return NotImplemented
        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented
        return _flat_sparse_tensor_numpy_func_impls[func](*args, **kwargs)

    @staticmethod
    @implements(np.copy)
    def _copy(x):
        return x.__class__(q_labels=x.q_labels.copy(), shapes=x.shapes.copy(), data=x.data.copy(), idxs=x.idxs.copy())

    def copy(self):
        return np.copy(self)

    @staticmethod
    @implements(np.linalg.norm)
    def _norm(x):
        return np.linalg.norm(x.data)

    def norm(self):
        return np.linalg.norm(self.data)

    def kron_sum_info(self, *idxs, pattern=None):
        idxs = np.array([i if i >= 0 else self.ndim +
                         i for i in idxs], dtype=np.int32)
        # using minimal fused dimension
        return flat_sparse_kron_sum_info(
            self.q_labels[:, idxs], self.shapes[:, idxs], pattern=pattern)

    def kron_product_info(self, *idxs, pattern=None):
        idxs = np.array([i if i >= 0 else self.ndim +
                         i for i in idxs], dtype=np.int32)
        return flat_sparse_kron_product_info(np.array(self.infos)[idxs], pattern=pattern)

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
                A str of '+'/'-'. Only required when info is not specified (if not in fast mode).
                Indicating how quantum numbers are linearly combined.
        """
        idxs = np.array([i if i >= 0 else self.ndim +
                         i for i in idxs], dtype=np.int32)
        if info is None:
            info = self.kron_sum_info(*idxs, pattern=pattern)
        if pattern is None:
            if hasattr(info, "pattern"):
                pattern = info.pattern
            else:
                pattern = "+" * len(idxs)
        r = flat_sparse_fuse(self.q_labels, self.shapes, self.data,
                             self.idxs, idxs, info, pattern)
        x = FlatSparseTensor(*r)
        return x

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

        return a.__class__(*flat_sparse_tensordot(
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
            return b.__class__(b.q_labels, b.shapes, data, b.idxs)
        elif isinstance(b, numbers.Number):
            data = a.data + b
            return a.__class__(a.q_labels, a.shapes, data, a.idxs)
        elif b.n_blocks == 0:
            return a
        elif a.n_blocks == 0:
            return b
        else:
            return a.__class__(*flat_sparse_add(a.q_labels, a.shapes, a.data,
                                                a.idxs, b.q_labels, b.shapes, b.data, b.idxs))

    def add(self, b):
        return self._add(self, b)

    @staticmethod
    @implements(np.subtract)
    def _subtract(a, b):
        if isinstance(a, numbers.Number):
            data = a - b.data
            return b.__class__(b.q_labels, b.shapes, data, b.idxs)
        elif isinstance(b, numbers.Number):
            data = a.data - b
            return a.__class__(a.q_labels, a.shapes, data, a.idxs)
        elif b.n_blocks == 0:
            return a
        elif a.n_blocks == 0:
            return b.__class__(b.q_labels, b.shapes, -b.data, b.idxs)
        else:
            return a.__class__(*flat_sparse_add(a.q_labels, a.shapes, a.data,
                                                a.idxs, b.q_labels, b.shapes, -b.data, b.idxs))

    def subtract(self, b):
        return self._subtract(self, b)

    @staticmethod
    def _kron_add(a, b, infos=None):
        """
        Direct sum of first and last legs.
        Middle legs are summed.

        Args:
            infos : (BondInfo, BondInfo) or None
                BondInfo of first and last legs for the result.
        """
        if infos is None:
            abi, bbi = a.infos, b.infos
            infos = (abi[0] + bbi[0], abi[-1] + bbi[-1])

        return a.__class__(
            *flat_sparse_kron_add(a.q_labels, a.shapes, a.data,
                                  a.idxs, b.q_labels, b.shapes,
                                  b.data, b.idxs, infos[0], infos[1]))

    def kron_add(self, b, infos=None):
        """Direct sum of first and last legs.
        Middle legs are summed."""
        return self._kron_add(self, b, infos=infos)

    def left_canonicalize(self, mode='reduced'):
        """
        Left canonicalization (using QR factorization).
        Left canonicalization needs to collect all left indices for each specific right index.
        So that we will only have one R, but left dim of q is unchanged.

        Returns:
            q, r : tuple(FlatSparseTensor)
        """
        assert mode == 'reduced'
        r = flat_sparse_left_canonicalize(
            self.q_labels, self.shapes, self.data, self.idxs)
        return self.__class__(*r[:4]), self.__class__(*r[4:])

    def right_canonicalize(self, mode='reduced'):
        """
        Right canonicalization (using QR factorization).

        Returns:
            l, q : tuple(FlatSparseTensor)
        """
        assert mode == 'reduced'
        r = flat_sparse_right_canonicalize(
            self.q_labels, self.shapes, self.data, self.idxs)
        return self.__class__(*r[:4]), self.__class__(*r[4:])

    def left_svd(self, full_matrices=False):
        assert not full_matrices
        lsr = flat_sparse_left_svd(
            self.q_labels, self.shapes, self.data, self.idxs)
        return self.__class__(*lsr[:4]), self.__class__(*lsr[4:8]), self.__class__(*lsr[8:])

    def right_svd(self, full_matrices=False):
        assert not full_matrices
        lsr = flat_sparse_right_svd(
            self.q_labels, self.shapes, self.data, self.idxs)
        return self.__class__(*lsr[:4]), self.__class__(*lsr[4:8]), self.__class__(*lsr[8:])

    def tensor_svd(self, idx=2, linfo=None, rinfo=None, pattern=None, full_matrices=False):
        """
        Separate tensor in the middle, collecting legs as [0, idx) and [idx, ndim), then perform SVD.

        Returns:
            l, s, r : tuple(FlatSparseTensor)
        """
        assert full_matrices == False
        assert idx >= 1 and idx <= self.ndim - 1
        if pattern is None:
            pattern = '+' * self.ndim
        if linfo is None:
            linfo = self.kron_sum_info(*range(0, idx), pattern=pattern[:idx])
        if rinfo is None:
            rpat = ''.join(['-' if x == '+' else '+' for x in pattern[idx:]])
            rinfo = self.kron_sum_info(
                *range(idx, self.ndim), pattern=rpat)
        lsr = flat_sparse_tensor_svd(
            self.q_labels, self.shapes, self.data, self.idxs, idx, linfo, rinfo, pattern)
        return self.__class__(*lsr[:4]), self.__class__(*lsr[4:8]), self.__class__(*lsr[8:])

    @staticmethod
    def truncate_svd(l, s, r, max_bond_dim=-1, cutoff=0.0, max_dw=0.0, norm_cutoff=0.0, eigen_values=False):
        """
        Truncate tensors obtained from SVD.

        Args:
            l, s, r : tuple(FlatSparseTensor)
                SVD tensors.
            max_bond_dim : int
                Maximal total bond dimension.
                If `k == -1`, no restriction in total bond dimension.
            cutoff : double
                Minimal kept singular value.
            max_dw : double
                Maximal sum of square of discarded singular values.
            norm_cutoff : double
                Blocks with norm smaller than norm_cutoff will be deleted.
            eigen_values : bool
                If True, treat `s` as eigenvalues.

        Returns:
            l, s, r : tuple(FlatSparseTensor)
                SVD decomposition.
            error : float
                Truncation error (same unit as singular value squared).
        """
        lsre = flat_sparse_truncate_svd(
            l.q_labels, l.shapes, l.data, l.idxs,
            s.q_labels, s.shapes, s.data, s.idxs,
            r.q_labels, r.shapes, r.data, r.idxs,
            max_bond_dim, cutoff, max_dw, norm_cutoff, eigen_values)
        return l.__class__(*lsre[:4]), s.__class__(*lsre[4:8]), r.__class__(*lsre[8:12]), lsre[12]

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
            return v.__class__(
                q_labels=v.q_labels[mask, 0], shapes=shapes,
                data=np.concatenate([np.diag(v.data[i:j].reshape(sh, sh)) for i, j, sh in zip(v.idxs[:-1][mask], v.idxs[1:][mask], shapes)]))
        elif v.ndim == 1:
            return v.__class__(
                q_labels=np.repeat(v.q_labels, 2, axis=1),
                shapes=np.repeat(v.shapes, 2, axis=1),
                data=np.concatenate([np.diag(v.data[i:j]).flatten() for i, j in zip(v.idxs, v.idxs[1:])]))
        elif v.n_blocks != 0:
            raise RuntimeError("ndim for np.diag must be 1 or 2.")
        else:
            return v

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
            axes = np.array(axes, dtype=np.int32)
            return a.__class__(*flat_sparse_transpose(a.q_labels, a.shapes, a.data, a.idxs, axes))

    def transpose(self, axes=None):
        return np.transpose(self, axes=axes)

    @staticmethod
    def _to_dense(a, infos=None):
        return a.to_dense(infos=infos)

    def to_dense(self, infos=None):
        return self.to_sparse().to_dense(infos=infos)

    def to_flat(self):
        return self


_flat_fermion_tensor_numpy_func_impls = {}
_numpy_func_impls = _flat_fermion_tensor_numpy_func_impls


class FlatFermionTensor(FermionTensor):
    ZERO = FlatSparseTensor.get_zero()
    """
    flat block-sparse tensor with fermion factors.

    Attributes:
        odd : FlatSparseTensor
            Including blocks with odd fermion parity.
        even : FlatSparseTensor
            Including blocks with even fermion parity.
    """

    def __init__(self, odd=None, even=None):
        self.odd = odd if odd is not None else FlatFermionTensor.ZERO
        self.even = even if even is not None else FlatFermionTensor.ZERO

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
    def zeros(bond_infos, pattern=None, dq=None, dtype=float):
        spt = FlatSparseTensor.zeros(
            bond_infos, pattern=pattern, dq=dq, dtype=dtype)
        if dq is not None and dq.is_fermion:
            return FlatFermionTensor(odd=spt)
        else:
            return FlatFermionTensor(even=spt)

    @staticmethod
    def ones(bond_infos, pattern=None, dq=None, dtype=float):
        spt = FlatSparseTensor.ones(
            bond_infos, pattern=pattern, dq=dq, dtype=dtype)
        if dq is not None and dq.is_fermion:
            return FlatFermionTensor(odd=spt)
        else:
            return FlatFermionTensor(even=spt)

    @staticmethod
    def random(bond_infos, pattern=None, dq=None, dtype=float):
        spt = FlatSparseTensor.random(
            bond_infos, pattern=pattern, dq=dq, dtype=dtype)
        if dq is not None and dq.is_fermion:
            return FlatFermionTensor(odd=spt)
        else:
            return FlatFermionTensor(even=spt)

    def deflate(self, cutoff=0):
        return FlatFermionTensor(odd=self.odd.deflate(cutoff), even=self.even.deflate(cutoff))

    @staticmethod
    @implements(np.real)
    def _real(x):
        return x.__class__(odd=x.odd.real, even=x.even.real)

    @staticmethod
    @implements(np.imag)
    def _imag(x):
        return x.__class__(odd=x.odd.imag, even=x.even.imag)

    @staticmethod
    @implements(np.copy)
    def _copy(x):
        return FlatFermionTensor(odd=x.odd.copy(), even=x.even.copy())

    def copy(self):
        return np.copy(self)

    @staticmethod
    @implements(np.linalg.norm)
    def _norm(x):
        return np.sqrt(np.linalg.norm(x.odd) ** 2 + np.linalg.norm(x.even) ** 2)

    def norm(self):
        return np.sqrt(np.linalg.norm(self.odd) ** 2 + np.linalg.norm(self.even) ** 2)

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

    def kron_sum_info(self, *idxs, pattern=None):
        idxs = np.array([i if i >= 0 else self.ndim +
                         i for i in idxs], dtype=np.int32)
        if self.odd.n_blocks == 0:
            qs, shs = self.even.q_labels[:, idxs], self.even.shapes[:, idxs]
        elif self.even.n_blocks == 0:
            qs, shs = self.odd.q_labels[:, idxs], self.odd.shapes[:, idxs]
        else:
            qs = np.concatenate(
                (self.odd.q_labels[:, idxs], self.even.q_labels[:, idxs]))
            shs = np.concatenate(
                (self.odd.shapes[:, idxs], self.even.shapes[:, idxs]))
        # using minimal fused dimension
        return flat_sparse_kron_sum_info(qs, shs, pattern=pattern)

    def kron_product_info(self, *idxs, pattern=None):
        idxs = np.array([i if i >= 0 else self.ndim +
                         i for i in idxs], dtype=np.int32)
        return flat_sparse_kron_product_info(np.array(self.infos)[idxs], pattern=pattern)

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
        idxs = np.array([i if i >= 0 else self.ndim +
                         i for i in idxs], dtype=np.int32)
        if info is None:
            info = self.kron_sum_info(*idxs, pattern=pattern)
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
            if len(idxa) == 0 and len(idxb) == 0:
                assert a.ndim % 2 == 0
                d = a.ndim // 2
                idx = range(d, d + d) if d != 1 else d
                blocks = [odd_b, even_a]
            # horizontal
            elif list(idxa) == [a.ndim - 1] and list(idxb) == [0]:
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
                        if SZ.is_flat_fermion(q):
                            np.negative(x.data[i:j], out=x.data[i:j])
        else:
            for x in blocks:
                if x.n_blocks != 0:
                    for i, j, qs in zip(x.idxs, x.idxs[1:], x.q_labels[:, idx]):
                        if np.logical_xor.reduce([SZ.is_flat_fermion(q) for q in qs]):
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
            infos = (abi[0] + bbi[0], abi[-1] + bbi[-1])

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

    @staticmethod
    def _to_flat_sparse(a):
        return a.to_flat_sparse()

    def to_flat_sparse(self):
        if self.odd.n_blocks == 0:
            return self.even
        elif self.even.n_blocks == 0:
            return self.odd
        qs = np.concatenate([self.odd.q_labels, self.even.q_labels], axis=0)
        shs = np.concatenate([self.odd.shapes, self.even.shapes], axis=0)
        data = np.concatenate([self.odd.data, self.even.data], axis=0)
        return FlatSparseTensor(qs, shs, data)

    def split(self, xidx, nodd):
        assert self.even.n_blocks == 0
        ii = np.arange(0, len(xidx), dtype=int)
        mo, me = ii[xidx < nodd], ii[xidx >= nodd]
        if len(mo) == 0:
            return FlatFermionTensor(even=self.odd)
        elif len(me) == 0:
            return self
        dt, ix = self.odd.data, self.odd.idxs
        odata = np.concatenate([dt[st:ed]
                                for st, ed in zip(ix[mo], ix[mo + 1])])
        edata = np.concatenate([dt[st:ed]
                                for st, ed in zip(ix[me], ix[me + 1])])
        odd = FlatSparseTensor(
            self.odd.q_labels[mo], self.odd.shapes[mo], odata)
        even = FlatSparseTensor(
            self.odd.q_labels[me], self.odd.shapes[me], edata)
        return FlatFermionTensor(odd=odd, even=even)

    def left_canonicalize(self, mode='reduced'):
        """
        Left canonicalization (using QR factorization).
        Left canonicalization needs to collect all left indices for each specific right index.
        So that we will only have one R, but left dim of q is unchanged.

        Returns:
            q, r : (FlatFermionTensor, FlatSparseTensor (gauge))
        """
        assert mode == 'reduced'
        a = self.to_flat_sparse()
        qr, xidx = flat_sparse_left_canonicalize_indexed(
            a.q_labels, a.shapes, a.data, a.idxs)
        return FlatFermionTensor(odd=FlatSparseTensor(*qr[0:4])).split(xidx, self.odd.n_blocks), \
            FlatSparseTensor(*qr[4:8])

    def right_canonicalize(self, mode='reduced'):
        """
        Right canonicalization (using QR factorization).

        Returns:
            l, q : (FlatSparseTensor (gauge), FlatFermionTensor)
        """
        assert mode == 'reduced'
        a = self.to_flat_sparse()
        lq, xidx = flat_sparse_right_canonicalize_indexed(
            a.q_labels, a.shapes, a.data, a.idxs)
        return FlatSparseTensor(*lq[0:4]), \
            FlatFermionTensor(odd=FlatSparseTensor(
                *lq[4:8])).split(xidx, self.odd.n_blocks)

    def left_svd(self, full_matrices=True):
        """
        Left svd needs to collect all left indices for each specific right index.

        Returns:
            l, s, r : (FlatFermionTensor, FlatSparseTensor (vector), FlatSparseTensor (gauge))
        """
        assert full_matrices == False
        a = self.to_flat_sparse()
        lsr, xidx = flat_sparse_left_svd_indexed(
            a.q_labels, a.shapes, a.data, a.idxs)
        return FlatFermionTensor(odd=FlatSparseTensor(*lsr[0:4])).split(xidx, self.odd.n_blocks), \
            FlatSparseTensor(*lsr[4:8]), FlatSparseTensor(*lsr[8:12])

    def right_svd(self, full_matrices=True):
        """
        Right svd needs to collect all right indices for each specific left index.

        Returns:
            l, s, r : (FlatSparseTensor (gauge), FlatSparseTensor (vector), FlatFermionTensor)
        """
        assert full_matrices == False
        a = self.to_flat_sparse()
        lsr, xidx = flat_sparse_right_svd_indexed(
            a.q_labels, a.shapes, a.data, a.idxs)
        return FlatSparseTensor(*lsr[0:4]), FlatSparseTensor(*lsr[4:8]), \
            FlatFermionTensor(odd=FlatSparseTensor(
                *lsr[8:12])).split(xidx, self.odd.n_blocks)

    @staticmethod
    def truncate_svd(l, s, r, max_bond_dim=-1, cutoff=0.0, max_dw=0.0, norm_cutoff=0.0):
        """
        Truncate tensors obtained from SVD.

        Args:
            l, s, r : tuple(FlatSparseTensor/FlatFermionTensor)
                SVD tensors.
            max_bond_dim : int
                Maximal total bond dimension.
                If `k == -1`, no restriction in total bond dimension.
            cutoff : double
                Minimal kept singular value.
            max_dw : double
                Maximal sum of square of discarded singular values.
            norm_cutoff : double
                Blocks with norm smaller than norm_cutoff will be deleted.

        Returns:
            l, s, r : tuple(FlatSparseTensor/FlatFermionTensor)
                SVD decomposition.
            error : float
                Truncation error (same unit as singular value squared).
        """
        ll = (l.odd, l.even) if isinstance(l, FlatFermionTensor) else (l, l)
        rr = (r.odd, r.even) if isinstance(r, FlatFermionTensor) else (r, r)
        lsre = flat_sparse_truncate_svd(
            ll[0].q_labels, ll[0].shapes, ll[0].data, ll[0].idxs, s.q_labels,
            s.shapes, s.data, s.idxs, rr[0].q_labels, rr[0].shapes, rr[0].data,
            rr[0].idxs, max_bond_dim, cutoff, max_dw, norm_cutoff)
        if isinstance(l, FlatFermionTensor) or isinstance(r, FlatFermionTensor):
            lsre2 = flat_sparse_truncate_svd(
                ll[1].q_labels, ll[1].shapes, ll[1].data, ll[1].idxs, s.q_labels,
                s.shapes, s.data, s.idxs, rr[1].q_labels, rr[1].shapes, rr[1].data,
                rr[1].idxs, max_bond_dim, cutoff, max_dw, norm_cutoff)
        nl = FlatSparseTensor(*lsre[0:4])
        if isinstance(l, FlatFermionTensor):
            nl = FlatFermionTensor(odd=nl, even=FlatSparseTensor(*lsre2[0:4]))
        ns = FlatSparseTensor(*lsre[4:8])
        nr = FlatSparseTensor(*lsre[8:12])
        if isinstance(r, FlatFermionTensor):
            nr = FlatFermionTensor(odd=nr, even=FlatSparseTensor(*lsre2[8:12]))
        return nl, ns, nr, lsre[12]

    @staticmethod
    @implements(np.linalg.qr)
    def _qr(a, mode='reduced'):
        return a.left_canonicalize(mode)

    def qr(self, mode='reduced'):
        return self.left_canonicalize(mode)

    @staticmethod
    def _lq(a, mode='reduced'):
        return a.right_canonicalize(mode)

    def lq(self, mode='reduced'):
        return self.right_canonicalize(mode)

    @staticmethod
    @implements(np.linalg.svd)
    def _svd(a, full_matrices=True):
        return a.left_svd(full_matrices)

    def to_flat(self):
        return self
