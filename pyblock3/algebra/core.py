
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

"""
Basic definitions for block-sparse tensors
and block-sparse tensors with fermion factors.
"""

import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin
import numbers
from collections import Counter
from itertools import accumulate, groupby

from .symmetry import BondInfo, BondFusingInfo


def method_alias(name):
    """Make method callable from algebra.funcs."""
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
    """Wrapper for overwritting numpy methods."""
    global _numpy_func_impls
    return lambda f: (_numpy_func_impls.update({np_func: f})
                      if np_func not in _numpy_func_impls else None,
                      _numpy_func_impls[np_func])[1]


_sub_tensor_numpy_func_impls = {}
_numpy_func_impls = _sub_tensor_numpy_func_impls


class SubTensor(np.ndarray):
    """
    A block in block-sparse tensor.

    Attributes:
        q_labels : tuple(SZ..)
            Quantum labels for this sub-tensor block.
            Each element in the tuple corresponds one leg of the tensor.
    """
    def __new__(cls, reduced, q_labels=None):
        obj = np.asarray(reduced).view(cls)
        if q_labels is not None and not isinstance(q_labels, tuple):
            q_labels = tuple(q_labels)
        obj.q_labels = q_labels
        return obj

    def __init__(self, reduced, q_labels=None):
        pass

    def __array_finalize__(self, obj):
        if obj is not None:
            self.q_labels = getattr(obj, 'q_labels', None)

    def __repr__(self):
        return "(Q=) %r (R=) %r" % (self.q_labels, np.asarray(self))

    def __str__(self):
        return "(Q=) %r (R=) %r" % (self.q_labels, np.asarray(self))

    def __array_function__(self, func, types, args, kwargs):
        if func not in _sub_tensor_numpy_func_impls:
            return NotImplemented
        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented
        return _sub_tensor_numpy_func_impls[func](*args, **kwargs)

    def __reduce__(self):
        r = super().__reduce__()
        return r[:2] + (r[2] + (self.q_labels, ), )

    def __setstate__(self, state):
        self.q_labels = state[-1]
        super().__setstate__(state[:-1])

    @classmethod
    def zeros(cls, shape, q_labels=None, dtype=float):
        obj = np.asarray(np.zeros(shape, dtype=dtype)).view(cls)
        obj.q_labels = q_labels
        return obj

    @classmethod
    def ones(cls, shape, q_labels=None, dtype=float):
        obj = np.asarray(np.ones(shape, dtype=dtype)).view(cls)
        obj.q_labels = q_labels
        return obj

    @classmethod
    def random(cls, shape, q_labels=None, dtype=float):
        if dtype == float:
            obj = np.random.random(shape)
        elif dtype == complex:
            obj = np.random.random(shape) + np.random.random(shape) * 1j
        else:
            return NotImplementedError('dtype %r not supported!' % dtype)
        obj = np.asarray(obj).view(cls)
        obj.q_labels = q_labels
        return obj

    @staticmethod
    @implements(np.copy)
    def _copy(x):
        return SubTensor(reduced=np.asarray(x).copy(), q_labels=x.q_labels)

    def copy(self):
        return np.copy(self)

    def conj(self):
        return SubTensor(reduced=np.asarray(self).conj(), q_labels=self.q_labels)

    @staticmethod
    @implements(np.real)
    def _real(x):
        return SubTensor(reduced=np.ascontiguousarray(np.asarray(x).real), q_labels=x.q_labels)

    @property
    def real(self):
        return np.real(self)

    @staticmethod
    @implements(np.imag)
    def _imag(x):
        return SubTensor(reduced=np.ascontiguousarray(np.asarray(x).imag), q_labels=x.q_labels)

    @property
    def imag(self):
        return np.imag(self)

    @staticmethod
    @implements(np.linalg.norm)
    def _norm(x):
        return np.linalg.norm(np.asarray(x))

    def norm(self):
        return np.linalg.norm(np.asarray(self))

    @staticmethod
    @implements(np.tensordot)
    def _tensordot(a, b, axes=2):
        """
        Contract two SubTensor to form a new SubTensor.

        Args:
            a : SubTensor
                SubTensor a, as left operand.
            b : SubTensor
                SubTensor b, as right operand.
            axes : int or (2,) array_like
                If an int N, sum over the last N axes of a and the first N axes of b in order.
                Or, a list of axes to be summed over, first sequence applying to a, second to b.

        Returns:
            SubTensor : SubTensor
                The contracted SubTensor.
        """
        if isinstance(axes, int):
            idxa, idxb = list(range(-axes, 0)), list(range(0, axes))
        else:
            idxa, idxb = axes
        idxa = [x if x >= 0 else a.ndim + x for x in idxa]
        idxb = [x if x >= 0 else b.ndim + x for x in idxb]
        out_idx_a = sorted(list(set(range(0, a.ndim)) - set(idxa)))
        out_idx_b = sorted(list(set(range(0, b.ndim)) - set(idxb)))

        assert all(a.q_labels[ia] == b.q_labels[ib]
                   for ia, ib in zip(idxa, idxb))

        r = np.tensordot(np.asarray(a), np.asarray(
            b), axes=(idxa, idxb)).view(a.__class__)
        r.q_labels = tuple(a.q_labels[id] for id in out_idx_a) \
            + tuple(b.q_labels[id] for id in out_idx_b)
        return r

    def tensordot(self, b, axes=2):
        return np.tensordot(self, b, axes)

    @staticmethod
    @implements(np.diag)
    def _diag(v):
        """
        Extract a diagonal or construct a diagonal array.

        Args:
            v : SubTensor
                If v is a 2-D array, return a copy of its 0-th diagonal.
                If v is a 1-D array, return a 2-D array with v on the 0-th diagonal.
        """
        if v.ndim == 2:
            assert v.q_labels[0] == v.q_labels[1]
            return SubTensor(reduced=np.diag(np.asarray(v)), q_labels=v.q_labels[:1])
        elif v.ndim == 1:
            return SubTensor(reduced=np.diag(np.asarray(v)), q_labels=v.q_labels * 2)
        else:
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
            p : SubTensor
                a with its axes permuted. A view is returned whenever possible.
        """
        if axes is None:
            axes = range(a.ndim)[::-1]
        r = np.transpose(np.asarray(a), axes=axes).view(a.__class__)
        r.q_labels = tuple(a.q_labels[i] for i in axes)
        return r

    def transpose(self, axes=None):
        return np.transpose(self, axes=axes)


_sliceable_tensor_numpy_func_impls = {}
_numpy_func_impls = _sliceable_tensor_numpy_func_impls


class SliceableTensor(np.ndarray):
    """
    Dense tensor of zero and non-zero blocks.
    For zero blocks, the elemenet is zero.
    """
    def __new__(cls, reduced, infos=None):
        obj = np.asarray(reduced, dtype=object).view(cls)
        obj.infos = infos
        return obj

    def __init__(self, reduced, infos=None):
        pass

    def __array_finalize__(self, obj):
        if obj is not None:
            self.infos = getattr(obj, 'infos', None)

    def __repr__(self):
        idx = np.indices(self.shape).reshape((self.ndim, -1)).transpose()
        p = np.asarray(self)
        return "\n".join("%10s %r" % (ix, p[tuple(ix)]) for ix in idx if p[tuple(ix)] is not None)

    def __str__(self):
        idx = np.indices(self.shape).reshape((self.ndim, -1)).transpose()
        p = np.asarray(self)
        return "\n".join("%10s %r" % (ix, p[tuple(ix)]) for ix in idx if p[tuple(ix)] is not None)

    def __array_function__(self, func, types, args, kwargs):
        if func not in _sliceable_tensor_numpy_func_impls:
            return NotImplemented
        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented
        return _sliceable_tensor_numpy_func_impls[func](*args, **kwargs)

    def __reduce__(self):
        r = super().__reduce__()
        return r[:2] + (r[2] + (self.infos, ), )

    def __setstate__(self, state):
        self.infos = state[-1]
        super().__setstate__(state[:-1])

    def __getitem__(self, key):
        r = super().__getitem__(key)
        r.infos = list(r.infos)
        if isinstance(key, int):
            kk = sorted(r.infos[0])[key]
            r.infos[0] = r.infos[0].__class__({kk: r.infos[0][kk]})
        elif isinstance(key, tuple):
            for ik, k in enumerate(key):
                if isinstance(k, int):
                    kk = sorted(r.infos[ik])[k]
                    r.infos[ik] = r.infos[ik].__class__({kk: r.infos[ik][kk]})
        else:
            raise TypeError("Unknown index type %r" % key.__class__)
        r.infos = tuple(r.infos)
        return r

    @staticmethod
    @implements(np.copy)
    def _copy(x):
        return SliceableTensor(reduced=np.asarray(x).copy(), infos=tuple(x.infos))

    def copy(self):
        return np.copy(self)

    @staticmethod
    @implements(np.tensordot)
    def _tensordot(a, b, axes=2):
        """
        Contract two SliceableTensor to form a new SliceableTensor.

        Args:
            a : SliceableTensor
                SliceableTensor a, as left operand.
            b : SliceableTensor
                SliceableTensor b, as right operand.
            axes : int or (2,) array_like
                If an int N, sum over the last N axes of a and the first N axes of b in order.
                Or, a list of axes to be summed over, first sequence applying to a, second to b.

        Returns:
            SliceableTensor : SliceableTensor
                The contracted SliceableTensor.
        """
        if isinstance(axes, int):
            idxa, idxb = list(range(-axes, 0)), list(range(0, axes))
        else:
            idxa, idxb = axes
        idxa = [x if x >= 0 else len(a.infos) + x for x in idxa]
        idxb = [x if x >= 0 else len(b.infos) + x for x in idxb]
        out_idx_a = sorted(set(range(0, len(a.infos))) - set(idxa))
        out_idx_b = sorted(set(range(0, len(b.infos))) - set(idxb))
        new_infos = (*[a.infos[i] for i in out_idx_a],
                     *[b.infos[i] for i in out_idx_b])

        return np.tensordot(a.to_sparse(), b.to_sparse(), axes=axes).to_sliceable(infos=new_infos)

    def tensordot(self, b, axes=2):
        return np.tensordot(self, b, axes)

    @staticmethod
    @implements(np.dot)
    def _dot(a, b, out=None):
        """Contract the last dim of a and the first dim of b."""
        if isinstance(a, numbers.Number) or isinstance(b, numbers.Number):
            return np.multiply(a, b, out=out)

        assert isinstance(a, SliceableTensor) and isinstance(
            b, SliceableTensor)
        r = np.tensordot(a, b, axes=1)

        if out is not None:
            out[:] = np.asarray(r)
            out.infos = r.infos

        return r

    def dot(self, b, out=None):
        return np.dot(self, b, out=out)

    @property
    def density(self):
        """Ratio of number of non-zero elements to total number of elements."""
        idx = np.indices(self.shape).reshape((self.ndim, -1)).transpose()
        p = np.asarray(self)
        return len([0 for ix in idx if p[tuple(ix)] is not None]) / self.size

    @property
    def dtype(self):
        for v in self.flatten():
            if v != 0:
                return v.dtype
        return float

    @staticmethod
    def _to_sparse(a):
        return a.to_sparse()

    def to_sparse(self):
        blocks = np.asarray(self)[tuple(
            np.indices(self.shape).reshape((self.ndim, -1)))]
        return SparseTensor(blocks=blocks).quick_deflate()

    @staticmethod
    def _to_dense(a):
        return a.to_dense()

    def to_dense(self):
        """Convert to dense numpy.ndarray."""
        sh = tuple(info.n_bonds for info in self.infos)
        aw = np.indices(self.shape).reshape((self.ndim, -1)).transpose()
        r = np.zeros(sh, dtype=self.dtype)
        idxs = []
        for ii, info in enumerate(self.infos):
            idx = np.zeros((self.shape[ii] + 1, ), dtype=int)
            for ik, k in enumerate(sorted(info)):
                idx[ik + 1] = info[k] + idx[ik]
            idxs.append(idx)
        for ix in aw:
            p = np.asarray(self)
            if p[tuple(ix)] is not None:
                sl = tuple(slice(idx[k], idx[k + 1])
                           for k, idx in zip(ix, idxs))
                r[sl] = np.asarray(p[tuple(ix)])
        return r


_sparse_tensor_numpy_func_impls = {}
_numpy_func_impls = _sparse_tensor_numpy_func_impls


class SparseTensor(NDArrayOperatorsMixin):
    """
    block-sparse tensor.
    Represented as a list of :class:`SubTesnor`.
    """

    def __init__(self, blocks=None):
        self.blocks = blocks if blocks is not None else []

    @property
    def ndim(self):
        """Number of dimensions."""
        return 0 if len(self.blocks) == 0 else self.blocks[0].ndim

    @property
    def n_blocks(self):
        """Number of blocks."""
        return len(self.blocks)

    @property
    def nbytes(self):
        """Number bytes in memory."""
        return sum([b.nbytes for b in self.blocks])

    @property
    def T(self):
        """Transpose."""
        return self.transpose()

    @property
    def dtype(self):
        """Element datatype."""
        return self.blocks[0].dtype if self.n_blocks != 0 else float

    def item(self):
        """Return scalar element."""
        if self.n_blocks == 0:
            return 0
        assert self.n_blocks == 1 and self.blocks[0].q_labels == ()
        return self.blocks[0].item()

    def __str__(self):
        return "\n".join("%3d %r" % (ib, b) for ib, b in enumerate(self.blocks))

    def __repr__(self):
        return "\n".join("%3d %r" % (ib, b) for ib, b in enumerate(self.blocks))

    @staticmethod
    def _skeleton(bond_infos, pattern=None, dq=None):
        """Create tensor skeleton from tuple of BondInfo.
        dq will not have effects if ndim == 1
            (blocks with different dq will be allowed)."""
        it = np.zeros(tuple(len(i) for i in bond_infos), dtype=int)
        qsh = [sorted(i.items(), key=lambda x: x[0]) for i in bond_infos]
        q = [[k for k, v in i] for i in qsh]
        sh = [[v for k, v in i] for i in qsh]
        if pattern is None:
            pattern = ("+" * (len(bond_infos) - 1)) + "-"
        nit = np.nditer(it, flags=['multi_index'])
        for _ in nit:
            x = nit.multi_index
            ps = [iq[ix] if ip == '+' else -iq[ix]
                  for iq, ix, ip in zip(q, x, pattern)]
            if len(ps) == 1 or np.add.reduce(ps[:-1]) == (-ps[-1] if dq is None else dq - ps[-1]):
                xqs = tuple(iq[ix] for iq, ix in zip(q, x))
                xsh = tuple(ish[ix] for ish, ix in zip(sh, x))
                yield xsh, xqs

    @staticmethod
    def zeros(bond_infos, pattern=None, dq=None, dtype=float):
        """Create tensor from tuple of BondInfo with zero elements."""
        blocks = []
        for sh, qs in SparseTensor._skeleton(bond_infos, pattern=pattern, dq=dq):
            blocks.append(SubTensor.zeros(shape=sh, q_labels=qs, dtype=dtype))
        return SparseTensor(blocks=blocks)

    @staticmethod
    def ones(bond_infos, pattern=None, dq=None, dtype=float):
        """Create tensor from tuple of BondInfo with ones."""
        blocks = []
        for sh, qs in SparseTensor._skeleton(bond_infos, pattern=pattern, dq=dq):
            blocks.append(SubTensor.ones(shape=sh, q_labels=qs, dtype=dtype))
        return SparseTensor(blocks=blocks)

    @staticmethod
    def random(bond_infos, pattern=None, dq=None, dtype=float):
        """Create tensor from tuple of BondInfo with random elements."""
        blocks = []
        for sh, qs in SparseTensor._skeleton(bond_infos, pattern=pattern, dq=dq):
            blocks.append(SubTensor.random(shape=sh, q_labels=qs, dtype=dtype))
        return SparseTensor(blocks=blocks)

    @property
    def infos(self):
        """Return the quantum number layout of the SparseTensor, similar to numpy.ndarray.shape."""
        bond_infos = tuple(BondInfo() for _ in range(self.ndim))
        for block in self.blocks:
            for iq, q in enumerate(block.q_labels):
                if q in bond_infos[iq]:
                    assert block.shape[iq] == bond_infos[iq][q]
                else:
                    bond_infos[iq][q] = block.shape[iq]
        return bond_infos

    def conj(self):
        """
        Complex conjugate.
        Note that np.conj() is a ufunc (no need to be defined).
        But np.real and np.imag are array_functions
        """
        return self.__class__(blocks=[b.conj() for b in self.blocks])

    @staticmethod
    @implements(np.real)
    def _real(x):
        return x.__class__(blocks=[b.real for b in x.blocks])

    @property
    def real(self):
        return np.real(self)

    @staticmethod
    @implements(np.imag)
    def _imag(x):
        return x.__class__(blocks=[b.imag for b in x.blocks])

    @property
    def imag(self):
        return np.imag(self)

    def quick_deflate(self):
        return self.__class__(blocks=[b for b in self.blocks if b is not None])

    def deflate(self, cutoff=0):
        """Remove zero blocks."""
        blocks = [
            b for b in self.blocks if b is not None and np.linalg.norm(b) > cutoff]
        return self.__class__(blocks=blocks)

    def normalize_along_axis(self, axis):
        if axis < 0:
            axis += self.ndim
        normsq = {}
        for b in self.blocks:
            q = b.q_labels[axis]
            if q not in normsq:
                normsq[q] = 0.0
            normsq[q] += np.linalg.norm(b) ** 2
        for ib, b in enumerate(self.blocks):
            q = b.q_labels[axis]
            if normsq[q] > 1E-12:
                self.blocks[ib] = b / np.sqrt(normsq[q])

    def __getitem__(self, i, idx=-1):
        if isinstance(i, tuple):
            for j in range(self.n_blocks):
                if self.blocks[j].q_labels == i:
                    return self.blocks[j]
        elif isinstance(i, slice) or isinstance(i, int):
            return self.blocks[i]
        else:
            for j in range(self.n_blocks):
                if self.blocks[j].q_labels[idx] == i:
                    return self.blocks[j]

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if ufunc in _sparse_tensor_numpy_func_impls:
            types = tuple(
                x.__class__ for x in inputs if not isinstance(x, numbers.Number))
            return self.__array_function__(ufunc, types, inputs, kwargs)
        out = kwargs.pop("out", None)
        if out is not None and not all(isinstance(x, SparseTensor) for x in out):
            return NotImplemented
        if any(isinstance(x, FermionTensor) for x in inputs):
            return NotImplemented
        if method == "__call__":
            if ufunc.__name__ in ["matmul"]:
                a, b = inputs
                if isinstance(a, numbers.Number):
                    blocks = [a * block for block in b.blocks]
                elif isinstance(b, numbers.Number):
                    blocks = [block * b for block in a.blocks]
                else:
                    blocks = self._tensordot(a, b, axes=([-1], [0])).blocks
            elif ufunc.__name__ in ["multiply", "divide", "true_divide"]:
                a, b = inputs
                if isinstance(a, numbers.Number):
                    blocks = [getattr(ufunc, method)(a, block)
                              for block in b.blocks]
                elif isinstance(b, numbers.Number):
                    blocks = [getattr(ufunc, method)(block, b)
                              for block in a.blocks]
                else:
                    return NotImplemented
            elif len(inputs) == 1:
                blocks = [getattr(ufunc, method)(block)
                          for block in inputs[0].blocks]
            else:
                return NotImplemented
        else:
            return NotImplemented
        if out is not None:
            out[0].blocks = blocks
        return self.__class__(blocks=blocks)

    def __array_function__(self, func, types, args, kwargs):
        if func not in _sparse_tensor_numpy_func_impls:
            return NotImplemented
        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented
        return _sparse_tensor_numpy_func_impls[func](*args, **kwargs)

    @staticmethod
    @implements(np.copy)
    def _copy(x):
        return x.__class__(blocks=[b.copy() for b in x.blocks])

    def copy(self):
        return np.copy(self)

    @staticmethod
    @implements(np.linalg.norm)
    def _norm(x):
        return np.linalg.norm([np.linalg.norm(block) for block in x.blocks])

    def norm(self):
        return np.linalg.norm(self)

    def kron_sum_info(self, *idxs, pattern=None):
        """Kron sum of quantum numbers along selected indices, for fusing purpose."""
        idxs = [i if i >= 0 else self.ndim + i for i in idxs]
        items = []
        for block in self.blocks:
            qs = tuple(block.q_labels[i] for i in idxs)
            shs = tuple(block.shape[i] for i in idxs)
            items.append((qs, shs))
        # using minimal fused dimension
        return BondFusingInfo.kron_sum(items, pattern=pattern)

    def kron_product_info(self, *idxs, pattern=None):
        """Kron product of quantum numbers along selected indices, for fusing purpose."""
        idxs = np.array([i if i >= 0 else self.ndim +
                         i for i in idxs], dtype=np.int32)
        return BondFusingInfo.tensor_product(*np.array(self.infos)[idxs], pattern=pattern)

    @staticmethod
    def _unfuse(a, i, info):
        return a.unfuse(i, info)

    def unfuse(self, i, info):
        """Unfuse one leg to several legs.

        Args:
            i : int
                index of the leg to be unfused. The new unfused indices will be i, i + 1, ...
            info : BondFusingInfo
                Indicating how quantum numbers are collected.
        """
        blocks = []
        for block in self.blocks:
            qs = block.q_labels
            ns = list(block.shape)
            # fused q
            q = qs[i]
            if q not in info:
                continue
            assert ns[i] == info[q]
            # (unfused q, (starting index in fused dim, unfused shape))
            sl = [slice(None) for _ in range(len(ns))]
            for sqs, (k, sh) in info.finfo[q].items():
                nk = np.multiply.reduce(sh)
                sl[i] = slice(k, k + nk)
                new_ns = (ns[:i], *sh, ns[i + 1:])
                new_qs = (qs[:i], *sqs, qs[i + 1:])
                mat = np.asarray(block)[tuple(sl)].reshape(new_ns)
                blocks.append(SubTensor(mat, q_labels=new_qs))
        return self.__class__(blocks=blocks)

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
                len(pattern) == len(idxs)
        """
        blocks_map = {}
        idxs = [i if i >= 0 else self.ndim + i for i in idxs]
        if info is None:
            info = self.kron_sum_info(*idxs, pattern=pattern)
        if pattern is None:
            pattern = info.pattern
        for block in self.blocks:
            qs = block.q_labels
            ns = block.shape
            # unfused q
            sqs = tuple(qs[ix] for ix in idxs)
            # fused q
            q = np.add.reduce([iq if ip == "+" else -iq for iq,
                               ip in zip(sqs, pattern)])
            if q not in info:
                continue
            # fused shape for fused leg
            x = info[q]
            # starting index in fused dim for this block
            k = info.finfo[q][sqs][0]
            # shape in fused dim for this block
            nk = np.multiply.reduce([ns[ix] for ix in idxs])
            new_qs = tuple(iq if iiq != idxs[0] else q for iiq, iq in enumerate(
                qs) if iiq not in idxs[1:])
            new_ns = [ix if iiq != idxs[0] else x for iiq,
                      ix in enumerate(ns) if iiq not in idxs[1:]]
            if new_qs not in blocks_map:
                blocks_map[new_qs] = SubTensor.zeros(
                    tuple(new_ns), q_labels=new_qs, dtype=block.dtype)
            new_ns[idxs[0]] = nk
            sl = tuple(slice(None) if ix != idxs[0] else slice(
                k, k + nk) for ix in range(len(ns)) if ix not in idxs[1:])
            blocks_map[new_qs][sl] = np.asarray(block).reshape(tuple(new_ns))
        return self.__class__(blocks=list(blocks_map.values()))

    @staticmethod
    @implements(np.tensordot)
    def _tensordot(a, b, axes=2):
        """
        Contract two SparseTensor to form a new SparseTensor.

        Args:
            a : SparseTensor
                SparseTensor a, as left operand.
            b : SparseTensor
                SparseTensor b, as right operand.
            axes : int or (2,) array_like
                If an int N, sum over the last N axes of a and the first N axes of b in order.
                Or, a list of axes to be summed over, first sequence applying to a, second to b.

        Returns:
            SparseTensor : SparseTensor
                The contracted SparseTensor.
        """
        if isinstance(axes, int):
            idxa, idxb = list(range(-axes, 0)), list(range(0, axes))
        else:
            idxa, idxb = axes
        idxa = [x if x >= 0 else a.ndim + x for x in idxa]
        idxb = [x if x >= 0 else b.ndim + x for x in idxb]
        out_idx_a = sorted(list(set(range(0, a.ndim)) - set(idxa)))
        out_idx_b = sorted(list(set(range(0, b.ndim)) - set(idxb)))
        assert len(idxa) == len(idxb)

        map_idx_b = {}
        for block in b.blocks:
            ctrq = tuple(block.q_labels[id] for id in idxb)
            outq = tuple(block.q_labels[id] for id in out_idx_b)
            if ctrq not in map_idx_b:
                map_idx_b[ctrq] = []
            map_idx_b[ctrq].append((block, outq))

        blocks_map = {}
        for block_a in a.blocks:
            ctrq = tuple(block_a.q_labels[id] for id in idxa)
            if ctrq in map_idx_b:
                outqa = tuple(block_a.q_labels[id] for id in out_idx_a)
                for block_b, outqb in map_idx_b[ctrq]:
                    outq = outqa + outqb
                    mat = np.tensordot(np.asarray(block_a), np.asarray(
                        block_b), axes=(idxa, idxb)).view(block_a.__class__)
                    if outq not in blocks_map:
                        mat.q_labels = outq
                        blocks_map[outq] = mat
                    else:
                        blocks_map[outq] += mat

        return a.__class__(blocks=list(blocks_map.values()))

    def tensordot(self, b, axes=2):
        return np.tensordot(self, b, axes)

    @staticmethod
    def _hdot(a, b, out=None):
        """Horizontal contraction (contracting connected virtual dimensions)."""
        if isinstance(a, numbers.Number) or isinstance(b, numbers.Number):
            return np.multiply(a, b, out=out)

        assert isinstance(a, SparseTensor) and isinstance(b, SparseTensor)
        r = np.tensordot(a, b, axes=1)

        if out is not None:
            out.blocks = r.blocks

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

        assert isinstance(a, SparseTensor) and isinstance(b, SparseTensor)

        assert a.ndim == b.ndim
        d = a.ndim - 2
        aidx = list(range(1, d + 1))
        bidx = list(range(1, d + 1))
        tr = (0, 2, 1, 3)

        r = np.tensordot(a, b, axes=(aidx, bidx)).transpose(axes=tr)

        if out is not None:
            out.blocks = r.blocks

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
            blocks = [np.add(a, block) for block in b.blocks]
            return b.__class__(blocks=blocks)
        elif isinstance(b, numbers.Number):
            blocks = [np.add(block, b) for block in a.blocks]
            return a.__class__(blocks=blocks)
        else:
            blocks_map = {block.q_labels: block for block in a.blocks}
            for block in b.blocks:
                if block.q_labels in blocks_map:
                    mb = blocks_map[block.q_labels]
                    blocks_map[block.q_labels] = np.add(mb, block)
                else:
                    blocks_map[block.q_labels] = block
            blocks = list(blocks_map.values())
            return a.__class__(blocks=blocks)

    def add(self, b):
        return self._add(self, b)

    @staticmethod
    @implements(np.subtract)
    def _subtract(a, b):
        if isinstance(a, numbers.Number):
            blocks = [np.subtract(a, block) for block in b.blocks]
            return b.__class__(blocks=blocks)
        elif isinstance(b, numbers.Number):
            blocks = [np.subtract(block, b) for block in a.blocks]
            return a.__class__(blocks=blocks)
        else:
            blocks_map = {block.q_labels: block for block in a.blocks}
            for block in b.blocks:
                if block.q_labels in blocks_map:
                    mb = blocks_map[block.q_labels]
                    blocks_map[block.q_labels] = np.subtract(mb, block)
                else:
                    blocks_map[block.q_labels] = -block
            blocks = list(blocks_map.values())
            return a.__class__(blocks=blocks)

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

        lb, rb = infos[0], infos[-1]

        sub_mp = {}
        # find required new blocks and their shapes
        for block in a.blocks + b.blocks:
            qs = block.q_labels
            sh = block.shape
            if qs not in sub_mp:
                mshape = list(sh)
                mshape[0] = lb[qs[0]]
                mshape[-1] = rb[qs[-1]]
                sub_mp[qs] = SubTensor.zeros(shape=tuple(
                    mshape), q_labels=qs, dtype=a.dtype)
        # copy block self.blocks to smaller index in new block
        for block in a.blocks:
            qs = block.q_labels
            sh = block.shape
            sub_mp[qs][: sh[0], ..., : sh[-1]] += block
        # copy block other.blocks to greater index in new block
        for block in b.blocks:
            qs = block.q_labels
            sh = block.shape
            sub_mp[qs][-sh[0]:, ..., -sh[-1]:] += block
        return a.__class__(blocks=list(sub_mp.values()))

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
            q, r : tuple(SparseTensor)
        """
        collected_rows = {}
        for block in self.blocks:
            q_label_r = block.q_labels[-1]
            if q_label_r not in collected_rows:
                collected_rows[q_label_r] = []
            collected_rows[q_label_r].append(block)
        q_blocks, r_blocks = [], []
        for q_label_r, blocks in collected_rows.items():
            l_shapes = [np.prod(b.shape[:-1]) for b in blocks]
            mat = np.concatenate([b.view(np.ndarray).reshape((sh, -1))
                                  for sh, b in zip(l_shapes, blocks)], axis=0)
            q, r = np.linalg.qr(mat, mode=mode)
            r_blocks.append(
                SubTensor(reduced=r, q_labels=(q_label_r, q_label_r)))
            qs = np.split(q, list(accumulate(l_shapes[:-1])), axis=0)
            assert len(qs) == len(blocks)
            for q, b in zip(qs, blocks):
                mat = q.reshape(b.shape[:-1] + (r.shape[0], ))
                q_blocks.append(SubTensor(reduced=mat, q_labels=b.q_labels))
        return self.__class__(blocks=q_blocks), self.__class__(blocks=r_blocks)

    def right_canonicalize(self, mode='reduced'):
        """
        Right canonicalization (using QR factorization).

        Returns:
            l, q : tuple(SparseTensor)
        """
        collected_cols = {}
        for block in self.blocks:
            q_label_l = block.q_labels[0]
            if q_label_l not in collected_cols:
                collected_cols[q_label_l] = []
            collected_cols[q_label_l].append(block)
        l_blocks, q_blocks = [], []
        for q_label_l, blocks in collected_cols.items():
            r_shapes = [np.prod(b.shape[1:]) for b in blocks]
            mat = np.concatenate([b.view(np.ndarray).reshape((-1, sh)).T
                                  for sh, b in zip(r_shapes, blocks)], axis=0)
            q, r = np.linalg.qr(mat, mode=mode)
            l_blocks.append(
                SubTensor(reduced=r.T, q_labels=(q_label_l, q_label_l)))
            qs = np.split(q, list(accumulate(r_shapes[:-1])), axis=0)
            assert len(qs) == len(blocks)
            for q, b in zip(qs, blocks):
                mat = q.T.reshape((r.shape[0], ) + b.shape[1:])
                q_blocks.append(SubTensor(reduced=mat, q_labels=b.q_labels))
        return self.__class__(blocks=l_blocks), self.__class__(blocks=q_blocks)

    def left_svd(self, full_matrices=False):
        """
        Left svd needs to collect all left indices for each specific right index.

        Returns:
            l, s, r : tuple(SparseTensor)
        """
        collected_rows = {}
        for block in self.blocks:
            q_label_r = block.q_labels[-1]
            if q_label_r not in collected_rows:
                collected_rows[q_label_r] = []
            collected_rows[q_label_r].append(block)
        l_blocks, s_blocks, r_blocks = [], [], []
        for q_label_r, blocks in collected_rows.items():
            l_shapes = [np.prod(b.shape[:-1]) for b in blocks]
            mat = np.concatenate([np.asarray(b).reshape((sh, -1))
                                  for sh, b in zip(l_shapes, blocks)], axis=0)
            u, s, vh = np.linalg.svd(mat, full_matrices=full_matrices)
            qs = np.split(u, list(accumulate(l_shapes[:-1])), axis=0)
            assert len(qs) == len(blocks)
            for q, b in zip(qs, blocks):
                mat = q.reshape(b.shape[:-1] + (s.shape[0], ))
                l_blocks.append(SubTensor(reduced=mat, q_labels=b.q_labels))
            s_blocks.append(SubTensor(reduced=s, q_labels=(q_label_r, )))
            r_blocks.append(
                SubTensor(reduced=vh, q_labels=(q_label_r, q_label_r)))
        return self.__class__(blocks=l_blocks), self.__class__(blocks=s_blocks), self.__class__(blocks=r_blocks)

    def right_svd(self, full_matrices=False):
        """
        Right svd needs to collect all right indices for each specific left index.

        Returns:
            l, s, r : tuple(SparseTensor)
        """
        collected_cols = {}
        for block in self.blocks:
            q_label_l = block.q_labels[0]
            if q_label_l not in collected_cols:
                collected_cols[q_label_l] = []
            collected_cols[q_label_l].append(block)
        l_blocks, s_blocks, r_blocks = [], [], []
        for q_label_l, blocks in collected_cols.items():
            r_shapes = [np.prod(b.shape[1:]) for b in blocks]
            mat = np.concatenate([np.asarray(b).reshape((-1, sh))
                                  for sh, b in zip(r_shapes, blocks)], axis=1)
            u, s, vh = np.linalg.svd(mat, full_matrices=full_matrices)
            qs = np.split(vh, list(accumulate(r_shapes[:-1])), axis=1)
            assert len(qs) == len(blocks)
            for q, b in zip(qs, blocks):
                mat = q.reshape((vh.shape[0], ) + b.shape[1:])
                r_blocks.append(SubTensor(reduced=mat, q_labels=b.q_labels))
            s_blocks.append(SubTensor(reduced=s, q_labels=(q_label_l, )))
            l_blocks.append(
                SubTensor(reduced=u, q_labels=(q_label_l, q_label_l)))
        return self.__class__(blocks=l_blocks), self.__class__(blocks=s_blocks), self.__class__(blocks=r_blocks)

    def tensor_svd(self, idx=2, linfo=None, rinfo=None, pattern=None, full_matrices=False):
        """
        Separate tensor in the middle, collecting legs as [0, idx) and [idx, ndim), then perform SVD.

        Returns:
            l, s, r : tuple(SparseTensor)
        """
        assert idx >= 1 and idx <= self.ndim - 1
        if pattern is None:
            pattern = '+' * self.ndim
        if linfo is None:
            items = []
            for block in self.blocks:
                items.append((block.q_labels[:idx], block.shape[:idx]))
            linfo = BondFusingInfo.kron_sum(items, pattern=pattern[:idx])
        if rinfo is None:
            items = []
            for block in self.blocks:
                items.append((block.q_labels[idx:], block.shape[idx:]))
                rpat = ''.join(['-' if x == '+' else '+' for x in pattern[idx:]])
            rinfo = BondFusingInfo.kron_sum(items, pattern=rpat)
        info = linfo & rinfo
        mats = {}
        for q in info:
            mats[q] = np.zeros((linfo[q], rinfo[q]))
        rpat = ''.join(['-' if x == '+' else '+' for x in rinfo.pattern])
        pattern = linfo.pattern + rpat
        items = {}
        for block in self.blocks:
            qls, qrs = block.q_labels[:idx], block.q_labels[idx:]
            ql = np.add.reduce(
                [iq if ip == '+' else -iq for iq, ip in zip(qls, pattern[:idx])])
            qr = np.add.reduce(
                [iq if ip == '-' else -iq for iq, ip in zip(qrs, pattern[idx:])])
            assert ql == qr
            q = ql
            if q not in mats:
                continue
            if q not in items:
                items[q] = [], []
            items[q][0].append(qls)
            items[q][1].append(qrs)
            lk, lkn = linfo.finfo[q][qls][0], np.multiply.reduce(
                linfo.finfo[q][qls][1])
            rk, rkn = rinfo.finfo[q][qrs][0], np.multiply.reduce(
                rinfo.finfo[q][qrs][1])
            mats[q][lk:lk + lkn, rk:rk +
                    rkn] = np.asarray(block).reshape((lkn, rkn))
        l_blocks, s_blocks, r_blocks = [], [], []
        for q, mat in mats.items():
            u, s, vh = np.linalg.svd(mat, full_matrices=full_matrices)
            s_blocks.append(SubTensor(reduced=s, q_labels=(q, )))
            items[q][0].sort()
            items[q][1].sort()
            psqs = None
            for sqs in items[q][0]:
                if sqs == psqs:
                    continue
                k, sh = linfo.finfo[q][sqs]
                nk = np.multiply.reduce(sh)
                mat = u[k:k + nk, :].reshape(sh + (-1, ))
                l_blocks.append(SubTensor(reduced=mat, q_labels=sqs + (q, )))
                psqs = sqs
            psqs = None
            for sqs in items[q][1]:
                if sqs == psqs:
                    continue
                k, sh = rinfo.finfo[q][sqs]
                nk = np.multiply.reduce(sh)
                mat = vh[:, k:k + nk].reshape((-1, ) + sh)
                r_blocks.append(SubTensor(reduced=mat, q_labels=(q, ) + sqs))
                psqs = sqs
        return self.__class__(blocks=l_blocks), self.__class__(blocks=s_blocks), self.__class__(blocks=r_blocks)

    @staticmethod
    def truncate_svd(l, s, r, max_bond_dim=-1, cutoff=0.0, max_dw=0.0, norm_cutoff=0.0, eigen_values=False):
        """
        Truncate tensors obtained from SVD.

        Args:
            l, s, r : tuple(SparseTensor)
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
            l, s, r : tuple(SparseTensor)
                SVD decomposition.
            error : float
                Truncation error (same unit as singular value squared).
        """
        ss = [(i, j, v) for i, ps in enumerate(s.blocks)
              for j, v in enumerate(ps)]
        ss.sort(key=lambda x: -x[2])
        ss_trunc = ss
        if max_dw != 0:
            p, dw = 0, 0.0
            for x in ss_trunc[::-1]:
                dw += x[2] if eigen_values else x[2] * x[2]
                if dw <= max_dw:
                    p += 1
                else:
                    break
            ss_trunc = ss_trunc[:-p]
        if cutoff != 0:
            if not eigen_values:
                cutoff = np.sqrt(cutoff)
            ss_trunc = [x for x in ss_trunc if x[2] >= cutoff]
        if max_bond_dim != -1:
            ss_trunc = ss_trunc[:max_bond_dim]
        ss_trunc.sort(key=lambda x: (x[0], x[1]))
        l_blocks, s_blocks, r_blocks = [], [], []
        error = 0.0
        selected = [False] * len(s.blocks)
        ikl, ikr = 0, 0
        for ik, g in groupby(ss_trunc, key=lambda x: x[0]):
            gl = np.array([ig[1] for ig in g], dtype=int)
            gl_inv = np.array(
                sorted(list(set(range(0, len(s.blocks[ik]))) - set(gl))), dtype=int)
            while ikl < l.n_blocks and l.blocks[ikl].q_labels[-1] != s.blocks[ik].q_labels[0]:
                ikl += 1
            while ikl < l.n_blocks and l.blocks[ikl].q_labels[-1] == s.blocks[ik].q_labels[0]:
                l_blocks.append(l.blocks[ikl][..., gl])
                ikl += 1
            s_blocks.append(s.blocks[ik][gl])
            while ikr < r.n_blocks and r.blocks[ikr].q_labels[0] != s.blocks[ik].q_labels[-1]:
                ikr += 1
            while ikr < r.n_blocks and r.blocks[ikr].q_labels[0] == s.blocks[ik].q_labels[-1]:
                r_blocks.append(r.blocks[ikr][gl, ...])
                ikr += 1
            if eigen_values:
                error += s.blocks[ik][gl_inv].sum()
            else:
                error += (s.blocks[ik][gl_inv] ** 2).sum()
            selected[ik] = True
        for ik in range(len(s.blocks)):
            if not selected[ik]:
                if eigen_values:
                    error += s.blocks[ik].sum()
                else:
                    error += (s.blocks[ik] ** 2).sum()
        error = np.asarray(error).item()
        return l.__class__(blocks=l_blocks).deflate(cutoff=norm_cutoff), \
            s.__class__(blocks=s_blocks), r.__class__(
                blocks=r_blocks).deflate(cutoff=norm_cutoff), error

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

    @staticmethod
    @implements(np.diag)
    def _diag(v):
        """
        Extract a diagonal or construct a diagonal array.

        Args:
            v : SparseTensor
                If v is a 2-D array, return a copy of its 0-th diagonal.
                If v is a 1-D array, return a 2-D array with v on the 0-th diagonal.
        """
        blocks = []
        if v.ndim == 2:
            for block in v.blocks:
                if block.q_labels[0] == block.q_labels[1]:
                    blocks.append(np.diag(block))
        elif v.ndim == 1:
            for block in v.blocks:
                blocks.append(np.diag(block))
        elif len(v.blocks) != 0:
            raise RuntimeError("ndim for np.diag must be 1 or 2.")
        return v.__class__(blocks=blocks)

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
            p : SparseTensor
                a with its axes permuted. A view is returned whenever possible.
        """
        blocks = [np.transpose(block, axes=axes) for block in a.blocks]
        return a.__class__(blocks=blocks)

    def transpose(self, axes=None):
        return np.transpose(self, axes=axes)

    @staticmethod
    def _to_sliceable(a, infos=None):
        return a.to_sliceable(infos=infos)

    def to_sliceable(self, infos=None):
        if infos is None:
            infos = self.infos
        idx_infos = []
        for info in infos:
            idx_info = Counter()
            for ik, k in enumerate(sorted(info)):
                idx_info[k] = ik
            idx_infos.append(idx_info)
        sh = tuple(len(info) for info in infos)
        arr = np.zeros(sh, dtype=object)
        arr[:] = None
        for block in self.blocks:
            idx = tuple([info[q]
                         for q, info in zip(block.q_labels, idx_infos)])
            arr[idx] = block
        return SliceableTensor(reduced=arr, infos=infos)

    @staticmethod
    def _to_dense(a, infos=None):
        return a.to_dense(infos=infos)

    def to_dense(self, infos=None):
        return self.to_sliceable(infos=infos).to_dense()

    def to_sparse(self):
        return self


_fermion_tensor_numpy_func_impls = {}
_numpy_func_impls = _fermion_tensor_numpy_func_impls


class FermionTensor(NDArrayOperatorsMixin):
    """
    block-sparse tensor with fermion factors.

    Attributes:
        odd : SparseTensor
            Including blocks with odd fermion parity.
        even : SparseTensor
            Including blocks with even fermion parity.
    """

    def __init__(self, odd=None, even=None):
        self.odd = odd if odd is not None else []
        self.even = even if even is not None else []
        if isinstance(self.odd, list):
            self.odd = SparseTensor(blocks=self.odd)
        if isinstance(self.even, list):
            self.even = SparseTensor(blocks=self.even)

    @property
    def ndim(self):
        """Number of dimensions."""
        return self.odd.ndim | self.even.ndim

    @property
    def nbytes(self):
        """Number bytes in memory."""
        return self.odd.nbytes + self.even.nbytes

    @property
    def n_blocks(self):
        """Number of (non-zero) blocks."""
        return self.odd.n_blocks + self.even.n_blocks

    @property
    def dtype(self):
        """Element datatype."""
        return self.odd.dtype if self.odd.n_blocks != 0 else self.even.dtype

    def item(self):
        """Return scalar element."""
        if self.n_blocks == 0:
            return 0
        assert self.odd.n_blocks == 0
        return self.even.item()

    def __str__(self):
        ro = "\n".join(
            [" ODD-" + x for x in str(self.odd).split("\n") if x != ""])
        rd = "\n".join(
            ["EVEN-" + x for x in str(self.even).split("\n") if x != ""])
        return ro + "\n" + rd

    def __repr__(self):
        ro = "\n".join(
            [" ODD-" + x for x in repr(self.odd).split("\n") if x != ""])
        rd = "\n".join(
            ["EVEN-" + x for x in repr(self.even).split("\n") if x != ""])
        return ro + "\n" + rd

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if ufunc in _fermion_tensor_numpy_func_impls:
            types = tuple(
                x.__class__ for x in inputs if not isinstance(x, numbers.Number))
            return self.__array_function__(ufunc, types, inputs, kwargs)
        out = kwargs.pop("out", None)
        if out is not None and not all(isinstance(x, FermionTensor) for x in out):
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
            out[0].odd, out[0].even = blocks
        return FermionTensor(odd=blocks[0], even=blocks[1])

    def __array_function__(self, func, types, args, kwargs):
        if func not in _fermion_tensor_numpy_func_impls:
            return NotImplemented
        if not all(issubclass(t, self.__class__) or issubclass(t, self.odd.__class__) for t in types):
            return NotImplemented
        return _fermion_tensor_numpy_func_impls[func](*args, **kwargs)

    @staticmethod
    def zeros(bond_infos, pattern=None, dq=None, dtype=float):
        """Create operator tensor with zero elements."""
        spt = SparseTensor.zeros(
            bond_infos, pattern=pattern, dq=dq, dtype=dtype)
        if dq is not None and dq.is_fermion:
            return FermionTensor(odd=spt)
        else:
            return FermionTensor(even=spt)

    @staticmethod
    def ones(bond_infos, pattern=None, dq=None, dtype=float):
        """Create operator tensor with ones."""
        spt = SparseTensor.ones(
            bond_infos, pattern=pattern, dq=dq, dtype=dtype)
        if dq is not None and dq.is_fermion:
            return FermionTensor(odd=spt)
        else:
            return FermionTensor(even=spt)

    @staticmethod
    def random(bond_infos, pattern=None, dq=None, dtype=float):
        """Create operator tensor with random elements."""
        spt = SparseTensor.random(
            bond_infos, pattern=pattern, dq=dq, dtype=dtype)
        if dq is not None and dq.is_fermion:
            return FermionTensor(odd=spt)
        else:
            return FermionTensor(even=spt)

    @property
    def infos(self):
        """Return the quantum number layout of the FermionTensor, similar to numpy.ndarray.shape."""
        if self.odd.ndim == 0:
            return self.even.infos
        elif self.even.ndim == 0:
            return self.odd.infos
        else:
            return tuple(o | e for o, e in zip(self.odd.infos, self.even.infos))

    def conj(self):
        """
        Complex conjugate.
        Note that np.conj() is a ufunc (no need to be defined).
        But np.real and np.imag are array_functions
        """
        return self.__class__(odd=self.odd.conj(), even=self.even.conj())

    @staticmethod
    @implements(np.real)
    def _real(x):
        return x.__class__(odd=x.odd.real, even=x.even.real)

    @property
    def real(self):
        return np.real(self)

    @staticmethod
    @implements(np.imag)
    def _imag(x):
        return x.__class__(odd=x.odd.imag, even=x.even.imag)

    @property
    def imag(self):
        return np.imag(self)

    def deflate(self, cutoff=0):
        return FermionTensor(odd=self.odd.deflate(cutoff), even=self.even.deflate(cutoff))

    @staticmethod
    @implements(np.copy)
    def _copy(x):
        return FermionTensor(odd=x.odd.copy(), even=x.even.copy())

    def copy(self):
        return np.copy(self)

    def kron_sum_info(self, *idxs, pattern=None):
        idxs = [i if i >= 0 else self.ndim + i for i in idxs]
        items = []
        for block in self.odd.blocks + self.even.blocks:
            qs = tuple(block.q_labels[i] for i in idxs)
            shs = tuple(block.shape[i] for i in idxs)
            items.append((qs, shs))
        # using minimal fused dimension
        return BondFusingInfo.kron_sum(items, pattern=pattern)

    def kron_product_info(self, *idxs, pattern=None):
        idxs = np.array([i if i >= 0 else self.ndim +
                         i for i in idxs], dtype=np.int32)
        return BondFusingInfo.tensor_product(*np.array(self.infos)[idxs], pattern=pattern)

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
        return FermionTensor(odd=odd, even=even)

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
            info = self.kron_sum_info(*idxs, pattern=pattern)
        odd = self.odd.fuse(*idxs, info=info, pattern=pattern)
        even = self.even.fuse(*idxs, info=info, pattern=pattern)
        return FermionTensor(odd=odd, even=even)

    @staticmethod
    @implements(np.tensordot)
    def _tensordot(a, b, axes=2):
        """
        Contract two FermionTensor/SparseTensor to form a new FermionTensor/SparseTensor.
        Fermion factor will be handled in a case-by-case way.

        Args:
            a : FermionTensor/SparseTensor
                FermionTensor/SparseTensor a, as left operand.
            b : FermionTensor/SparseTensor
                FermionTensor/SparseTensor b, as right operand.
            axes : int or (2,) array_like
                If an int N, sum over the last N axes of a and the first N axes of b in order.
                Or, a list of axes to be summed over, first sequence applying to a, second to b.

        Returns:
            c : FermionTensor/SparseTensor
                The contracted FermionTensor/SparseTensor.
        """
        if isinstance(axes, int):
            idxa, idxb = list(range(-axes, 0)), list(range(0, axes))
        else:
            idxa, idxb = axes
        idxa = [x if x >= 0 else a.ndim + x for x in idxa]
        idxb = [x if x >= 0 else b.ndim + x for x in idxb]
        assert len(idxa) == len(idxb)

        blocks = []
        r = None
        # op x op
        if isinstance(a, FermionTensor) and isinstance(b, FermionTensor):
            odd_a = np.tensordot(a.odd, b.even, (idxa, idxb))
            odd_b = np.tensordot(a.even, b.odd, (idxa, idxb))
            even_a = np.tensordot(a.odd, b.odd, (idxa, idxb))
            even_b = np.tensordot(a.even, b.even, (idxa, idxb))
            r = FermionTensor(odd=odd_a + odd_b, even=even_a + even_b)
            # symbolic horizontal
            if idxa == [] and idxb == []:
                assert a.ndim % 2 == 0
                d = a.ndim // 2
                idx = range(d, d + d) if d != 1 else d
                blocks = odd_b.blocks + even_a.blocks
            # horizontal
            elif idxa == [a.ndim - 1] and idxb == [0]:
                assert a.ndim % 2 == 0
                d = (a.ndim - 2) // 2
                idx = range(d + 1, d + d + 1) if d != 1 else d + 1
                blocks = odd_b.blocks + even_a.blocks
            # vertical
            else:
                idx = a.ndim - len(idxa)
                # 1-site op x n-site op
                idx = range(idx, idx + min(idxb))
                blocks = odd_a.blocks + even_a.blocks
        # op x state
        elif isinstance(a, FermionTensor):
            idx = a.ndim - len(idxa)
            even = np.tensordot(a.even, b, (idxa, idxb))
            odd = np.tensordot(a.odd, b, (idxa, idxb))
            # op rotation / op x gauge (right multiply)
            if b.ndim - len(idxb) == 1:
                r = FermionTensor(odd=odd, even=even)
            else:
                # 1-site op x n-site state
                idx = range(idx, idx + min(idxb))
                blocks = odd.blocks
                r = odd + even
        # state x op
        elif isinstance(b, FermionTensor):
            idx = 0
            even = np.tensordot(a, b.even, (idxa, idxb))
            odd = np.tensordot(a, b.odd, (idxa, idxb))
            # op rotation / gauge x op (left multiply)
            if a.ndim - len(idxa) == 1:
                r = FermionTensor(odd=odd, even=even)
            else:
                # n-site state x 1-site op
                idx = range(idx, idx + min(idxa))
                blocks = odd.blocks
                r = odd + even
        else:
            raise TypeError('Unsupported tensordot for %r and %r' %
                            (a.__class__, b.__class__))

        # apply fermion factor
        if isinstance(idx, int):
            for block in blocks:
                if block.q_labels[idx].is_fermion:
                    np.negative(block, out=block)
        else:
            for block in blocks:
                if np.logical_xor.reduce([block.q_labels[ix].is_fermion for ix in idx]):
                    np.negative(block, out=block)

        return r

    def tensordot(self, b, axes=2):
        return np.tensordot(self, b, axes)

    @staticmethod
    def _shdot(a, b, out=None):
        """Horizontally contract operator tensors (matrices) in symbolic matrix."""
        if isinstance(a, numbers.Number) or isinstance(b, numbers.Number):
            return np.multiply(a, b, out=out)

        assert isinstance(a, FermionTensor) and isinstance(b, FermionTensor)
        r = np.tensordot(a, b, axes=0)

        if isinstance(a, FermionTensor) and isinstance(b, FermionTensor) and a.ndim % 2 == 0 and b.ndim % 2 == 0:
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

        assert isinstance(a, FermionTensor) or isinstance(b, FermionTensor)
        r = np.tensordot(a, b, axes=1)

        if isinstance(a, FermionTensor) and isinstance(b, FermionTensor) and a.ndim % 2 == 0 and b.ndim % 2 == 0:
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
            if isinstance(b, FermionTensor):
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
            elif isinstance(b, SparseTensor):
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

            assert isinstance(a, FermionTensor) or isinstance(b, FermionTensor)

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
                assert isinstance(a, FermionTensor)
                dau, db = a.ndim - b.ndim, b.ndim - 2
                aidx = list(range(dau + 1, dau + db + 1))
                bidx = list(range(1, db + 1))
                tr = tuple([0, dau + 2] + list(range(1, dau + 1)) +
                           [dau + 1, dau + 3])
            # MPS x MPO
            elif a.ndim < b.ndim:
                assert isinstance(b, FermionTensor)
                da, dbd = a.ndim - 2, b.ndim - a.ndim
                aidx = list(range(1, da + 1))
                bidx = list(range(1, da + 1))
                tr = tuple([0, 2] + list(range(3, dbd + 3)) + [1, dbd + 3])
            else:
                raise RuntimeError(
                    "Cannot matmul tensors with ndim: %d x %d" % (a.ndim, b.ndim))

            r = np.tensordot(a, b, axes=(aidx, bidx)).transpose(axes=tr)

        if out is not None:
            if isinstance(r, SparseTensor):
                out.blocks = r.blocks
            else:
                out.odd = r.odd
                out.even = r.even

        return r

    def pdot(self, b, out=None):
        return FermionTensor._pdot(self, b, out=out)

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
        return FermionTensor(odd=odd, even=even)

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
            q, r : (FermionTensor, SparseTensor (gauge))
        """
        collected_rows = {}
        for ip, blocks in [(1, self.odd.blocks), (0, self.even.blocks)]:
            for block in blocks:
                q_label_r = block.q_labels[-1]
                if q_label_r not in collected_rows:
                    collected_rows[q_label_r] = []
                collected_rows[q_label_r].append((ip, block))
        q_odd, q_even, r_blocks = [], [], []
        for q_label_r, blocks in collected_rows.items():
            l_shapes = [np.prod(b.shape[:-1]) for _, b in blocks]
            mat = np.concatenate([b.view(np.ndarray).reshape((sh, -1))
                                  for sh, (_, b) in zip(l_shapes, blocks)], axis=0)
            q, r = np.linalg.qr(mat, mode=mode)
            r_blocks.append(
                SubTensor(reduced=r, q_labels=(q_label_r, q_label_r)))
            qs = np.split(q, list(accumulate(l_shapes[:-1])), axis=0)
            assert(len(qs) == len(blocks))
            for q, (ip, b) in zip(qs, blocks):
                mat = q.reshape(b.shape[:-1] + (r.shape[0], ))
                if ip == 1:
                    q_odd.append(SubTensor(reduced=mat, q_labels=b.q_labels))
                else:
                    q_even.append(SubTensor(reduced=mat, q_labels=b.q_labels))
        return FermionTensor(odd=q_odd, even=q_even), SparseTensor(blocks=r_blocks)

    def right_canonicalize(self, mode='reduced'):
        """
        Right canonicalization (using QR factorization).

        Returns:
            l, q : (SparseTensor (gauge), FermionTensor)
        """
        collected_cols = {}
        for ip, blocks in [(1, self.odd.blocks), (0, self.even.blocks)]:
            for block in blocks:
                q_label_l = block.q_labels[0]
                if q_label_l not in collected_cols:
                    collected_cols[q_label_l] = []
                collected_cols[q_label_l].append((ip, block))
        l_blocks, q_odd, q_even = [], [], []
        for q_label_l, blocks in collected_cols.items():
            r_shapes = [np.prod(b.shape[1:]) for _, b in blocks]
            mat = np.concatenate([b.view(np.ndarray).reshape((-1, sh)).T
                                  for sh, (_, b) in zip(r_shapes, blocks)], axis=0)
            q, r = np.linalg.qr(mat, mode=mode)
            l_blocks.append(
                SubTensor(reduced=r.T, q_labels=(q_label_l, q_label_l)))
            qs = np.split(q, list(accumulate(r_shapes[:-1])), axis=0)
            assert(len(qs) == len(blocks))
            for q, (ip, b) in zip(qs, blocks):
                mat = q.T.reshape((r.shape[0], ) + b.shape[1:])
                if ip == 1:
                    q_odd.append(SubTensor(reduced=mat, q_labels=b.q_labels))
                else:
                    q_even.append(SubTensor(reduced=mat, q_labels=b.q_labels))
        return SparseTensor(blocks=l_blocks), FermionTensor(odd=q_odd, even=q_even)

    def left_svd(self, full_matrices=True):
        """
        Left svd needs to collect all left indices for each specific right index.

        Returns:
            l, s, r : (FermionTensor, SparseTensor (vector), SparseTensor (gauge))
        """
        collected_rows = {}
        for ip, blocks in [(1, self.odd.blocks), (0, self.even.blocks)]:
            for block in blocks:
                q_label_r = block.q_labels[-1]
                if q_label_r not in collected_rows:
                    collected_rows[q_label_r] = []
                collected_rows[q_label_r].append((ip, block))
        l_odd, l_even, s_blocks, r_blocks = [], [], [], []
        for q_label_r, blocks in collected_rows.items():
            l_shapes = [np.prod(b.shape[:-1]) for _, b in blocks]
            mat = np.concatenate([np.asarray(b).reshape((sh, -1))
                                  for sh, (_, b) in zip(l_shapes, blocks)], axis=0)
            u, s, vh = np.linalg.svd(mat, full_matrices=full_matrices)
            qs = np.split(u, list(accumulate(l_shapes[:-1])), axis=0)
            assert(len(qs) == len(blocks))
            for q, (ip, b) in zip(qs, blocks):
                mat = q.reshape(b.shape[:-1] + (s.shape[0], ))
                if ip == 1:
                    l_odd.append(SubTensor(reduced=mat, q_labels=b.q_labels))
                else:
                    l_even.append(SubTensor(reduced=mat, q_labels=b.q_labels))
            s_blocks.append(SubTensor(reduced=s, q_labels=(q_label_r, )))
            r_blocks.append(
                SubTensor(reduced=vh, q_labels=(q_label_r, q_label_r)))
        return FermionTensor(odd=l_odd, even=l_even), \
            SparseTensor(blocks=s_blocks), SparseTensor(blocks=r_blocks)

    def right_svd(self, full_matrices=True):
        """
        Right svd needs to collect all right indices for each specific left index.

        Returns:
            l, s, r : (SparseTensor (gauge), SparseTensor (vector), FermionTensor)
        """
        collected_cols = {}
        for ip, blocks in [(1, self.odd.blocks), (0, self.even.blocks)]:
            for block in blocks:
                q_label_l = block.q_labels[0]
                if q_label_l not in collected_cols:
                    collected_cols[q_label_l] = []
                collected_cols[q_label_l].append((ip, block))
        l_blocks, s_blocks, r_odd, r_even = [], [], [], []
        for q_label_l, blocks in collected_cols.items():
            r_shapes = [np.prod(b.shape[1:]) for _, b in blocks]
            mat = np.concatenate([np.asarray(b).reshape((-1, sh))
                                  for sh, (_, b) in zip(r_shapes, blocks)], axis=1)
            u, s, vh = np.linalg.svd(mat, full_matrices=full_matrices)
            qs = np.split(vh, list(accumulate(r_shapes[:-1])), axis=1)
            assert(len(qs) == len(blocks))
            for q, (ip, b) in zip(qs, blocks):
                mat = q.reshape((vh.shape[0], ) + b.shape[1:])
                if ip == 1:
                    r_odd.append(SubTensor(reduced=mat, q_labels=b.q_labels))
                else:
                    r_even.append(SubTensor(reduced=mat, q_labels=b.q_labels))
            s_blocks.append(SubTensor(reduced=s, q_labels=(q_label_l, )))
            l_blocks.append(
                SubTensor(reduced=u, q_labels=(q_label_l, q_label_l)))
        return SparseTensor(blocks=l_blocks), SparseTensor(blocks=s_blocks), \
            FermionTensor(odd=r_odd, even=r_even)

    @staticmethod
    def truncate_svd(l, s, r, max_bond_dim=-1, cutoff=0.0, max_dw=0.0, norm_cutoff=0.0):
        """
        Truncate tensors obtained from SVD.

        Args:
            l, s, r : tuple(SparseTensor/FermionTensor)
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
            l, s, r : tuple(SparseTensor/FermionTensor)
                SVD decomposition.
            error : float
                Truncation error (same unit as singular value squared).
        """
        ss = [(i, j, v) for i, ps in enumerate(s.blocks)
              for j, v in enumerate(ps)]
        ss.sort(key=lambda x: -x[2])
        ss_trunc = ss.copy()
        if max_dw != 0:
            p, dw = 0, 0.0
            for x in ss_trunc:
                dw += x[2] * x[2]
                if dw <= max_dw:
                    p += 1
                else:
                    break
            ss_trunc = ss_trunc[:-p]
        if cutoff != 0:
            ss_trunc = [x for x in ss_trunc if x[2] >= cutoff]
        if max_bond_dim != -1:
            ss_trunc = ss_trunc[:max_bond_dim]
        ss_trunc.sort(key=lambda x: (x[0], x[1]))
        if isinstance(l, FermionTensor):
            lmps = [Counter(), Counter()]
            lbs = [l.odd.blocks, l.even.blocks]
            l_blocks = [[], []]
            for ip, bs in enumerate(lbs):
                for ib, b in enumerate(bs):
                    if b.q_labels[-1] not in lmps[ip]:
                        lmps[ip][b.q_labels[-1]] = ib
        else:
            lmps = [Counter()]
            lbs = [l.blocks]
            l_blocks = [[]]
            for ib, b in enumerate(lbs[0]):
                if b.q_labels[-1] not in lmps[0]:
                    lmps[0][b.q_labels[-1]] = ib
        if isinstance(r, FermionTensor):
            rmps = [Counter(), Counter()]
            rbs = [r.odd.blocks, r.even.blocks]
            r_blocks = [[], []]
            for ip, bs in enumerate(rbs):
                for ib, b in enumerate(bs):
                    if b.q_labels[0] not in rmps[ip]:
                        rmps[ip][b.q_labels[0]] = ib
        else:
            rmps = [Counter()]
            rbs = [r.blocks]
            r_blocks = [[]]
            for ib, b in enumerate(rbs[0]):
                if b.q_labels[0] not in rmps[0]:
                    rmps[0][b.q_labels[0]] = ib
        s_blocks = []
        error = 0.0
        selected = [False] * len(s.blocks)
        for ik, g in groupby(ss_trunc, key=lambda x: x[0]):
            gl = np.array([ig[1] for ig in g], dtype=int)
            gl_inv = np.array(
                sorted(list(set(range(0, len(s.blocks[ik]))) - set(gl))), dtype=int)
            for il, lb in enumerate(lbs):
                ikl = lmps[il][s.blocks[ik].q_labels[0]]
                while ikl < len(lb) and lb[ikl].q_labels[-1] == s.blocks[ik].q_labels[0]:
                    l_blocks[il].append(lb[ikl][..., gl])
                    ikl += 1
            s_blocks.append(s.blocks[ik][gl])
            for ir, rb in enumerate(rbs):
                ikr = rmps[ir][s.blocks[ik].q_labels[-1]]
                while ikr < len(rb) and rb[ikr].q_labels[0] == s.blocks[ik].q_labels[-1]:
                    r_blocks[ir].append(rb[ikr][gl, ...])
                    ikr += 1
            error += (s.blocks[ik][gl_inv] ** 2).sum()
            selected[ik] = True
        for ik in range(len(s.blocks)):
            if not selected[ik]:
                error += (s.blocks[ik] ** 2).sum()
        if isinstance(l, FermionTensor):
            new_l = FermionTensor(odd=l_blocks[0], even=l_blocks[1]).deflate(
                cutoff=norm_cutoff)
        else:
            new_l = SparseTensor(blocks=l_blocks[0]).deflate(
                cutoff=norm_cutoff)
        if isinstance(r, FermionTensor):
            new_r = FermionTensor(odd=r_blocks[0], even=r_blocks[1]).deflate(
                cutoff=norm_cutoff)
        else:
            new_r = SparseTensor(blocks=r_blocks[0]).deflate(
                cutoff=norm_cutoff)
        error = np.asarray(error).item()
        return new_l, SparseTensor(blocks=s_blocks), new_r, error

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

    @staticmethod
    @implements(np.diag)
    def _diag(v):
        """
        Extract a diagonal or construct a diagonal array.

        Args:
            v : FermionTensor
                If v is a 2-D array, return a copy of its 0-th diagonal.
                If v is a 1-D array, return a 2-D array with v on the 0-th diagonal.
        """
        return FermionTensor(odd=np.diag(v.odd), even=np.diag(v.even))

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
            p : FermionTensor
                a with its axes permuted. A view is returned whenever possible.
        """
        odd = np.transpose(a.odd, axes=axes)
        even = np.transpose(a.even, axes=axes)
        return FermionTensor(odd=odd, even=even)

    def transpose(self, axes=None):
        return np.transpose(self, axes=axes)

    @staticmethod
    def _to_sparse(a):
        return a.to_sparse()

    def to_sparse(self):
        return SparseTensor(blocks=self.odd.blocks + self.even.blocks)

    @staticmethod
    def _to_sliceable(a, infos=None):
        return a.to_sliceable(infos=infos)

    def to_sliceable(self, infos=None):
        return self.to_sparse().to_sliceable(infos=infos)

    @staticmethod
    def _to_dense(a, infos=None):
        return a.to_dense(infos=infos)

    def to_dense(self, infos=None):
        return self.to_sparse().to_dense(infos=infos)
