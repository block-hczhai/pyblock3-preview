
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
General Fermionic sparse tensor.
Author: Yang Gao
"""

import numpy as np
from pyblock3.algebra.core import SparseTensor, SubTensor, _sparse_tensor_numpy_func_impls
from pyblock3.algebra.flat import FlatSparseTensor, flat_sparse_skeleton, _flat_sparse_tensor_numpy_func_impls
from pyblock3.algebra.symmetry import QPN
import numbers
from pyblock3.algebra.fermion_split import _run_sparse_fermion_svd, _run_flat_fermion_svd

from block3 import flat_fermion_tensor, flat_sparse_tensor, VectorMapUIntUInt

def flat_fermion_skeleton(bond_infos, pattern=None, dq=None):
    fdq = dq.to_flat() if dq is not None else QPN(0).to_flat()
    if pattern is None:
        pattern = _gen_default_pattern(bond_infos)
    return flat_sparse_tensor.skeleton(block3.VectorMapUIntUInt(bond_infos), pattern, fdq)

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

NEW_METHODS = [np.transpose, np.tensordot, np.add, np.subtract]

_sparse_fermion_tensor_numpy_func_impls = _sparse_tensor_numpy_func_impls.copy()
[_sparse_fermion_tensor_numpy_func_impls.pop(key) for key in NEW_METHODS]
_numpy_func_impls = _sparse_fermion_tensor_numpy_func_impls

def _gen_default_pattern(obj):
    ndim = obj if isinstance(obj, int) else len(obj)
    return "+" * (ndim-1) + '-'

def _flip_pattern(pattern):
    flip_dict = {"+":"-", "-":"+"}
    return "".join([flip_dict[ip] for ip in pattern])

def _contract_patterns(patterna, patternb, idxa, idxb):

    conc_a = "".join([patterna[ix] for ix in idxa])
    out_a = "".join([ip for ix, ip in enumerate(patterna) if ix not in idxa])
    conc_b = "".join([patternb[ix] for ix in idxb])
    out_b = "".join([ip for ix, ip in enumerate(patternb) if ix not in idxb])

    if conc_a == conc_b:
        out = out_a + _flip_pattern(out_b)
    elif conc_a == _flip_pattern(conc_b):
        out = out_a + out_b
    else:
        raise ValueError("Input patterns not valid for contractions")
    return out

def compute_phase(q_labels, axes, direction="left"):
    plist = [qpn.parity for qpn in q_labels]
    counted = []
    phase = 1
    for x in axes:
        if direction=="left":
            parity = sum([plist[i] for i in range(x) if i not in counted]) * plist[x]
        elif direction=="right":
            parity = sum([plist[i] for i in range(x+1, len(plist)) if i not in counted]) * plist[x]
        phase *= (-1) ** parity
        counted.append(x)
    return phase

def fermion_tensor_svd(tsr, left_idx, right_idx=None, **opts):
    pass
    '''
    if right_idx is None:
        right_idx = [idim for idim in range(tsr.ndim) if idim not in left_idx]
    neworder = tuple(left_idx) + tuple(right_idx)
    split_ax = len(left_idx)
    newtsr = tsr.transpose(neworder)
    if isinstance(tsr, SparseFermionTensor):
        u, s, v = _run_sparse_fermion_svd(newtsr, split_ax, **opts)
    elif isinstance(tsr, FlatFermionTensor):
        u, s, v = _run_flat_fermion_svd(newtsr, split_ax, **opts)
    else:
        raise TypeError("Tensor class not Sparse/FlatFermionTensor")
    return u, s, v
    '''

class SparseFermionTensor(SparseTensor):

    def __init__(self, blocks=None, pattern=None):
        self.blocks = blocks if blocks is not None else []
        if pattern is None:
            pattern = _gen_default_pattern(self.ndim)
        self._pattern = pattern

    @property
    def dq(self):
        dq = QPN()
        for qpn, sign in zip(self.blocks[0].q_labels, self.pattern):
            if sign == "+":
                dq += qpn
            else:
                dq -= qpn
        return dq

    @property
    def pattern(self):
        return self._pattern

    @pattern.setter
    def pattern(self, pattern_string):
        if not isinstance(pattern_string, str) or \
              not set(pattern_string).issubset(set("+-")):
            raise TypeError("Pattern must be a string of +-")

        elif len(pattern_string) != self.ndim:
            raise ValueError("Pattern string length must match the dimension of the tensor")
        self._pattern = pattern_string

    @property
    def dagger(self):
        axes = list(range(self.ndim))[::-1]
        blocks = [np.transpose(block.conj(), axes=axes) for block in self.blocks]
        return self.__class__(blocks=blocks, pattern=self.pattern[::-1])

    @property
    def parity(self):
        return self.parity_per_block[0]

    @property
    def parity_per_block(self):
        parity_list = []
        for block in self.blocks:
            parity = np.add.reduce(block.q_labels).parity
            parity_list.append(parity)
        return parity_list

    def conj(self):
        blks = [iblk.conj() for iblk in self.blocks]
        return self.__class__(blocks=blks, pattern=self.pattern)

    def check_sanity(self):
        parity_uniq = np.unique(self.parity_per_block)
        if len(parity_uniq) >1:
            raise ValueError("both odd/even parity found in blocks, \
                              this is prohibited")

    def _local_flip(self, axes):
        if isinstance(axes, int):
            axes = [axes]
        else:
            axes = list(axes)
        for blk in self.blocks:
            block_parity = sum([blk.q_labels[j].parity for j in axes]) % 2
            if block_parity == 1:
                blk *= -1

    def _global_flip(self):
        for blk in self.blocks:
            blk *= -1

    def to_flat(self):
        return FlatFermionTensor.from_sparse(self)

    @staticmethod
    def _skeleton(bond_infos, pattern=None, dq=None):
        """Create tensor skeleton from tuple of BondInfo.
        dq will not have effects if ndim == 1
            (blocks with different dq will not be allowed)."""
        if dq is None: dq = QPN(0,0)
        if not isinstance(dq, QPN):
            raise TypeError("dq is not an instance of QPN class")
        it = np.zeros(tuple(len(i) for i in bond_infos), dtype=int)
        qsh = [sorted(i.items(), key=lambda x: x[0]) for i in bond_infos]
        q = [[k for k, v in i] for i in qsh]
        sh = [[v for k, v in i] for i in qsh]
        if pattern is None:
            pattern = _gen_default_pattern(bond_infos)
        nit = np.nditer(it, flags=['multi_index'])
        for _ in nit:
            x = nit.multi_index
            ps = [iq[ix] if ip == '+' else -iq[ix]
                  for iq, ix, ip in zip(q, x, pattern)]
            if len(ps) == 1 or np.add.reduce(ps) == dq:
                xqs = tuple(iq[ix] for iq, ix in zip(q, x))
                xsh = tuple(ish[ix] for ish, ix in zip(sh, x))
                yield xsh, xqs

    @staticmethod
    def random(bond_infos, pattern=None, dq=None, dtype=float):
        """Create tensor from tuple of BondInfo with random elements."""
        blocks = []
        for sh, qs in SparseFermionTensor._skeleton(bond_infos, pattern=pattern, dq=dq):
            blocks.append(SubTensor.random(shape=sh, q_labels=qs, dtype=dtype))
        return SparseFermionTensor(blocks=blocks, pattern=pattern)

    @staticmethod
    def zeros(bond_infos, pattern=None, dq=None, dtype=float):
        """Create tensor from tuple of BondInfo with zero elements."""
        blocks = []
        for sh, qs in FermionTensor._skeleton(bond_infos, pattern=pattern, dq=dq):
            blocks.append(SubTensor.zeros(shape=sh, q_labels=qs, dtype=dtype))
        return SparseFermionTensor(blocks=blocks, pattern=pattern)

    @staticmethod
    def ones(bond_infos, pattern=None, dq=None, dtype=float):
        """Create tensor from tuple of BondInfo with one elements."""
        blocks = []
        for sh, qs in SparseFermionTensor._skeleton(bond_infos, pattern=pattern, dq=dq):
            blocks.append(SubTensor.ones(shape=sh, q_labels=qs, dtype=dtype))
        return SparseFermionTensor(blocks=blocks, pattern=pattern)

    @staticmethod
    def eye(bond_info):
        """Create tensor from BondInfo with Identity matrix."""
        blocks = []
        pattern = "+-"
        for sh, qs in SparseFermionTensor._skeleton((bond_info, bond_info), pattern=pattern):
            blocks.append(SubTensor(reduced=np.eye(sh[0]), q_labels=qs))
        return SparseFermionTensor(blocks=blocks, pattern=pattern)

    @staticmethod
    def symmetry_compatible(a, b):
        if a.pattern == b.pattern:
            return a.dq == b.dq
        elif a.pattern == _flip_pattern(b.pattern):
            return a.dq == -b.dq
        else:
            return False

    @staticmethod
    @implements(np.add)
    def _add(a, b):
        if isinstance(a, numbers.Number):
            blocks = [np.add(a, block) for block in b.blocks]
            return b.__class__(blocks=blocks, pattern=b.pattern)
        elif isinstance(b, numbers.Number):
            blocks = [np.add(block, b) for block in a.blocks]
            return a.__class__(blocks=blocks, pattern=a.pattern)
        else:
            if not SparseFermionTensor.symmetry_compatible(a, b):
                raise ValueError("Tensors must have same QPN symmetry for addtion")

            blocks_map = {block.q_labels: block for block in a.blocks}
            for block in b.blocks:
                if block.q_labels in blocks_map:
                    mb = blocks_map[block.q_labels]
                    blocks_map[block.q_labels] = np.add(mb, block)
                else:
                    blocks_map[block.q_labels] = block
            blocks = list(blocks_map.values())
            return a.__class__(blocks=blocks, pattern=a.pattern)

    def add(self, b):
        return self._add(self, b)

    @staticmethod
    @implements(np.subtract)
    def _subtract(a, b):
        if isinstance(a, numbers.Number):
            blocks = [np.subtract(a, block) for block in b.blocks]
            return b.__class__(blocks=blocks, pattern=b.pattern)
        elif isinstance(b, numbers.Number):
            blocks = [np.subtract(block, b) for block in a.blocks]
            return a.__class__(blocks=blocks, pattern=a.pattern)
        else:
            if not SparseFermionTensor.symmetry_compatible(a, b):
                raise ValueError("Tensors must have same QPN symmetry for subtraction")

            blocks_map = {block.q_labels: block for block in a.blocks}
            for block in b.blocks:
                if block.q_labels in blocks_map:
                    mb = blocks_map[block.q_labels]
                    blocks_map[block.q_labels] = np.subtract(mb, block)
                else:
                    blocks_map[block.q_labels] = -block
            blocks = list(blocks_map.values())
            return a.__class__(blocks=blocks, pattern=a.pattern)

    def subtract(self, b):
        return self._subtract(self, b)

    def __array_function__(self, func, types, args, kwargs):
        if func not in _sparse_fermion_tensor_numpy_func_impls:
            return NotImplemented
        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented
        return _sparse_fermion_tensor_numpy_func_impls[func](*args, **kwargs)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if ufunc in _sparse_fermion_tensor_numpy_func_impls:
            types = tuple(
                x.__class__ for x in inputs if not isinstance(x, numbers.Number))
            return self.__array_function__(ufunc, types, inputs, kwargs)
        out = kwargs.pop("out", None)
        if out is not None and not all(isinstance(x, self.__class__) for x in out):
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
                out_pattern = _contract_patterns(a.pattern, b.pattern, [a.ndim-1], [0])
            elif ufunc.__name__ in ["multiply", "divide", "true_divide"]:
                a, b = inputs
                if isinstance(a, numbers.Number):
                    blocks = [getattr(ufunc, method)(a, block)
                              for block in b.blocks]
                    out_pattern = b.pattern
                elif isinstance(b, numbers.Number):
                    blocks = [getattr(ufunc, method)(block, b)
                              for block in a.blocks]
                    out_pattern = a.pattern
                else:
                    return NotImplemented
            elif len(inputs) == 1:
                blocks = [getattr(ufunc, method)(block)
                          for block in inputs[0].blocks]
                out_pattern = inputs[0].pattern
            else:
                return NotImplemented
        else:
            return NotImplemented
        if out is not None:
            out[0].blocks = blocks
            out[0].pattern = out_pattern
        return self.__class__(blocks=blocks, pattern=out_pattern)

    @staticmethod
    @implements(np.tensordot)
    def _tensordot(a, b, axes=2):
        if isinstance(axes, int):
            idxa, idxb = list(range(-axes, 0)), list(range(0, axes))
        else:
            idxa, idxb = axes
        idxa = [x if x >= 0 else a.ndim + x for x in idxa]
        idxb = [x if x >= 0 else b.ndim + x for x in idxb]

        out_pattern = _contract_patterns(a.pattern, b.pattern, idxa, idxb)
        out_idx_a = list(set(range(0, a.ndim)) - set(idxa))
        out_idx_b = list(set(range(0, b.ndim)) - set(idxb))
        assert len(idxa) == len(idxb)

        map_idx_b = {}
        for block in b.blocks:
            ctrq = tuple(block.q_labels[id] for id in idxb)
            outq = tuple(block.q_labels[id] for id in out_idx_b)
            if ctrq not in map_idx_b:
                map_idx_b[ctrq] = []
            map_idx_b[ctrq].append((block, outq))

        def _compute_phase(nlist, idx, direction='left'):
            counted = []
            phase = 1
            for x in idx:
                if direction == 'left':
                    counter = sum([nlist[i] for i in range(x) if i not in counted]) * nlist[x]
                elif direction == 'right':
                    counter = sum([nlist[i] for i in range(x+1, len(nlist)) if i not in counted]) * nlist[x]
                phase *= (-1) ** counter
                counted.append(x)
            return phase

        blocks_map = {}
        for block_a in a.blocks:
            phase_a = compute_phase(block_a.q_labels, idxa, 'right')
            ctrq = tuple(block_a.q_labels[id] for id in idxa)
            if ctrq in map_idx_b:
                outqa = tuple(block_a.q_labels[id] for id in out_idx_a)
                for block_b, outqb in map_idx_b[ctrq]:
                    phase_b = compute_phase(block_b.q_labels, idxb, 'left')
                    phase = phase_a * phase_b
                    outq = outqa + outqb
                    mat = np.tensordot(np.asarray(block_a), np.asarray(
                        block_b), axes=(idxa, idxb)).view(block_a.__class__)
                    if outq not in blocks_map:
                        mat.q_labels = outq
                        blocks_map[outq] = mat * phase
                    else:
                        blocks_map[outq] += mat * phase

        return a.__class__(blocks=list(blocks_map.values()), pattern=out_pattern)

    @staticmethod
    @implements(np.transpose)
    def _transpose(a, axes=None):
        if axes is None:
            axes = list(range(a.ndim))[::-1]
        phase = [compute_phase(block.q_labels, axes) for block in a.blocks]
        blocks = [np.transpose(block, axes=axes)*phase[ibk] for ibk, block in enumerate(a.blocks)]
        pattern = "".join([a.pattern[ix] for ix in axes])
        return a.__class__(blocks=blocks, pattern=pattern)

    def tensor_svd(self, left_idx, right_idx=None, qpn_info=None, **opts):
        return _run_sparse_fermion_svd(self, left_idx, right_idx=right_idx, qpn_info=qpn_info, **opts)

_flat_fermion_tensor_numpy_func_impls = _flat_sparse_tensor_numpy_func_impls.copy()
[_flat_fermion_tensor_numpy_func_impls.pop(key) for key in NEW_METHODS]
_numpy_func_impls = _flat_fermion_tensor_numpy_func_impls

class FlatFermionTensor(FlatSparseTensor):

    def __init__(self, q_labels, shapes, data, pattern=None, idxs=None):
        self.n_blocks = len(q_labels)
        self.ndim = q_labels.shape[1] if self.n_blocks != 0 else 0
        self.shapes = shapes
        self.q_labels = q_labels
        self.data = data

        if pattern is None:
            pattern = _gen_default_pattern(self.ndim)
        self._pattern = pattern

        if idxs is None:
            self.idxs = np.zeros((self.n_blocks + 1, ), dtype=shapes.dtype)
            self.idxs[1:] = np.cumsum(shapes.prod(axis=1))
        else:
            self.idxs = idxs
        if self.n_blocks != 0:
            assert shapes.shape == (self.n_blocks, self.ndim)
            assert q_labels.shape == (self.n_blocks, self.ndim)

    @property
    def dq(self):
        dq = QPN()
        for qpn, sign in zip(self.q_labels[0], self.pattern):
            if sign == "+":
                dq += QPN.from_flat(qpn)
            else:
                dq -= QPN.from_flat(qpn)
        return dq

    @property
    def pattern(self):
        return self._pattern

    @pattern.setter
    def pattern(self, pattern_string):
        if not isinstance(pattern_string, str) or \
              not set(pattern_string).issubset(set("+-")):
            raise TypeError("Pattern must be a string of +-")

        elif len(pattern_string) != self.ndim:
            raise ValueError("Pattern string length must match the dimension of the tensor")
        self._pattern = pattern_string

    @property
    def dagger(self):
        axes = list(range(self.ndim))[::-1]
        axes = np.array(axes, dtype=np.int32)
        data = np.zeros_like(self.data)
        flat_sparse_tensor.transpose(self.shapes, self.data.conj(), self.idxs, axes, data)
        return self.__class__(self.q_labels[:, axes], self.shapes[:, axes], data, pattern=self.pattern[::-1], idxs=self.idxs)

    @property
    def parity(self):
        return self.parity_per_block[0]

    @property
    def parity_per_block(self):
        parity_list = []
        for q_label in self.q_labels:
            parity = np.add.reduce([QPN.from_flat(qlab) for qlab in q_label]).parity
            parity_list.append(parity)
        return parity_list

    def conj(self):
        return self.__class__(self.q_labels, self.shapes, self.data.conj(), pattern=self.pattern, idxs=self.idxs)

    def check_sanity(self):
        parity_uniq = np.unique(self.parity_per_block)
        if len(parity_uniq) >1:
            raise ValueError("both odd/even parity found in blocks, \
                              this is prohibited")

    def _local_flip(self, axes):
        if isinstance(axes, int):
            ax = [axes]
        else:
            ax = list(axes)
        idx = self.idxs
        for i in range(len(self.q_labels)):
            block_parity = np.add.reduce([QPN.from_flat(self.q_labels[i][j]) for j in ax]).parity
            if block_parity == 1:
                self.data[idx[i]:idx[i+1]] *=-1

    def _global_flip(self):
        self.data *= -1

    def to_sparse(self):
        blocks = [None] * self.n_blocks
        for i in range(self.n_blocks):
            qs = tuple(map(QPN.from_flat, self.q_labels[i]))
            blocks[i] = SubTensor(
                self.data[self.idxs[i]:self.idxs[i + 1]].reshape(self.shapes[i]), q_labels=qs)
        return SparseFermionTensor(blocks=blocks, pattern=self.pattern)

    @staticmethod
    def from_sparse(spt):
        ndim = spt.ndim
        n_blocks = spt.n_blocks
        shapes = np.zeros((n_blocks, ndim), dtype=np.uint32)
        q_labels = np.zeros((n_blocks, ndim), dtype=np.uint32)
        for i in range(n_blocks):
            shapes[i] = spt.blocks[i].shape
            q_labels[i] = list(map(QPN.to_flat, spt.blocks[i].q_labels))
        idxs = np.zeros((n_blocks + 1, ), dtype=np.uint32)
        idxs[1:] = np.cumsum(shapes.prod(axis=1))
        data = np.zeros((idxs[-1], ), dtype=spt.dtype)
        for i in range(n_blocks):
            data[idxs[i]:idxs[i + 1]] = spt.blocks[i].flatten()
        return FlatFermionTensor(q_labels, shapes, data, spt.pattern, idxs)

    @staticmethod
    def random(bond_infos, pattern=None, dq=None, dtype=float):
        if pattern is None:
            pattern = _gen_default_pattern(bond_infos)
        qs, shs, idxs = flat_fermion_skeleton(bond_infos, pattern, dq)
        if dtype == float:
            data = np.random.random((idxs[-1], ))
        elif dtype == complex:
            data = np.random.random(
                (idxs[-1], )) + np.random.random((idxs[-1], )) * 1j
        else:
            return NotImplementedError('dtype %r not supported!' % dtype)
        return FlatFermionTensor(qs, shs, data, pattern, idxs)

    @staticmethod
    def zeros(bond_infos, pattern=None, dq=None, dtype=float):
        if pattern is None:
            pattern = _gen_default_pattern(bond_infos)
        qs, shs, idxs = flat_fermion_skeleton(bond_infos, pattern, dq)
        data = np.zeros((idxs[-1], ), dtype=dtype)
        return FlatFermionTensor(qs, shs, data, pattern, idxs)

    @staticmethod
    def ones(bond_infos, pattern=None, dq=None, dtype=float):
        """Create tensor from tuple of BondInfo with one elements."""
        if pattern is None:
            pattern = _gen_default_pattern(bond_infos)
        qs, shs, idxs = flat_fermion_skeleton(bond_infos, pattern, dq)
        data = np.ones((idxs[-1], ), dtype=dtype)
        return FlatFermionTensor(qs, shs, data, pattern, idxs)

    @staticmethod
    def symmetry_compatible(a, b):
        if a.pattern == b.pattern:
            return a.dq == b.dq
        elif a.pattern == _flip_pattern(b.pattern):
            return a.dq == -b.dq
        else:
            return False

    @staticmethod
    @implements(np.add)
    def _add(a, b):
        if isinstance(a, numbers.Number):
            data = a + b.data
            return b.__class__(b.q_labels, b.shapes, data, b.pattern, b.idxs)
        elif isinstance(b, numbers.Number):
            data = a.data + b
            return a.__class__(a.q_labels, a.shapes, data, a.pattern, a.idxs)
        elif b.n_blocks == 0:
            return a
        elif a.n_blocks == 0:
            return b
        else:
            if not FlatFermionTensor.symmetry_compatible(a, b):
                raise ValueError("Tensors must have same parity for addition")
            q_labels, shapes, data, idxs = flat_sparse_tensor.add(a.q_labels, a.shapes, a.data,
                                                a.idxs, b.q_labels, b.shapes, b.data, b.idxs)
            return a.__class__(q_labels, shapes, data, a.pattern, idxs)

    def add(self, b):
        return self._add(self, b)

    @staticmethod
    @implements(np.subtract)
    def _subtract(a, b):
        if isinstance(a, numbers.Number):
            data = a - b.data
            return b.__class__(b.q_labels, b.shapes, data, b.pattern, b.idxs)
        elif isinstance(b, numbers.Number):
            data = a.data - b
            return a.__class__(a.q_labels, a.shapes, data, a.pattern, a.idxs)
        elif b.n_blocks == 0:
            return a
        elif a.n_blocks == 0:
            return b
        else:
            if not FlatFermionTensor.symmetry_compatible(a, b):
                raise ValueError("Tensors must have same parity for subtraction")
            q_labels, shapes, data, idxs = flat_sparse_tensor.add(a.q_labels, a.shapes, a.data,
                                                a.idxs, b.q_labels, b.shapes, -b.data, b.idxs)
            return a.__class__(q_labels, shapes, data, a.pattern, idxs)

    def subtract(self, b):
        return self._subtract(self, b)

    def __array_function__(self, func, types, args, kwargs):
        if func not in _flat_fermion_tensor_numpy_func_impls:
            return NotImplemented
        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented
        return _flat_fermion_tensor_numpy_func_impls[func](*args, **kwargs)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if ufunc in _flat_fermion_tensor_numpy_func_impls:
            types = tuple(
                x.__class__ for x in inputs if not isinstance(x, numbers.Number))
            return self.__array_function__(ufunc, types, inputs, kwargs)
        out = kwargs.pop("out", None)
        if out is not None and not all(isinstance(x, self.__class__) for x in out):
            return NotImplemented

        if method == "__call__":
            if ufunc.__name__ in ["matmul"]:
                a, b = inputs
                if isinstance(a, numbers.Number):
                    shs, qs, data, idxs = b.shapes, b.q_labels, a * b.data, b.idxs
                    out_pattern = b.pattern
                elif isinstance(b, numbers.Number):
                    shs, qs, data, idxs = a.shapes, a.q_labels, a.data * b, a.idxs
                    out_pattern = a.pattern
                else:
                    c = self._tensordot(a, b, axes=([-1], [0]))
                    shs, qs, data, idxs = c.shapes, c.q_labels, c.data, c.idxs
                    out_pattern = _contract_patterns(a.pattern, b.pattern, [a.ndim-1], [0])
            elif ufunc.__name__ in ["multiply", "divide", "true_divide"]:
                a, b = inputs
                if isinstance(a, numbers.Number):
                    shs, qs, data, idxs = b.shapes, b.q_labels, getattr(
                        ufunc, method)(a, b.data), b.idxs
                    out_pattern = b.pattern
                elif isinstance(b, numbers.Number):
                    shs, qs, data, idxs = a.shapes, a.q_labels, getattr(
                        ufunc, method)(a.data, b), a.idxs
                    out_pattern = a.pattern
                else:
                    return NotImplemented
            elif len(inputs) == 1:
                a = inputs[0]
                shs, qs, data, idxs = a.shapes, a.q_labels, getattr(
                    ufunc, method)(a.data), a.idxs
                out_pattern = a.pattern
            else:
                return NotImplemented
        else:
            return NotImplemented
        if out is not None:
            out[0].shapes[...] = shs
            out[0].q_labels[...] = qs
            out[0].data[...] = data
            out[0].idxs[...] = idxs
            out[0].pattern = out_pattern
        return FlatFermionTensor(q_labels=qs, shapes=shs, data=data, pattern=out_pattern, idxs=idxs)

    @staticmethod
    @implements(np.tensordot)
    def _tensordot(a, b, axes=2):
        if isinstance(axes, int):
            idxa = np.arange(-axes, 0, dtype=np.int32)
            idxb = np.arange(0, axes, dtype=np.int32)
        else:
            idxa = np.array(axes[0], dtype=np.int32)
            idxb = np.array(axes[1], dtype=np.int32)
        idxa[idxa < 0] += a.ndim
        idxb[idxb < 0] += b.ndim
        q_labels, shapes, data, idxs = flat_fermion_tensor.tensordot(
                    a.q_labels, a.shapes, a.data, a.idxs,
                    b.q_labels, b.shapes, b.data, b.idxs,
                    idxa, idxb)
        out_pattern = _contract_patterns(a.pattern, b.pattern, idxa, idxb)
        return a.__class__(q_labels, shapes, data, out_pattern, idxs)

    @staticmethod
    @implements(np.transpose)
    def _transpose(a, axes=None):
        if axes is None:
            axes = np.arange(a.ndim)[::-1]
        if a.n_blocks == 0:
            return a
        else:
            data = np.zeros_like(a.data)
            axes = np.array(axes, dtype=np.int32)
            flat_fermion_tensor.transpose(a.q_labels, a.shapes, a.data, a.idxs, axes, data)
            pattern = "".join([a.pattern[ix] for ix in axes])
            return a.__class__(a.q_labels[:,axes], a.shapes[:,axes], \
                               data, pattern, a.idxs)

    def tensor_svd(self, left_idx, right_idx=None, qpn_info=None, **opts):
        return _run_flat_fermion_svd(self, left_idx, right_idx=right_idx, qpn_info=qpn_info, **opts)
