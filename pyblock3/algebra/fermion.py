
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

import numbers
from functools import reduce
import numpy as np
from pyblock3.algebra.core import SparseTensor, SubTensor, _sparse_tensor_numpy_func_impls
from pyblock3.algebra.flat import FlatSparseTensor, _flat_sparse_tensor_numpy_func_impls
from pyblock3.algebra.symmetry import BondInfo
from pyblock3.algebra.fermion_symmetry import U1, Z4, Z2, U11
import block3.sz as block3

def get_backend(symmetry):
    if isinstance(symmetry, str):
        key = symmetry.upper()
    else:
        key = symmetry.__name__
    if key == "U11":
        import block3.u11 as backend
    elif key == "U1":
        import block3.u1 as backend
    elif key == "Z2":
        import block3.z2 as backend
    elif key == "Z4":
        import block3.z4 as backend
    else:
        raise NotImplementedError
    return backend

DEFAULT_SYMMETRY = U1
SVD_SCREENING = 1e-10

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

NEW_METHODS = [np.transpose, np.tensordot, np.add, np.subtract, np.copy]

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

def _trim_singular_vals(s, cutoff, cutoff_mode):
    """Find the number of singular values to keep of ``s`` given ``cutoff`` and
    ``cutoff_mode``.
    Parameters
    ----------
    s : array
        Singular values.
    cutoff : float
        Cutoff.
    cutoff_mode : {1, 2, 3, 4, 5, 6}
        How to perform the trim:
            - 1: ['abs'], trim values below ``cutoff``
            - 2: ['rel'], trim values below ``s[0] * cutoff``
            - 3: ['sum2'], trim s.t. ``sum(s_trim**2) < cutoff``.
            - 4: ['rsum2'], trim s.t. ``sum(s_trim**2) < sum(s**2) * cutoff``.
            - 5: ['sum1'], trim s.t. ``sum(s_trim**1) < cutoff``.
            - 6: ['rsum1'], trim s.t. ``sum(s_trim**1) < sum(s**1) * cutoff``.
    """
    if cutoff_mode == 1:
        n_chi = np.sum(s > cutoff)

    elif cutoff_mode == 2:
        n_chi = np.sum(s > cutoff * s[0])

    elif cutoff_mode in (3, 4, 5, 6):
        if cutoff_mode in (3, 4):
            p = 2
        else:
            p = 1

        target = cutoff
        if cutoff_mode in (4, 6):
            target *= np.sum(s**p)

        n_chi = s.size
        ssum = 0.0
        for i in range(s.size - 1, -1, -1):
            s2 = s[i]**p
            if not np.isnan(s2):
                ssum += s2
            if ssum > target:
                break
            n_chi -= 1

    return max(n_chi, 1)

def _renorm_singular_vals(s, n_chi, renorm_power):
    """Find the normalization constant for ``s`` such that the new sum squared
    of the ``n_chi`` largest values equals the sum squared of all the old ones.
    """
    s_tot_keep = 0.0
    s_tot_lose = 0.0
    for i in range(s.size):
        s2 = s[i]**renorm_power
        if not np.isnan(s2):
            if i < n_chi:
                s_tot_keep += s2
            else:
                s_tot_lose += s2
    return ((s_tot_keep + s_tot_lose) / s_tot_keep)**(1 / renorm_power)

def _trim_and_renorm_SVD(U, s, VH, **svdopts):
    cutoff = svdopts.pop("cutoff", -1.0)
    cutoff_mode = svdopts.pop("cutoff_mode", 3)
    max_bond = svdopts.pop("max_bond", -1)
    renorm_power = svdopts.pop("renorm_power", 0)

    if cutoff > 0.0:
        n_chi = _trim_singular_vals(s, cutoff, cutoff_mode)

        if max_bond > 0:
            n_chi = min(n_chi, max_bond)

        if n_chi < s.size:
            if renorm_power > 0:
                s = s[:n_chi] * _renorm_singular_vals(s, n_chi, renorm_power)
            else:
                s = s[:n_chi]

            U = U[..., :n_chi]
            VH = VH[:n_chi, ...]

    elif max_bond != -1:
        U = U[..., :max_bond]
        s = s[:max_bond]
        VH = VH[:max_bond, ...]

    s = np.ascontiguousarray(s)
    return U, s, VH

def _absorb_svd(u, s, v, absorb):
    if absorb == -1:
        u = u * s.reshape((1, -1))
    elif absorb == 1:
        v = v * s.reshape((-1, 1))
    else:
        s **= 0.5
        u = u * s.reshape((1, -1))
        v = v * s.reshape((-1, 1))
    return u, None, v

def _svd_preprocess(T, left_idx, right_idx, qpn_partition, absorb):
    cls = T.dq.__class__
    if qpn_partition is None:
        if absorb ==-1:
            qpn_partition = (T.dq, cls(0))
        else:
            qpn_partition = (cls(0), T.dq)
    else:
        assert np.sum(qpn_partition) == T.dq

    if right_idx is None:
        right_idx = [idim for idim in range(T.ndim) if idim not in left_idx]
    elif left_idx is None:
        left_idx = [idim for idim in range(T.ndim) if idim not in right_idx]
    neworder = list(left_idx) + list(right_idx)
    if neworder == list(range(T.ndim)):
        new_T = T
    else:
        new_T = T.transpose(neworder)

    return new_T, left_idx, right_idx, qpn_partition, cls

def sparse_svd(T, left_idx, right_idx=None, absorb=0, qpn_partition=None, qpn_cutoff_func=None, **opts):
    new_T, left_idx, right_idx, qpn_partition, symmetry = _svd_preprocess(T, left_idx, right_idx, qpn_partition, absorb)
    if len(left_idx) == T.ndim:
        raise NotImplementedError
    elif len(right_idx) == T.ndim:
        raise NotImplementedError
    split_ax = len(left_idx)
    left_pattern = new_T.pattern[:split_ax]
    right_pattern = new_T.pattern[split_ax:]
    data_map = {}
    for iblk in new_T.blocks:
        left_q = symmetry._compute(left_pattern, iblk.q_labels[:split_ax], offset=("-", qpn_partition[0]))
        if callable(qpn_cutoff_func):
            if not qpn_cutoff_func(left_q):
                continue
        if left_q not in data_map:
            data_map[left_q] = []
        data_map[left_q].append(iblk)
    ublocks = []
    vblocks = []
    sblocks = []
    for slab, datasets in data_map.items():
        row_len = col_len = 0
        row_map = {}
        col_map = {}
        for iblk in datasets:
            lq = tuple(iblk.q_labels[:split_ax]) + (slab,)
            rq = (slab,) + tuple(iblk.q_labels[split_ax:])
            if lq not in row_map:
                new_row_len = row_len + np.prod(iblk.shape[:split_ax], dtype=int)
                row_map[lq] = (row_len, new_row_len, iblk.shape[:split_ax])
                row_len = new_row_len
            if rq not in col_map:
                new_col_len = col_len + np.prod(iblk.shape[split_ax:], dtype=int)
                col_map[rq] = (col_len, new_col_len, iblk.shape[split_ax:])
                col_len = new_col_len
        data = np.zeros([row_len, col_len], dtype=T.dtype)
        for iblk in datasets:
            lq = tuple(iblk.q_labels[:split_ax]) + (slab,)
            rq = (slab,) + tuple(iblk.q_labels[split_ax:])
            ist, ied = row_map[lq][:2]
            jst, jed = col_map[rq][:2]
            data[ist:ied,jst:jed] = np.asarray(iblk).reshape(ied-ist, jst-jed)
        if data.size ==0:
            continue

        u, s, v = np.linalg.svd(data, full_matrices=False)
        u, s, v = _trim_and_renorm_SVD(u, s, v, **opts)
        if absorb is None:
            s = np.diag(s)
            sblocks.append(SubTensor(reduced=s, q_labels=(slab, slab)))
        else:
            u, s, v= _absorb_svd(u, s, v, absorb)

        for lq, (lst, led, lsh) in row_map.items():
            if abs(u[lst:led]).max()<SVD_SCREENING:
                continue
            ublocks.append(SubTensor(reduced=u[lst:led].reshape(tuple(lsh)+(-1,)), q_labels=lq))

        for rq, (rst, red, rsh) in col_map.items():
            if abs(v[:,rst:red]).max()<SVD_SCREENING:
                continue
            vblocks.append(SubTensor(reduced=v[:,rst:red].reshape((-1,)+tuple(rsh)), q_labels=rq))
    if absorb is None:
        s = T.__class__(blocks=sblocks, pattern="+-")
    u = T.__class__(blocks=ublocks, pattern=new_T.pattern[:split_ax]+"-")
    v = T.__class__(blocks=vblocks, pattern="+"+new_T.pattern[split_ax:])
    return u, s, v

def flat_svd(T, left_idx, right_idx=None, qpn_partition=None, qpn_cutoff_func=None, **opts):
    absorb = opts.pop("absorb", 0)
    new_T, left_idx, right_idx, qpn_partition, symmetry = _svd_preprocess(T, left_idx, right_idx, qpn_partition, absorb)
    if left_idx is not None and len(left_idx) == T.ndim:
        raise NotImplementedError
    elif right_idx is not None and len(right_idx) == T.ndim:
        raise NotImplementedError

    split_ax = len(left_idx)
    left_q = symmetry._compute(new_T.pattern[:split_ax], new_T.q_labels[:,:split_ax], offset=("-", qpn_partition[0]))
    right_q = symmetry._compute(new_T.pattern[split_ax:], new_T.q_labels[:,split_ax:], offset=("-", qpn_partition[1]), neg=True)

    if not callable(qpn_cutoff_func):
        aux_q = list(set(np.unique(left_q)) & set(np.unique(right_q)))
    else:
        tmp_q = list(set(np.unique(left_q)) & set(np.unique(right_q)))
        aux_q = []
        for iq in tmp_q:
            if qpn_cutoff_func(symmetry.from_flat(iq)):
                aux_q.append(iq)
    full_left = np.hstack([new_T.q_labels[:,:split_ax], left_q.reshape(-1,1)])
    full_right = np.hstack([left_q.reshape(-1,1), new_T.q_labels[:,split_ax:]])
    full = [(tuple(il), tuple(ir)) for il, ir in zip(full_left, full_right)]

    udata = []
    vdata = []
    sdata = []

    qu = []
    qv = []
    qs = []
    shu = []
    shv = []
    shs = []
    row_shapes = np.prod(new_T.shapes[:,:split_ax], axis=1, dtype=int)
    col_shapes = np.prod(new_T.shapes[:,split_ax:], axis=1, dtype=int)
    for slab in aux_q:
        blocks = np.where(left_q == slab)[0]
        row_map = {}
        col_map = {}
        row_len = 0
        col_len = 0
        qs.append([slab, slab])
        alldatas = {}
        for iblk in blocks:
            lq, rq = full[iblk]
            if lq not in row_map:
                new_row_len = row_shapes[iblk] + row_len
                row_map[lq] = (row_len, new_row_len, new_T.shapes[iblk,:split_ax])
                ist, ied = row_len, new_row_len
                row_len = new_row_len
            else:
                ist, ied = row_map[lq][:2]
            if rq not in col_map:
                new_col_len = col_shapes[iblk] + col_len
                col_map[rq] = (col_len, new_col_len, new_T.shapes[iblk,split_ax:])
                jst, jed = col_len, new_col_len
                col_len = new_col_len
            else:
                jst, jed = col_map[rq][:2]
            xst, xed = new_T.idxs[iblk], new_T.idxs[iblk+1]
            alldatas[(ist,ied,jst,jed)] = new_T.data[xst:xed].reshape(ied-ist, jed-jst)

        data = np.zeros([row_len, col_len], dtype=new_T.dtype)
        for (ist, ied, jst, jed), val in alldatas.items():
            data[ist:ied,jst:jed] = val

        if data.size==0:
            continue
        u, s, v = np.linalg.svd(data, full_matrices=False)
        u, s, v = _trim_and_renorm_SVD(u, s, v, **opts)
        if absorb is None:
            s = np.diag(s)
            shs.append(s.shape)
            sdata.append(s.ravel())
        else:
            u, s, v= _absorb_svd(u, s, v, absorb)

        for lq, (lst, led, lsh) in row_map.items():
            if abs(u[lst:led]).max()<SVD_SCREENING:
                continue
            udata.append(u[lst:led].ravel())
            qu.append(lq)
            shu.append(tuple(lsh)+(u.shape[-1],))

        for rq, (rst, red, rsh) in col_map.items():
            if abs(v[:,rst:red]).max()<SVD_SCREENING:
                continue
            vdata.append(v[:,rst:red].ravel())
            qv.append(rq)
            shv.append((v.shape[0],)+tuple(rsh))

    if absorb is None:
        qs = np.asarray(qs, dtype=np.uint32)
        shs = np.asarray(shs, dtype=np.uint32)
        sdata = np.concatenate(sdata)
        s = T.__class__(qs, shs, sdata, pattern="+-", symmetry=T.symmetry)

    qu = np.asarray(qu, dtype=np.uint32)
    shu = np.asarray(shu, dtype=np.uint32)
    udata = np.concatenate(udata)

    qv = np.asarray(qv, dtype=np.uint32)
    shv = np.asarray(shv, dtype=np.uint32)
    vdata = np.concatenate(vdata)
    u = T.__class__(qu, shu, udata, pattern=new_T.pattern[:split_ax]+"-", symmetry=T.symmetry)
    v = T.__class__(qv, shv, vdata, pattern="+"+new_T.pattern[split_ax:], symmetry=T.symmetry)
    return u, s, v

def symmetry_compatible(a, b):
    return a.pattern == b.pattern or a.pattern==_flip_pattern(b.pattern)

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

def eye(bond_info, FLAT=True):
    """Create tensor from BondInfo with Identity matrix."""
    blocks = []
    for sh, qs in SparseFermionTensor._skeleton((bond_info, bond_info)):
        blocks.append(SubTensor(reduced=np.eye(sh[0]), q_labels=qs))
    T = SparseFermionTensor(blocks=blocks, pattern="+-")
    if FLAT:
        T = T.to_flat()
    return T

class SparseFermionTensor(SparseTensor):

    def __init__(self, blocks=None, pattern=None):
        self.blocks = blocks if blocks is not None else []
        if pattern is None:
            pattern = _gen_default_pattern(self.ndim)
        self._pattern = pattern

    @property
    def dq(self):
        cls = self.blocks[0].q_labels[0].__class__
        dq = cls._compute(self.pattern, self.blocks[0].q_labels)
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

    @staticmethod
    @implements(np.copy)
    def _copy(x):
        return x.__class__(blocks=[b.copy() for b in x.blocks], pattern=x.pattern)

    def copy(self):
        return np.copy(self)

    @property
    def parity(self):
        return self.dq.parity

    def conj(self):
        blks = [iblk.conj() for iblk in self.blocks]
        return self.__class__(blocks=blks, pattern=self.pattern)

    def _local_flip(self, axes):
        if isinstance(axes, int):
            axes = [axes]
        else:
            axes = list(axes)
        for blk in self.blocks:
            block_parity = np.add.reduce([blk.q_labels[j] for j in axes]).parity
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
        if dq is None:
            symmetry = list(bond_infos[0].keys())[0].__class__
            dq = symmetry(0)
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
        for sh, qs in SparseFermionTensor._skeleton(bond_infos, pattern=pattern, dq=dq):
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

    def expand_dim(self, axis=0, dq=None, direction="left", return_full=False):
        if dq is None:
            dq = self.blocks[0].q_labels.__class__(0)
        assert direction in ["left", "right"]
        blocks=[]
        for iblk in self.blocks:
            shape = iblk.shape
            new_shape = shape[:axis] + (1,) + shape[axis:]
            if direction=="left":
                passed_symmetry = iblk.q_labels[:axis]
            else:
                passed_symmetry = iblk.q_labels[axis:]
            if len(passed_symmetry) >0:
                passed_parity = np.sum(passed_symmetry).parity
            else:
                passed_parity = 0
            phase = -1 if passed_parity * dq.parity==1 else 1
            dat = np.asarray(iblk).reshape(new_shape) * phase
            qpn = iblk.q_labels[:axis] + (dq,) + iblk.q_labels[axis:]
            blocks.append(SubTensor(reduced=dat, q_labels=qpn))
        pattern = self.pattern[:axis] + "+" + self.pattern[axis:]
        new_T = self.__class__(blocks=blocks, pattern=pattern)

        if return_full:
            comp = self.__class__(
                    blocks=[SubTensor(reduced=np.ones(1), q_labels=(dq,))], pattern="-")
            identity = self.__class__.eye(BondInfo({dq:1}))
            return new_T, comp, identity
        else:
            return new_T

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
            if not symmetry_compatible(a, b):
                raise ValueError("Tensors must have same symmetry for addtion")

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
            if not symmetry_compatible(a, b):
                raise ValueError("Tensors must have same symmetry for subtraction")

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

    def tensor_svd(self, left_idx, right_idx=None, qpn_partition=None, qpn_cutoff_func=None, **opts):
        return sparse_svd(self, left_idx, right_idx=right_idx, qpn_partition=qpn_partition, qpn_cutoff_func=qpn_cutoff_func, **opts)

    def to_exponential(self, x):
        return get_sparse_exponential(self,x)

_flat_fermion_tensor_numpy_func_impls = _flat_sparse_tensor_numpy_func_impls.copy()
[_flat_fermion_tensor_numpy_func_impls.pop(key) for key in NEW_METHODS]
_numpy_func_impls = _flat_fermion_tensor_numpy_func_impls

class FlatFermionTensor(FlatSparseTensor):

    def __init__(self, q_labels, shapes, data, pattern=None, idxs=None, symmetry=DEFAULT_SYMMETRY):
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
        self.symmetry = symmetry

    @property
    def dq(self):
        cls = self.symmetry
        dq = cls._compute(self.pattern, self.q_labels[0][None])[0]
        x = cls._compute(self.pattern, self.q_labels[0][None])[0]
        return cls.from_flat(int(dq))

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
        backend = get_backend(self.symmetry)
        backend.flat_sparse_tensor.transpose(self.shapes, self.data.conj(), self.idxs, axes, data)
        return self.__class__(self.q_labels[:, axes], self.shapes[:, axes], data, pattern=self.pattern[::-1], idxs=self.idxs, symmetry=self.symmetry)

    @staticmethod
    @implements(np.copy)
    def _copy(x):
        return x.__class__(q_labels=x.q_labels.copy(), shapes=x.shapes.copy(), data=x.data.copy(), pattern=x.pattern, idxs=x.idxs.copy(), symmetry=x.symmetry)

    def copy(self):
        return np.copy(self)

    @property
    def parity(self):
        return self.dq.parity

    @property
    def shape(self):
        return tuple(np.amax(self.shapes, axis=0))

    def conj(self):
        return self.__class__(self.q_labels, self.shapes, self.data.conj(), pattern=self.pattern, idxs=self.idxs, symmetry=self.symmetry)

    def _local_flip(self, axes):
        if isinstance(axes, int):
            ax = [axes]
        else:
            ax = list(axes)
        idx = self.idxs
        q_labels = np.stack([self.q_labels[:,ix] for ix in axes], axis=1)
        pattern = "".join([self.pattern[ix] for ix in axes])
        net_q = self.symmetry._compute(pattern, q_labels)
        parities = self.symmetry.flat_to_parity(net_q)
        inds = np.where(parities==1)[0]
        for i in inds:
            self.data[idx[i]:idx[i+1]] *=-1

    def _global_flip(self):
        self.data *= -1

    def to_sparse(self):
        blocks = [None] * self.n_blocks
        for i in range(self.n_blocks):
            qs = tuple(map(self.symmetry.from_flat, self.q_labels[i]))
            blocks[i] = SubTensor(
                self.data[self.idxs[i]:self.idxs[i + 1]].reshape(self.shapes[i]), q_labels=qs)
        return SparseFermionTensor(blocks=blocks, pattern=self.pattern)

    def expand_dim(self, axis=0, dq=None, direction="left", return_full=False):
        if dq is None:
            dq = self.symmetry(0)
        const = dq.to_flat()
        q_labels = self.q_labels
        qpn = np.insert(q_labels, axis, const, axis=1)
        shapes = np.insert(self.shapes, axis, 1, axis=1)
        pattern = self.pattern[:axis] + "+" + self.pattern[axis:]
        if direction=="left":
            passed_symmetry = self.q_labels[:,:axis]
        else:
            passed_symmetry = self.q_labels[:,axis:]

        data = self.data.copy()
        if dq.parity!=0:
            if passed_symmetry.size !=0:
                passed_parity = self.symmetry._compute(self.pattern[:axis], passed_symmetry)
                passed_parity = self.symmetry.flat_to_parity(passed_parity) * dq.parity
                blocks = np.where(passed_parity==1)[0]
                for iblk in blocks:
                    ist, ied = self.idxs[iblk], self.idxs[iblk+1]
                    data[ist:ied] *= -1

        new_T = self.__class__(qpn, shapes, data, pattern=pattern, symmetry=self.symmetry)
        if return_full:
            bond = BondInfo({dq:1})
            comp = SparseFermionTensor.ones((bond,), pattern="-").to_flat()
            identity = SparseFermionTensor.eye(bond).to_flat()
            return new_T, comp, identity
        else:
            return new_T

    @staticmethod
    def from_sparse(spt):
        ndim = spt.ndim
        n_blocks = spt.n_blocks
        shapes = np.zeros((n_blocks, ndim), dtype=np.uint32)
        q_labels = np.zeros((n_blocks, ndim), dtype=np.uint32)
        cls = spt.blocks[0].q_labels[0].__class__
        for i in range(n_blocks):
            shapes[i] = spt.blocks[i].shape
            q_labels[i] = list(map(cls.to_flat, spt.blocks[i].q_labels))
        idxs = np.zeros((n_blocks + 1, ), dtype=np.uint32)
        idxs[1:] = np.cumsum(shapes.prod(axis=1))
        data = np.zeros((idxs[-1], ), dtype=spt.dtype)
        for i in range(n_blocks):
            data[idxs[i]:idxs[i + 1]] = spt.blocks[i].flatten()
        return FlatFermionTensor(q_labels, shapes, data, spt.pattern, idxs, symmetry=cls)

    @staticmethod
    @implements(np.add)
    def _add(a, b):
        if isinstance(a, numbers.Number):
            data = a + b.data
            return b.__class__(b.q_labels, b.shapes, data, b.pattern, b.idxs, b.symmetry)
        elif isinstance(b, numbers.Number):
            data = a.data + b
            return a.__class__(a.q_labels, a.shapes, data, a.pattern, a.idxs, a.symmetry)
        elif b.n_blocks == 0:
            return a
        elif a.n_blocks == 0:
            return b
        else:
            if not symmetry_compatible(a, b):
                raise ValueError("Tensors must have same symmetry for addition")
            q_labels, shapes, data, idxs = block3.flat_sparse_tensor.add(a.q_labels, a.shapes, a.data,
                                                a.idxs, b.q_labels, b.shapes, b.data, b.idxs)
            return a.__class__(q_labels, shapes, data, a.pattern, idxs, a.symmetry)

    def add(self, b):
        return self._add(self, b)

    @staticmethod
    @implements(np.subtract)
    def _subtract(a, b):
        if isinstance(a, numbers.Number):
            data = a - b.data
            return b.__class__(b.q_labels, b.shapes, data, b.pattern, b.idxs, b.symmetry)
        elif isinstance(b, numbers.Number):
            data = a.data - b
            return a.__class__(a.q_labels, a.shapes, data, a.pattern, a.idxs, a.symmetry)
        elif b.n_blocks == 0:
            return a
        elif a.n_blocks == 0:
            return b
        else:
            if not symmetry_compatible(a, b):
                raise ValueError("Tensors must have same symmetry for subtraction")
            q_labels, shapes, data, idxs = block3.flat_sparse_tensor.add(a.q_labels, a.shapes, a.data,
                                                a.idxs, b.q_labels, b.shapes, -b.data, b.idxs)
            return a.__class__(q_labels, shapes, data, a.pattern, idxs, a.symmetry)

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
                    symmetry = b.symmetry
                elif isinstance(b, numbers.Number):
                    shs, qs, data, idxs = a.shapes, a.q_labels, a.data * b, a.idxs
                    out_pattern = a.pattern
                    symmetry = a.symmetry
                else:
                    c = self._tensordot(a, b, axes=([-1], [0]))
                    shs, qs, data, idxs = c.shapes, c.q_labels, c.data, c.idxs
                    out_pattern = _contract_patterns(a.pattern, b.pattern, [a.ndim-1], [0])
                    symmetry = a.symmetry
            elif ufunc.__name__ in ["multiply", "divide", "true_divide"]:
                a, b = inputs
                if isinstance(a, numbers.Number):
                    shs, qs, data, idxs = b.shapes, b.q_labels, getattr(
                        ufunc, method)(a, b.data), b.idxs
                    out_pattern = b.pattern
                    symmetry = b.symmetry
                elif isinstance(b, numbers.Number):
                    shs, qs, data, idxs = a.shapes, a.q_labels, getattr(
                        ufunc, method)(a.data, b), a.idxs
                    out_pattern = a.pattern
                    symmetry = a.symmetry
                else:
                    return NotImplemented
            elif len(inputs) == 1:
                a = inputs[0]
                shs, qs, data, idxs = a.shapes, a.q_labels, getattr(
                    ufunc, method)(a.data), a.idxs
                out_pattern = a.pattern
                symmetry = a.symmetry
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
            out[0].symmetry = symmetry
        return FlatFermionTensor(q_labels=qs, shapes=shs, data=data, pattern=out_pattern, idxs=idxs, symmetry=symmetry)

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
        backend = get_backend(a.symmetry)
        q_labels, shapes, data, idxs = backend.flat_fermion_tensor.tensordot(
                    a.q_labels, a.shapes, a.data, a.idxs,
                    b.q_labels, b.shapes, b.data, b.idxs,
                    idxa, idxb)
        out_pattern = _contract_patterns(a.pattern, b.pattern, idxa, idxb)
        return a.__class__(q_labels, shapes, data, out_pattern, idxs, a.symmetry)

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
            backend = get_backend(a.symmetry)
            backend.flat_fermion_tensor.transpose(a.q_labels, a.shapes, a.data, a.idxs, axes, data)
            pattern = "".join([a.pattern[ix] for ix in axes])
            return a.__class__(a.q_labels[:,axes], a.shapes[:,axes], \
                               data, pattern, a.idxs, a.symmetry)

    def tensor_svd(self, left_idx, right_idx=None, qpn_partition=None, qpn_cutoff_func=None, **opts):
        return flat_svd(self, left_idx, right_idx=right_idx, qpn_partition=qpn_partition, qpn_cutoff_func=qpn_cutoff_func, **opts)

    def to_exponential(self, x):
        return get_flat_exponential(self, x)
