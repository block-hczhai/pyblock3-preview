
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
from .core import SparseTensor, SubTensor, _sparse_tensor_numpy_func_impls
from .flat import FlatSparseTensor, flat_sparse_skeleton, _flat_sparse_tensor_numpy_func_impls
from .symmetry import SZ
import numbers
from block3 import flat_fermion_tensor, flat_sparse_tensor, VectorMapUIntUInt

def flat_fermion_skeleton(bond_infos, dq=None):
    fdq = dq.to_flat() if dq is not None else SZ(0).to_flat()
    return flat_fermion_tensor.skeleton(VectorMapUIntUInt(bond_infos), fdq)

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

def compute_phase(q_labels, axes):
    nlist = [qlab.n for qlab in q_labels]
    counted = []
    phase = 1
    for x in axes:
        parity = sum([nlist[i] for i in range(x) if i not in counted]) * nlist[x]
        phase *= (-1) ** parity
        counted.append(x)
    return phase

def _unpack_sparse_tensor(data, imap, ax, blklist):
    for q_label, (isec, ishape) in imap.items():
        ist, ied = isec
        if ax==0:
            tmp = data[ist:ied].reshape(ishape+(-1,))
        else:
            tmp = data[:,ist:ied].reshape((-1,)+ishape)
        blklist.append(SubTensor(tmp, q_labels=q_label))

def _unpack_flat_tensor(data, imap, ax, adata, qlst, shapelst):
    for q_label, (isec, ishape) in imap.items():
        qlst.append([SZ.to_flat(iq) for iq in q_label])
        ist, ied = isec
        if ax==0:
            tmp = data[ist:ied]
            shape = tuple(ishape) + (tmp.size//np.prod(ishape), )
        else:
            tmp = data[:,ist:ied]
            shape = (tmp.size//np.prod(ishape), ) + tuple(ishape)
        adata.append(tmp.ravel())
        shapelst.append(shape)

def _pack_sparse_tensor(tsr, s_label, ax):
    parity = tsr.parity
    u_map = {}
    v_map = {}
    left_offset = 0
    right_offset = 0
    for iblk in tsr.blocks:
        q_label_left = iblk.q_labels[:ax]
        q_label_right = iblk.q_labels[ax:]
        left_parity = np.mod(sum([iq.n for iq in q_label_left]) ,2)
        right_parity = np.mod(sum([iq.n for iq in q_label_right]) + parity ,2)
        q_label_left += (SZ(left_parity), )
        q_label_right = (SZ(right_parity), ) + q_label_right
        q_label_mid = (SZ(left_parity), SZ(right_parity))
        if q_label_mid != s_label: continue
        len_left = np.prod(iblk.shape[:ax])
        len_right = np.prod(iblk.shape[ax:])
        if q_label_left not in u_map.keys():
            u_map[q_label_left] = ((left_offset, left_offset+len_left), iblk.shape[:ax])
            left_offset += len_left
        if q_label_right not in v_map.keys():
            v_map[q_label_right] = ((right_offset, right_offset+len_right), iblk.shape[ax:])
            right_offset += len_right

    data = np.zeros((left_offset, right_offset), dtype=tsr.dtype)
    for iblk in tsr.blocks:
        q_label_left = iblk.q_labels[:ax]
        q_label_right = iblk.q_labels[ax:]
        left_parity = np.mod(sum([iq.n for iq in q_label_left]) ,2)
        right_parity = np.mod(sum([iq.n for iq in q_label_right]) + parity ,2)
        q_label_left += (SZ(left_parity), )
        q_label_right = (SZ(right_parity), ) + q_label_right
        q_label_mid = (SZ(left_parity), SZ(right_parity))
        if q_label_mid != s_label: continue
        ist, ied = u_map[q_label_left][0]
        jst, jed = v_map[q_label_right][0]
        data[ist:ied,jst:jed] = np.asarray(iblk).reshape(ied-ist, jed-jst)
    return data, u_map, v_map

def _pack_flat_tensor(tsr, s_label, ax):
    nblks, ndim = tsr.q_labels.shape
    parity = tsr.parity
    u_map = {}
    v_map = {}
    left_offset = 0
    right_offset = 0
    for iblk in range(nblks):
        q_label_left = tuple(SZ.from_flat(iq) for iq in tsr.q_labels[iblk][:ax])
        q_label_right = tuple(SZ.from_flat(iq) for iq in tsr.q_labels[iblk][ax:])
        left_parity = np.mod(sum([iq.n for iq in q_label_left]) ,2)
        right_parity = np.mod(sum([iq.n for iq in q_label_right]) + parity ,2)
        q_label_left += (SZ(left_parity), )
        q_label_right = (SZ(right_parity), ) + q_label_right
        q_label_mid = (SZ(left_parity), SZ(right_parity))
        if q_label_mid != s_label: continue
        len_left = int(np.prod(tsr.shapes[iblk][:ax]))
        len_right = int(np.prod(tsr.shapes[iblk][ax:]))
        if q_label_left not in u_map.keys():
            u_map[q_label_left] = ((left_offset, left_offset+len_left), tsr.shapes[iblk][:ax])
            left_offset += len_left
        if q_label_right not in v_map.keys():
            v_map[q_label_right] = ((right_offset, right_offset+len_right), tsr.shapes[iblk][ax:])
            right_offset += len_right

    data = np.zeros((left_offset, right_offset), dtype=tsr.dtype)
    for iblk in range(nblks):
        q_label_left = tuple(SZ.from_flat(iq) for iq in tsr.q_labels[iblk][:ax])
        q_label_right = tuple(SZ.from_flat(iq) for iq in tsr.q_labels[iblk][ax:])
        left_parity = np.mod(sum([iq.n for iq in q_label_left]) ,2)
        right_parity = np.mod(sum([iq.n for iq in q_label_right]) + parity ,2)
        q_label_left += (SZ(left_parity), )
        q_label_right = (SZ(right_parity), ) + q_label_right
        q_label_mid = (SZ(left_parity), SZ(right_parity))
        if q_label_mid != s_label: continue
        ist, ied = u_map[q_label_left][0]
        jst, jed = v_map[q_label_right][0]
        blkst, blked = tsr.idxs[iblk], tsr.idxs[iblk+1]
        data[ist:ied,jst:jed] = np.asarray(tsr.data[blkst:blked]).reshape(ied-ist, jed-jst)
    return data, u_map, v_map

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

def _absorb_svd(u, s, v, absorb, s_label, umap, vmap):
    flip_parity = (s_label[0]!=s_label[1])
    if absorb == -1:
        u = u * s.reshape((1, -1))
        if flip_parity:
            qlst = list(umap.keys())
            for qlab in qlst:
                val = umap.pop(qlab)
                newqlab = qlab[:-1] + (s_label[-1], )
                umap[newqlab] = val
    elif absorb == 1:
        v = v * s.reshape((-1, 1))
        if flip_parity:
            qlst = list(vmap.keys())
            for qlab in qlst:
                val = vmap.pop(qlab)
                newqlab = (s_label[0], ) + qlab[1:]
                vmap[newqlab] = val
    else:
        s **= 0.5
        u = u * s.reshape((1, -1))
        v = v * s.reshape((-1, 1))
        if flip_parity:
            qlst = list(umap.keys())
            for qlab in qlst:
                val = umap.pop(qlab)
                newqlab = qlab[:-1] + (s_label[-1], )
                umap[newqlab] = val
    return u, None, v, umap, vmap

def _run_sparse_fermion_svd(tsr, ax=2, **svd_opts):
    full_matrices = svd_opts.pop("full_matrices", False)
    absorb = svd_opts.pop("absorb", 0)

    s_labels = [(SZ(0), SZ(0)), (SZ(1), SZ(1))]
    ublk, sblk, vblk = [],[],[]
    for s_label in s_labels:
        data, umap, vmap = _pack_sparse_tensor(tsr, s_label, ax)
        if data.size==0:
            continue
        u, s, v = np.linalg.svd(data, full_matrices=full_matrices)
        u, s, v = _trim_and_renorm_SVD(u, s, v, **svd_opts)
        if absorb is None:
            sblk.append(SubTensor(np.diag(s), q_labels=s_label))
        else:
            u, s, v, umap, vmap = _absorb_svd(u, s, v, absorb, s_label, umap, vmap)

        _unpack_sparse_tensor(u, umap, 0, ublk)
        _unpack_sparse_tensor(v, vmap, 1, vblk)

    u = SparseFermionTensor(blocks=ublk)
    if absorb is None:
        s = SparseFermionTensor(blocks=sblk)
    v = SparseFermionTensor(blocks=vblk)
    return u, s, v

def _run_flat_fermion_svd(tsr, ax=2, **svd_opts):
    full_matrices = svd_opts.pop("full_matrices", False)
    absorb = svd_opts.pop("absorb", 0)
    nblk, ndim = tsr.q_labels.shape
    s_labels = [(SZ(0), SZ(0)), (SZ(1), SZ(1))]
    udata, sdata, vdata = [],[],[]
    uq, sq, vq = [],[],[]
    ushapes, sshapes, vshapes = [],[],[]
    for s_label in s_labels:
        data, umap, vmap = _pack_flat_tensor(tsr, s_label, ax)
        if data.size==0:
            continue
        u, s, v = np.linalg.svd(data, full_matrices=full_matrices)
        u, s, v = _trim_and_renorm_SVD(u, s, v, **svd_opts)
        if absorb is None:
            s = np.diag(s)
            sq.append([SZ.to_flat(iq) for iq in s_label])
            sshapes.append(s.shape)
            sdata.append(s.ravel())
        else:
            u, s, v, umap, vmap = _absorb_svd(u, s, v, absorb, s_label, umap, vmap)
        _unpack_flat_tensor(u, umap, 0, udata, uq, ushapes)
        _unpack_flat_tensor(v, vmap, 1, vdata, vq, vshapes)

    if absorb is None:
        sq = np.asarray(sq, dtype=np.uint32)
        sshapes = np.asarray(sshapes, dtype=np.uint32)
        sdata = np.concatenate(sdata)
        s = FlatFermionTensor(sq, sshapes, sdata)

    uq = np.asarray(uq, dtype=np.uint32)
    ushapes = np.asarray(ushapes, dtype=np.uint32)
    udata = np.concatenate(udata)

    vq = np.asarray(vq, dtype=np.uint32)
    vshapes = np.asarray(vshapes, dtype=np.uint32)
    vdata = np.concatenate(vdata)
    u = FlatFermionTensor(uq, ushapes, udata)
    v = FlatFermionTensor(vq, vshapes, vdata)
    return u, s ,v

def fermion_tensor_svd(tsr, left_idx, right_idx=None, **opts):
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

def matricize(tsr, row_idx):
    col_idx = tuple(i for i in range(tsr.ndim) if i not in row_idx)
    col_dic = {}
    row_dic = {}
    col_off = 0
    row_off = 0
    for iblk in tsr:
        row_q = tuple(iblk.q_labels[i] for i in row_idx)
        col_q = tuple(iblk.q_labels[i] for i in col_idx)
        row_sp = tuple(iblk.shape[i] for i in row_idx)
        col_sp = tuple(iblk.shape[i] for i in col_idx)
        if row_q not in row_dic:
            row_dic[row_q] = (row_off, row_off+np.prod(row_sp), row_sp)
            row_off += np.prod(row_sp)
        if col_q not in col_dic:
            col_dic[col_q] = (col_off, col_off+np.prod(col_sp), col_sp)
            col_off += np.prod(col_sp)
    row_size = max([val[1] for val in row_dic.values()])
    col_size = max([val[1] for val in col_dic.values()])
    mat = np.zeros([row_size, col_size], dtype=tsr.dtype)
    for iblk in tsr:
        row_q = tuple(iblk.q_labels[i] for i in row_idx)
        col_q = tuple(iblk.q_labels[i] for i in col_idx)
        row_sp = tuple(iblk.shape[i] for i in row_idx)
        col_sp = tuple(iblk.shape[i] for i in col_idx)
        ist, ied = row_dic[row_q][:2]
        jst, jed = col_dic[col_q][:2]
        mat[ist:ied, jst:jed] = iblk.reshape(ied-ist, jed-jst)
    return mat, row_dic, col_dic

NEW_METHODS = [np.transpose, np.tensordot]

_sparse_fermion_tensor_numpy_func_impls = _sparse_tensor_numpy_func_impls.copy()
[_sparse_fermion_tensor_numpy_func_impls.pop(key) for key in NEW_METHODS]
_numpy_func_impls = _sparse_fermion_tensor_numpy_func_impls

class SparseFermionTensor(SparseTensor):

    def __init__(self, *args, **kwargs):
        SparseTensor.__init__(self, *args, **kwargs)
        self.check_sanity()

    @property
    def parity(self):
        return self.parity_per_block[0]

    @property
    def parity_per_block(self):
        parity_list = []
        for block in self.blocks:
            pval = sum([q_label.n for q_label in block.q_labels])
            parity_list.append(int(pval)%2)
        return parity_list

    def conj(self):
        blks = [iblk.conj() for iblk in self.blocks]
        return self.__class__(blocks=blks)

    def check_sanity(self):
        parity_uniq = np.unique(self.parity_per_block)
        if len(parity_uniq) >1:
            raise ValueError("both odd/even parity detected in blocks, \
                              this is prohibited")

    def _local_flip(self, axes):
        if isinstance(axes, int):
            ax = [axes]
        else:
            ax = list(axes)
        for i in range(self.n_blocks):
            block_parity = sum([self.blocks[i].q_labels[j].n for j in ax]) %2
            if block_parity == 1:
                self.blocks[i] *= -1

    def _global_flip(self):
        for i in range(self.n_blocks):
            self.blocks[i] *= -1

    def to_matrix(self, row_idx, return_dic=False):
        if return_dic:
            return matricize(self, row_idx)
        else:
            return matricize(self, row_idx)[0]

    def to_flat(self):
        return FlatFermionTensor.from_sparse(self)

    def _skeleton(bond_infos, dq=None):
        if dq is None:
            dq = SZ(0,0,0)
        if not isinstance(dq, SZ):
            raise TypeError("dq is not an instance of SZ class")
        it = np.zeros(tuple(len(i) for i in bond_infos), dtype=np.uint32)
        qsh = [sorted(i.items(), key=lambda x: x[0]) for i in bond_infos]
        q = [[k for k, v in i] for i in qsh]
        sh = [[v for k, v in i] for i in qsh]
        nit = np.nditer(it, flags=['multi_index'])
        for _ in nit:
            x = nit.multi_index
            ps = [iq[ix] for iq, ix in zip(q, x)]
            parity = np.mod(sum([iq[ix].n for iq, ix in zip(q, x)]), 2)
            if parity == dq.n:
                xqs = tuple(iq[ix] for iq, ix in zip(q, x))
                xsh = tuple(ish[ix] for ish, ix in zip(sh, x))
                yield xsh, xqs

    def tensor_svd(self, left_idx, right_idx=None, **svd_opts):
        return fermion_tensor_svd(self, left_idx, right_idx=right_idx, **svd_opts)

    @staticmethod
    def zeros(bond_infos, dq=None, dtype=float):
        """Create tensor from tuple of BondInfo with zero elements."""
        blocks = []
        for sh, qs in SparseFermionTensor._skeleton(bond_infos, dq=dq):
            blocks.append(SubTensor.zeros(shape=sh, q_labels=qs, dtype=dtype))
        return SparseFermionTensor(blocks=blocks)

    @staticmethod
    def ones(bond_infos, dq=None, dtype=float):
        """Create tensor from tuple of BondInfo with ones."""
        blocks = []
        for sh, qs in SparseFermionTensor._skeleton(bond_infos, dq=dq):
            blocks.append(SubTensor.ones(shape=sh, q_labels=qs, dtype=dtype))
        return SparseFermionTensor(blocks=blocks)

    @staticmethod
    def random(bond_infos, dq=None, dtype=float):
        """Create tensor from tuple of BondInfo with random elements."""
        blocks = []
        for sh, qs in SparseFermionTensor._skeleton(bond_infos, dq=dq):
            blocks.append(SubTensor.random(shape=sh, q_labels=qs, dtype=dtype))
        return SparseFermionTensor(blocks=blocks)

    @staticmethod
    def eye(bond_info):
        """Create tensor from BondInfo with Identity matrix."""
        blocks = []
        for sh, qs in SparseFermionTensor._skeleton((bond_info, bond_info)):
            blocks.append(SubTensor(reduced=np.eye(sh[0]), q_labels=qs))
        return SparseFermionTensor(blocks=blocks)

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
            out.blocks = blocks
        return self.__class__(blocks=blocks)

    @staticmethod
    def _hdot(a, b, out=None):
        """Horizontal contraction (contracting connected virtual dimensions)."""
        if isinstance(a, numbers.Number) or isinstance(b, numbers.Number):
            return np.multiply(a, b, out=out)

        assert isinstance(a, SparseFermionTensor) and isinstance(b, SparseFermionTensor)
        r = np.tensordot(a, b, axes=1)

        if out is not None:
            out.blocks = r.blocks

        return r

    @staticmethod
    def _pdot(a, b, out=None):
        """Vertical contraction (all middle/physical dims)."""
        if isinstance(a, numbers.Number) or isinstance(b, numbers.Number):
            return np.multiply(a, b, out=out)

        assert isinstance(a, SparseFermionTensor) and isinstance(b, SparseFermionTensor)

        assert a.ndim == b.ndim
        d = a.ndim - 2
        aidx = list(range(1, d + 1))
        bidx = list(range(1, d + 1))
        tr = (0, 2, 1, 3)

        r = np.tensordot(a, b, axes=(aidx, bidx)).transpose(axes=tr)

        if out is not None:
            out.blocks = r.blocks

        return r

    @staticmethod
    @implements(np.tensordot)
    def _tensordot(a, b, axes=2):
        if isinstance(axes, int):
            idxa, idxb = list(range(-axes, 0)), list(range(0, axes))
        else:
            idxa, idxb = axes
        idxa = [x if x >= 0 else a.ndim + x for x in idxa]
        idxb = [x if x >= 0 else b.ndim + x for x in idxb]
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
            ptca = [i.n for i in block_a.q_labels]
            phase_a = _compute_phase(ptca, idxa, 'right')
            ctrq = tuple(block_a.q_labels[id] for id in idxa)
            if ctrq in map_idx_b:
                outqa = tuple(block_a.q_labels[id] for id in out_idx_a)
                for block_b, outqb in map_idx_b[ctrq]:
                    ptcb = [i.n for i in block_b.q_labels]
                    phase_b = _compute_phase(ptcb, idxb, 'left')
                    phase = phase_a * phase_b
                    outq = outqa + outqb
                    mat = np.tensordot(np.asarray(block_a), np.asarray(
                        block_b), axes=(idxa, idxb)).view(block_a.__class__)
                    if outq not in blocks_map:
                        mat.q_labels = outq
                        blocks_map[outq] = mat * phase
                    else:
                        blocks_map[outq] += mat * phase

        return a.__class__(blocks=list(blocks_map.values()))

    @staticmethod
    @implements(np.transpose)
    def _transpose(a, axes=None):
        phase = [compute_phase(block.q_labels, axes) for block in a.blocks]
        blocks = [np.transpose(block, axes=axes)*phase[ibk] for ibk, block in enumerate(a.blocks)]
        return a.__class__(blocks=blocks)

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
        if out is not None and not all(isinstance(x, SparseFermionTensor) for x in out):
            return NotImplemented
        if any(isinstance(x, SparseFermionTensor) for x in inputs):
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
            out.blocks = blocks
        return self.__class__(blocks=blocks)

_flat_fermion_tensor_numpy_func_impls = _flat_sparse_tensor_numpy_func_impls.copy()
[_flat_fermion_tensor_numpy_func_impls.pop(key) for key in NEW_METHODS]
_numpy_func_impls = _flat_fermion_tensor_numpy_func_impls

class FlatFermionTensor(FlatSparseTensor):

    def __init__(self, *args, **kwargs):
        FlatSparseTensor.__init__(self, *args, **kwargs)
        self.check_sanity()

    @property
    def parity(self):
        self.check_sanity()
        parity_uniq = np.unique(self.parity_per_block)
        return parity_uniq[0]

    @property
    def parity_per_block(self):
        parity_list = []
        for q_label in self.q_labels:
            pval = sum([SZ.from_flat(qlab).n for qlab in q_label])
            parity_list.append(int(pval)%2)
        return parity_list

    def conj(self):
        return self.__class__(self.q_labels, self.shapes, self.data.conj(), idxs=self.idxs)

    @property
    def shape(self):
        return tuple(np.amax(self.shapes, axis=0))

    def check_sanity(self):
        parity_uniq = np.unique(self.parity_per_block)
        if len(parity_uniq) >1:
            raise ValueError("both odd/even parity detected in blocks, \
                              this is prohibited")

    def _local_flip(self, axes):
        if isinstance(axes, int):
            ax = [axes]
        else:
            ax = list(axes)
        idx = self.idxs
        for i in range(len(self.q_labels)):
            block_parity = sum([SZ.from_flat(self.q_labels[i][j]).n for j in ax]) % 2
            if block_parity == 1:
                self.data[idx[i]:idx[i+1]] *=-1

    def _global_flip(self):
        self.data *= -1

    def to_matrix(self, row_idx, return_dic=False):
        tsr = self.to_sparse()
        if return_dic:
            return matricize(tsr, row_idx)
        else:
            return matricize(tsr, row_idx)[0]

    def to_sparse(self):
        blocks = [None] * self.n_blocks
        for i in range(self.n_blocks):
            qs = tuple(map(SZ.from_flat, self.q_labels[i]))
            blocks[i] = SubTensor(
                self.data[self.idxs[i]:self.idxs[i + 1]].reshape(self.shapes[i]), q_labels=qs)
        return SparseFermionTensor(blocks=blocks)

    def tensor_svd(self, left_idx, **svd_opts):
        return fermion_tensor_svd(self, left_idx, **svd_opts)

    @staticmethod
    def get_zero():
        zu = np.zeros((0, 0), dtype=np.uint32)
        zd = np.zeros((0, ), dtype=float)
        iu = np.zeros((1, ), dtype=np.uint32)
        return FlatFermionTensor(zu, zu, zd, iu)

    @staticmethod
    def from_sparse(spt):
        ndim = spt.ndim
        n_blocks = spt.n_blocks
        shapes = np.zeros((n_blocks, ndim), dtype=np.uint32)
        q_labels = np.zeros((n_blocks, ndim), dtype=np.uint32)
        for i in range(n_blocks):
            shapes[i] = spt.blocks[i].shape
            q_labels[i] = list(map(SZ.to_flat, spt.blocks[i].q_labels))
        idxs = np.zeros((n_blocks + 1, ), dtype=np.uint32)
        idxs[1:] = np.cumsum(shapes.prod(axis=1))
        data = np.zeros((idxs[-1], ), dtype=spt.dtype)
        for i in range(n_blocks):
            data[idxs[i]:idxs[i + 1]] = spt.blocks[i].flatten()
        return FlatFermionTensor(q_labels, shapes, data, idxs)


    @staticmethod
    def zeros(bond_infos, dq=None, dtype=float):
        """Create tensor from tuple of BondInfo with zero elements."""
        qs, shs, idxs = flat_fermion_skeleton(bond_infos, dq)
        data = np.zeros((idxs[-1], ), dtype=dtype)
        return FlatFermionTensor(qs, shs, data, idxs)

    @staticmethod
    def ones(bond_infos, dq=None, dtype=float):
        """Create tensor from tuple of BondInfo with ones."""
        qs, shs, idxs = flat_fermion_skeleton(bond_infos, dq)
        data = np.ones((idxs[-1], ), dtype=dtype)
        return FlatFermionTensor(qs, shs, data, idxs)

    @staticmethod
    def random(bond_infos, dq=None, dtype=float):
        """Create tensor from tuple of BondInfo with random elements."""
        qs, shs, idxs = flat_fermion_skeleton(bond_infos, dq)
        if dtype == float:
            data = np.random.random((idxs[-1], ))
        elif dtype == complex:
            data = np.random.random(
                (idxs[-1], )) + np.random.random((idxs[-1], )) * 1j
        else:
            return NotImplementedError('dtype %r not supported!' % dtype)
        return FlatSparseTensor(qs, shs, data, idxs)

    @staticmethod
    def eye(bond_info):
        """Create tensor from BondInfo with Identity matrix."""
        blocks = []
        for sh, qs in SparseFermionTensor._skeleton((bond_info, bond_info)):
            blocks.append(SubTensor(reduced=np.eye(sh[0]), q_labels=qs))
        return SparseFermionTensor(blocks=blocks).to_flat()

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
        return FlatFermionTensor(q_labels=qs, shapes=shs, data=data, idxs=idxs)


    @staticmethod
    def _hdot(a, b, out=None):
        """Horizontal contraction (contracting connected virtual dimensions)."""
        if isinstance(a, numbers.Number) or isinstance(b, numbers.Number):
            return np.multiply(a, b, out=out)

        assert isinstance(a, FlatFermionTensor) and isinstance(
            b, FlatFermionTensor)
        r = np.tensordot(a, b, axes=1)

        if out is not None:
            out.q_labels[...] = r.q_labels
            out.shapes[...] = r.shapes
            out.data[...] = r.data
            out.idxs[...] = r.idxs

        return r


    @staticmethod
    def _pdot(a, b, out=None):
        """Vertical contraction (all middle/physical dims)."""
        if isinstance(a, numbers.Number) or isinstance(b, numbers.Number):
            return np.multiply(a, b, out=out)

        assert isinstance(a, FlatFermionTensor) and isinstance(
            b, FlatFermionTensor)

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
            return a.__class__(a.q_labels[:,axes], a.shapes[:,axes], \
                               data, a.idxs)

    def permute(self, axes=None):
        if axes is None:
            axes = np.arange(self.ndim)[::-1]
        if self.n_blocks == 0:
            return self
        else:
            axes = np.array(axes, dtype=np.int32)
            data = np.zeros_like(self.data)
            flat_sparse_tensor.transpose(self.shapes, self.data, self.idxs, axes, data)
            return self.__class__(self.q_labels[:, axes], self.shapes[:, axes], data, self.idxs)


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
        return a.__class__(*flat_fermion_tensor.tensordot(
            a.q_labels, a.shapes, a.data, a.idxs,
            b.q_labels, b.shapes, b.data, b.idxs,
            idxa, idxb))

    @staticmethod
    @implements(np.linalg.norm)
    def _norm(a):
        return np.linalg.norm(a.data)

    @staticmethod
    @implements(np.reshape)
    def _reshape(a, newshape, *args, **kwargs):
        return np.reshape(a.data, newshape, *args, **kwargs)

    def reshape(self, newshape, *args, **kwargs):
        return np.reshape(self, newshape, *args, **kwargs)

    def astype(self, dtype, *args, **kwargs):
        return self.__class__(self.q_labels, self.shapes, \
                              self.data.astype(dtype, *args, **kwargs), \
                              self.idxs)
