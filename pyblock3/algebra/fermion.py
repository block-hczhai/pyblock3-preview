
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
import block3.sz as _block3
from .core import SparseTensor, SubTensor, _sparse_tensor_numpy_func_impls
from .flat import FlatSparseTensor, _flat_sparse_tensor_numpy_func_impls
from .fermion_symmetry import U1, Z4, Z2, U11, Z22
from .symmetry import BondInfo
from . import fermion_setting as setting

SVD_SCREENING = setting.SVD_SCREENING

def get_backend(symmetry):
    """Get the C++ backend for the input symmetry

    Parameters
    ----------
    symmetry : str or symmetry object
    """
    if not setting.DEFAULT_FERMION:
        return _block3
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
    elif key == "Z22":
        import block3.z22 as backend
    else:
        raise NotImplementedError("symmetry %s not supported"%key)
    return backend

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
    """Generate a default algebraic pattern
    """
    ndim = obj if isinstance(obj, int) else len(obj)
    pattern = "+" * (ndim-1) + '-'
    return pattern

def _flip_pattern(pattern):
    """Flip the algebraic pattern, eg, "+" -> "-" and "-" -> "+"
    """
    flip_dict = {"+":"-", "-":"+"}
    return "".join([flip_dict[ip] for ip in pattern])

def _contract_patterns(patterna, patternb, idxa, idxb):
    """Get the output pattern from contracting two input patterns. Return the axes that needs to be flipped for the second pattern if the two patterns does not fit directly
    """
    conc_a = "".join([patterna[ix] for ix in idxa])
    out_a = "".join([ip for ix, ip in enumerate(patterna) if ix not in idxa])
    conc_b = "".join([patternb[ix] for ix in idxb])
    out_b = "".join([ip for ix, ip in enumerate(patternb) if ix not in idxb])

    b_flip_axes = []
    if conc_a != _flip_pattern(conc_b):
        for ind, ixa, ixb in zip(idxb, conc_a, conc_b):
            if ixa == ixb:
                b_flip_axes.append(ind)
    return out_a + out_b, b_flip_axes

def _trim_singular_vals(s_data, cutoff, cutoff_mode, max_bond=None):
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
    if cutoff_mode in (1, 2):
        s = None
        if cutoff_mode == 1:
            n_chis = [np.sum(sblk>cutoff) for sblk in s_data]
        else:
            s = np.concatenate(s_data)
            smax = s.max()
            n_chis = [np.sum(sblk>cutoff*smax) for sblk in s_data]

        if max_bond is not None and max_bond>0:
            n_chi = np.sum(n_chis)
            extra_bonds = n_chi - max_bond
            if extra_bonds >0:
                if s is None:
                    s = np.concatenate(s_data)
                s_ind = np.argsort(s)
                ind_map = []
                for ix, sblk in enumerate(s_data):
                    ind_map += [ix,] * sblk.size
                for i in range(extra_bonds):
                    ind = s_ind[i+s.size-n_chi]
                    n_chis[ind_map[ind]] -= 1

    elif cutoff_mode in (3, 4, 5, 6):
        if cutoff_mode in (3, 4):
            p = 2
        else:
            p = 1

        target = cutoff

        s = np.concatenate(s_data) ** p
        if cutoff_mode in (4, 6):
            target *= np.sum(s)
        s_ind = np.argsort(s)
        s_sorted = np.cumsum(np.sort(s))
        ind_map = []
        for ix, sblk in enumerate(s_data):
            ind_map += [ix,] * sblk.size

        n_chis = [sblk.size for sblk in s_data]
        ncut = np.sum(s_sorted<=target)
        if max_bond is not None and max_bond>0:
            ncut = max(ncut, s.size-max_bond)
        for i in range(ncut):
            group_ind = ind_map[s_ind[i]]
            n_chis[group_ind] -= 1
    return n_chis

def _renorm_singular_vals(s_data, n_chis, renorm_power):
    """Find the normalization constant for ``s`` such that the new sum squared
    of the ``n_chi`` largest values equals the sum squared of all the old ones.
    """
    s_tot_keep = 0.0
    s_tot_lose = 0.0
    for sblk, n_chi in zip(s_data, n_chis):
        for i in range(sblk.size):
            s2 = sblk[i]**renorm_power
            if not np.isnan(s2):
                if i < n_chi:
                    s_tot_keep += s2
                else:
                    s_tot_lose += s2

    return ((s_tot_keep + s_tot_lose) / s_tot_keep)**(1 / renorm_power)

def _trim_and_renorm_SVD(s_data, uv_data, **svdopts):
    cutoff = svdopts.pop("cutoff", 1e-12)
    cutoff_mode = svdopts.pop("cutoff_mode", 3)
    max_bond = svdopts.pop("max_bond", None)
    renorm_power = svdopts.pop("renorm_power", 0)
    n_chis = _trim_singular_vals(s_data, cutoff, cutoff_mode, max_bond)
    n_chi = np.sum(n_chis)
    all_size = np.sum([iblk.size for iblk in s_data])
    if n_chi < all_size and renorm_power > 0:
        renorm_fac = _renorm_singular_vals(s_data, n_chis, renorm_power)
        for sblk in s_data:
            sblk *= renorm_fac

    for ix, n_chi in enumerate(n_chis):
        s_data[ix] = s_data[ix][:n_chi]
        U, VH = uv_data[ix]
        uv_data[ix] = (U[...,:n_chi], VH[:n_chi,...])

    return s_data, uv_data

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

def _index_partition(T, left_idx, right_idx):
    if right_idx is None:
        right_idx = [idim for idim in range(T.ndim) if idim not in left_idx]
    elif left_idx is None:
        left_idx = [idim for idim in range(T.ndim) if idim not in right_idx]
    neworder = list(left_idx) + list(right_idx)
    if neworder == list(range(T.ndim)):
        new_T = T
    else:
        new_T = T.transpose(neworder)
    return new_T, left_idx, right_idx

def _svd_preprocess(T, left_idx, right_idx, qpn_partition, absorb):
    cls = T.dq.__class__
    if qpn_partition is None:
        if absorb ==-1:
            qpn_partition = (T.dq, cls(0))
        else:
            qpn_partition = (cls(0), T.dq)
    else:
        assert np.sum(qpn_partition) == T.dq

    new_T, left_idx, right_idx = _index_partition(T, left_idx, right_idx)
    return new_T, left_idx, right_idx, qpn_partition, cls

def flat_svd(T, left_idx, right_idx=None, qpn_partition=None, **opts):
    absorb = opts.pop("absorb", 0)
    new_T, left_idx, right_idx, qpn_partition, symmetry = _svd_preprocess(T, left_idx, right_idx, qpn_partition, absorb)
    if left_idx is not None and len(left_idx) == T.ndim:
        raise NotImplementedError
    elif right_idx is not None and len(right_idx) == T.ndim:
        raise NotImplementedError

    split_ax = len(left_idx)
    left_q = symmetry._compute(new_T.pattern[:split_ax], new_T.q_labels[:,:split_ax], offset=("-", qpn_partition[0]))
    right_q = symmetry._compute(new_T.pattern[split_ax:], new_T.q_labels[:,split_ax:], offset=("-", qpn_partition[1]), neg=True)

    aux_q = list(set(np.unique(left_q)) & set(np.unique(right_q)))

    full_left = np.hstack([new_T.q_labels[:,:split_ax], left_q.reshape(-1,1)])
    full_right = np.hstack([left_q.reshape(-1,1), new_T.q_labels[:,split_ax:]])
    full = [(tuple(il), tuple(ir)) for il, ir in zip(full_left, full_right)]

    row_shapes = np.prod(new_T.shapes[:,:split_ax], axis=1, dtype=int)
    col_shapes = np.prod(new_T.shapes[:,split_ax:], axis=1, dtype=int)
    s_data =[]
    uv_data = []
    all_maps = []
    for slab in aux_q:
        blocks = np.where(left_q == slab)[0]
        row_map = {}
        col_map = {}
        row_len = 0
        col_len = 0
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
        ind = s > SVD_SCREENING
        s = s[ind]
        if s.size ==0:
            continue
        u = u[:,ind]
        v = v[ind,:]
        s_data.append(s)
        uv_data.append([u,v])
        all_maps.append([slab, row_map, col_map])

    s_data, uv_data = _trim_and_renorm_SVD(s_data, uv_data, **opts)

    if absorb is not None:
        for iblk in range(len(uv_data)):
            s = s_data[iblk]
            if s.size==0:
                continue
            u, v = uv_data[iblk]
            u, s, v = _absorb_svd(u, s, v, absorb)
            uv_data[iblk] = (u, v)
            s_data[iblk] = s

    udata = []
    vdata = []
    sdata = []

    qu = []
    qv = []
    qs = []
    shu = []
    shv = []
    shs = []

    for s, (u, v), (slab, row_map, col_map) in zip(s_data, uv_data, all_maps):
        if u.size==0:
            continue
        if absorb is None:
            s = np.diag(s)
            shs.append(s.shape)
            sdata.append(s.ravel())
            qs.append([slab, slab])

        for lq, (lst, led, lsh) in row_map.items():
            udata.append(u[lst:led].ravel())
            qu.append(lq)
            shu.append(tuple(lsh)+(u.shape[-1],))

        for rq, (rst, red, rsh) in col_map.items():
            vdata.append(v[:,rst:red].ravel())
            qv.append(rq)
            shv.append((v.shape[0],)+tuple(rsh))

    if absorb is None:
        qs = np.asarray(qs, dtype=np.uint32)
        shs = np.asarray(shs, dtype=np.uint32)
        sdata = np.concatenate(sdata)
        s = T.__class__(qs, shs, sdata, pattern="+-", symmetry=T.symmetry)
    else:
        s = None

    qu = np.asarray(qu, dtype=np.uint32)
    shu = np.asarray(shu, dtype=np.uint32)
    udata = np.concatenate(udata)

    qv = np.asarray(qv, dtype=np.uint32)
    shv = np.asarray(shv, dtype=np.uint32)
    vdata = np.concatenate(vdata)
    u = T.__class__(qu, shu, udata, pattern=new_T.pattern[:split_ax]+"-", symmetry=T.symmetry)
    v = T.__class__(qv, shv, vdata, pattern="+"+new_T.pattern[split_ax:], symmetry=T.symmetry)
    return u, s, v

def sparse_svd(T, left_idx, right_idx=None, qpn_partition=None, **opts):
    absorb = opts.pop("absorb", 0)
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
        if left_q not in data_map:
            data_map[left_q] = []
        data_map[left_q].append(iblk)

    s_data =[]
    uv_data = []
    all_maps = []

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
        ind = s > SVD_SCREENING
        s = s[ind]
        if s.size ==0:
            continue
        u = u[:,ind]
        v = v[ind,:]
        s_data.append(s)
        uv_data.append([u,v])
        all_maps.append([slab, row_map, col_map])

    s_data, uv_data = _trim_and_renorm_SVD(s_data, uv_data, **opts)

    if absorb is not None:
        for iblk in range(len(uv_data)):
            s = s_data[iblk]
            if s.size==0:
                continue
            u, v = uv_data[iblk]
            u, s, v = _absorb_svd(u, s, v, absorb)
            uv_data[iblk] = (u, v)
            s_data[iblk] = s

    ublocks = []
    vblocks = []
    sblocks = []

    for s, (u, v), (slab, row_map, col_map) in zip(s_data, uv_data, all_maps):
        if u.size==0:
            continue
        if absorb is None:
            s = np.diag(s)
            sblocks.append(SubTensor(reduced=s, q_labels=(slab,)*2))

        for lq, (lst, led, lsh) in row_map.items():
            ublocks.append(SubTensor(reduced=u[lst:led].reshape(tuple(lsh)+(-1,)), q_labels=lq))

        for rq, (rst, red, rsh) in col_map.items():
            vblocks.append(SubTensor(reduced=v[:,rst:red].reshape((-1,)+tuple(rsh)), q_labels=rq))

    if absorb is None:
        s = T.__class__(blocks=sblocks, pattern="+-")
    u = T.__class__(blocks=ublocks, pattern=new_T.pattern[:split_ax]+"-")
    v = T.__class__(blocks=vblocks, pattern="+"+new_T.pattern[split_ax:])
    return u, s, v

def sparse_qr(T, left_idx, right_idx=None, mod="qr"):
    assert mod in ["qr", "lq"]
    new_T, left_idx, right_idx = _index_partition(T, left_idx, right_idx)
    if len(left_idx) == T.ndim or len(right_idx)==T.ndim:
        raise NotImplementedError
    symmetry = T.dq.__class__
    dq = {"lq": symmetry(0),
          "qr": T.dq}[mod]
    split_ax = len(left_idx)
    left_pattern = new_T.pattern[:split_ax]
    data_map = {}
    for iblk in new_T.blocks:
        left_q = symmetry._compute(left_pattern, iblk.q_labels[:split_ax], offset=("-", dq))
        if left_q not in data_map:
            data_map[left_q] = []
        data_map[left_q].append(iblk)
    qblocks = []
    rblocks = []
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
        if mod=="qr":
            q, r = np.linalg.qr(data)
        else:
            r, q = np.linalg.qr(data.T)
            q, r = q.T, r.T

        for lq, (lst, led, lsh) in row_map.items():
            qblocks.append(SubTensor(reduced=q[lst:led].reshape(tuple(lsh)+(-1,)), q_labels=lq))

        for rq, (rst, red, rsh) in col_map.items():
            rblocks.append(SubTensor(reduced=r[:,rst:red].reshape((-1,)+tuple(rsh)), q_labels=rq))
    q = T.__class__(blocks=qblocks, pattern=new_T.pattern[:split_ax]+"-")
    r = T.__class__(blocks=rblocks, pattern="+"+new_T.pattern[split_ax:])
    return q, r

def flat_qr(T, left_idx, right_idx=None, mod="qr"):
    assert mod in ["qr", "lq"]
    new_T, left_idx, right_idx = _index_partition(T, left_idx, right_idx)
    if len(left_idx) == T.ndim or len(right_idx)==T.ndim:
        raise NotImplementedError
    symmetry = T.dq.__class__
    dq = {"lq": symmetry(0),
          "qr": T.dq}[mod]
    split_ax = len(left_idx)
    left_q = symmetry._compute(new_T.pattern[:split_ax], new_T.q_labels[:,:split_ax], offset=("-", dq))
    aux_q = list(set(np.unique(left_q)) )
    full_left = np.hstack([new_T.q_labels[:,:split_ax], left_q.reshape(-1,1)])
    full_right = np.hstack([left_q.reshape(-1,1), new_T.q_labels[:,split_ax:]])
    full = [(tuple(il), tuple(ir)) for il, ir in zip(full_left, full_right)]

    qdata = []
    rdata = []
    qq = []
    qr = []
    shq = []
    shr = []
    row_shapes = np.prod(new_T.shapes[:,:split_ax], axis=1, dtype=int)
    col_shapes = np.prod(new_T.shapes[:,split_ax:], axis=1, dtype=int)
    for slab in aux_q:
        blocks = np.where(left_q == slab)[0]
        row_map = {}
        col_map = {}
        row_len = 0
        col_len = 0
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
        if data.size == 0: continue
        for (ist, ied, jst, jed), val in alldatas.items():
            data[ist:ied,jst:jed] = val

        if mod=="qr":
            q, r = np.linalg.qr(data)
        else:
            r, q = np.linalg.qr(data.T)
            q, r = q.T, r.T

        for lq, (lst, led, lsh) in row_map.items():
            qdata.append(q[lst:led].ravel())
            qq.append(lq)
            shq.append(tuple(lsh)+(q.shape[-1],))

        for rq, (rst, red, rsh) in col_map.items():
            rdata.append(r[:,rst:red].ravel())
            qr.append(rq)
            shr.append((r.shape[0],)+tuple(rsh))

    qq = np.asarray(qq, dtype=np.uint32)
    shq = np.asarray(shq, dtype=np.uint32)
    qdata = np.concatenate(qdata)

    qr = np.asarray(qr, dtype=np.uint32)
    shr = np.asarray(shr, dtype=np.uint32)
    rdata = np.concatenate(rdata)
    q = T.__class__(qq, shq, qdata, pattern=new_T.pattern[:split_ax]+"-", symmetry=T.symmetry)
    r = T.__class__(qr, shr, rdata, pattern="+"+new_T.pattern[split_ax:], symmetry=T.symmetry)
    return q, r

def flat_qr_fast(T, left_idx, right_idx=None, mod="qr"):
    assert mod in ["qr", "lq"]
    new_T, left_idx, right_idx = _index_partition(T, left_idx, right_idx)
    if len(left_idx) == T.ndim or len(right_idx)==T.ndim:
        raise NotImplementedError
    split_ax = len(left_idx)
    backend = get_backend(T.symmetry)
    qq, shq, qdata, qidxs, qr, shr, rdata, ridxs = backend.flat_fermion_tensor.tensor_qr(
        new_T.q_labels, new_T.shapes, new_T.data, new_T.idxs, split_ax, new_T.pattern, mod == "qr")
    q = T.__class__(qq, shq, qdata, pattern=new_T.pattern[:split_ax]+"-", idxs=qidxs, symmetry=T.symmetry)
    r = T.__class__(qr, shr, rdata, pattern="+"+new_T.pattern[split_ax:], idxs=ridxs, symmetry=T.symmetry)
    return q, r

def _adjust_block(block, flip_axes):
    if len(flip_axes)==0:
        return block
    else:
        new_q_labels = list(block.q_labels)
        for ix in flip_axes:
            new_q_labels[ix] = - block.q_labels[ix]
        new_block = SubTensor(reduced=np.asarray(block), q_labels=tuple(new_q_labels))
        return new_block

def _adjust_q_labels(symmetry, q_labels, flip_axes):
    if len(flip_axes)==0:
        return q_labels
    else:
        new_q_labels = q_labels.copy()
        for ix in flip_axes:
            new_q_labels[:,ix] = symmetry.flip_flat(q_labels[:,ix])
        return new_q_labels

def compute_phase(q_labels, axes, direction="left"):
    if not setting.DEFAULT_FERMION: return 1
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

def eye(bond_info, flat=None):
    """Create tensor from BondInfo with Identity matrix."""
    flat = setting.dispatch_settings(flat=flat)
    blocks = []
    for sh, qs in SparseFermionTensor._skeleton((bond_info, bond_info)):
        blocks.append(SubTensor(reduced=np.eye(sh[0]), q_labels=qs))
    T = SparseFermionTensor(blocks=blocks, pattern="+-")
    if flat:
        T = T.to_flat()
    return T

class SparseFermionTensor(SparseTensor):

    def __init__(self, blocks=None, pattern=None):
        self.blocks = blocks if blocks is not None else []
        if pattern is None:
            pattern = _gen_default_pattern(self.ndim)
        self._pattern = pattern
        shapes = [iblk.shape for iblk in self]
        self._shape = tuple(np.sum(shapes, axis=0))

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
        return self.__class__(blocks=blocks, pattern=_flip_pattern(self.pattern[::-1]))

    @property
    def shape(self):
        return self._shape

    @staticmethod
    @implements(np.copy)
    def _copy(x):
        return x.__class__(blocks=[b.copy() for b in x.blocks], pattern=x.pattern)

    def copy(self):
        return np.copy(self)

    @property
    def parity(self):
        return self.dq.parity

    def get_bond_info(self, ax):
        bond = dict()
        ipattern = self.pattern[ax]
        for iblk in self:
            q = iblk.q_labels[ax]
            dim = iblk.shape[ax]
            if ipattern == "+":
                bond.update({q:dim})
            else:
                bond.update({-q:dim})
        return BondInfo(bond)

    def conj(self):
        blks = [iblk.conj() for iblk in self.blocks]
        return self.__class__(blocks=blks, pattern=self.pattern)

    def _local_flip(self, axes):
        if not setting.DEFAULT_FERMION: return
        if isinstance(axes, int):
            axes = [axes]
        else:
            axes = list(axes)
        for blk in self.blocks:
            block_parity = np.add.reduce([blk.q_labels[j] for j in axes]).parity
            if block_parity == 1:
                blk *= -1

    def _global_flip(self):
        if not setting.DEFAULT_FERMION: return
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
            flip_axes = [ix for ix in range(b.ndim) if a.pattern[ix]!=b.pattern[ix]]

            blocks_map = {block.q_labels: block for block in a.blocks}
            for iblock in b.blocks:
                block = _adjust_block(iblock, flip_axes)
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
            flip_axes = [ix for ix in range(b.ndim) if a.pattern[ix]!=b.pattern[ix]]
            blocks_map = {block.q_labels: block for block in a.blocks}
            for iblock in b.blocks:
                block = _adjust_block(iblock, flip_axes)
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
                out_pattern, _ = _contract_patterns(a.pattern, b.pattern, [a.ndim-1], [0])
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

        out_pattern, b_flip_axes = _contract_patterns(a.pattern, b.pattern, idxa, idxb)
        out_idx_a = sorted(list(set(range(0, a.ndim)) - set(idxa)))
        out_idx_b = sorted(list(set(range(0, b.ndim)) - set(idxb)))
        assert len(idxa) == len(idxb)

        map_idx_b = {}
        for iblock in b.blocks:
            block = _adjust_block(iblock, b_flip_axes)
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
        if len(out_idx_a) == 0 and len(out_idx_b) == 0:
            return np.asarray(list(blocks_map.values())[0]).item()
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

    def tensor_qr(self, left_idx, right_idx=None, mod="qr"):
        return sparse_qr(self, left_idx, right_idx=right_idx, mod=mod)

    def to_exponential(self, x):
        from pyblock3.algebra.fermion_ops import get_sparse_exponential
        return get_sparse_exponential(self, x)

_flat_fermion_tensor_numpy_func_impls = _flat_sparse_tensor_numpy_func_impls.copy()
[_flat_fermion_tensor_numpy_func_impls.pop(key) for key in NEW_METHODS]
_numpy_func_impls = _flat_fermion_tensor_numpy_func_impls

class FlatFermionTensor(FlatSparseTensor):

    def __init__(self, q_labels, shapes, data, pattern=None, idxs=None, symmetry=None):
        self.n_blocks = len(q_labels)
        self.ndim = q_labels.shape[1] if self.n_blocks != 0 else 0
        self.shapes = shapes
        self.q_labels = q_labels
        self.data = data

        if pattern is None:
            pattern = _gen_default_pattern(self.ndim)
        self._pattern = pattern

        if idxs is None:
            self.idxs = np.zeros((self.n_blocks + 1, ), dtype=np.uint64)
            self.idxs[1:] = np.cumsum(shapes.prod(axis=1), dtype=np.uint64)
        else:
            self.idxs = idxs
        if self.n_blocks != 0:
            assert shapes.shape == (self.n_blocks, self.ndim)
            assert q_labels.shape == (self.n_blocks, self.ndim)
        self._shape = tuple(np.sum(shapes, axis=0))
        self.symmetry = setting.dispatch_settings(symmetry=symmetry)

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
        return self.__class__(self.q_labels[:, axes], self.shapes[:, axes], data, pattern=_flip_pattern(self.pattern[::-1]), idxs=self.idxs, symmetry=self.symmetry)

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
        return self._shape

    def get_bond_info(self, ax):
        ipattern = self.pattern[ax]
        if ipattern=="+":
            sz = [self.symmetry.from_flat(ix) for ix in self.q_labels[:,ax]]
        else:
            sz = [-self.symmetry.from_flat(ix) for ix in self.q_labels[:,ax]]
        sp = self.shapes[:,ax]
        bond = dict(zip(sz, sp))
        return BondInfo(bond)

    def conj(self):
        return self.__class__(self.q_labels, self.shapes, self.data.conj(), pattern=self.pattern, idxs=self.idxs, symmetry=self.symmetry)

    def _local_flip(self, axes):
        if not setting.DEFAULT_FERMION: return
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
        if not setting.DEFAULT_FERMION: return
        self.data *= -1

    def to_sparse(self):
        blocks = [None] * self.n_blocks
        for i in range(self.n_blocks):
            qs = tuple(map(self.symmetry.from_flat, self.q_labels[i]))
            blocks[i] = SubTensor(
                self.data[self.idxs[i]:self.idxs[i + 1]].reshape(self.shapes[i]), q_labels=qs)
        return SparseFermionTensor(blocks=blocks, pattern=self.pattern)

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
        idxs = np.zeros((n_blocks + 1, ), dtype=np.uint64)
        idxs[1:] = np.cumsum(shapes.prod(axis=1), dtype=np.uint64)
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
            flip_axes = [ix for ix in range(b.ndim) if a.pattern[ix]!=b.pattern[ix]]
            q_labels_b = _adjust_q_labels(b.symmetry, b.q_labels, flip_axes)
            q_labels, shapes, data, idxs = _block3.flat_sparse_tensor.add(a.q_labels, a.shapes, a.data,
                                                a.idxs, q_labels_b, b.shapes, b.data, b.idxs)
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
            flip_axes = [ix for ix in range(b.ndim) if a.pattern[ix]!=b.pattern[ix]]
            q_labels_b = _adjust_q_labels(b.symmetry, b.q_labels, flip_axes)
            q_labels, shapes, data, idxs = _block3.flat_sparse_tensor.add(a.q_labels, a.shapes, a.data,
                                                a.idxs, q_labels_b, b.shapes, -b.data, b.idxs)
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
                    out_pattern, _ = _contract_patterns(a.pattern, b.pattern, [a.ndim-1], [0])
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

        out_pattern, b_flip_axes = _contract_patterns(a.pattern, b.pattern, idxa, idxb)
        q_labels_b = _adjust_q_labels(b.symmetry, b.q_labels, b_flip_axes)
        backend = get_backend(a.symmetry)
        if setting.DEFAULT_FERMION:
            q_labels, shapes, data, idxs = backend.flat_fermion_tensor.tensordot(
                        a.q_labels, a.shapes, a.data, a.idxs,
                        q_labels_b, b.shapes, b.data, b.idxs,
                        idxa, idxb)
        else:
            q_labels, shapes, data, idxs = backend.flat_sparse_tensor.tensordot(
                        a.q_labels, a.shapes, a.data, a.idxs,
                        q_labels_b, b.shapes, b.data, b.idxs,
                        idxa, idxb)

        if len(idxa) == a.ndim and len(idxb) == b.ndim:
            return data[0]
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
            if setting.DEFAULT_FERMION:
                backend.flat_fermion_tensor.transpose(a.q_labels, a.shapes, a.data, a.idxs, axes, data)
            else:
                backend.flat_sparse_tensor.transpose(a.shapes, a.data, a.idxs, axes, data)
            pattern = "".join([a.pattern[ix] for ix in axes])
            return a.__class__(a.q_labels[:,axes], a.shapes[:,axes], \
                               data, pattern, a.idxs, a.symmetry)

    def tensor_svd(self, left_idx, right_idx=None, qpn_partition=None, qpn_cutoff_func=None, **opts):
        return flat_svd(self, left_idx, right_idx=right_idx, qpn_partition=qpn_partition, qpn_cutoff_func=qpn_cutoff_func, **opts)

    def tensor_qr(self, left_idx, right_idx=None, mod="qr"):
        return flat_qr(self, left_idx, right_idx=right_idx, mod=mod)

    def to_exponential(self, x):
        from pyblock3.algebra.fermion_ops import get_flat_exponential
        return get_flat_exponential(self, x)
