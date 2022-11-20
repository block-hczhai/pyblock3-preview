
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
Fermionic Tensor with Block Sparsity from Symmetry.
Author: Yang Gao
Modified: Huanchen Zhai, Nov 14 2022
"""

import numbers
import numpy as np
import string
from .core import (SparseTensor, SubTensor,
                   _sparse_tensor_numpy_func_impls)
from .core import jax_pytree_node
from ..symmetry import BondInfo
from .. import fermion_setting as setting
import time

from . import ENABLE_AUTORAY, ENABLE_JAX, ENABLE_EINSUM

if ENABLE_AUTORAY:
    from autoray import numpy as jnp
elif ENABLE_JAX:
    import jax.numpy as jnp
else:
    jnp = np

SVD_SCREENING = setting.SVD_SCREENING
Q_LABELS_DTYPE = SHAPES_DTYPE = np.uint32
INDEX_DTYPE = np.uint64

_timings = {}

def format_timing():
    from ..fermion_symmetry import _timings as _stimings, clear_timing as sclear_timing
    global _timings
    rt = " ".join(["%s = %.3f" % (k, v) for k, v in list(_timings.items()) + list(_stimings.items())])
    rt += " total = %.3f" % sum(_timings.values())
    clear_timing()
    sclear_timing()
    return rt

def clear_timing():
    global _timings
    _timings.clear()

def timing(tm_key):
    global _timings
    def tf(f):
        def ttf(*args, **kwargs):
            tx = time.perf_counter()
            ret = f(*args, **kwargs)
            _timings[tm_key] = _timings.get(tm_key, 0.0) + time.perf_counter() - tx
            return ret
        return ttf
    return tf

def implements(np_func):
    global _numpy_func_impls
    return lambda f: (_numpy_func_impls.update({np_func: f})
                      if np_func not in _numpy_func_impls else None,
                      _numpy_func_impls[np_func])[1]

NEW_METHODS = [np.transpose, np.add, np.subtract, np.copy]

_sparse_fermion_tensor_numpy_func_impls = _sparse_tensor_numpy_func_impls.copy()
[_sparse_fermion_tensor_numpy_func_impls.pop(key) for key in NEW_METHODS]
_numpy_func_impls = _sparse_fermion_tensor_numpy_func_impls

def _gen_default_pattern(obj):
    """Get a default algebraic pattern for given dimension

    Parameters
    ----------
    obj : int or list/array list object

    Returns
    ----------
    pattern : str
    """
    ndim = obj if isinstance(obj, int) else len(obj)
    pattern = "+" * (ndim-1) + '-'
    return pattern

def _flip_pattern(pattern):
    """Flip the algebraic pattern, eg, "+" -> "-" and "-" -> "+"

    Parameters
    ----------
    pattern : str made of '+' and '-'
    """
    flip_dict = {"+":"-", "-":"+"}
    return "".join([flip_dict[ip] for ip in pattern])

def _contract_patterns(patterna, patternb, idxa, idxb):
    """Get the output pattern from contracting two input patterns.
    Compute the axes that needs to be flipped for the second pattern
    if the two patterns does not fit directly

    Parameters
    ----------
    patterna : str
        symmetry pattern for the first input
    patternb : str
        symmetry pattern for the second input
    idxa: list/tuple of int
        indices to be contracted on the first input
    idxb: list/tuple of int
        indices to be contracted on the second input

    Returns
    -------

    output_pattern: str
        symmetry pattern for the output
    b_flip_idxs: list of int
        the indices that needs to be flipped on b due to input pattern misfit
    """
    conc_a = "".join([patterna[ix] for ix in idxa])
    out_a = "".join([ip for ix, ip in enumerate(patterna)
                    if ix not in idxa])
    conc_b = "".join([patternb[ix] for ix in idxb])
    out_b = "".join([ip for ix, ip in enumerate(patternb)
                    if ix not in idxb])

    b_flip_idxs = []
    if conc_a != _flip_pattern(conc_b):
        for ind, ixa, ixb in zip(idxb, conc_a, conc_b):
            if ixa == ixb:
                b_flip_idxs.append(ind)
    output_pattern = out_a + out_b
    return output_pattern, b_flip_idxs

def _trace_preprocess(T, ax1, ax2):
    """Preprocessing for the tracing operation Tr(T)_{ax1, ax2}

    Parameters
    ----------
    T : SparseFermionTensor or FlatFermionTensor
        input tensor
    ax1 : int or list/tuple of int
        sequence of the first dimensions
    ax2: int or list/tuple of int
        sequence of the second dimensions, must correpsond
        to ax1

    Returns
    -------

    new_ax1: list of int
        first sequence of contracted axes for the transposed input
    new_ax2: list of int
        second sequence of contracted axes for the transposed input
    einsum_subs: str
        einsum subscript for the trace operation (using transposed input)
    output_axes: list of int
        remaining axes in the output
    output_pattern: str
        symmetry pattern in the output
    transpose_order: list of int
        transposition order so that the trace operation is equivalent
        to normal trace operation
    """
    if isinstance(ax1, int):
        ax1 = (ax1, )
    if isinstance(ax2, int):
        ax2 = (ax2, )
    input_subs = output_subs = string.ascii_lowercase[:T.ndim]
    transpose_order = []
    new_ax1 = []
    new_ax2 = []
    for ix, iy in zip(ax1, ax2):
        ix, iy = sorted([ix, iy])
        new_ax1.append(ix)
        new_ax2.append(iy)
        transpose_order.extend([ix, iy])
        output_subs = output_subs.replace(input_subs[ix], "")
        output_subs = output_subs.replace(input_subs[iy], "")
        input_subs = input_subs.replace(input_subs[iy],input_subs[ix])
    einsum_subs = input_subs + "->" + output_subs
    output_axes = [ix for ix in range(T.ndim) if ix not in ax1+ax2]
    output_pattern = "".join([T.pattern[ix]
                        for ix in output_axes])
    return (new_ax1, new_ax2, einsum_subs,
            output_axes, output_pattern, transpose_order)

def _trim_singular_vals(
    s_data,
    cutoff,
    cutoff_mode,
    max_bond=None
):
    """Find the number of singular values to keep of ``s`` given ``cutoff`` and
    ``cutoff_mode``.

    Parameters
    ----------
    s_data : a list of array
        Singular values for different blocks.
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

    Returns
    -------
    n_chis: a list of int
        The number of remaining singular values for each block
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
            n_chi = np.sum([x.get() if hasattr(x, 'get') else x for x in n_chis])
            extra_bonds = n_chi - max_bond
            if extra_bonds >0:
                if s is None:
                    s = np.concatenate(s_data)
                s_ind = np.argsort(s)
                s_ind = [x.get() if hasattr(x, 'get') else x for x in s_ind]
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
        s_ind = [x.get() if hasattr(x, 'get') else x for x in s_ind]
        s_sorted = np.cumsum(np.sort(s))
        ind_map = []
        for ix, sblk in enumerate(s_data):
            ind_map += [ix,] * sblk.size

        n_chis = [sblk.size for sblk in s_data]
        ncut = np.sum(s_sorted<=target)
        ncut = ncut.get() if hasattr(ncut, 'get') else ncut
        if max_bond is not None and max_bond>0:
            ncut = max(ncut, s.size-max_bond)
        for i in range(ncut):
            group_ind = ind_map[s_ind[i]]
            n_chis[group_ind] -= 1
    return n_chis

def _renorm_singular_vals(s_data, n_chis, renorm):
    """Find the normalization constant for ``s`` such that the new sum squared
    of the ``n_chi`` largest values equals the sum squared of all the old ones.
    """
    s_tot_keep = 0.0
    s_tot_lose = 0.0
    for sblk, n_chi in zip(s_data, n_chis):
        for i in range(sblk.size):
            s2 = sblk[i]**renorm
            if not np.isnan(s2):
                if i < n_chi:
                    s_tot_keep += s2
                else:
                    s_tot_lose += s2

    return ((s_tot_keep + s_tot_lose) / s_tot_keep)**(1 / renorm)

def _trim_and_renorm_SVD(
    s_data,
    uv_data,
    cutoff=SVD_SCREENING,
    cutoff_mode=3,
    max_bond=None,
    renorm=0
):
    """Truncate and renormalize the singular values for each block
    """
    if isinstance(max_bond, (tuple, list)):
        if len(max_bond) != len(s_data):
            raise ValueError("max_bond must be an integer or a tuple "
                             "with same length as the number of blocks")
        if len(set(max_bond)) != 1:
            raise ValueError("max_bond in each dimension must be equal")
        n_chis = max_bond
    else:

        n_chis = _trim_singular_vals(s_data, cutoff,
                                cutoff_mode, max_bond)
    n_chi = np.sum([x.get() if hasattr(x, 'get') else x for x in n_chis])
    tot_size = np.sum([iblk.size for iblk in s_data])
    if n_chi < tot_size and renorm > 0:
        renorm_fac = _renorm_singular_vals(s_data,
                        n_chis, renorm)
        for sblk in s_data:
            sblk *= renorm_fac

    for ix, n_chi in enumerate(n_chis):
        s_data[ix] = s_data[ix][:n_chi]
        U, VH = uv_data[ix]
        uv_data[ix] = (U[...,:n_chi], VH[:n_chi,...])

    return s_data, uv_data

def _absorb_svd(u, s, v, absorb):
    """absorb the singular values s onto left u or right side v

    Parameters
    ----------
    u : array
    s : array
    v : array
    absorb : {1, -1, None}
        absorpsion mode:
            - -1: s onto u
            - 1: s onto `v`
            - None: ``s**.5`` onto both u and v
    """
    if absorb == -1:
        u = u * s.reshape((1, -1))
    elif absorb == 1:
        v = v * s.reshape((-1, 1))
    else:
        s **= 0.5
        u = u * s.reshape((1, -1))
        v = v * s.reshape((-1, 1))
    return u, None, v

def _maybe_transpose_tensor(T, left_idx, right_idx):
    """transpose the tensor if needed to make sure
    it's in specified order

    Parameters
    ----------
    T : SparseFermionTensor or FlatFermionTensor
    left_idx : tuple/list of int
    right_idx : tuple/list of int

    Returns
    -------
    new_T: SparseFermionTensor or FlatFermionTensor in the right order
    """
    neworder = list(left_idx) + list(right_idx)
    if neworder == list(range(T.ndim)):
        new_T = T
    else:
        new_T = T.transpose(neworder)
    return new_T

def _svd_preprocess(T, left_idx, right_idx, qpn_partition, absorb):
    """transpose the tensor if needed to make sure
    it's in specified order

    Parameters
    ----------
    T : SparseFermionTensor or FlatFermionTensor
    left_idx : tuple/list of int
        left indices for SVD
    right_idx : tuple/list of int
        right indices for SVD
    qpn_partition: tuple/list of symmetry object
        partition of symmetry on the left and right
    absorb: {-1,1,0,None}
        absorpsion mode:
            - -1: s onto left
            - 1: s onto right
            - 0: do not absorb s onto left/right
            - None: ``s**.5`` onto both u and v
    Returns
    -------
    new_T: SparseFermionTensor or FlatFermionTensor in the right order
    qpn_partition: tuple of symmetry object
        partition of symmetry on the left and right
    symmetry_cls: symmetry class
        symmetry class for the undering data
    """
    symmetry_cls = T.dq.__class__
    if max(len(left_idx), len(right_idx)) == T.ndim:
        raise ValueError
    if qpn_partition is None:
        if absorb ==-1:
            qpn_partition = (T.dq, symmetry_cls(0))
        else:
            qpn_partition = (symmetry_cls(0), T.dq)
    else:
        assert np.sum(qpn_partition) == T.dq
    new_T = _maybe_transpose_tensor(T, left_idx, right_idx)
    return new_T, qpn_partition, symmetry_cls

def sparse_svd(
    T,
    left_idx,
    right_idx=None,
    qpn_partition=None,
    **opts
):
    """Perform tensor SVD on SparseFermionTensor

    Parameters
    ----------
    T : SparseFermionTensor
    left_idx : tuple/list of int
        left indices for SVD
    right_idx : optional, tuple/list of int
        right indices for SVD
    qpn_partition: optional, tuple/list of symmetry object
        partition of symmetry on the left and right,
        must add up to total symmetry of the input tensor

    Returns
    -------
    u: SparseFermionTensor
    s: SparseFermionTensor or None
    v: SparseFermionTensor
    """
    absorb = opts.pop("absorb", 0)
    if right_idx is None:
        right_idx = [idim for idim in range(T.ndim)
                        if idim not in left_idx]
    new_T, qpn_partition, symmetry = _svd_preprocess(
                    T, left_idx, right_idx,
                    qpn_partition, absorb)
    split_ax = len(left_idx)
    left_pattern = new_T.pattern[:split_ax]
    data_map = {}
    for iblk in new_T.blocks:
        left_q = symmetry._compute(left_pattern,
                    iblk.q_labels[:split_ax],
                    offset=("-", qpn_partition[0]))
        if left_q not in data_map:
            data_map[left_q] = []
        data_map[left_q].append(iblk)

    s_data =[]
    uv_data = []
    all_maps = []

    for sblk_q_label, datasets in data_map.items():
        row_len = col_len = 0
        row_map = {}
        col_map = {}
        for iblk in datasets:
            lq = tuple(iblk.q_labels[:split_ax]) + (sblk_q_label,)
            rq = (sblk_q_label,) + tuple(iblk.q_labels[split_ax:])
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
            lq = tuple(iblk.q_labels[:split_ax]) + (sblk_q_label,)
            rq = (sblk_q_label,) + tuple(iblk.q_labels[split_ax:])
            ist, ied = row_map[lq][:2]
            jst, jed = col_map[rq][:2]
            data[ist:ied,jst:jed] = iblk.data.reshape(ied-ist, jst-jed)
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
        all_maps.append([sblk_q_label, row_map, col_map])

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

    for s, (u, v), (sblk_q_label, row_map, col_map) \
            in zip(s_data, uv_data, all_maps):
        if u.size==0:
            continue
        if absorb is None:
            s = np.diag(s)
            sblocks.append(
                SubTensor(data=s,
                          q_labels=(sblk_q_label,)*2))

        for lq, (lst, led, lsh) in row_map.items():
            utrunk = u[lst:led].reshape(tuple(lsh)+(-1,))
            ublocks.append(SubTensor(data=utrunk, q_labels=lq))

        for rq, (rst, red, rsh) in col_map.items():
            vtrunk = v[:,rst:red].reshape((-1,)+tuple(rsh))
            vblocks.append(SubTensor(data=vtrunk, q_labels=rq))

    if absorb is None:
        s = T.__class__(blocks=sblocks, pattern="+-")
    u_pattern = new_T.pattern[:split_ax]+"-"
    v_pattern = "+"+new_T.pattern[split_ax:]
    u = T.__class__(blocks=ublocks, pattern=u_pattern)
    v = T.__class__(blocks=vblocks, pattern=v_pattern)
    # make sure shape is consistent
    u.shape = new_T.shape[:split_ax] + (u.shape[-1],)
    v.shape = (v.shape[0],)+new_T.shape[split_ax:]
    return u, s, v

def _gen_null_qr_info(T, mod):
    """
    Generate the information for a fake QR/LQ decomposition
    T = QR/LQ where Q is 1
    """
    return {"qr": ("-"+T.pattern, 0, slice(None)),
            "lq": (T.pattern+"-", T.ndim, slice(None,None,-1))}[mod]

def sparse_qr(T,
    left_idx,
    right_idx=None,
    mod="qr"
):
    """Perform tensor QR on SparseFermionTensor, this will partition
    the quantum number fully on Q and net zero symmetry on R

    Parameters
    ----------
    T : SparseFermionTensor
    left_idx : tuple/list of int
        left indices for QR
    right_idx : optional, tuple/list of int
        right indices for QR
    mod: optional, {"qr", "lq"}

    """
    assert mod in ["qr", "lq"]
    if right_idx is None:
        right_idx = [idim for idim in range(T.ndim)
        if idim not in left_idx]
    new_T = _maybe_transpose_tensor(T, left_idx, right_idx)
    if len(left_idx) == T.ndim or len(right_idx)==T.ndim:
        dq = T.dq
        qblks = [SubTensor(data=np.ones([1]), q_labels=(dq, ))]
        Q = T.__class__(blocks=qblks, pattern="+")

        def _get_new_blk_qr(blk, ax):
            new_q_labels = blk.q_labels[:ax] + (dq,) + blk.q_labels[ax:]
            new_shape = blk.data.shape[:ax] + (1,) + blk.data.shape[ax:]
            arr = blk.data.reshape(new_shape)
            return blk.__class__(data=arr, q_labels=new_q_labels)

        new_pattern, inds, return_order = _gen_null_qr_info(T, mod)
        if return_order == slice(None):
            shape = (1,)+T.shape
        else:
            shape = T.shape + (1,)
        rblks = [_get_new_blk_qr(iblk, inds) for iblk in T.blocks]
        R = T.__class__(blocks=rblks, pattern=new_pattern, shape=shape)
        return (Q, R)[return_order]

    symmetry = T.dq.__class__
    dq = {"lq": symmetry(0),
          "qr": T.dq}[mod]
    split_ax = len(left_idx)
    left_pattern = new_T.pattern[:split_ax]
    data_map = {}
    for iblk in new_T.blocks:
        left_q = symmetry._compute(left_pattern,
                    iblk.q_labels[:split_ax], offset=("-", dq))
        if left_q not in data_map:
            data_map[left_q] = []
        data_map[left_q].append(iblk)
    qblocks = []
    rblocks = []
    for sblk_q_label, datasets in data_map.items():
        row_len = col_len = 0
        row_map = {}
        col_map = {}
        for iblk in datasets:
            lq = tuple(iblk.q_labels[:split_ax]) + (sblk_q_label,)
            rq = (sblk_q_label,) + tuple(iblk.q_labels[split_ax:])
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
            lq = tuple(iblk.q_labels[:split_ax]) + (sblk_q_label,)
            rq = (sblk_q_label,) + tuple(iblk.q_labels[split_ax:])
            ist, ied = row_map[lq][:2]
            jst, jed = col_map[rq][:2]
            data[ist:ied,jst:jed] = iblk.data.reshape(ied-ist, jst-jed)
        if data.size ==0:
            continue
        if mod=="qr":
            q, r = np.linalg.qr(data)
        else:
            r, q = np.linalg.qr(data.T)
            q, r = q.T, r.T

        for lq, (lst, led, lsh) in row_map.items():
            qtrunk = q[lst:led].reshape(tuple(lsh)+(-1,))
            qblocks.append(SubTensor(data=qtrunk,
                                     q_labels=lq))
        for rq, (rst, red, rsh) in col_map.items():
            rtrunk = r[:,rst:red].reshape((-1,)+tuple(rsh))
            rblocks.append(SubTensor(data=rtrunk, q_labels=rq))

    q_pattern = new_T.pattern[:split_ax]+"-"
    r_pattern = "+"+new_T.pattern[split_ax:]
    q = T.__class__(blocks=qblocks, pattern=q_pattern)
    r = T.__class__(blocks=rblocks, pattern=r_pattern)
    q.shape = new_T.shape[:split_ax] + (q.shape[-1],)
    r.shape = (r.shape[0],)+new_T.shape[split_ax:]
    return q, r

def _adjust_block(block, flip_axes):
    if len(flip_axes)==0:
        return block
    else:
        raise NotImplementedError

def _adjust_q_labels(symmetry, q_labels, flip_axes):
    if len(flip_axes)==0:
        return q_labels
    else:
        raise NotImplementedError

def compute_phase(
    q_labels,
    axes,
    direction="left",
    symmetry=None
):
    if not setting.DEFAULT_FERMION: return 1
    if isinstance(q_labels, np.ndarray):
        if symmetry is None:
            symmetry = setting.DEFAULT_SYMMETRY
        plist = [symmetry.flat_to_parity(qpn) for qpn in q_labels]
    else:
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

def eye(bond_info, flat=None, large=None):
    """Create tensor from BondInfo with Identity matrix."""
    flat, large = setting.dispatch_settings(flat=flat, large=large)
    blocks = []
    for sh, qs in SparseFermionTensor._skeleton((bond_info, bond_info)):
        blocks.append(SubTensor(data=np.eye(sh[0]), q_labels=qs))
    T = SparseFermionTensor(blocks=blocks, pattern="+-")
    if large:
        T = T.to_flat().to_large()
    elif flat:
        T = T.to_flat()
    return T

@jax_pytree_node
class SparseFermionTensor(SparseTensor):

    def __init__(self, blocks=None, pattern=None, shape=None):
        self.blocks = blocks if blocks is not None else []
        if pattern is None:
            pattern = _gen_default_pattern(self.ndim)
        self._pattern = pattern
        self._shape = shape

    def pytree_flatten(self):
        traced = tuple(self.blocks)
        others = (self.pattern, self.shape)
        return traced, others

    @classmethod
    def pytree_unflatten(cls, others, traced):
        blocks = list(traced)
        pattern, shape = others
        return cls(blocks=blocks, pattern=pattern, shape=shape)

    @property
    def dq(self):
        symmetry_cls = self.blocks[0].q_labels[0].__class__
        dq = symmetry_cls._compute(self.pattern,
                self.blocks[0].q_labels)
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
        blocks = [np.transpose(block.conj(), axes=axes)
                                for block in self.blocks]
        shape = self.shape[::-1]
        return self.__class__(blocks=blocks, pattern=_flip_pattern(self.pattern[::-1]), shape=shape)

    @property
    def shape(self):
        if self._shape is None:
            counted_block = [[] for _ in range(self.ndim)]
            shape = [0,] * self.ndim
            for iblk in self:
                for ix, iq in enumerate(iblk.q_labels):
                    if iq in counted_block[ix]:
                        continue
                    else:
                        shape[ix] += iblk.shape[ix]
                        counted_block[ix].append(iq)
            self._shape = tuple([int(ish) for ish in shape])
        return self._shape

    @shape.setter
    def shape(self, sh):
        assert len(sh) == self.ndim
        self._shape = tuple(sh)

    def to_constructor(self, axes):
        return Constructor.from_sparse_tensor(self, axes)

    def new_like(self, blocks, **kwargs):
        pattern = kwargs.pop("pattern", self.pattern)
        shape = kwargs.pop("shape", self.shape)
        r = self.__class__(blocks=blocks, pattern=pattern, shape=shape)
        if hasattr(self, '_tree'):
            r._tree = self._tree
        return r

    @staticmethod
    @implements(np.copy)
    def _copy(x):
        return x.new_like([b.copy() for b in x.blocks])

    def copy(self):
        return np.copy(self)

    @property
    def parity(self):
        return self.dq.parity

    def get_bond_info(self, ax, flip=True):
        bond = dict()
        ipattern = self.pattern[ax]
        for iblk in self:
            q = iblk.q_labels[ax]
            dim = iblk.shape[ax]
            if ipattern == "-" and flip:
                bond.update({-q:dim})
            else:
                bond.update({q:dim})

        return BondInfo(bond)

    def conj(self):
        blks = [iblk.conj() for iblk in self.blocks]
        return self.new_like(blks)

    def _local_flip(self, axes):
        if not setting.DEFAULT_FERMION: return
        if isinstance(axes, int):
            axes = [axes]
        else:
            axes = list(axes)
        for ib, blk in enumerate(self.blocks):
            block_parity = np.add.reduce([blk.q_labels[j] for j in axes]).parity
            if block_parity == 1:
                self.blocks[ib] = -blk

    def _global_flip(self):
        if not setting.DEFAULT_FERMION: return
        for ib, blk in enumerate(self.blocks):
            self.blocks[ib] = -blk

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
    def random(
        bond_infos,
        pattern=None,
        dq=None,
        dtype=float,
        shape=None
    ):
        """Create tensor from tuple of BondInfo with random elements."""
        blocks = []
        for sh, qs in SparseFermionTensor._skeleton(bond_infos, pattern=pattern, dq=dq):
            blocks.append(SubTensor.random(shape=sh, q_labels=qs, dtype=dtype))
        return SparseFermionTensor(blocks=blocks, pattern=pattern, shape=shape)

    @staticmethod
    def zeros(
        bond_infos,
        pattern=None,
        dq=None,
        dtype=float,
        shape=None
    ):
        """Create tensor from tuple of BondInfo with zero elements."""
        blocks = []
        for sh, qs in SparseFermionTensor._skeleton(bond_infos, pattern=pattern, dq=dq):
            blocks.append(SubTensor.zeros(shape=sh, q_labels=qs, dtype=dtype))
        return SparseFermionTensor(blocks=blocks, pattern=pattern, shape=shape)

    @staticmethod
    def ones(
        bond_infos,
        pattern=None,
        dq=None,
        dtype=float,
        shape=None
    ):
        """Create tensor from tuple of BondInfo with one elements."""
        blocks = []
        for sh, qs in SparseFermionTensor._skeleton(bond_infos, pattern=pattern, dq=dq):
            blocks.append(SubTensor.ones(shape=sh, q_labels=qs, dtype=dtype))
        return SparseFermionTensor(blocks=blocks, pattern=pattern, shape=shape)

    @staticmethod
    def eye(bond_info):
        """Create tensor from BondInfo with Identity matrix."""
        blocks = []
        pattern = "+-"
        for sh, qs in SparseFermionTensor._skeleton((bond_info, bond_info), pattern=pattern):
            blocks.append(SubTensor(data=np.eye(sh[0]), q_labels=qs))
        return SparseFermionTensor(blocks=blocks, pattern=pattern)

    @staticmethod
    @implements(np.add)
    def _add(a, b):
        if isinstance(a, numbers.Number):
            blocks = [np.add(a, block) for block in b.blocks]
            return b.new_like(blocks)
        elif isinstance(b, numbers.Number):
            blocks = [np.add(block, b) for block in a.blocks]
            return a.new_like(blocks)
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
            return a.new_like(blocks, shape=None)

    def add(self, b):
        return self._add(self, b)

    @staticmethod
    @implements(np.subtract)
    def _subtract(a, b):
        if isinstance(a, numbers.Number):
            blocks = [np.subtract(a, block) for block in b.blocks]
            return b.new_like(blocks)
        elif isinstance(b, numbers.Number):
            blocks = [np.subtract(block, b) for block in a.blocks]
            return a.new_like(blocks)
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
            return a.new_like(blocks)

    def subtract(self, b):
        return self._subtract(self, b)

    def trace(self, ax1, ax2):
        ax1, ax2, einsum_subs, output_axes, \
            output_pattern, transpose_order= _trace_preprocess(self, ax1, ax2)
        symmetry = self.dq.__class__
        if len(ax1+ax2) == self.ndim and self.dq != symmetry(0):
            return 0
        blks_to_trace = []
        for iblk in self.blocks:
            aligned = True
            for axi, axj in zip(ax1, ax2):
                q_labels = [iblk.q_labels[axi], iblk.q_labels[axj]]
                pattern = self.pattern[axi] + self.pattern[axj]
                dq = symmetry._compute(pattern, q_labels)
                aligned = aligned and dq == symmetry(0)
                if not aligned:
                    break
            if aligned:
                blks_to_trace.append(iblk)
        data = {}
        for iblk in blks_to_trace:
            phase = compute_phase(iblk.q_labels, transpose_order)
            out = np.einsum(einsum_subs, iblk.data) * phase
            q_labels = tuple([iblk.q_labels[ix] for ix in output_axes])
            if output_pattern:
                out = iblk.__class__(data=out, q_labels=q_labels)
            if q_labels not in data:
                data[q_labels] = out
            else:
                data[q_labels] += out
        data = list(data.values())
        shape = [self.shape[ix] for ix in range(self.ndim) if ix not in ax1+ax2]
        if output_pattern:
            return self.__class__(blocks=data, pattern=output_pattern, shape=tuple(shape))
        else:
            return sum(data)

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
        out_shape = None
        if method == "__call__":
            if ufunc.__name__ in ["matmul"]:
                a, b = inputs
                if isinstance(a, numbers.Number):
                    blocks = [a * block for block in b.blocks]
                    out_shape = a.shape
                elif isinstance(b, numbers.Number):
                    blocks = [block * b for block in a.blocks]
                    out_shape = b.shape
                else:
                    blocks = self._tensordot(a, b, axes=([-1], [0])).blocks
                    out.shape = blocks.shape
                out_pattern, _ = _contract_patterns(a.pattern, b.pattern, [a.ndim-1], [0])
            elif ufunc.__name__ in ["multiply", "divide", "true_divide"]:
                a, b = inputs
                if isinstance(a, numbers.Number):
                    blocks = [getattr(ufunc, method)(a, block)
                              for block in b.blocks]
                    out_shape = b.shape
                    out_pattern = b.pattern
                elif isinstance(b, numbers.Number):
                    blocks = [getattr(ufunc, method)(block, b)
                              for block in a.blocks]
                    out_shape = a.shape
                    out_pattern = a.pattern
                else:
                    return NotImplemented
            elif len(inputs) == 1:
                blocks = [getattr(ufunc, method)(block)
                          for block in inputs[0].blocks]
                out_pattern = inputs[0].pattern
                out_shape = inputs[0].shape
            else:
                return NotImplemented
        else:
            return NotImplemented
        if out is not None:
            out[0].blocks = blocks
            out[0].pattern = out_pattern
            out[0].shape=out_shape
        return self.__class__(blocks=blocks, pattern=out_pattern, shape=out_shape)


    @staticmethod
    def _tensordot_basic(a, b, axes=2):

        if isinstance(axes, int):
            idxa, idxb = list(range(-axes, 0)), list(range(0, axes))
        else:
            idxa, idxb = axes
        idxa = [x if x >= 0 else a.ndim + x for x in idxa]
        idxb = [x if x >= 0 else b.ndim + x for x in idxb]

        out_pattern, b_flip_axes = _contract_patterns(a.pattern, b.pattern, idxa, idxb)
        out_idx_a = sorted(list(set(range(0, a.ndim)) - set(idxa)))
        out_idx_b = sorted(list(set(range(0, b.ndim)) - set(idxb)))
        out_shape = [a.shape[ix] for ix in out_idx_a] + \
                    [b.shape[ix] for ix in out_idx_b]
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
                    matd = jnp.tensordot(
                        block_a.data, block_b.data, axes=(idxa, idxb))
                    mat = block_a.__class__(matd, q_labels=None)
                    if outq not in blocks_map:
                        mat.q_labels = outq
                        blocks_map[outq] = mat * phase
                    else:
                        blocks_map[outq] += mat * phase

        return a.__class__(blocks=list(blocks_map.values()), pattern=out_pattern, shape=tuple(out_shape))


    @staticmethod
    def _tensordot_fused(a, b, axes=2):

        if isinstance(axes, int):
            idxa, idxb = list(range(-axes, 0)), list(range(0, axes))
        else:
            idxa, idxb = axes
        idxa = [i if i >= 0 else a.ndim + i for i in idxa]
        idxb = [i if i >= 0 else b.ndim + i for i in idxb]
        a_bond_inds = tuple(idxa)
        b_bond_inds = tuple(idxb)
        a_out_inds = tuple(i for i in range(a.ndim) if i not in idxa)
        b_out_inds = tuple(i for i in range(b.ndim) if i not in idxb)
        a_out_shape = tuple(a.shape[ix] for ix in a_out_inds)
        b_out_shape = tuple(b.shape[ix] for ix in b_out_inds)

        a_reorder_inds = a_out_inds + a_bond_inds
        b_reorder_inds = b_bond_inds[::-1] + b_out_inds

        if a_reorder_inds != tuple(range(0, a.ndim)):
            a = np.transpose(a, axes=a_reorder_inds)
        if b_reorder_inds != tuple(range(0, b.ndim)):
            b = np.transpose(b, axes=b_reorder_inds)

        axes = list(range(-len(a_bond_inds), 0)), list(range(0, len(b_bond_inds)))[::-1]

        z = super()._tensordot_fused(a, b, axes=axes)
        z._shape = a_out_shape + b_out_shape
        return z

    @staticmethod
    @implements(np.transpose)
    def _transpose(a, axes=None):
        if axes is None:
            axes = list(range(a.ndim))[::-1]
        phase = [compute_phase(block.q_labels, axes) for block in a.blocks]
        blocks = [np.transpose(block, axes=axes)*phase[ibk] for ibk, block in enumerate(a.blocks)]
        pattern = "".join([a.pattern[ix] for ix in axes])
        shape = [a.shape[ix] for ix in axes]
        return a.__class__(blocks=blocks, pattern=pattern, shape=tuple(shape))

    def tensor_svd(
        self,
        left_idx,
        right_idx=None,
        qpn_partition=None,
        **opts
    ):
        return sparse_svd(self, left_idx,
                          right_idx=right_idx,
                          qpn_partition=qpn_partition,
                          **opts)

    def tensor_qr(
        self,
        left_idx,
        right_idx=None,
        mod="qr"
    ):
        return sparse_qr(self, left_idx,
                         right_idx=right_idx, mod=mod)

    def to_exponential(self, x):
        from .fermion_ops import get_sparse_exponential
        return get_sparse_exponential(self, x)
