import numpy as np
from pyblock3.algebra.symmetry import QPN
from pyblock3.algebra.core import SubTensor


def _compute_qpn(q_labels, pattern):
    dq = QPN()
    for qpn, sign in zip(q_labels, pattern):
        if sign=="+":
            dq += qpn
        else:
            dq -= qpn
    return dq

def _pack_sparse_tensor(tsr, s_label, ax, qpn_info,parity_axes=None):
    u_map = {}
    v_map = {}
    left_offset = 0
    right_offset = 0
    for iblk in tsr.blocks:
        qpn_left = iblk.q_labels[:ax]
        qpn_right = iblk.q_labels[ax:]
        left_netq = _compute_qpn(qpn_left, tsr.pattern[:ax]) - qpn_info[0]
        right_netq = qpn_info[1] - _compute_qpn(qpn_right, tsr.pattern[ax:])
        if left_netq != s_label or right_netq != s_label:
            continue
        qpn_left += (left_netq, )
        qpn_right = (right_netq, ) + qpn_right
        len_left = np.prod(iblk.shape[:ax])
        len_right = np.prod(iblk.shape[ax:])
        if qpn_left not in u_map.keys():
            u_map[qpn_left] = ((left_offset, left_offset+len_left), iblk.shape[:ax])
            left_offset += len_left
        if qpn_right not in v_map.keys():
            v_map[qpn_right] = ((right_offset, right_offset+len_right), iblk.shape[ax:])
            right_offset += len_right

    data = np.zeros((left_offset, right_offset), dtype=tsr.dtype)
    for iblk in tsr.blocks:
        qpn_left = iblk.q_labels[:ax]
        qpn_right = iblk.q_labels[ax:]
        left_netq = _compute_qpn(qpn_left, tsr.pattern[:ax]) - qpn_info[0]
        right_netq = qpn_info[1] - _compute_qpn(qpn_right, tsr.pattern[ax:])
        if left_netq != s_label or right_netq != s_label:
            continue
        qpn_left += (left_netq, )
        qpn_right = (right_netq, ) + qpn_right
        sign = 1
        if parity_axes is not None:
            l_parity = [iblk.q_labels[iq].parity for iq in parity_axes]
            count_parity = l_parity.count(1)
            if np.mod(count_parity, 4) > 1:
                sign = -1
        ist, ied = u_map[qpn_left][0]
        jst, jed = v_map[qpn_right][0]
        data[ist:ied,jst:jed] = np.asarray(iblk).reshape(ied-ist, jed-jst) * sign
    return data, u_map, v_map

def _unpack_sparse_tensor(data, imap, ax, blklist, parity_axes=None):
    for q_label, (isec, ishape) in imap.items():
        sign = 1
        if parity_axes is not None:
            count_parity = [q_label[iq].parity for iq in parity_axes].count(1)
            if np.mod(count_parity, 4) > 1:
                sign = -1
        ist, ied = isec
        if ax==0:
            tmp = data[ist:ied].reshape(ishape+(-1,)) * sign
        else:
            tmp = data[:,ist:ied].reshape((-1,)+ishape) * sign
        blklist.append(SubTensor(tmp, q_labels=q_label))

def _fetch_aux_qpn(tsr, ax, reverse=False, dq=None):
    if dq is None: dq = QPN(0)
    left_qpn = []
    for iblk in tsr.blocks:
        left = np.add.reduce([iblk.q_labels[ix] if tsr.pattern[ix]=="+" else -iblk.q_labels[ix] for ix in ax]) - dq
        if reverse:
            left_qpn.append(-left)
        else:
            left_qpn.append(left)
    return set(left_qpn)

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
    if absorb == -1:
        u = u * s.reshape((1, -1))
    elif absorb == 1:
        v = v * s.reshape((-1, 1))
    else:
        s **= 0.5
        u = u * s.reshape((1, -1))
        v = v * s.reshape((-1, 1))
    return u, None, v, umap, vmap


def _run_sparse_fermion_svd(tsr, left_idx, right_idx=None, qpn_info=None, **opts):
    absorb = opts.pop("absorb", 0)
    if qpn_info is None:
        if absorb ==-1:
            qpn_info = (tsr.dq, QPN(0))
        else:
            qpn_info = (QPN(0), tsr.dq)

    if right_idx is None:
        right_idx = [idim for idim in range(tsr.ndim) if idim not in left_idx]
    neworder = tuple(left_idx) + tuple(right_idx)
    split_ax = len(left_idx)
    newtsr = tsr.transpose(neworder)
    ublk, sblk, vblk = [],[],[]

    s_left = _fetch_aux_qpn(tsr, left_idx, dq=qpn_info[0])
    s_right = _fetch_aux_qpn(tsr, right_idx, reverse=True, dq=qpn_info[1])
    s_labels = list(s_left | s_right)

    for slab in s_labels:
        data, umap, vmap = _pack_sparse_tensor(newtsr, slab, split_ax, qpn_info)
        if data.size==0:
            continue
        u, s, v = np.linalg.svd(data, full_matrices=False)
        u, s, v = _trim_and_renorm_SVD(u, s, v, **opts)
        if absorb is None:
            sblk.append(SubTensor(np.diag(s), q_labels=(slab,)*2))
        else:
            u, s, v, umap, vmap = _absorb_svd(u, s, v, absorb, slab, umap, vmap)
        _unpack_sparse_tensor(u, umap, 0, ublk)
        _unpack_sparse_tensor(v, vmap, 1, vblk)

    u = tsr.__class__(blocks=ublk, pattern=newtsr.pattern[:split_ax]+"-")
    if absorb is None:
        s = tsr.__class__(blocks=sblk, pattern="+-")
    v = tsr.__class__(blocks=vblk, pattern="+"+newtsr.pattern[split_ax:])
    return u, s, v
