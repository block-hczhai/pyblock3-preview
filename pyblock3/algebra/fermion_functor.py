import numpy as np
from pyblock3.algebra.symmetry import SZ, BondInfo
from pyblock3.algebra.fermion import SparseFermionTensor, FlatFermionTensor
from pyblock3.algebra.core import SubTensor

def _unpack_tensor(data, imap, ax, blklist):
    for q_label, (isec, ishape) in imap.items():
        ist, ied = isec
        if ax==0:
            tmp = data[ist:ied].reshape(ishape+(-1,))
        else:
            tmp = data[:,ist:ied].reshape((-1,)+ishape)
        blklist.append(SubTensor(tmp, q_labels=q_label))

def _pack_tensor(tsr, s_label, ax):
    u_map = {}
    v_map = {}
    left_offset = 0
    right_offset = 0
    for iblk in tsr.blocks:
        q_label_left = iblk.q_labels[:ax]
        q_label_right = iblk.q_labels[ax:]
        left_parity = np.mod(sum([iq.n for iq in q_label_left]) ,2)
        right_parity = np.mod(sum([iq.n for iq in q_label_right]) ,2)
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
        right_parity = np.mod(sum([iq.n for iq in q_label_right]) ,2)
        q_label_left += (SZ(left_parity), )
        q_label_right = (SZ(right_parity), ) + q_label_right
        q_label_mid = (SZ(left_parity), SZ(right_parity))
        if q_label_mid != s_label: continue
        ist, ied = u_map[q_label_left][0]
        jst, jed = v_map[q_label_right][0]
        data[ist:ied,jst:jed] = np.asarray(iblk).reshape(ied-ist, jed-jst)
    return data, u_map, v_map

def _run_svd(tsr, ax=2, **svd_opts):
    cutoff = svd_opts.pop("cutoff", None)
    full_matrices = svd_opts.pop("full_matrices", False)
    max_bond = svd_opts.pop("max_bond", None)

    s_labels = {0: [(SZ(0), SZ(0)), (SZ(1), SZ(1))],
                1: [(SZ(0), SZ(1)), (SZ(1), SZ(0))]}[tsr.parity]
    ublk = []
    sblk = []
    vblk = []
    for s_label in s_labels:
        data, umap, vmap = _pack_tensor(tsr, s_label, ax)
        u, s, v = np.linalg.svd(data, full_matrices=full_matrices)
        if cutoff is None:
            _unpack_tensor(u, umap, 0, ublk)
            _unpack_tensor(v, vmap, 1, vblk)
            sblk.append(SubTensor(np.diag(s), q_labels=s_label))
        else:
            idx = s > cutoff
            _unpack_tensor(u[:,idx], umap, 0, ublk)
            _unpack_tensor(v[idx], vmap, 1, vblk)
            sblk.append(SubTensor(np.diag(s[idx]), q_labels=s_label))
    u = SparseFermionTensor(blocks=ublk)
    s = SparseFermionTensor(blocks=sblk)
    v = SparseFermionTensor(blocks=vblk)
    return u, s, v

def tensor_svd(tsr, left_idx, **opts):
    right_idx = [idim for idim in range(tsr.ndim) if idim not in left_idx]
    neworder = tuple(left_idx) + tuple(right_idx)
    split_ax = len(left_idx)
    newtsr = tsr.transpose(neworder)
    u, s, v = _run_svd(newtsr, split_ax, **opts)
    return u, s, v

if __name__ == "__main__":
    np.random.seed(3)
    sx = SZ(0,0,0)
    sy = SZ(1,0,0)
    infox = BondInfo({sx:8, sy: 11})
    tsr = SparseFermionTensor.random((infox,infox,infox,infox,infox), dq=sy)

    u, s, v = _run_svd(tsr, ax=2)
    out = np.tensordot(u,s,axes=((2,),(0,)))
    out = np.tensordot(out,v,axes=((2,),(0,)))

    delta = out-tsr
    err = 0.
    for blk in delta:
        err += abs(np.asarray(blk)).sum()
    print("SVD Error=%.10f"%err)

    u, s, v = tensor_svd(tsr, (1,3))
    out1 = np.tensordot(u,s,axes=((2,),(0,)))
    out1 = np.tensordot(out1,v,axes=((2,),(0,)))
    out1 = out1.transpose((2,0,3,1,4))
    delta = out1-tsr
    err = 0.
    for blk in delta:
        err += abs(np.asarray(blk)).sum()
    print("transposed SVD Error=%.10f"%err)
