import numpy as np
from pyblock3.algebra.symmetry import SZ, BondInfo
from pyblock3.algebra.fermion import SparseFermionTensor, FlatFermionTensor
from pyblock3.algebra.core import SubTensor

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

def _pack_flat_tensor(tsr, s_label, ax):
    nblks, ndim = tsr.q_labels.shape
    u_map = {}
    v_map = {}
    left_offset = 0
    right_offset = 0
    for iblk in range(nblks):
        q_label_left = tuple(SZ.from_flat(iq) for iq in tsr.q_labels[iblk][:ax])
        q_label_right = tuple(SZ.from_flat(iq) for iq in tsr.q_labels[iblk][ax:])
        left_parity = np.mod(sum([iq.n for iq in q_label_left]) ,2)
        right_parity = np.mod(sum([iq.n for iq in q_label_right]) ,2)
        q_label_left += (SZ(left_parity), )
        q_label_right = (SZ(right_parity), ) + q_label_right
        q_label_mid = (SZ(left_parity), SZ(right_parity))
        if q_label_mid != s_label: continue
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
        right_parity = np.mod(sum([iq.n for iq in q_label_right]) ,2)
        q_label_left += (SZ(left_parity), )
        q_label_right = (SZ(right_parity), ) + q_label_right
        q_label_mid = (SZ(left_parity), SZ(right_parity))
        if q_label_mid != s_label: continue
        ist, ied = u_map[q_label_left][0]
        jst, jed = v_map[q_label_right][0]
        blkst, blked = tsr.idxs[iblk], tsr.idxs[iblk+1]
        data[ist:ied,jst:jed] = np.asarray(tsr.data[blkst:blked]).reshape(ied-ist, jed-jst)
    return data, u_map, v_map

def _run_sparse_fermion_svd(tsr, ax=2, **svd_opts):
    cutoff = svd_opts.pop("cutoff", None)
    full_matrices = svd_opts.pop("full_matrices", False)
    max_bond = svd_opts.pop("max_bond", None)

    s_labels = {0: [(SZ(0), SZ(0)), (SZ(1), SZ(1))],
                1: [(SZ(0), SZ(1)), (SZ(1), SZ(0))]}[tsr.parity]
    ublk, sblk, vblk = [],[],[]
    for s_label in s_labels:
        data, umap, vmap = _pack_sparse_tensor(tsr, s_label, ax)
        u, s, v = np.linalg.svd(data, full_matrices=full_matrices)
        if cutoff is None:
            _unpack_sparse_tensor(u, umap, 0, ublk)
            _unpack_sparse_tensor(v, vmap, 1, vblk)
            sblk.append(SubTensor(np.diag(s), q_labels=s_label))
        else:
            idx = s > cutoff
            _unpack_sparse_tensor(u[:,idx], umap, 0, ublk)
            _unpack_sparse_tensor(v[idx], vmap, 1, vblk)
            sblk.append(SubTensor(np.diag(s[idx]), q_labels=s_label))
    u = SparseFermionTensor(blocks=ublk)
    s = SparseFermionTensor(blocks=sblk)
    v = SparseFermionTensor(blocks=vblk)
    return u, s, v

def _run_flat_fermion_svd(tsr, ax=2, **svd_opts):
    cutoff = svd_opts.pop("cutoff", None)
    full_matrices = svd_opts.pop("full_matrices", False)
    max_bond = svd_opts.pop("max_bond", None)
    nblk, ndim = tsr.q_labels.shape
    s_labels = {0: [(SZ(0), SZ(0)), (SZ(1), SZ(1))],
                1: [(SZ(0), SZ(1)), (SZ(1), SZ(0))]}[tsr.parity]
    udata, sdata, vdata = [],[],[]
    uq, sq, vq = [],[],[]
    ushapes, sshapes, vshapes = [],[],[]
    for s_label in s_labels:
        data, umap, vmap = _pack_flat_tensor(tsr, s_label, ax)
        u, s, v = np.linalg.svd(data, full_matrices=full_matrices)
        if cutoff is not None:
            idx = s > cutoff
            u = u[:,idx]
            v = v[idx]
            s = s[idx]

        s = np.diag(s)
        sq.append([SZ.to_flat(iq) for iq in s_label])
        sshapes.append(s.shape)
        sdata.append(s.ravel())
        _unpack_flat_tensor(u, umap, 0, udata, uq, ushapes)
        _unpack_flat_tensor(v, vmap, 1, vdata, vq, vshapes)

    sq = np.asarray(sq, dtype=int)
    sshapes = np.asarray(sshapes, dtype=int)
    sdata = np.concatenate(sdata)

    uq = np.asarray(uq, dtype=int)
    ushapes = np.asarray(ushapes, dtype=int)
    udata = np.concatenate(udata)

    vq = np.asarray(vq, dtype=int)
    vshapes = np.asarray(vshapes, dtype=int)
    vdata = np.concatenate(vdata)
    u = FlatFermionTensor(uq, ushapes, udata)
    s = FlatFermionTensor(sq, sshapes, sdata)
    v = FlatFermionTensor(vq, vshapes, vdata)
    return u, s ,v

def fermion_tensor_svd(tsr, left_idx, **opts):
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

if __name__ == "__main__":
    np.random.seed(3)
    sx = SZ(0,0,0)
    sy = SZ(1,0,0)
    infox = BondInfo({sx:8, sy: 11})
    tsr = SparseFermionTensor.random((infox,infox,infox,infox,infox), dq=sy)
    tsr_f = FlatFermionTensor.from_sparse(tsr)

    u, s, v = _run_sparse_fermion_svd(tsr, ax=2)
    us, ss, vs = _run_flat_fermion_svd(tsr_f, ax=2)

    out = np.tensordot(u,s,axes=((2,),(0,)))
    out = np.tensordot(out,v,axes=((2,),(0,)))

    outs = np.tensordot(us,ss,axes=((2,),(0,)))
    outs = np.tensordot(outs,vs,axes=((2,),(0,)))

    delta = out-tsr
    err = 0.
    for blk in delta:
        err += abs(np.asarray(blk)).sum()

    errs = np.amax((outs - tsr_f).data)
    print("Sparse SVD Error=%.10f"%err)
    print("Flat SVD Error=%.10f"%errs)

    u, s, v = fermion_tensor_svd(tsr, (1,3))
    u1, s1, v1 = fermion_tensor_svd(tsr_f, (1,3))

    out = np.tensordot(u,s,axes=((2,),(0,)))
    out = np.tensordot(out,v,axes=((2,),(0,)))
    out = out.transpose((2,0,3,1,4))

    out1 = np.tensordot(u1,s1,axes=((2,),(0,)))
    out1 = np.tensordot(out1,v1,axes=((2,),(0,)))
    out1 = out1.transpose((2,0,3,1,4))

    delta = out-tsr
    err = 0.
    for blk in delta:
        err += abs(np.asarray(blk)).sum()

    errs = np.amax((out1 - tsr_f).data)

    print("transposed Sparse SVD Error=%.10f"%err)
    print("transposed Flat SVD Error=%.10f"%errs)
