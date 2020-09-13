
import numpy as np
from itertools import accumulate
from ..symmetry import SZ, BondInfo


def flat_sparse_tensordot(aqs, ashs, adata, aidxs, bqs, bshs, bdata, bidxs, idxa, idxb):
    if len(aqs) == 0:
        return aqs, ashs, adata, aidxs
    elif len(bqs) == 0:
        return bqs, bshs, bdata, bidxs
    la, ndima = aqs.shape
    lb, ndimb = bqs.shape
    out_idx_a = np.delete(np.arange(0, ndima), idxa)
    out_idx_b = np.delete(np.arange(0, ndimb), idxb)
    ctrqas, outqas = aqs[:, idxa], aqs[:, out_idx_a]
    ctrqbs, outqbs = bqs[:, idxb], bqs[:, out_idx_b]

    map_idx_b = {}
    for ib in range(lb):
        ctrq = ctrqbs[ib].tobytes()
        if ctrq not in map_idx_b:
            map_idx_b[ctrq] = []
        map_idx_b[ctrq].append(ib)

    blocks_map = {}
    idxs = [0]
    qs = []
    shapes = []
    mats = []
    for ia in range(la):
        ctrq = ctrqas[ia].tobytes()
        mata = adata[aidxs[ia]: aidxs[ia + 1]].reshape(ashs[ia])
        if ctrq in map_idx_b:
            for ib in map_idx_b[ctrq]:
                outq = np.concatenate((outqas[ia], outqbs[ib]))
                matb = bdata[bidxs[ib]: bidxs[ib + 1]].reshape(bshs[ib])
                mat = np.tensordot(mata, matb, axes=(idxa, idxb))
                outqk = outq.tobytes()
                if outqk not in blocks_map:
                    blocks_map[outqk] = len(qs)
                    idxs.append(idxs[-1] + mat.size)
                    qs.append(outq)
                    shapes.append(mat.shape)
                    mats.append(mat.flatten())
                else:
                    mats[blocks_map[outqk]] += mat.flatten()

    return (np.array(qs, dtype=np.uint32),
            np.array(shapes, dtype=np.uint32),
            np.concatenate(mats) if len(mats) != 0 else np.array(
                [], dtype=np.float64),
            np.array(idxs, dtype=np.uint32))


def flat_sparse_add(aqs, ashs, adata, aidxs, bqs, bshs, bdata, bidxs):
    blocks_map = {q.tobytes(): iq for iq, q in enumerate(aqs)}
    data = adata.copy()
    dd = []
    bmats = []
    for ib in range(bqs.shape[0]):
        q = bqs[ib].tobytes()
        if q in blocks_map:
            ia = blocks_map[q]
            data[aidxs[ia]:aidxs[ia + 1]
                 ] += bdata[bidxs[ib]:bidxs[ib + 1]]
            dd.append(ib)
        else:
            bmats.append(bdata[bidxs[ib]:bidxs[ib + 1]])

    return (np.concatenate((aqs, np.delete(bqs, dd, axis=0))),
            np.concatenate((ashs, np.delete(bshs, dd, axis=0))),
            np.concatenate((data, *bmats)),
            None)


def flat_sparse_left_canonicalize(aqs, ashs, adata, aidxs):
    collected_rows = {}
    for i, q in enumerate(aqs[:, -1]):
        if q not in collected_rows:
            collected_rows[q] = [i]
        else:
            collected_rows[q].append(i)
    nblocks_r = len(collected_rows)
    qmats = [None] * aqs.shape[0]
    rmats = [None] * nblocks_r
    qqs = aqs
    qshs = ashs.copy()
    rqs = np.zeros((nblocks_r, 2), aqs.dtype)
    rshs = np.zeros((nblocks_r, 2), ashs.dtype)
    ridxs = np.zeros((nblocks_r + 1), aidxs.dtype)
    for ir, (qq, v) in enumerate(collected_rows.items()):
        pashs = ashs[v, :-1]
        l_shapes = np.prod(pashs, axis=1)
        mat = np.concatenate([adata[aidxs[ia]:aidxs[ia + 1]].reshape((sh, -1))
                              for sh, ia in zip(l_shapes, v)], axis=0)
        q, r = np.linalg.qr(mat, mode='reduced')
        rqs[ir, :] = qq
        rshs[ir] = r.shape
        ridxs[ir + 1] = ridxs[ir] + r.size
        rmats[ir] = r.flatten()
        qs = np.split(q, list(accumulate(l_shapes[:-1])), axis=0)
        assert len(qs) == len(v)
        qshs[v, -1] = r.shape[0]
        for q, ia in zip(qs, v):
            qmats[ia] = q.flatten()
    return (qqs, qshs, np.concatenate(qmats), None, rqs, rshs, np.concatenate(rmats), ridxs)


def flat_sparse_right_canonicalize(aqs, ashs, adata, aidxs):
    collected_cols = {}
    for i, q in enumerate(aqs[:, 0]):
        if q not in collected_cols:
            collected_cols[q] = [i]
        else:
            collected_cols[q].append(i)
    nblocks_l = len(collected_cols)
    lmats = [None] * nblocks_l
    qmats = [None] * aqs.shape[0]
    lqs = np.zeros((nblocks_l, 2), aqs.dtype)
    lshs = np.zeros((nblocks_l, 2), ashs.dtype)
    lidxs = np.zeros((nblocks_l + 1), aidxs.dtype)
    qqs = aqs
    qshs = ashs.copy()
    for il, (qq, v) in enumerate(collected_cols.items()):
        pashs = ashs[v, 1:]
        r_shapes = np.prod(pashs, axis=1)
        mat = np.concatenate(
            [adata[aidxs[ia]:aidxs[ia + 1]].reshape((-1, sh)).T for sh, ia in zip(r_shapes, v)], axis=0)
        q, r = np.linalg.qr(mat, mode='reduced')
        lqs[il, :] = qq
        lshs[il] = r.shape[::-1]
        lidxs[il + 1] = lidxs[il] + r.size
        lmats[il] = r.T.flatten()
        qs = np.split(q, list(accumulate(r_shapes[:-1])), axis=0)
        assert len(qs) == len(v)
        qshs[v, 0] = r.shape[0]
        for q, ia in zip(qs, v):
            qmats[ia] = q.T.flatten()
    return (lqs, lshs, np.concatenate(lmats), lidxs, qqs, qshs, np.concatenate(qmats), None)


def flat_sparse_transpose(aqs, ashs, adata, aidxs, axes):
    data = np.concatenate(
        [np.transpose(adata[i:j].reshape(sh), axes=axes).flatten()
         for i, j, sh in zip(aidxs, aidxs[1:], ashs)])
    return (aqs[:, axes], ashs[:, axes], data, aidxs)


def flat_sparse_get_infos(aqs, ashs):
    if len(aqs) == 0:
        return ()
    ndim = aqs.shape[1]
    bond_infos = tuple(BondInfo() for _ in range(ndim))
    for j in range(ndim):
        qs = aqs[:, j]
        shs = ashs[:, j]
        bis = bond_infos[j]
        for q, s in zip(qs, shs):
            bis[SZ.from_flat(q)] = s
    return bond_infos

def flat_sparse_skeleton(bond_infos, pattern=None, dq=None):
    """Create tensor skeleton from tuple of BondInfo."""
    it = np.zeros(tuple(len(i) for i in bond_infos), dtype=int)
    qsh = [sorted(i.items(), key=lambda x: x[0]) for i in bond_infos]
    nl, nd = it.ndim, np.max(it.shape)
    q = np.zeros((nl, nd), dtype=np.uint32)
    pq = np.zeros((nl, nd), dtype=object)
    sh = np.zeros((nl, nd), dtype=np.uint32)
    ix = np.arange(0, nl, dtype=int)
    if pattern is None:
        pattern = "+" * (nl - 1) + "-"
    for ii, i in enumerate(qsh):
        q[ii, :len(i)] = [k.to_flat() for k, v in i]
        pq[ii, :len(i)] = [k for k, v in i] if pattern[ii] == '+' else [-k for k, v in i]
        sh[ii, :len(i)] = [v for k, v in i]
    nit = np.nditer(it, flags=['multi_index'])
    xxs = []
    for _ in nit:
        x = nit.multi_index
        ps = pq[ix, x]
        if len(ps) == 1 or np.add.reduce(ps[:-1]) == (-ps[-1] if dq is None else dq - ps[-1]):
            xxs.append(x)
    xxs = np.array(xxs, dtype=np.uint32)
    q_labels = np.array(q[ix, xxs], dtype=np.uint32)
    shapes = np.array(sh[ix, xxs], dtype=np.uint32)
    idxs = np.zeros((len(xxs) + 1, ), dtype=np.uint32)
    idxs[1:] = np.cumsum(shapes.prod(axis=1))
    return q_labels, shapes, idxs
