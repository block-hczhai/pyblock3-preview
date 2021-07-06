
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
python implementation for Flat version of block-sparse tensors
and block-sparse tensors with fermion factors.

This should only be used for debugging C++ code purpose.
"""

import numpy as np
from itertools import accumulate, groupby
from collections import Counter
from ..symmetry import SZ, BondInfo, BondFusingInfo


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
                [], dtype=adata.dtype),
            np.array(idxs, dtype=np.uint64))


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


def flat_sparse_tensor_svd(aqs, ashs, adata, aidxs, idx, linfo, rinfo, pattern):
    info = linfo & rinfo
    mats = {}
    for q in info:
        mats[q] = np.zeros((linfo[q], rinfo[q]), dtype=adata.dtype)
    items = {}
    xqls = [tuple(SZ.from_flat(q) for q in aq)
            for ia, aq in enumerate(aqs[:, :idx])]
    xqrs = [tuple(SZ.from_flat(q) for q in aq)
            for ia, aq in enumerate(aqs[:, idx:])]
    for iq in range(len(xqls)):
        qls, qrs = xqls[iq], xqrs[iq]
        ql = np.add.reduce([iq if ip == "+" else -iq for iq,
                            ip in zip(qls, pattern[:idx])])
        qr = np.add.reduce([iq if ip == "-" else -iq for iq,
                            ip in zip(qrs, pattern[idx:])])
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
        mats[q][lk:lk + lkn, rk:rk + rkn] = adata[aidxs[iq]
            :aidxs[iq + 1]].reshape((lkn, rkn))
    n = len(mats)
    sqs = np.zeros((n, 1), dtype=aqs.dtype)
    sshs = np.zeros((n, 1), dtype=ashs.dtype)
    smats = [None] * n
    l_blocks, r_blocks = [], []
    for iq, (q, mat) in enumerate(mats.items()):
        l, s, r = np.linalg.svd(mat, full_matrices=False)
        sqs[iq, 0] = q.to_flat()
        sshs[iq] = s.shape
        smats[iq] = s
        items[q][0].sort()
        items[q][1].sort()
        pqs = None
        for qs in items[q][0]:
            if qs == pqs:
                continue
            k, sh = linfo.finfo[q][qs]
            nk = np.multiply.reduce(sh)
            qq = np.array([x.to_flat() for x in qs + (q, )], dtype=aqs.dtype)
            sh = np.array(sh + (l.shape[-1], ), dtype=ashs.dtype)
            l_blocks.append((qq, sh, l[k:k + nk, :].flatten()))
            pqs = qs
        pqs = None
        for qs in items[q][1]:
            if qs == pqs:
                continue
            k, sh = rinfo.finfo[q][qs]
            nk = np.multiply.reduce(sh)
            qq = np.array([x.to_flat() for x in (q, ) + qs], dtype=aqs.dtype)
            sh = np.array((r.shape[0], ) + sh, dtype=ashs.dtype)
            r_blocks.append((qq, sh, r[:, k:k + nk].flatten()))
            pqs = qs
    lqs = np.array([xl[0] for xl in l_blocks], dtype=aqs.dtype)
    lshs = np.array([xl[1] for xl in l_blocks], dtype=ashs.dtype)
    ldata = np.concatenate([xl[2] for xl in l_blocks])
    rqs = np.array([xr[0] for xr in r_blocks], dtype=aqs.dtype)
    rshs = np.array([xr[1] for xr in r_blocks], dtype=ashs.dtype)
    rdata = np.concatenate([xr[2] for xr in r_blocks])
    sdata = np.concatenate(smats)
    return lqs, lshs, ldata, None, sqs, sshs, sdata, None, rqs, rshs, rdata, None


def flat_sparse_left_svd(aqs, ashs, adata, aidxs, indexed=False):
    collected_rows = {}
    for i, q in enumerate(aqs[:, -1]):
        if q not in collected_rows:
            collected_rows[q] = [i]
        else:
            collected_rows[q].append(i)
    nblocks_l = aqs.shape[0]
    nblocks_r = len(collected_rows)
    lmats = [None] * nblocks_l
    rmats = [None] * nblocks_r
    smats = [None] * nblocks_r
    lqs = aqs
    lshs = ashs.copy()
    sqs = np.zeros((nblocks_r, 1), aqs.dtype)
    sshs = np.zeros((nblocks_r, 1), ashs.dtype)
    sidxs = np.zeros((nblocks_r + 1), aidxs.dtype)
    rqs = np.zeros((nblocks_r, 2), aqs.dtype)
    rshs = np.zeros((nblocks_r, 2), ashs.dtype)
    ridxs = np.zeros((nblocks_r + 1), aidxs.dtype)
    lidx = np.zeros((nblocks_l, ), dtype=int)
    ill = 0
    for ir, (qq, v) in enumerate(collected_rows.items()):
        pashs = ashs[v, :-1]
        l_shapes = np.prod(pashs, axis=1)
        mat = np.concatenate([adata[aidxs[ia]:aidxs[ia + 1]].reshape((sh, -1))
                              for sh, ia in zip(l_shapes, v)], axis=0)
        l, s, r = np.linalg.svd(mat, full_matrices=False)
        rqs[ir, :] = qq
        rshs[ir] = r.shape
        ridxs[ir + 1] = ridxs[ir] + r.size
        rmats[ir] = r.flatten()
        sqs[ir, 0] = qq
        sshs[ir, 0] = s.shape[0]
        sidxs[ir + 1] = sidxs[ir] + s.shape[0]
        smats[ir] = s
        ls = np.split(l, list(accumulate(l_shapes[:-1])), axis=0)
        assert len(ls) == len(v)
        lshs[v, -1] = l.shape[-1]
        lidx[ill:ill + len(v)] = v
        for q, ia in zip(ls, v):
            lmats[ia] = q.flatten()
            ill += 1
    assert ill == nblocks_l
    rr = lqs[lidx], lshs[lidx], np.concatenate([lmats[x] for x in lidx]), None, \
        sqs, sshs, np.concatenate(smats), sidxs, \
        rqs, rshs, np.concatenate(rmats), ridxs
    return (rr, lidx) if indexed else r


def flat_sparse_right_svd(aqs, ashs, adata, aidxs, indexed=False):
    collected_cols = {}
    for i, q in enumerate(aqs[:, 0]):
        if q not in collected_cols:
            collected_cols[q] = [i]
        else:
            collected_cols[q].append(i)
    nblocks_l = len(collected_cols)
    nblocks_r = aqs.shape[0]
    lmats = [None] * nblocks_l
    rmats = [None] * nblocks_r
    smats = [None] * nblocks_l
    lqs = np.zeros((nblocks_l, 2), aqs.dtype)
    lshs = np.zeros((nblocks_l, 2), ashs.dtype)
    lidxs = np.zeros((nblocks_l + 1), aidxs.dtype)
    sqs = np.zeros((nblocks_l, 1), aqs.dtype)
    sshs = np.zeros((nblocks_l, 1), ashs.dtype)
    sidxs = np.zeros((nblocks_l + 1), aidxs.dtype)
    rqs = aqs.copy()
    rshs = ashs.copy()
    ridx = np.zeros((nblocks_r, ), dtype=int)
    irr = 0
    for il, (qq, v) in enumerate(collected_cols.items()):
        pashs = ashs[v, 1:]
        r_shapes = np.prod(pashs, axis=1)
        mat = np.concatenate(
            [adata[aidxs[ia]:aidxs[ia + 1]].reshape((-1, sh)) for sh, ia in zip(r_shapes, v)], axis=1)
        l, s, r = np.linalg.svd(mat, full_matrices=False)
        lqs[il, :] = qq
        lshs[il] = l.shape
        lidxs[il + 1] = lidxs[il] + l.size
        lmats[il] = l.flatten()
        sqs[il, 0] = qq
        sshs[il, 0] = s.shape[0]
        sidxs[il + 1] = sidxs[il] + s.shape[0]
        smats[il] = s
        rs = np.split(r, list(accumulate(r_shapes[:-1])), axis=1)
        assert len(rs) == len(v)
        rshs[v, 0] = r.shape[0]
        ridx[irr:irr + len(v)] = v
        for q, ia in zip(rs, v):
            rmats[ia] = q.flatten()
            irr += 1
    assert irr == nblocks_r
    rr = lqs, lshs, np.concatenate(lmats), lidxs, \
        sqs, sshs, np.concatenate(smats), sidxs, \
        rqs[ridx], rshs[ridx], np.concatenate([rmats[x] for x in ridx]), None
    return (rr, ridx) if indexed else rr


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
    return qqs, qshs, np.concatenate(qmats), None, rqs, rshs, np.concatenate(rmats), ridxs


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
    return lqs, lshs, np.concatenate(lmats), lidxs, qqs, qshs, np.concatenate(qmats), None


def flat_sparse_left_svd_indexed(aqs, ashs, adata, aidxs):
    return flat_sparse_left_svd(aqs, ashs, adata, aidxs, indexed=True)


def flat_sparse_right_svd_indexed(aqs, ashs, adata, aidxs):
    return flat_sparse_right_svd(aqs, ashs, adata, aidxs, indexed=True)


def flat_sparse_left_canonicalize_indexed(aqs, ashs, adata, aidxs):
    return flat_sparse_left_canonicalize(aqs, ashs, adata, aidxs), np.arange(0, aqs.shape[0], dtype=int)


def flat_sparse_right_canonicalize_indexed(aqs, ashs, adata, aidxs):
    return flat_sparse_right_canonicalize(aqs, ashs, adata, aidxs), np.arange(0, aqs.shape[0], dtype=int)


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
    """Create tensor skeleton from tuple of BondInfo.
    dq will not have effects if ndim == 1
        (blocks with different dq will be allowed)."""
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
        pq[ii, :len(i)] = [
            k for k, v in i] if pattern[ii] == '+' else [-k for k, v in i]
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
    idxs = np.zeros((len(xxs) + 1, ), dtype=np.uint64)
    idxs[1:] = np.cumsum(shapes.prod(axis=1))
    return q_labels, shapes, idxs


def flat_sparse_kron_sum_info(aqs, ashs, pattern):
    items = list(zip((tuple(SZ.from_flat(q) for q in qs) for qs in aqs),
                     (tuple(s.tolist()) for s in ashs)))
    return BondFusingInfo.kron_sum(items, pattern=pattern)


def flat_sparse_kron_product_info(infos, pattern):
    return BondFusingInfo.tensor_product(*infos, pattern=pattern)


def flat_sparse_fuse(aqs, ashs, adata, aidxs, idxs, info, pattern):
    if len(aqs) == 0:
        return aqs, ashs, adata, aidxs
    # unfused size
    fshs = np.prod(ashs[:, idxs], axis=1, dtype=np.uint32)
    # unfused q
    faqs = [(ia, tuple(SZ.from_flat(q) for q in aq))
            for ia, aq in enumerate(aqs[:, idxs])]
    # fused q
    fqs = [np.add.reduce([iq if ip == "+" else -iq for iq,
                          ip in zip(faq, pattern)]) for _, faq in faqs]
    blocks_map = {}
    for ia, faq in faqs:
        q = fqs[ia]
        if q not in info:
            continue
        x = info[q]
        k = info.finfo[q][faq][0]
        # shape in fused dim for this block
        nk = fshs[ia]
        rq = np.array([*aqs[ia, :idxs[0]], q.to_flat(),
                       * aqs[ia, idxs[-1] + 1:]], dtype=np.uint32)
        rsh = np.array([*ashs[ia, :idxs[0]], x,
                        * ashs[ia, idxs[-1] + 1:]], dtype=np.uint32)
        zrq = rq.tobytes()
        if zrq not in blocks_map:
            rdata = np.zeros(rsh, dtype=float)
            blocks_map[zrq] = (rq, rsh, rdata)
        else:
            rdata = blocks_map[zrq][2]
        rsh = rsh.copy()
        rsh[idxs[0]] = nk
        sl = tuple(slice(None) if ix != idxs[0] else slice(
            k, k + nk) for ix in range(len(rq)))
        rdata[sl] = adata[aidxs[ia]:aidxs[ia + 1]].reshape(rsh)
    rqs, rshs, rdata = zip(*blocks_map.values())
    return np.array(rqs, dtype=np.uint32), np.array(rshs, dtype=np.uint32), np.concatenate([d.flatten() for d in rdata]), None


def flat_sparse_trans_fusing_info(info):
    import block3.sz as block3
    minfo = block3.MapFusing()
    for k, v in info.items():
        mp = block3.MapVUIntPUV()
        for kk, (vv, tvv) in info.finfo[k].items():
            vk = block3.VectorUInt([x.to_flat() for x in kk])
            mp[vk] = (vv, block3.VectorUInt(tvv))
        minfo[k.to_flat()] = (v, mp)
    return minfo


def flat_sparse_kron_add(aqs, ashs, adata, aidxs, bqs, bshs, bdata, bidxs, infol, infor):
    if len(aqs) == 0:
        return bqs, bshs, bdata, bidxs
    elif len(bqs) == 0:
        return aqs, ashs, adata, aidxs
    lb = {k.to_flat(): v for k, v in infol.items()}
    rb = {k.to_flat(): v for k, v in infor.items()}
    na, nb = aqs.shape[0], bqs.shape[0]
    ndim = aqs.shape[1]
    assert ndim == bqs.shape[1]
    xaqs = [(i, q.tobytes()) for i, q in enumerate(aqs)]
    xbqs = [(i, q.tobytes()) for i, q in enumerate(bqs)]

    # find required new blocks and their shapes
    sub_mp = Counter()
    for ia, q in xaqs:
        if q not in sub_mp:
            sub_mp[q] = ia
    for ib, q in xbqs:
        if q not in sub_mp:
            sub_mp[q] = ib + na
    nc = len(sub_mp)
    cqs = np.zeros((nc, ndim), dtype=np.uint32)
    cshs = np.zeros((nc, ndim), dtype=np.uint32)
    cidxs = np.zeros((nc + 1, ), dtype=np.uint64)
    ic = 0
    ic_map = np.zeros((na + nb, ), dtype=int)
    for iab in sub_mp.values():
        ic_map[iab] = ic
        if iab < na:
            ia = iab
            cqs[ic] = aqs[ia]
            cshs[ic, 1:-1] = ashs[ia][1:-1]
            cshs[ic, 0] = lb[aqs[ia, 0]]
            cshs[ic, -1] = rb[aqs[ia, -1]]
        else:
            ib = iab - na
            cqs[ic] = bqs[ib]
            cshs[ic, 1:-1] = bshs[ib][1:-1]
            cshs[ic, 0] = lb[bqs[ib, 0]]
            cshs[ic, -1] = rb[bqs[ib, -1]]
        cidxs[ic + 1] = cidxs[ic] + np.prod(cshs[ic], dtype=np.uint64)
        ic += 1
    cdata = np.zeros((cidxs[-1], ), dtype=float)

    # copy a blocks to smaller index in new block
    for ia, q in xaqs:
        ic = ic_map[sub_mp[q]]
        mat = cdata[cidxs[ic]:cidxs[ic + 1]]
        mat.shape = cshs[ic]
        mat[: ashs[ia, 0], ..., : ashs[ia, -1]] += \
            adata[aidxs[ia]:aidxs[ia + 1]].reshape(ashs[ia])

    # copy b blocks to smaller index in new block
    for ib, q in xbqs:
        ic = ic_map[sub_mp[q]]
        mat = cdata[cidxs[ic]:cidxs[ic + 1]]
        mat.shape = cshs[ic]
        mat[-int(bshs[ib, 0]):, ..., -int(bshs[ib, -1]):] += \
            bdata[bidxs[ib]:bidxs[ib + 1]].reshape(bshs[ib])

    return cqs, cshs, cdata, cidxs


def flat_sparse_truncate_svd(lqs, lshs, ldata, lidxs, sqs, sshs, sdata, sidxs,
                             rqs, rshs, rdata, ridxs, max_bond_dim=-1, cutoff=0.0,
                             max_dw=0.0, norm_cutoff=0.0, eigen_values=False):
    ss = [(i, j, v) for i, (ist, ied) in enumerate(zip(sidxs, sidxs[1:]))
          for j, v in enumerate(sdata[ist:ied])]
    ss.sort(key=lambda x: -x[2])
    ss_trunc = ss.copy()
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
    l_blocks, r_blocks = [], []
    error = 0.0
    selected = np.array([False] * len(sqs), dtype=bool)
    iks, iksz = 0, 0
    nl, nr = len(lqs), len(rqs)
    nsdata = np.zeros((len(ss_trunc), ), dtype=sdata.dtype)
    gs = [(ik, list(g)) for ik, g in groupby(ss_trunc, key=lambda x: x[0])]
    nsshs = np.zeros((len(gs), 1), dtype=sshs.dtype)
    idxgl = np.zeros((len(sqs), ), dtype=int)
    idxgr = np.zeros((len(sqs), ), dtype=int)
    ikl, ikr = 0, 0
    for ik, q in enumerate(sqs[:, 0]):
        if ikl < nl and lqs[ikl, -1] == q:
            idxgl[ik] = ikl
            while ikl < nl and lqs[ikl, -1] == q:
                ikl += 1
        if ikr < nr and rqs[ikr, 0] == q:
            idxgr[ik] = ikr
            while ikr < nr and rqs[ikr, 0] == q:
                ikr += 1
    assert ikl == nl and ikr == nr
    for ik, g in gs:
        gl = np.array([ig[1] for ig in g], dtype=int)
        ns, ng = sidxs[ik + 1] - sidxs[ik], len(gl)
        gl_inv = np.array(list(set(range(ns)) - set(gl)), dtype=int)
        ikl = idxgl[ik]
        nkl, nkr = 0, 0
        while ikl + nkl < nl and lqs[ikl + nkl, -1] == sqs[ik, 0]:
            nkl += 1
        if nkl != 0:
            sh = lshs[ikl:ikl + nkl].copy()
            sh[:, -1] = ng
            dt = ldata[lidxs[ikl]:lidxs[ikl + nkl]
                       ].reshape((-1, ns))[:, gl].flatten()
            l_blocks.append((lqs[ikl:ikl + nkl], sh, dt,
                             np.arange(ikl, ikl + nkl, dtype=int)))
        nsshs[iks, 0] = ng
        nsdata[iksz:iksz + ng] = sdata[sidxs[ik]:sidxs[ik + 1]][gl]
        ikr = idxgr[ik]
        while ikr + nkr < nr and rqs[ikr + nkr, 0] == sqs[ik, -1]:
            nkr += 1
        if nkr != 0:
            sh = rshs[ikr:ikr + nkr].copy()
            sh[:, 0] = ng
            rrx = ridxs[ikr:ikr + nkr + 1]
            rrxr = (rrx - ridxs[ikr]) // ns
            dt = np.concatenate(
                [rdata[irst:ired].reshape((ns, -1))
                    for irst, ired in zip(rrx[:-1], rrx[1:])], axis=1)[gl, :]
            dt = np.concatenate(
                [dt[:, irst:ired].flatten() for irst, ired in zip(rrxr[:-1], rrxr[1:])])
            r_blocks.append((rqs[ikr:ikr + nkr], sh, dt,
                             np.arange(ikr, ikr + nkr, dtype=int)))
        if eigen_values:
            error += sdata[sidxs[ik]:sidxs[ik + 1]][gl_inv].sum()
        else:
            error += (sdata[sidxs[ik]:sidxs[ik + 1]][gl_inv] ** 2).sum()
        selected[ik] = True
        iks += 1
        iksz += ng
    for ik in range(len(sqs)):
        if not selected[ik]:
            if eigen_values:
                error += sdata[sidxs[ik]:sidxs[ik + 1]].sum()
            else:
                error += (sdata[sidxs[ik]:sidxs[ik + 1]] ** 2).sum()
    error = np.asarray(error).item()
    if len(l_blocks) == 0 or len(r_blocks) == 0:
        zq = np.zeros((0, lqs.shape[1]), dtype=lqs.dtype)
        zi = np.zeros((1, ), dtype=lidxs.dtype)
        zd = np.zeros((0, ), dtype=ldata.dtype)
    if len(l_blocks) != 0:
        nlqs = np.concatenate([xl[0] for xl in l_blocks], axis=0)
        nlshs = np.concatenate([xl[1] for xl in l_blocks], axis=0)
        nldata = np.concatenate([xl[2] for xl in l_blocks])
        nlidxs = None
    else:
        nlqs, nlshs, nldata, nlidxs = zq, zq, zd, zi
    if len(r_blocks) != 0:
        nrqs = np.concatenate([xr[0] for xr in r_blocks], axis=0)
        nrshs = np.concatenate([xr[1] for xr in r_blocks], axis=0)
        nrdata = np.concatenate([xr[2] for xr in r_blocks])
        nridxs = None
    else:
        nrqs, nrshs, nrdata, nridxs = zq, zq, zd, zi
    return nlqs, nlshs, nldata, nlidxs, sqs[selected], nsshs, nsdata, None, nrqs, nrshs, nrdata, nridxs, error
