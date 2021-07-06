
/*
 * pyblock3: An Efficient python MPS/DMRG Library
 * Copyright (C) 2020 The pyblock3 developers. All Rights Reserved.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 *
 */

#include "flat_functor.hpp"
#include <algorithm>
#include <cstring>
#include <map>

template <typename Q>
tuple<py::array_t<uint32_t>, py::array_t<uint32_t>, py::array_t<double>,
      py::array_t<uint64_t>>
flat_sparse_tensor_diag(const py::array_t<uint32_t> &aqs,
                        const py::array_t<uint32_t> &ashs,
                        const py::array_t<double> &adata,
                        const py::array_t<uint64_t> &aidxs,
                        const py::array_t<int> &idxa,
                        const py::array_t<int> &idxb) {
    if (aqs.shape()[0] == 0)
        return std::make_tuple(aqs, ashs, adata, aidxs);
    int n_blocks_a = (int)aqs.shape()[0], ndima = (int)aqs.shape()[1];
    int nctr = (int)idxa.shape()[0], nctrb = (int)idxb.shape()[0];
    assert(nctr == nctrb && nctr != 0);
    const ssize_t asi = aqs.strides()[0] / sizeof(uint32_t),
                  asj = aqs.strides()[1] / sizeof(uint32_t);
    assert(memcmp(aqs.strides(), ashs.strides(), 2 * sizeof(ssize_t)) == 0);

    // sort contracted indices (for tensor a)
    int pidxa[nctr], pidxb[nctr], ctr_idx[nctr];
    const int *ppidxa = idxa.data(), *ppidxb = idxb.data();
    for (int i = 0; i < nctr; i++)
        ctr_idx[i] = i;
    sort(ctr_idx, ctr_idx + nctr,
         [ppidxa](int a, int b) { return ppidxa[a] < ppidxa[b]; });
    for (int i = 0; i < nctr; i++)
        pidxa[i] = ppidxa[ctr_idx[i]], pidxb[i] = ppidxb[ctr_idx[i]];

    assert(pidxa[nctr - 1] - pidxa[0] == nctr - 1);
    assert(pidxb[nctr - 1] - pidxb[0] == nctr - 1);
    assert(pidxa[nctr - 1] + 1 == pidxb[0]);
    assert(is_sorted(pidxb, pidxb + nctr));

    // free indices
    int ndiml = pidxa[0], ndimr = ndima - 1 - pidxb[nctr - 1];
    int outa[ndiml], outb[ndimr];
    for (int i = 0; i < ndiml; i++)
        outa[i] = i;
    for (int i = 0; i < ndimr; i++)
        outb[i] = i + pidxb[nctr - 1] + 1;

    // free and contracted dims
    vector<int> a_free_dim(n_blocks_a), b_free_dim(n_blocks_a),
        a_ctr_dim(n_blocks_a), b_ctr_dim(n_blocks_a);
    const uint32_t *psh = ashs.data();
    for (int i = 0; i < n_blocks_a; i++) {
        a_free_dim[i] = a_ctr_dim[i] = b_ctr_dim[i] = b_free_dim[i] = 1;
        int p = 0;
        for (int j = 0; j < ndiml; j++, p++)
            a_free_dim[i] *= psh[i * asi + p * asj];
        for (int j = 0; j < nctr; j++, p++)
            a_ctr_dim[i] *= psh[i * asi + p * asj];
        for (int j = 0; j < nctr; j++, p++)
            b_ctr_dim[i] *= psh[i * asi + p * asj];
        for (int j = 0; j < ndimr; j++, p++)
            b_free_dim[i] *= psh[i * asi + p * asj];
        assert(p == ndima);
    }

    int n_blocks_c = 0, ndimc = ndima - nctr;
    ssize_t csize = 0;
    vector<bool> oks(n_blocks_a);
    for (int ia = 0; ia < n_blocks_a; ia++) {
        bool ok = true;
        psh = aqs.data() + ia * asi;
        for (int j = 0; j < nctr && ok; j++)
            if (psh[(j + ndiml) * asj] != psh[(j + ndiml + nctr) * asj])
                ok = false;
        oks[ia] = ok;
        if (!ok)
            continue;
        assert(a_ctr_dim[ia] == b_ctr_dim[ia]);
        csize += (ssize_t)a_free_dim[ia] * a_ctr_dim[ia] * b_free_dim[ia];
        n_blocks_c++;
    }

    vector<ssize_t> sh = {n_blocks_c, ndimc};
    py::array_t<uint32_t> cqs(sh), cshs(sh);
    py::array_t<uint64_t> cidxs(vector<ssize_t>{n_blocks_c + 1});
    assert(cqs.strides()[1] == sizeof(uint32_t));
    assert(cshs.strides()[1] == sizeof(uint32_t));
    py::array_t<double> cdata(vector<ssize_t>{csize});
    uint32_t *pcqs = cqs.mutable_data(), *pcshs = cshs.mutable_data();
    uint64_t *pcidxs = cidxs.mutable_data();
    double *pc = cdata.mutable_data();
    const double *pa = adata.data();
    const uint64_t *pia = aidxs.data();
    pcidxs[0] = 0;
    for (int ia = 0, ic = 0; ia < n_blocks_a; ia++) {
        if (!oks[ia])
            continue;
        psh = aqs.data() + ia * asi;
        for (int j = 0; j < ndiml + nctr; j++)
            pcqs[ic * ndimc + j] = psh[j * asj];
        for (int j = ndiml + nctr; j < ndimr + ndiml + nctr; j++)
            pcqs[ic * ndimc + j] = psh[(j + nctr) * asj];
        psh = ashs.data() + ia * asi;
        for (int j = 0; j < ndiml + nctr; j++)
            pcshs[ic * ndimc + j] = psh[j * asj];
        for (int j = ndiml + nctr; j < ndimr + ndiml + nctr; j++)
            pcshs[ic * ndimc + j] = psh[(j + nctr) * asj];
        csize = (ssize_t)a_free_dim[ia] * a_ctr_dim[ia] * b_free_dim[ia];
        int shape_ai = a_ctr_dim[ia] * a_ctr_dim[ia] * b_free_dim[ia];
        int shape_ci = a_ctr_dim[ia] * b_free_dim[ia];
        int inca = (a_ctr_dim[ia] + 1) * b_free_dim[ia];
        int incc = b_free_dim[ia];
        for (int i = 0; i < a_free_dim[ia]; i++)
            for (int j = 0; j < b_free_dim[ia]; j++)
                dcopy(&a_ctr_dim[ia], &pa[pia[ia] + i * shape_ai + j], &inca,
                      &pc[pcidxs[ic] + i * shape_ci + j], &incc);
        pcidxs[ic + 1] = pcidxs[ic] + csize;
        ic++;
    }
    return std::make_tuple(cqs, cshs, cdata, cidxs);
}

template <typename Q, typename FL>
size_t flat_sparse_tensor_matmul(const py::array_t<int32_t> &plan,
                                 const py::array_t<FL> &adata,
                                 const py::array_t<FL> &bdata,
                                 py::array_t<FL> &cdata) {
    int nb = (int)plan.shape()[0], ndimp = (int)plan.shape()[1];
    assert(plan.strides()[1] == sizeof(int32_t));
    assert(ndimp == 9);
    const FL *pa = adata.data(), *pb = bdata.data();
    FL *pc = cdata.mutable_data();
    size_t nflop = 0;
    const CBLAS_LAYOUT layout = CblasRowMajor;
    vector<CBLAS_TRANSPOSE> ta(nb), tb(nb);
    vector<int> n(nb), m(nb), k(nb), gp(nb, 1), lda(nb), ldb(nb), ldc(nb);
    vector<FL> alpha(nb), beta(nb, 1.0);
    vector<const FL *> a(nb), b(nb);
    vector<FL *> c(nb);
    for (int i = 0, j = 0; i < nb; i++) {
        const int32_t *pp = plan.data() + ndimp * i;
        const int trans_b = pp[0], trans_a = pp[1];
        const int nn = pp[2], mm = pp[3], kk = pp[4];
        const int pib = pp[5], pia = pp[6], pic = pp[7];
        alpha[i] = pp[8];
        ta[i] = trans_a == -1 ? CblasNoTrans : CblasTrans;
        tb[i] = trans_b == 1 ? CblasNoTrans : CblasTrans;
        m[i] = mm, n[i] = nn, k[i] = kk;
        ldb[i] = trans_b == 1 ? nn : kk;
        lda[i] = trans_a == -1 ? kk : mm;
        ldc[i] = nn;
        a[i] = pa + pia, b[i] = pb + pib, c[i] = pc + pic;
        if (i == nb - 1 || (plan.data() + ndimp * (i + 1))[7] <= pic) {
            cblas_xgemm_batch<FL>(layout, &ta[j], &tb[j], &m[j], &n[j], &k[j],
                                  &alpha[j], &a[j], &lda[j], &b[j], &ldb[j],
                                  &beta[j], &c[j], &ldc[j], i + 1 - j, &gp[j]);
            j = i + 1;
        }
        nflop += (size_t)mm * nn * kk;
    }
    return nflop;
}

template <typename Q>
tuple<int, int, vector<map_uint_uint<Q>>, vector<map_uint_uint<Q>>>
flat_sparse_tensor_matmul_init(
    const py::array_t<uint32_t> &loqs, const py::array_t<uint32_t> &loshs,
    const py::array_t<uint32_t> &leqs, const py::array_t<uint32_t> &leshs,
    const py::array_t<uint32_t> &roqs, const py::array_t<uint32_t> &roshs,
    const py::array_t<uint32_t> &reqs, const py::array_t<uint32_t> &reshs) {
    int n_blocks_l = (int)loqs.shape()[0] + (int)leqs.shape()[0];
    int ndiml =
        loqs.shape()[0] != 0 ? (int)loqs.shape()[1] : (int)leqs.shape()[1];
    int n_blocks_r = (int)roqs.shape()[0] + (int)reqs.shape()[0];
    int ndimr =
        roqs.shape()[0] != 0 ? (int)roqs.shape()[1] : (int)reqs.shape()[1];
    vector<map_uint_uint<Q>> lqs(ndiml);
    vector<map_uint_uint<Q>> rqs(ndimr);
    if (loqs.shape()[0] != 0) {
        const ssize_t asi = loqs.strides()[0] / sizeof(uint32_t),
                      asj = loqs.strides()[1] / sizeof(uint32_t);
        uint32_t n = loqs.shape()[0];
        for (int i = 0; i < ndiml; i++)
            for (int j = 0; j < n; j++)
                lqs[i][loqs.data()[j * asi + i * asj]] =
                    loshs.data()[j * asi + i * asj];
    }
    if (leqs.shape()[0] != 0) {
        const ssize_t asi = leqs.strides()[0] / sizeof(uint32_t),
                      asj = leqs.strides()[1] / sizeof(uint32_t);
        uint32_t n = leqs.shape()[0];
        for (int i = 0; i < ndiml; i++)
            for (int j = 0; j < n; j++)
                lqs[i][leqs.data()[j * asi + i * asj]] =
                    leshs.data()[j * asi + i * asj];
    }
    if (roqs.shape()[0] != 0) {
        const ssize_t asi = roqs.strides()[0] / sizeof(uint32_t),
                      asj = roqs.strides()[1] / sizeof(uint32_t);
        uint32_t n = roqs.shape()[0];
        for (int i = 0; i < ndimr; i++)
            for (int j = 0; j < n; j++)
                rqs[i][roqs.data()[j * asi + i * asj]] =
                    roshs.data()[j * asi + i * asj];
    }
    if (reqs.shape()[0] != 0) {
        const ssize_t asi = reqs.strides()[0] / sizeof(uint32_t),
                      asj = reqs.strides()[1] / sizeof(uint32_t);
        uint32_t n = reqs.shape()[0];
        for (int i = 0; i < ndimr; i++)
            for (int j = 0; j < n; j++)
                rqs[i][reqs.data()[j * asi + i * asj]] =
                    reshs.data()[j * asi + i * asj];
    }
    int dl = (ndiml - 2) / 2, dr = (ndimr - 2) / 2;
    vector<map_uint_uint<Q>> vinfos(dl + dr + 2);
    vector<map_uint_uint<Q>> cinfos(dl + dr + 2);
    cinfos[0] = lqs[0];
    vinfos[0] = lqs[0];
    for (int i = 0; i < dl; i++) {
        vinfos[i + 1] = lqs[i + 1];
        cinfos[i + 1].insert(lqs[i + 1 + dl].begin(), lqs[i + 1 + dl].end());
    }
    for (int i = 0; i < dr; i++) {
        vinfos[i + 1 + dl] = rqs[i + 1];
        cinfos[i + 1 + dl].insert(rqs[i + 1 + dr].begin(),
                                  rqs[i + 1 + dr].end());
    }
    cinfos[dl + dr + 1] = rqs[ndimr - 1];
    vinfos[dl + dr + 1] = rqs[ndimr - 1];
    return std::make_tuple(dl, dr, cinfos, vinfos);
}

template <typename Q>
tuple<py::array_t<uint32_t>, py::array_t<uint32_t>, py::array_t<uint64_t>>
flat_sparse_tensor_tensordot_skeleton(const py::array_t<uint32_t> &aqs,
                                      const py::array_t<uint32_t> &ashs,
                                      const py::array_t<uint32_t> &bqs,
                                      const py::array_t<uint32_t> &bshs,
                                      const py::array_t<int> &idxa,
                                      const py::array_t<int> &idxb) {
    int n_blocks_a = (int)aqs.shape()[0], ndima = (int)aqs.shape()[1];
    int n_blocks_b = (int)bqs.shape()[0], ndimb = (int)bqs.shape()[1];
    if (n_blocks_a == 0) {
        py::array_t<uint32_t> cidxs{vector<ssize_t>{1}};
        cidxs.mutable_data()[0] = 0;
        return std::make_tuple(aqs, ashs, cidxs);
    } else if (n_blocks_b == 0) {
        py::array_t<uint32_t> cidxs{vector<ssize_t>{1}};
        cidxs.mutable_data()[0] = 0;
        return std::make_tuple(bqs, bshs, cidxs);
    }
    int nctr = (int)idxa.shape()[0];
    int ndimc = ndima - nctr + ndimb - nctr;
    const ssize_t asi = aqs.strides()[0] / sizeof(uint32_t),
                  asj = aqs.strides()[1] / sizeof(uint32_t);
    const ssize_t bsi = bqs.strides()[0] / sizeof(uint32_t),
                  bsj = bqs.strides()[1] / sizeof(uint32_t);
    assert(memcmp(aqs.strides(), ashs.strides(), 2 * sizeof(ssize_t)) == 0);
    assert(memcmp(bqs.strides(), bshs.strides(), 2 * sizeof(ssize_t)) == 0);

    // free indices
    int maska[ndima], maskb[ndimb], outa[ndima - nctr], outb[ndimb - nctr];
    memset(maska, -1, ndima * sizeof(int));
    memset(maskb, -1, ndimb * sizeof(int));
    for (int i = 0; i < nctr; i++)
        maska[idxa.data()[i]] = i, maskb[idxb.data()[i]] = i;
    for (int i = 0, j = 0; i < ndima; i++)
        if (maska[i] == -1)
            outa[j++] = i;
    for (int i = 0, j = 0; i < ndimb; i++)
        if (maskb[i] == -1)
            outb[j++] = i;

    // free and contracted dims
    vector<int> a_free_dim(n_blocks_a), b_free_dim(n_blocks_b);
    vector<vector<int>> a_free_dims(n_blocks_a), b_free_dims(n_blocks_b);
    const uint32_t *psh = ashs.data();
    for (int i = 0; i < n_blocks_a; i++) {
        a_free_dims[i].resize(ndima - nctr);
        a_free_dim[i] = 1;
        for (int j = 0; j < ndima - nctr; j++)
            a_free_dim[i] *= (a_free_dims[i][j] = psh[i * asi + outa[j] * asj]);
    }
    psh = bshs.data();
    for (int i = 0; i < n_blocks_b; i++) {
        b_free_dims[i].resize(ndimb - nctr);
        b_free_dim[i] = 1;
        for (int j = 0; j < ndimb - nctr; j++)
            b_free_dim[i] *= (b_free_dims[i][j] = psh[i * bsi + outb[j] * bsj]);
    }

    // contracted q_label hashs
    vector<size_t> ctrqas(n_blocks_a), ctrqbs(n_blocks_b), outqas(n_blocks_a),
        outqbs(n_blocks_b);
    psh = aqs.data();
    for (int i = 0; i < n_blocks_a; i++) {
        ctrqas[i] = q_labels_hash(psh + i * asi, nctr, idxa.data(), asj);
        outqas[i] = q_labels_hash(psh + i * asi, ndima - nctr, outa, asj);
    }
    psh = bqs.data();
    for (int i = 0; i < n_blocks_b; i++) {
        ctrqbs[i] = q_labels_hash(psh + i * bsi, nctr, idxb.data(), bsj);
        outqbs[i] = q_labels_hash(psh + i * bsi, ndimb - nctr, outb, bsj);
    }

    unordered_map<size_t, vector<int>> map_idx_b;
    for (int i = 0; i < n_blocks_b; i++)
        map_idx_b[ctrqbs[i]].push_back(i);

    unordered_map<size_t,
                  vector<pair<vector<uint32_t>, vector<pair<int, int>>>>>
        map_out_q;
    ssize_t csize = 0;
    int n_blocks_c = 0;
    for (int ia = 0; ia < n_blocks_a; ia++) {
        if (map_idx_b.count(ctrqas[ia])) {
            const auto &vb = map_idx_b.at(ctrqas[ia]);
            vector<uint32_t> q_out(ndimc);
            psh = aqs.data() + ia * asi;
            for (int i = 0; i < ndima - nctr; i++)
                q_out[i] = psh[outa[i] * asj];
            for (int ib : vb) {
                bool same = true;
                // check hashing conflicits
                for (int ict = 0; ict < nctr; ict++)
                    if (*aqs.data(ia, idxa.data()[ict]) !=
                        *bqs.data(ib, idxb.data()[ict])) {
                        same = false;
                        break;
                    }
                if (!same)
                    continue;
                size_t hout = outqas[ia];
                hout ^= outqbs[ib] + 0x9E3779B9 + (hout << 6) + (hout >> 2);
                psh = bqs.data() + ib * bsi;
                for (int i = 0; i < ndimb - nctr; i++)
                    q_out[i + ndima - nctr] = psh[outb[i] * bsj];
                int iq = 0;
                if (map_out_q.count(hout)) {
                    auto &vq = map_out_q.at(hout);
                    for (; iq < (int)vq.size() && q_out != vq[iq].first; iq++)
                        ;
                    if (iq == (int)vq.size()) {
                        vq.push_back(make_pair(
                            q_out, vector<pair<int, int>>{make_pair(ia, ib)}));
                        n_blocks_c++;
                    } else
                        vq[iq].second.push_back(make_pair(ia, ib));
                } else {
                    map_out_q[hout].push_back(make_pair(
                        q_out, vector<pair<int, int>>{make_pair(ia, ib)}));
                    n_blocks_c++;
                }
            }
        }
    }

    vector<ssize_t> sh = {n_blocks_c, ndimc};
    py::array_t<uint32_t> cqs(sh), cshs(sh);
    py::array_t<uint64_t> cidxs(vector<ssize_t>{n_blocks_c + 1});
    assert(cqs.strides()[1] == sizeof(uint32_t));
    assert(cshs.strides()[1] == sizeof(uint32_t));
    uint32_t *pcqs = cqs.mutable_data(), *pcshs = cshs.mutable_data();
    uint64_t *pcidxs = cidxs.mutable_data();
    vector<uint32_t> psha(n_blocks_a * ndima), pshb(n_blocks_b * ndimb);
    for (int i = 0; i < n_blocks_a; i++)
        for (int j = 0; j < ndima; j++)
            psha[i * ndima + j] = ashs.data()[i * asi + j * asj];
    for (int i = 0; i < n_blocks_b; i++)
        for (int j = 0; j < ndimb; j++)
            pshb[i * ndimb + j] = bshs.data()[i * bsi + j * bsj];

    pcidxs[0] = 0;
    for (auto &mq : map_out_q) {
        for (auto &mmq : mq.second) {
            int xia = mmq.second[0].first, xib = mmq.second[0].second;
            memcpy(pcqs, mmq.first.data(), ndimc * sizeof(uint32_t));
            memcpy(pcshs, a_free_dims[xia].data(),
                   (ndima - nctr) * sizeof(uint32_t));
            memcpy(pcshs + (ndima - nctr), b_free_dims[xib].data(),
                   (ndimb - nctr) * sizeof(uint32_t));
            pcidxs[1] = pcidxs[0] + (uint32_t)a_free_dim[xia] * b_free_dim[xib];
            pcqs += ndimc;
            pcshs += ndimc;
            pcidxs++;
        }
    }

    return std::make_tuple(cqs, cshs, cidxs);
}

template <typename Q>
py::array_t<int32_t> flat_sparse_tensor_matmul_plan(
    const py::array_t<uint32_t> &aqs, const py::array_t<uint32_t> &ashs,
    const py::array_t<uint64_t> &aidxs, const py::array_t<uint32_t> &bqs,
    const py::array_t<uint32_t> &bshs, const py::array_t<uint64_t> &bidxs,
    const py::array_t<int> &idxa, const py::array_t<int> &idxb,
    const py::array_t<uint32_t> &cqs, const py::array_t<uint64_t> &cidxs,
    bool ferm_op) {

    assert(bqs.shape()[0] != 0);
    if (aqs.shape()[0] == 0)
        return py::array_t<int32_t>(vector<ssize_t>{0, 9});

    int n_blocks_a = (int)aqs.shape()[0], ndima = (int)aqs.shape()[1];
    int n_blocks_b = (int)bqs.shape()[0], ndimb = (int)bqs.shape()[1];
    int n_blocks_c = (int)cqs.shape()[0], ndimc = (int)cqs.shape()[1];
    int nctr = (int)idxa.shape()[0];
    assert(ndimc == ndima - nctr + ndimb - nctr);
    const ssize_t asi = aqs.strides()[0] / sizeof(uint32_t),
                  asj = aqs.strides()[1] / sizeof(uint32_t);
    const ssize_t bsi = bqs.strides()[0] / sizeof(uint32_t),
                  bsj = bqs.strides()[1] / sizeof(uint32_t);
    const ssize_t csi = cqs.strides()[0] / sizeof(uint32_t),
                  csj = cqs.strides()[1] / sizeof(uint32_t);
    assert(memcmp(aqs.strides(), ashs.strides(), 2 * sizeof(ssize_t)) == 0);
    assert(memcmp(bqs.strides(), bshs.strides(), 2 * sizeof(ssize_t)) == 0);

    // sort contracted indices (for tensor a)
    int pidxa[nctr], pidxb[nctr], ctr_idx[nctr];
    const int *ppidxa = idxa.data(), *ppidxb = idxb.data();
    for (int i = 0; i < nctr; i++)
        ctr_idx[i] = i;
    sort(ctr_idx, ctr_idx + nctr,
         [ppidxa](int a, int b) { return ppidxa[a] < ppidxa[b]; });
    for (int i = 0; i < nctr; i++)
        pidxa[i] = ppidxa[ctr_idx[i]], pidxb[i] = ppidxb[ctr_idx[i]];

    // checking whether permute is necessary
    int trans_a = 0, trans_b = 0;
    if (nctr == 0)
        trans_a = 1;
    else if (pidxa[nctr - 1] - pidxa[0] == nctr - 1) {
        if (pidxa[0] == 0 ||
            is_shape_one(ashs.data(), n_blocks_a, pidxa[0], asi, asj))
            trans_a = 1;
        else if (pidxa[nctr - 1] == ndima - 1 ||
                 is_shape_one(ashs.data() + (pidxa[nctr - 1] + 1) * asj,
                              n_blocks_a, ndima - (pidxa[nctr - 1] + 1), asi,
                              asj))
            trans_a = -1;
    }

    if (nctr == 0)
        trans_b = 1;
    else if (is_sorted(pidxb, pidxb + nctr) &&
             pidxb[nctr - 1] - pidxb[0] == nctr - 1) {
        if (pidxb[0] == 0 ||
            is_shape_one(bshs.data(), n_blocks_b, pidxb[0], bsi, bsj))
            trans_b = 1;
        else if (pidxb[nctr - 1] == ndimb - 1 ||
                 is_shape_one(bshs.data() + (pidxb[nctr - 1] + 1) * bsj,
                              n_blocks_b, ndimb - (pidxb[nctr - 1] + 1), bsi,
                              bsj))
            trans_b = -1;
    }

    assert(trans_a != 0 && trans_b != 0);

    // free indices
    int maska[ndima], maskb[ndimb], outa[ndima - nctr], outb[ndimb - nctr];
    memset(maska, -1, ndima * sizeof(int));
    memset(maskb, -1, ndimb * sizeof(int));
    for (int i = 0; i < nctr; i++)
        maska[pidxa[i]] = i, maskb[pidxb[i]] = i;
    for (int i = 0, j = 0; i < ndima; i++)
        if (maska[i] == -1)
            outa[j++] = i;
    for (int i = 0, j = 0; i < ndimb; i++)
        if (maskb[i] == -1)
            outb[j++] = i;

    // free and contracted dims
    vector<int> a_free_dim(n_blocks_a), b_free_dim(n_blocks_b),
        ctr_dim(n_blocks_a);
    vector<vector<int>> a_free_dims(n_blocks_a), b_free_dims(n_blocks_b);
    const uint32_t *psh = ashs.data();
    for (int i = 0; i < n_blocks_a; i++) {
        a_free_dims[i].resize(ndima - nctr);
        a_free_dim[i] = ctr_dim[i] = 1;
        for (int j = 0; j < nctr; j++)
            ctr_dim[i] *= psh[i * asi + pidxa[j] * asj];
        for (int j = 0; j < ndima - nctr; j++)
            a_free_dim[i] *= (a_free_dims[i][j] = psh[i * asi + outa[j] * asj]);
    }
    psh = bshs.data();
    for (int i = 0; i < n_blocks_b; i++) {
        b_free_dims[i].resize(ndimb - nctr);
        b_free_dim[i] = 1;
        for (int j = 0; j < ndimb - nctr; j++)
            b_free_dim[i] *= (b_free_dims[i][j] = psh[i * bsi + outb[j] * bsj]);
    }

    // contracted q_label hashs
    vector<size_t> ctrqas(n_blocks_a), ctrqbs(n_blocks_b), outqas(n_blocks_a),
        outqbs(n_blocks_b), allqcs(n_blocks_c);
    psh = aqs.data();
    for (int i = 0; i < n_blocks_a; i++) {
        ctrqas[i] = q_labels_hash(psh + i * asi, nctr, pidxa, asj);
        outqas[i] = q_labels_hash(psh + i * asi, ndima - nctr, outa, asj);
    }
    psh = bqs.data();
    for (int i = 0; i < n_blocks_b; i++) {
        ctrqbs[i] = q_labels_hash(psh + i * bsi, nctr, pidxb, bsj);
        outqbs[i] = q_labels_hash(psh + i * bsi, ndimb - nctr, outb, bsj);
    }

    psh = cqs.data();
    for (int i = 0; i < n_blocks_c; i++) {
        size_t hl = q_labels_hash(psh + i * csi, ndima - nctr, csj);
        size_t hr = q_labels_hash(psh + i * csi + (ndima - nctr) * csj,
                                  ndimb - nctr, csj);
        allqcs[i] = hl ^ (hr + 0x9E3779B9 + (hl << 6) + (hl >> 2));
    }

    unordered_map<size_t, vector<int>> map_idx_b, map_idx_c;
    vector<vector<uint32_t>> vqcs(n_blocks_c);
    for (int i = 0; i < n_blocks_b; i++)
        map_idx_b[ctrqbs[i]].push_back(i);
    for (int i = 0; i < n_blocks_c; i++) {
        map_idx_c[allqcs[i]].push_back(i);
        vqcs[i].resize(ndimc);
        psh = cqs.data() + i * csi;
        for (int j = 0; j < ndimc; j++)
            vqcs[i][j] = psh[j * csj];
    }

    uint64_t *pia = (uint64_t *)aidxs.data(), *pib = (uint64_t *)bidxs.data();
    uint64_t *pic = (uint64_t *)cidxs.data();
    map<int, vector<vector<int>>> mr;
    int maxz = 0;
    for (int ia = 0; ia < n_blocks_a; ia++) {
        if (map_idx_b.count(ctrqas[ia])) {
            const auto &vb = map_idx_b.at(ctrqas[ia]);
            vector<uint32_t> q_out(ndimc);
            psh = aqs.data() + ia * asi;
            for (int i = 0; i < ndima - nctr; i++)
                q_out[i] = psh[outa[i] * asj];
            for (int ib : vb) {
                // check hashing conflicits
                bool same = true;
                for (int ict = 0; ict < nctr; ict++)
                    if (*aqs.data(ia, pidxa[ict]) !=
                        *bqs.data(ib, pidxb[ict])) {
                        same = false;
                        break;
                    }
                if (!same)
                    continue;
                size_t hout = outqas[ia];
                hout ^= outqbs[ib] + 0x9E3779B9 + (hout << 6) + (hout >> 2);
                psh = bqs.data() + ib * bsi;
                int xferm = 0;
                for (int i = 0; i < ndimb - nctr; i++)
                    q_out[i + ndima - nctr] = psh[outb[i] * bsj];
                if (ferm_op)
                    for (int i = 0; i < pidxb[0]; i++)
                        xferm ^= (psh[i * bsj] & 8) != 0;
                assert(map_idx_c.count(hout));
                auto &vq = map_idx_c.at(hout);
                int iq = 0;
                for (; iq < (int)vq.size() && q_out != vqcs[vq[iq]]; iq++)
                    ;
                assert(iq < (int)vq.size());
                int ic = vq[iq];
                mr[ic].push_back(vector<int>{trans_b, trans_a, b_free_dim[ib],
                                             a_free_dim[ia], ctr_dim[ia],
                                             (int)pib[ib], (int)pia[ia],
                                             (int)pic[ic], xferm ? -1 : 1});
                maxz = max(maxz, (int)mr[ic].size());
            }
        }
    }
    vector<vector<int>> r;
    for (int i = 0; i < maxz; i++) {
        for (auto &mmr : mr)
            if (i < mmr.second.size())
                r.push_back(mmr.second[i]);
    }
    if (r.size() == 0)
        return py::array_t<int32_t>(vector<ssize_t>{0, 9});
    assert(r.size() != 0);
    ssize_t rz = (ssize_t)r[0].size();
    vector<ssize_t> sh = {(ssize_t)r.size(), rz};
    py::array_t<int32_t> ar(sh);
    assert(ar.strides()[1] == sizeof(int32_t));
    for (size_t i = 0; i < r.size(); i++)
        memcpy(ar.mutable_data() + i * rz, r[i].data(), sizeof(int32_t) * rz);

    return ar;
}

#define TMPL_NAME flat_functor

#include "symmetry_tmpl.hpp"
#define TMPL_FL double
#include "symmetry_tmpl.hpp"
#undef TMPL_FL
#define TMPL_FL complex<double>
#include "symmetry_tmpl.hpp"
#undef TMPL_FL

#undef TMPL_NAME
