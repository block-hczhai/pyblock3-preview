
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

#include "flat_sparse.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <list>
#include <math.h>
#include <numeric>

template <typename Q, typename FL>
void flat_fermion_tensor_transpose(const py::array_t<uint32_t> &aqs,
                                   const py::array_t<uint32_t> &ashs,
                                   const py::array_t<FL> &adata,
                                   const py::array_t<uint64_t> &aidxs,
                                   const py::array_t<int32_t> &perm,
                                   py::array_t<FL> &cdata) {
    int n_blocks_a = (int)ashs.shape()[0], ndima = (int)ashs.shape()[1];
    const ssize_t asi = ashs.strides()[0] / sizeof(uint32_t),
                  asj = ashs.strides()[1] / sizeof(uint32_t);
    const int *perma = (const int *)perm.data();
    const FL *pa = adata.data();
    const uint64_t *pia = aidxs.data();
    const uint32_t *psha = ashs.data();
    FL *pc = cdata.mutable_data();

    const uint32_t *apqs = aqs.data();
    vector<int> phase_a(n_blocks_a);
    for (int ia = 0; ia < n_blocks_a; ia++) {
        int pnuma[ndima];
        for (int j = 0; j < ndima; j++) {
            pnuma[j] = Q::to_q(apqs[ia * asi + j * asj]).parity();
        }
        int aparity_counter = 0;
        list<int> acounted;
        for (int xid = 0; xid < ndima; xid++) {
            int idx_a = perma[xid];
            for (int j = 0; j < idx_a; j++) {
                bool a_not_counted =
                    !(std::find(acounted.begin(), acounted.end(), j) !=
                      acounted.end());
                if (a_not_counted) {
                    aparity_counter += pnuma[j] * pnuma[idx_a];
                }
            }
            acounted.push_back(idx_a);
        }
        phase_a[ia] = pow(-1.0, (double)aparity_counter);
    }

    for (int ia = 0; ia < n_blocks_a; ia++) {
        const FL *a = pa + pia[ia];
        FL *c = pc + pia[ia];
        int shape_a[ndima];
        for (int i = 0; i < ndima; i++)
            shape_a[i] = psha[ia * asi + i * asj];
        uint64_t size_a = (uint64_t)(pia[ia + 1] - pia[ia]);
        tensor_transpose_impl<FL>(ndima, size_a, perma, shape_a, a, c,
                                  (FL)phase_a[ia], 0.0);
    }
}

template <typename Q, typename FL>
tuple<py::array_t<uint32_t>, py::array_t<uint32_t>, py::array_t<FL>,
      py::array_t<uint64_t>>
flat_fermion_tensor_tensordot(
    const py::array_t<uint32_t> &aqs, const py::array_t<uint32_t> &ashs,
    const py::array_t<FL> &adata, const py::array_t<uint64_t> &aidxs,
    const py::array_t<uint32_t> &bqs, const py::array_t<uint32_t> &bshs,
    const py::array_t<FL> &bdata, const py::array_t<uint64_t> &bidxs,
    const py::array_t<int> &idxa, const py::array_t<int> &idxb) {
    if (aqs.shape()[0] == 0)
        return std::make_tuple(aqs, ashs, adata, aidxs);
    else if (bqs.shape()[0] == 0)
        return std::make_tuple(bqs, bshs, bdata, bidxs);

    int n_blocks_a = (int)aqs.shape()[0], ndima = (int)aqs.shape()[1];
    int n_blocks_b = (int)bqs.shape()[0], ndimb = (int)bqs.shape()[1];
    int nctr = (int)idxa.shape()[0];
    int ndimc = ndima - nctr + ndimb - nctr;
    const ssize_t asi = aqs.strides()[0] / sizeof(uint32_t),
                  asj = aqs.strides()[1] / sizeof(uint32_t);
    const ssize_t bsi = bqs.strides()[0] / sizeof(uint32_t),
                  bsj = bqs.strides()[1] / sizeof(uint32_t);
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

    // permutation
    vector<int> perma(ndima, -1);
    vector<int> permb(ndimb, -1);
    vector<uint64_t> viatr(n_blocks_a + 1, -1);
    vector<uint64_t> vibtr(n_blocks_b + 1, -1);
    uint64_t *piatr = viatr.data(), *pibtr = vibtr.data();
    if (trans_a == 0) {
        for (int i = 0; i < nctr; i++)
            perma[i] = pidxa[i];
        for (int i = nctr; i < ndima; i++)
            perma[i] = outa[i - nctr];
    }
    if (trans_b == 0) {
        for (int i = 0; i < nctr; i++)
            permb[i] = pidxb[i];
        for (int i = nctr; i < ndimb; i++)
            permb[i] = outb[i - nctr];
    }

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
        outqbs(n_blocks_b);
    vector<FL> phase_a(n_blocks_a), phase_b(n_blocks_b);

    psh = aqs.data();
    const uint32_t *apqs = aqs.data();

    for (int i = 0; i < n_blocks_a; i++) {
        ctrqas[i] = q_labels_hash(psh + i * asi, nctr, pidxa, asj);
        outqas[i] = q_labels_hash(psh + i * asi, ndima - nctr, outa, asj);

        int pnuma[ndima];
        for (int j = 0; j < ndima; j++) {
            pnuma[j] = Q::to_q(apqs[i * asi + j * asj]).parity();
        }
        int aparity_counter = 0;
        list<int> acounted;
        for (int xid = 0; xid < nctr; xid++) {
            int idx_a = ppidxa[xid];
            for (int j = idx_a + 1; j < ndima; j++) {
                bool a_not_counted =
                    !(std::find(acounted.begin(), acounted.end(), j) !=
                      acounted.end());
                if (a_not_counted) {
                    aparity_counter += pnuma[j] * pnuma[idx_a];
                }
            }
            acounted.push_back(idx_a);
        }
        phase_a[i] = pow(-1.0, (double)aparity_counter);
    }

    psh = bqs.data();
    const uint32_t *bpqs = bqs.data();
    for (int i = 0; i < n_blocks_b; i++) {
        ctrqbs[i] = q_labels_hash(psh + i * bsi, nctr, pidxb, bsj);
        outqbs[i] = q_labels_hash(psh + i * bsi, ndimb - nctr, outb, bsj);

        int pnumb[ndimb];
        for (int j = 0; j < ndimb; j++) {
            pnumb[j] = Q::to_q(bpqs[i * bsi + j * bsj]).parity();
        }
        int bparity_counter = 0;
        list<int> bcounted;
        for (int xid = 0; xid < nctr; xid++) {
            int idx_b = ppidxb[xid];
            for (int j = 0; j < idx_b; j++) {
                bool b_not_counted =
                    !(std::find(bcounted.begin(), bcounted.end(), j) !=
                      bcounted.end());
                if (b_not_counted) {
                    bparity_counter += pnumb[j] * pnumb[idx_b];
                }
            }
            bcounted.push_back(idx_b);
        }
        phase_b[i] = pow(-1.0, (double)bparity_counter);
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
            piatr[ia] = 0;
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
                pibtr[ib] = 0;
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
                        csize += (ssize_t)a_free_dim[ia] * b_free_dim[ib],
                            n_blocks_c++;
                    } else
                        vq[iq].second.push_back(make_pair(ia, ib));
                } else {
                    map_out_q[hout].push_back(make_pair(
                        q_out, vector<pair<int, int>>{make_pair(ia, ib)}));
                    csize += (ssize_t)a_free_dim[ia] * b_free_dim[ib],
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
    py::array_t<FL> cdata(vector<ssize_t>{csize});
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

    // transpose
    FL *pa = (FL *)adata.data(), *pb = (FL *)bdata.data();
    uint64_t *pia = (uint64_t *)aidxs.data(), *pib = (uint64_t *)bidxs.data();
    if (trans_a == 0) {
        uint64_t iatr = 0;
        for (int ia = 0; ia < n_blocks_a; ia++)
            if (piatr[ia] != -1)
                piatr[ia] = iatr, iatr += pia[ia + 1] - pia[ia];
        FL *new_pa = new FL[iatr];
        uint64_t *new_pia = piatr;
        for (int ia = 0; ia < n_blocks_a; ia++)
            if (piatr[ia] != -1) {
                FL *a = pa + pia[ia], *new_a = new_pa + new_pia[ia];
                const int *shape_a = (const int *)(psha.data() + ia * ndima);
                uint64_t size_a = (uint64_t)(pia[ia + 1] - pia[ia]);
                tensor_transpose_impl<FL>(ndima, size_a, perma.data(), shape_a,
                                          a, new_a, 1.0, 0.0);
            }
        trans_a = 1;
        pa = new_pa;
        pia = new_pia;
    }

    if (trans_b == 0) {
        uint64_t ibtr = 0;
        for (int ib = 0; ib < n_blocks_b; ib++)
            if (pibtr[ib] != -1)
                pibtr[ib] = ibtr, ibtr += pib[ib + 1] - pib[ib];
        FL *new_pb = new FL[ibtr];
        uint64_t *new_pib = pibtr;
        for (int ib = 0; ib < n_blocks_b; ib++)
            if (pibtr[ib] != -1) {
                FL *b = pb + pib[ib], *new_b = new_pb + new_pib[ib];
                const int *shape_b = (const int *)(pshb.data() + ib * ndimb);
                uint64_t size_b = (uint64_t)(pib[ib + 1] - pib[ib]);
                tensor_transpose_impl<FL>(ndimb, size_b, permb.data(), shape_b,
                                          b, new_b, 1.0, 0.0);
            }
        trans_b = 1;
        pb = new_pb;
        pib = new_pib;
    }

    auto tra = trans_a == -1 ? "n" : "t";
    auto trb = trans_b == 1 ? "n" : "t";
    const FL alpha = 1.0, beta = 0.0;

    FL *pc = cdata.mutable_data();
    for (auto &mq : map_out_q) {
        for (auto &mmq : mq.second) {
            int xia = 0, xib = 0;
            for (size_t i = 0; i < mmq.second.size(); i++) {
                xia = mmq.second[i].first, xib = mmq.second[i].second;
                FL phase = phase_a[xia] * phase_b[xib];
                int ldb = trans_b == 1 ? b_free_dim[xib] : ctr_dim[xia];
                int lda = trans_a == -1 ? ctr_dim[xia] : a_free_dim[xia];
                int ldc = b_free_dim[xib];
                xgemm<FL>(trb, tra, &b_free_dim[xib], &a_free_dim[xia],
                          &ctr_dim[xia], &phase, pb + pib[xib], &ldb,
                          pa + pia[xia], &lda, i == 0 ? &beta : &alpha, pc,
                          &ldc);
            }
            pc += (uint32_t)a_free_dim[xia] * b_free_dim[xib];
        }
    }

    if (pa != adata.data())
        delete[] pa;
    if (pb != bdata.data())
        delete[] pb;

    return std::make_tuple(cqs, cshs, cdata, cidxs);
}

template <typename Q>
tuple<py::array_t<uint32_t>, py::array_t<uint32_t>, py::array_t<uint64_t>>
flat_fermion_tensor_skeleton(const vector<map_uint_uint<Q>> &infos,
                             uint32_t fdq) {
    int ndim = (int)infos.size();
    size_t nx = 1;
    for (int i = 0; i < ndim; i++) {
        nx *= infos[i].size();
    }
    const string pattern(ndim, '+');
    vector<uint32_t> qs, shs;
    vector<uint64_t> idxs(1, 0);
    vector<vector<pair<Q, uint32_t>>> infox;
    bond_info_trans<Q>(infos, pattern, infox, true);
    Q dq = Q::to_q(fdq);
    vector<uint32_t> qk(ndim), shk(ndim);
    for (size_t x = 0; x < nx; x++) {
        size_t xp = x;
        Q xq;
        for (int i = ndim - 1; i >= 0; xp /= infox[i].size(), i--)
            xq = xq + infox[i][xp % infox[i].size()].first;

        if (xq.parity() == dq.parity()) {
            uint32_t sz = 1;
            xp = x;
            for (int i = ndim - 1; i >= 0; xp /= infox[i].size(), i--) {
                auto &r = infox[i][xp % infox[i].size()];
                qk[i] = Q::from_q(r.first);
                shk[i] = r.second, sz *= r.second;
            }
            qs.insert(qs.end(), qk.begin(), qk.end());
            shs.insert(shs.end(), shk.begin(), shk.end());
            idxs.push_back(idxs.back() + sz);
        }
    }
    vector<ssize_t> sh = {(ssize_t)qs.size() / ndim, ndim};
    py::array_t<uint32_t> cqs(sh), cshs(sh),
        cidxs(vector<ssize_t>{(ssize_t)idxs.size()});
    assert(cqs.strides()[1] == sizeof(uint32_t));
    assert(cshs.strides()[1] == sizeof(uint32_t));
    memcpy(cqs.mutable_data(), qs.data(), qs.size() * sizeof(uint32_t));
    memcpy(cshs.mutable_data(), shs.data(), shs.size() * sizeof(uint32_t));
    memcpy(cidxs.mutable_data(), idxs.data(), idxs.size() * sizeof(uint64_t));
    return std::make_tuple(cqs, cshs, cidxs);
}

template <typename Q>
map_fusing flat_fermion_tensor_kron_sum_info(const py::array_t<uint32_t> &aqs,
                                             const py::array_t<uint32_t> &ashs,
                                             const string &pattern, int idxa,
                                             int idxb) {
    map_fusing r;
    if (aqs.shape()[0] == 0)
        return r;
    const int n_blocks_a = (int)aqs.shape()[0], ndima = (int)aqs.shape()[1];
    const ssize_t asi = aqs.strides()[0] / sizeof(uint32_t),
                  asj = aqs.strides()[1] / sizeof(uint32_t);
    assert(memcmp(aqs.strides(), ashs.strides(), 2 * sizeof(ssize_t)) == 0);
    vector<Q> qs(idxb - idxa);
    vector<uint32_t> shs(idxb - idxa);
    const uint32_t *pshs = ashs.data(), *pqs = aqs.data();
    unordered_map<uint32_t,
                  vector<pair<vector<Q>, pair<uint32_t, vector<uint32_t>>>>>
        xr;
    for (int i = 0; i < n_blocks_a; i++) {
        Q xq;
        for (int j = idxa; j < idxb; j++) {
            qs[j - idxa] = Q::to_q(pqs[i * asi + j * asj]);
            shs[j - idxa] = pshs[i * asi + j * asj];
            xq = xq + (pattern[j] == '+' ? qs[j - idxa] : -qs[j - idxa]);
        }
        uint32_t xxq = Q::from_q(xq);
        assert(Q::to_q(xxq) == xq);
        uint32_t sz =
            accumulate(shs.begin(), shs.end(), 1, multiplies<uint32_t>());
        xr[xxq].push_back(make_pair(qs, make_pair(sz, shs)));
    }
    vector<uint32_t> xqs(idxb - idxa);
    for (auto &m : xr) {
        vector<int> midx(m.second.size());
        for (size_t i = 0; i < m.second.size(); i++)
            midx[i] = i;
        sort(midx.begin(), midx.end(), [&m](size_t i, size_t j) {
            return less_pvsz<Q, pair<uint32_t, vector<uint32_t>>>(m.second[i],
                                                                  m.second[j]);
        });
        auto &mr = r[m.first];
        mr.first = 0;
        for (size_t im = 0; im < m.second.size(); im++) {
            auto &mm = m.second[midx[im]];
            bool same = mr.first == 0 ? false : true;
            for (int j = idxa; j < idxb; j++) {
                uint32_t qq = Q::from_q(mm.first[j - idxa]);
                same = same && qq == xqs[j - idxa];
                xqs[j - idxa] = qq;
            }
            if (same) {
                midx[im] = -1;
                continue;
            }
            mr.first += mm.second.first;
        }
        sort(midx.begin(), midx.end());
        mr.first = 0;
        for (size_t im = 0; im < m.second.size(); im++) {
            if (midx[im] == -1)
                continue;
            auto &mm = m.second[midx[im]];
            for (int j = idxa; j < idxb; j++) {
                uint32_t qq = Q::from_q(mm.first[j - idxa]);
                xqs[j - idxa] = qq;
            }
            mr.second[xqs] = make_pair(mr.first, mm.second.second);
            mr.first += mm.second.first;
        }
    }
    return r;
}

template <typename Q, typename FL>
tuple<py::array_t<uint32_t>, py::array_t<uint32_t>, py::array_t<FL>,
      py::array_t<uint64_t>, py::array_t<uint32_t>, py::array_t<uint32_t>,
      py::array_t<FL>, py::array_t<uint64_t>>
flat_fermion_tensor_qr(const py::array_t<uint32_t> &aqs,
                       const py::array_t<uint32_t> &ashs,
                       const py::array_t<FL> &adata,
                       const py::array_t<uint64_t> &aidxs, int idx,
                       const string &pattern, bool is_qr) {
    if (aqs.shape()[0] == 0)
        return std::make_tuple(aqs, ashs, adata, aidxs, aqs, ashs, adata,
                               aidxs);
    const int n_blocks_a = (int)aqs.shape()[0], ndima = (int)aqs.shape()[1];
    const ssize_t asi = aqs.strides()[0] / sizeof(uint32_t),
                  asj = aqs.strides()[1] / sizeof(uint32_t);
    assert(memcmp(aqs.strides(), ashs.strides(), 2 * sizeof(ssize_t)) == 0);

    const map_fusing linfo =
        flat_fermion_tensor_kron_sum_info<Q>(aqs, ashs, pattern, 0, idx);
    const map_fusing rinfo =
        flat_fermion_tensor_kron_sum_info<Q>(aqs, ashs, pattern, idx, ndima);

    vector<vector<uint32_t>> ufqsl(n_blocks_a), ufqsr(n_blocks_a);
    vector<pair<uint32_t, uint32_t>> fqs(n_blocks_a);
    unordered_map<uint32_t, size_t> mat_mp;
    const uint32_t *pshs = ashs.data(), *pqs = aqs.data();
    const uint64_t *pia = aidxs.data();
    const FL *pa = adata.data();
    size_t mat_size = 0;
    unordered_map<uint32_t, vector<int>> mat_idxl, mat_idxr;
    int max_lshape = 0, max_rshape = 0, max_mshape = 0;
    size_t max_tmp_size = 0;
    bool has_dq = false;
    Q dq;
    for (int ia = 0; ia < n_blocks_a; ia++) {
        Q xql, xqr;
        ufqsl[ia].resize(idx);
        ufqsr[ia].resize(ndima - idx);
        for (int j = 0; j < idx; j++) {
            ufqsl[ia][j] = pqs[ia * asi + j * asj];
            xql = xql + (pattern[j] == '+' ? Q::to_q(ufqsl[ia][j])
                                           : -Q::to_q(ufqsl[ia][j]));
        }
        for (int j = idx; j < ndima; j++) {
            ufqsr[ia][j - idx] = pqs[ia * asi + j * asj];
            xqr = xqr + (pattern[j] == '+' ? Q::to_q(ufqsr[ia][j - idx])
                                           : -Q::to_q(ufqsr[ia][j - idx]));
        }
        if (!has_dq)
            dq = xql + xqr, has_dq = true;
        else {
            assert(xql + xqr == dq);
            assert(dq - xql == xqr);
            assert(dq - xqr == xql);
        }
        uint32_t ql = Q::from_q(xql), qr = Q::from_q(xqr);
        fqs[ia] = make_pair(ql, qr);
        assert(linfo.count(ql) && rinfo.count(qr));
        mat_idxl[ql].push_back(ia);
        mat_idxr[qr].push_back(ia);
        if (!mat_mp.count(ql)) {
            mat_mp[ql] = mat_size;
            int ml = linfo.at(ql).first, mr = rinfo.at(qr).first;
            mat_size += (size_t)ml * mr;
            max_lshape = max(ml, max_lshape);
            max_rshape = max(mr, max_rshape);
            max_mshape = max(min(ml, mr), max_mshape);
            max_tmp_size = max((size_t)ml * mr, max_tmp_size);
        }
    }

    vector<FL> mat_data(mat_size);
    vector<pair<int, int>> lkns(n_blocks_a), rkns(n_blocks_a);
    for (int ia = 0; ia < n_blocks_a; ia++) {
        const uint32_t ql = fqs[ia].first, qr = fqs[ia].second;
        const vector<uint32_t> &qls = ufqsl[ia], &qrs = ufqsr[ia];
        assert(mat_mp.count(ql));
        int lfn = linfo.at(ql).first, rfn = rinfo.at(qr).first;
        const pair<uint32_t, vector<uint32_t>> &lpk =
            linfo.at(ql).second.at(qls);
        const pair<uint32_t, vector<uint32_t>> &rpk =
            rinfo.at(qr).second.at(qrs);
        int lk = (int)lpk.first, rk = (int)rpk.first;
        int lkn = (int)accumulate(lpk.second.begin(), lpk.second.end(), 1,
                                  multiplies<uint32_t>());
        int rkn = (int)accumulate(rpk.second.begin(), rpk.second.end(), 1,
                                  multiplies<uint32_t>());
        xlacpy<FL>("N", &rkn, &lkn, pa + pia[ia], &rkn,
                   mat_data.data() + mat_mp[ql] + rk + (size_t)lk * rfn, &rfn);
        lkns[ia] = make_pair(lk, lkn), rkns[ia] = make_pair(rk, rkn);
    }

    int n_blocks_s = (int)mat_mp.size();
    int n_blocks_l = 0, n_blocks_r = 0;
    size_t l_size = 0, r_size = 0;
    for (auto &v : mat_idxl) {
        int mm = min(linfo.at(v.first).first,
                     rinfo.at(Q::from_q(dq - Q::to_q(v.first))).first);
        sort(v.second.begin(), v.second.end(),
             [&lkns](int i, int j) { return lkns[i].first < lkns[j].first; });
        for (size_t i = 1; i < v.second.size(); i++) {
            const int vi = v.second[i], vip = v.second[i - 1];
            if (lkns[vi].first == lkns[vip].first)
                v.second[i - 1] = -1;
        }
        for (auto &vi : v.second)
            if (vi != -1)
                l_size += (size_t)mm * lkns[vi].second, n_blocks_l++;
    }
    for (auto &v : mat_idxr) {
        int mm = min(linfo.at(Q::from_q(dq - Q::to_q(v.first))).first,
                     rinfo.at(v.first).first);
        sort(v.second.begin(), v.second.end(),
             [&rkns](int i, int j) { return rkns[i].first < rkns[j].first; });
        for (size_t i = 1; i < v.second.size(); i++) {
            const int vi = v.second[i], vip = v.second[i - 1];
            if (rkns[vi].first == rkns[vip].first)
                v.second[i - 1] = -1;
        }
        for (auto &vi : v.second)
            if (vi != -1)
                r_size += (size_t)mm * rkns[vi].second, n_blocks_r++;
    }

    py::array_t<uint32_t> lqs(vector<ssize_t>{n_blocks_l, idx + 1}),
        lshs(vector<ssize_t>{n_blocks_l, idx + 1});
    py::array_t<uint64_t> lidxs(vector<ssize_t>{n_blocks_l + 1});
    py::array_t<uint32_t> rqs(vector<ssize_t>{n_blocks_r, ndima - idx + 1}),
        rshs(vector<ssize_t>{n_blocks_r, ndima - idx + 1});
    py::array_t<uint64_t> ridxs(vector<ssize_t>{n_blocks_r + 1});
    py::array_t<FL> ldata(vector<ssize_t>{(ssize_t)l_size});
    py::array_t<FL> rdata(vector<ssize_t>{(ssize_t)r_size});

    uint32_t *plqs = lqs.mutable_data(), *plshs = lshs.mutable_data();
    uint64_t *plidxs = lidxs.mutable_data();
    uint32_t *prqs = rqs.mutable_data(), *prshs = rshs.mutable_data();
    uint64_t *pridxs = ridxs.mutable_data();
    FL *pl = ldata.mutable_data(), *pr = rdata.mutable_data();

    int lwork = (is_qr ? max_rshape : max_lshape) * 34, info = 0;
    vector<FL> tau(max_mshape), work(lwork), tmpl(max_tmp_size),
        tmpr(max_tmp_size);
    int iis = 0, iil = 0, iir = 0;
    plidxs[0] = pridxs[0] = 0;
    if (is_qr)
        memset(pr, 0, sizeof(FL) * r_size);
    else
        memset(pl, 0, sizeof(FL) * l_size);
    for (auto &mq : mat_mp) {
        const uint32_t ql = mq.first, qr = Q::from_q(dq - Q::to_q(mq.first));
        FL *mat = mat_data.data() + mq.second;
        int ml = linfo.at(ql).first, mr = rinfo.at(qr).first, mm = min(ml, mr);
        const int lshape = ml, rshape = mr, mshape = mm;
        int lwork = max(ml, mr) * 34;
        if (is_qr) {
            memcpy(tmpr.data(), mat, sizeof(FL) * lshape * rshape);
            xgelqf<FL>(&rshape, &lshape, tmpr.data(), &rshape, tau.data(),
                       work.data(), &lwork, &info);
            assert(info == 0);
            memcpy(tmpl.data(), tmpr.data(), sizeof(FL) * lshape * rshape);
            xunglq<FL>(&mshape, &lshape, &mshape, tmpl.data(), &rshape,
                       tau.data(), work.data(), &lwork, &info);
            assert(info == 0);
        } else {
            memcpy(tmpl.data(), mat, sizeof(FL) * lshape * rshape);
            xgeqrf<FL>(&rshape, &lshape, tmpl.data(), &rshape, tau.data(),
                       work.data(), &lwork, &info);
            assert(info == 0);
            memcpy(tmpr.data(), tmpl.data(), sizeof(FL) * lshape * rshape);
            xungqr<FL>(&rshape, &mshape, &mshape, tmpr.data(), &rshape,
                       tau.data(), work.data(), &lwork, &info);
            assert(info == 0);
        }
        int isl = 0, isr = 0;
        for (auto &v : mat_idxl[ql]) {
            if (v == -1)
                continue;
            plidxs[iil + isl + 1] =
                plidxs[iil + isl] + (uint32_t)mm * lkns[v].second;
            if (!is_qr)
                for (int j = 0; j < lkns[v].second; j++)
                    memcpy(pl + plidxs[iil + isl] + j * mshape,
                           tmpl.data() + (lkns[v].first + j) * rshape,
                           sizeof(FL) * min(mshape, lkns[v].first + j + 1));
            else
                xlacpy<FL>("N", &mm, &lkns[v].second,
                           tmpl.data() + lkns[v].first * mr, &mr,
                           pl + plidxs[iil + isl], &mm);
            for (int i = 0; i < idx; i++) {
                plqs[(iil + isl) * (idx + 1) + i] = pqs[v * asi + i * asj];
                plshs[(iil + isl) * (idx + 1) + i] = pshs[v * asi + i * asj];
            }
            plqs[(iil + isl) * (idx + 1) + idx] =
                is_qr ? Q::from_q(Q::to_q(ql) - dq) : ql;
            plshs[(iil + isl) * (idx + 1) + idx] = mm;
            isl++;
        }
        for (auto &v : mat_idxr[qr]) {
            if (v == -1)
                continue;
            pridxs[iir + isr + 1] =
                pridxs[iir + isr] + (uint32_t)mm * rkns[v].second;
            if (is_qr) {
                int pxdr = rkns[v].first;
                for (int j = 0; j < min(mm, pxdr + rkns[v].second); j++) {
                    memcpy(pr + pridxs[iir + isr] + j * rkns[v].second +
                               max(j, pxdr) - pxdr,
                           tmpr.data() + max(j, pxdr) + j * rshape,
                           sizeof(FL) * (rkns[v].second + pxdr - max(j, pxdr)));
                }
            } else
                xlacpy<FL>("N", &rkns[v].second, &mm,
                           tmpr.data() + rkns[v].first, &mr,
                           pr + pridxs[iir + isr], &rkns[v].second);
            for (int i = idx; i < ndima; i++) {
                prqs[(iir + isr) * (ndima - idx + 1) + i - idx + 1] =
                    pqs[v * asi + i * asj];
                prshs[(iir + isr) * (ndima - idx + 1) + i - idx + 1] =
                    pshs[v * asi + i * asj];
            }
            prqs[(iir + isr) * (ndima - idx + 1) + 0] =
                is_qr ? Q::from_q(-Q::to_q(qr)) : Q::from_q(dq - Q::to_q(qr));
            prshs[(iir + isr) * (ndima - idx + 1) + 0] = mm;
            isr++;
        }
        iil += isl, iir += isr;
    }
    assert(iil == n_blocks_l && iir == n_blocks_r);
    assert(plidxs[iil] == l_size && pridxs[iir] == r_size);
    return std::make_tuple(lqs, lshs, ldata, lidxs, rqs, rshs, rdata, ridxs);
}

#define TMPL_NAME flat_fermion

#include "symmetry_tmpl.hpp"
#define TMPL_FL double
#include "symmetry_tmpl.hpp"
#undef TMPL_FL
#define TMPL_FL complex<double>
#include "symmetry_tmpl.hpp"
#undef TMPL_FL

#undef TMPL_NAME
