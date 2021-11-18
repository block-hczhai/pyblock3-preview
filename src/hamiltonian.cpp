
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

#include "hamiltonian.hpp"
#include "flat_sparse.hpp"
#include "max_flow.hpp"
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <numeric>
#include <unordered_map>

inline SZ from_op(int32_t op, const int32_t *orb_sym, const int32_t m_site,
                  const int32_t m_op) noexcept {
    int n = op / m_op ? -1 : 1;
    int twos = (op % m_site) ^ (op / m_op) ? -1 : 1;
    int pg = orb_sym[(op % m_op) / m_site];
    return SZ(n, twos, pg);
}

inline size_t op_hash(const int32_t *terms, int n,
                      const int32_t init = 0) noexcept {
    size_t h = (size_t)init;
    for (int i = 0; i < n; i++)
        h ^= (size_t)terms[i] + 0x9E3779B9 + (h << 6) + (h >> 2);
    return h;
}

typedef tuple<py::array_t<uint32_t>, py::array_t<uint32_t>,
              py::array_t<uint64_t>>
    op_skeleton;

inline void op_matmul(const op_skeleton &ska, const op_skeleton &skb,
                      const op_skeleton &skc, const double *pa,
                      const double *pb, double *pc) {
    int na = get<0>(ska).shape()[0], nb = get<0>(skb).shape()[0],
        nc = get<0>(skc).shape()[0];
    const uint32_t *pqa = get<0>(ska).data(), *pqb = get<0>(skb).data(),
                   *pqc = get<0>(skc).data();
    const uint32_t *psha = get<1>(ska).data(), *pshb = get<1>(skb).data(),
                   *pshc = get<1>(skc).data();
    const uint64_t *pia = get<2>(ska).data(), *pib = get<2>(skb).data(),
                   *pic = get<2>(skc).data();
    const double scale = 1.0, cfactor = 1.0;
    for (int ic = 0; ic < nc; ic++)
        for (int ia = 0; ia < na; ia++) {
            if (pqa[ia * 2 + 0] != pqc[ic * 2 + 0])
                continue;
            for (int ib = 0; ib < nb; ib++) {
                if (pqb[ib * 2 + 1] != pqc[ic * 2 + 1] ||
                    pqb[ib * 2 + 0] != pqa[ia * 2 + 1])
                    continue;
                int m = psha[ia * 2 + 0], n = pshb[ib * 2 + 1],
                    k = pshb[ib * 2 + 0];
                dgemm("N", "N", &n, &m, &k, &scale, pb + pib[ib], &n,
                      pa + pia[ia], &k, &cfactor, pc + pic[ic], &n);
            }
        }
}

// max_bond_dim >= -1: SVD
// max_bond_dim = -2: NC
// max_bond_dim = -3: CN
// max_bond_dim = -4: bipartite O(K^5)
// max_bond_dim = -5: fast bipartite O(K^4)
// max_bond_dim = -6: SVD (rescale)
// max_bond_dim = -7: SVD (rescale, fast)
// max_bond_dim = -8: SVD (fast)
vector<tuple<py::array_t<uint32_t>, py::array_t<uint32_t>, py::array_t<double>,
             py::array_t<uint64_t>>>
build_mpo(py::array_t<int32_t> orb_sym, py::array_t<double> h_values,
          py::array_t<int32_t> h_terms, double cutoff, int max_bond_dim) {
    bool rescale = false, fast_k4 = false;
    if (max_bond_dim == -6)
        rescale = true, fast_k4 = false, max_bond_dim = -1;
    else if (max_bond_dim == -7)
        rescale = true, fast_k4 = true, max_bond_dim = -1;
    else if (max_bond_dim == -8)
        rescale = false, fast_k4 = true, max_bond_dim = -1;
    else if (max_bond_dim == -5)
        fast_k4 = true, max_bond_dim = -4;
    const int m_site = 2, m_op = 16384;
    int n_sites = (int)orb_sym.shape()[0];
    long long int n_values = (long long int)h_values.shape()[0];
    long long int n_terms = (long long int)h_terms.shape()[0];
    int term_len = (int)h_terms.shape()[1];
    assert(n_terms == n_values);
    vector<SZ> left_q = {SZ(0, 0, 0)};
    unordered_map<uint32_t, uint32_t> info_l, info_r;
    info_l[SZ::from_q(left_q[0])] = 1;
    // terms
    vector<int32_t> term_sorted(n_terms * term_len);
    // length of each term; starting index of each term
    // at the beginning, term_i is all zero
    vector<int> term_l(n_terms), term_i(n_terms, 0);
    // index of current terms
    vector<vector<long long int>> cur_terms(1);
    cur_terms[0].resize(n_terms);
    // multiplying left matrix
    vector<vector<double>> cur_values(1);
    cur_values[0].resize(n_terms);
    const int32_t *pt = h_terms.data(), *porb = orb_sym.data();
    const double *pv = h_values.data();
    const ssize_t hsi = h_terms.strides()[0] / sizeof(uint32_t),
                  hsj = h_terms.strides()[1] / sizeof(uint32_t);
    vector<int> term_site(term_len);
    // pre-processing
    for (long long int it = 0; it < n_terms; it++) {
        int ix;
        SZ q(0, 0, 0);
        for (ix = 0; ix < term_len; ix++) {
            int32_t op = pt[it * hsi + ix * hsj];
            if (op == -1)
                break;
            q = q + from_op(op, porb, m_site, m_op);
            term_site[ix] = (op % m_op) / m_site;
            term_sorted[it * term_len + ix] = op;
        }
        if (q != SZ(0, 0, 0)) {
            cout << "Hamiltonian term #" << it
                 << " has a non-vanishing q: " << q << endl;
            abort();
        }
        term_l[it] = ix;
        cur_terms[0][it] = it;
        int ffactor = 1;
        for (int i = 0; i < ix; i++)
            for (int j = i + 1; j < ix; j++)
                if (term_site[i] > term_site[j])
                    ffactor = -ffactor;
        cur_values[0][it] = pv[it] * ffactor;
        stable_sort(term_sorted.data() + it * term_len,
                    term_sorted.data() + it * term_len + ix,
                    [m_op, m_site](int32_t i, int32_t j) {
                        return (i % m_op) / m_site < (j % m_op) / m_site;
                    });
    }
    vector<long long int> prefix_part;
    vector<long long int> prefix_terms;
    vector<double> prefix_values;
    // to save time, divide O(K^4) terms into K groups
    // for each iteration on site k, only O(K^3) terms are processed
    if (fast_k4) {
        vector<long long int> len_m(n_sites, 0);
        for (long long int it = 0; it < n_terms; it++)
            if (term_l[it] != 0)
                len_m[(term_sorted[it * term_len] % m_op) / m_site]++;
        long long int new_n_terms = 0;
        for (int ii = 0; ii < n_sites; ii++)
            new_n_terms += len_m[ii];
        prefix_part.resize(n_sites + 1, 0);
        for (int ii = 1; ii < n_sites; ii++)
            prefix_part[ii + 1] = prefix_part[ii] + len_m[ii - 1];
        prefix_terms.resize(new_n_terms);
        prefix_values = cur_values[0];
        for (long long int it = 0; it < n_terms; it++)
            if (term_l[it] != 0) {
                int ii = (term_sorted[it * term_len] % m_op) / m_site;
                long long int x = prefix_part[ii + 1]++;
                prefix_terms[x] = it;
            }
        assert(prefix_part[n_sites] == new_n_terms);
        cur_terms[0].resize(0);
        cur_values[0].resize(0);
    }
    // result mpo
    vector<tuple<py::array_t<uint32_t>, py::array_t<uint32_t>,
                 py::array_t<double>, py::array_t<uint64_t>>>
        rr(n_sites * 2);
    // do svd from left to right
    // time complexity: O(KDLN(log N))
    // K: n_sites, D: max_bond_dim, L: term_len, N: n_terms
    // using block-structure according to left q number
    // this is the map from left q number to its block index
    unordered_map<SZ, int> q_map;
    // for each iq block, a map from hashed repr of string of op in left block
    // to (mpo index, term index, left block string of op index)
    vector<unordered_map<size_t, vector<pair<pair<int, long long int>, int>>>>
        map_ls;
    // for each iq block, a map from hashed repr of string of op in right block
    // to (term index, right block string of op index)
    vector<unordered_map<size_t, vector<pair<long long int, int>>>> map_rs;
    // sparse repr of the connection (edge) matrix for each block
    vector<vector<pair<pair<int, int>, double>>> mats;
    // for each block, the nrow and ncol of the block
    vector<pair<long long int, long long int>> nms;
    vector<int> cur_term_i(n_terms, -1);
    py::array_t<int32_t> perm(vector<ssize_t>{4});
    perm.mutable_data()[0] = 0, perm.mutable_data()[1] = 2,
    perm.mutable_data()[2] = 3, perm.mutable_data()[3] = 1;
    const int32_t *pperm = perm.data();
    double rsc_factor = 1;
    for (int ii = 0; ii < n_sites; ii++) {
        cout << "MPO site" << setw(4) << ii << " / " << n_sites << endl;
        q_map.clear();
        map_ls.clear();
        map_rs.clear();
        mats.clear();
        nms.clear();
        info_r.clear();
        // site basis
        unordered_map<uint32_t, uint32_t> basis;
        basis[SZ::from_q(SZ(0, 0, 0))] = 1;
        basis[SZ::from_q(SZ(1, 1, porb[ii]))] = 1;
        basis[SZ::from_q(SZ(1, -1, porb[ii]))] = 1;
        basis[SZ::from_q(SZ(2, 0, 0))] = 1;
        long long int pholder_term = -1;
        // iter over all mpos
        for (int ip = 0; ip < (int)cur_values.size(); ip++) {
            SZ qll = left_q[ip];
            long long int cn = (long long int)cur_terms[ip].size(), cnr = cn;
            if (prefix_part.size() != 0 && ip == 0) {
                cn += prefix_part[ii + 1] - prefix_part[ii];
                if (prefix_part[ii + 1] != prefix_part[n_sites]) {
                    pholder_term = prefix_terms[prefix_part[ii + 1]];
                    q_map[qll] = 0;
                    map_ls.emplace_back();
                    map_rs.emplace_back();
                    mats.emplace_back();
                    nms.push_back(make_pair(1, 1));
                    map_ls[0][0].push_back(
                        make_pair(make_pair(0, pholder_term), 0));
                    map_rs[0][0].push_back(make_pair(pholder_term, 0));
                    mats[0].push_back(
                        make_pair(make_pair(0, 0),
                                  prefix_values[pholder_term] * rsc_factor));
                    cur_term_i[pholder_term] = 0;
                }
            }
            for (long long int ic = 0; ic < cn; ic++) {
                long long int it =
                    ic < cnr ? cur_terms[ip][ic]
                             : prefix_terms[ic - cnr + prefix_part[ii]];
                double itv = ic < cnr ? cur_values[ip][ic]
                                      : prefix_values[it] * rsc_factor;
                int ik = term_i[it], k = ik, kmax = term_l[it];
                long long int itt = it * term_len;
                // separate the current product into two parts
                // (left block part and right block part)
                for (; k < kmax && (term_sorted[itt + k] % m_op) / m_site <= ii;
                     k++)
                    ;
                // first right site position
                cur_term_i[it] = k;
                size_t hl = op_hash(term_sorted.data() + itt + ik, k - ik, ip);
                size_t hr = op_hash(term_sorted.data() + itt + k, kmax - k);
                SZ ql = qll;
                for (int i = ik; i < k; i++)
                    ql = ql + from_op(term_sorted[itt + i], porb, m_site, m_op);
                if (q_map.count(ql) == 0) {
                    q_map[ql] = (int)q_map.size();
                    map_ls.emplace_back();
                    map_rs.emplace_back();
                    mats.emplace_back();
                    nms.push_back(make_pair(0, 0));
                }
                int iq = q_map.at(ql), il = -1, ir = -1;
                long long int &nml = nms[iq].first, &nmr = nms[iq].second;
                auto &mpl = map_ls[iq];
                auto &mpr = map_rs[iq];
                if (mpl.count(hl)) {
                    int iq = 0;
                    auto &vq = mpl.at(hl);
                    for (; iq < vq.size(); iq++) {
                        int vip = vq[iq].first.first;
                        long long int vit = vq[iq].first.second;
                        long long int vitt = vit * term_len;
                        int vik = term_i[vit], vk = cur_term_i[vit];
                        if (vip == ip && vk - vik == k - ik &&
                            equal(term_sorted.data() + vitt + vik,
                                  term_sorted.data() + vitt + vk,
                                  term_sorted.data() + itt + ik))
                            break;
                    }
                    if (iq == (int)vq.size())
                        vq.push_back(make_pair(make_pair(ip, it), il = nml++));
                    else
                        il = vq[iq].second;
                } else
                    mpl[hl].push_back(make_pair(make_pair(ip, it), il = nml++));
                if (mpr.count(hr)) {
                    int iq = 0;
                    auto &vq = mpr.at(hr);
                    for (; iq < vq.size(); iq++) {
                        int vit = vq[iq].first, vitt = vit * term_len;
                        int vkmax = term_l[vit], vk = cur_term_i[vit];
                        if (vkmax - vk == kmax - k &&
                            equal(term_sorted.data() + vitt + vk,
                                  term_sorted.data() + vitt + vkmax,
                                  term_sorted.data() + itt + k))
                            break;
                    }
                    if (iq == (int)vq.size())
                        vq.push_back(make_pair(it, ir = nmr++));
                    else
                        ir = vq[iq].second;
                } else
                    mpr[hr].push_back(make_pair(it, ir = nmr++));
                mats[iq].push_back(make_pair(make_pair(il, ir), itv));
            }
        }
        vector<array<vector<double>, 3>> svds;
        vector<array<vector<int>, 2>> mvcs;
        if (max_bond_dim == -4)
            mvcs.resize(q_map.size());
        else
            svds.resize(q_map.size());
        vector<SZ> qs(q_map.size());
        int s_kept_total = 0, nr_total = 0;
        double res_s_sum = 0, res_factor = 1;
        size_t res_s_count = 0;
        for (auto &mq : q_map) {
            int iq = mq.second;
            qs[iq] = mq.first;
            auto &matvs = mats[iq];
            auto &nm = nms[iq];
            int szl = nm.first, szr = nm.second, szm;
            if (max_bond_dim == -2) // NC
                szm = szl;
            else if (max_bond_dim == -3) // CN
                szm = szr;
            else // bipartitie (-4/-5) / SVD (>= -1)
                szm = min(szl, szr);
            if (max_bond_dim != -4) {
                if (pholder_term != -1 && iq == 0)
                    szm = min(szl - 1, szr) + 1;
                svds[iq][0].resize((size_t)szm * szl);
                svds[iq][1].resize(szm);
                svds[iq][2].resize((size_t)szm * szr);
            }
            int s_kept = 0;
            if (max_bond_dim == -4) { // bipartite
                FLOW flow(szl + szr);
                for (auto &lrv : matvs)
                    flow.resi[lrv.first.first][lrv.first.second + szl] = 1;
                for (int i = 0; i < szl; i++)
                    flow.resi[szl + szr][i] = 1;
                for (int i = 0; i < szr; i++)
                    flow.resi[szl + i][szl + szr + 1] = 1;
                flow.MVC(0, szl, szl, szr, mvcs[iq][0], mvcs[iq][1]);
                // placeholder I * O(K^4) term must be of Normal/Complementary
                // type
                if (pholder_term != -1 && iq == 0) {
                    if ((mvcs[iq][0].size() == 0 || mvcs[iq][0][0] != 0))
                        mvcs[iq][0].push_back(0);
                    if (mvcs[iq][1].size() != 0 && mvcs[iq][1][0] == 0)
                        mvcs[iq][1] = vector<int>(mvcs[iq][1].begin() + 1,
                                                  mvcs[iq][1].end());
                }
                s_kept = (int)mvcs[iq][0].size() + (int)mvcs[iq][1].size();
            } else if (max_bond_dim == -2) { // NC
                memset(svds[iq][0].data(), 0,
                       sizeof(double) * svds[iq][0].size());
                memset(svds[iq][2].data(), 0,
                       sizeof(double) * svds[iq][2].size());
                for (auto &lrv : matvs)
                    svds[iq][2][lrv.first.first * szr + lrv.first.second] +=
                        lrv.second;
                for (int i = 0; i < szm; i++)
                    svds[iq][0][i * szm + i] = svds[iq][1][i] = 1;
                s_kept = szm;
            } else if (max_bond_dim == -3) { // CN
                memset(svds[iq][0].data(), 0,
                       sizeof(double) * svds[iq][0].size());
                memset(svds[iq][2].data(), 0,
                       sizeof(double) * svds[iq][2].size());
                for (auto &lrv : matvs)
                    svds[iq][0][lrv.first.first * szr + lrv.first.second] +=
                        lrv.second;
                for (int i = 0; i < szm; i++)
                    svds[iq][2][i * szr + i] = svds[iq][1][i] = 1;
                s_kept = szm;
            } else { // SVD
                int lwork = max(szl, szr) * 34, info;
                vector<double> mat((size_t)szl * szr, 0), work(lwork);
                if (pholder_term != -1 && iq == 0) {
                    for (auto &lrv : matvs)
                        if (lrv.first.first == 0)
                            svds[iq][2][lrv.first.second] += lrv.second;
                        else
                            mat[(lrv.first.first - 1) * szr +
                                lrv.first.second] += lrv.second;
                    szl--;
                    svds[iq][1][0] = 1;
                    svds[iq][0][0] = 1;
                    dgesvd("S", "S", &szr, &szl, mat.data(), &szr,
                           svds[iq][1].data() + 1, svds[iq][2].data() + szr,
                           &szr, svds[iq][0].data() + 1 + szm, &szm,
                           work.data(), &lwork, &info);
                    szl++;
                } else {
                    for (auto &lrv : matvs)
                        mat[lrv.first.first * szr + lrv.first.second] +=
                            lrv.second;
                    dgesvd("S", "S", &szr, &szl, mat.data(), &szr,
                           svds[iq][1].data(), svds[iq][2].data(), &szr,
                           svds[iq][0].data(), &szm, work.data(), &lwork,
                           &info);
                }
                res_s_sum += accumulate(svds[iq][1].begin(), svds[iq][1].end(),
                                        0, plus<double>());
                res_s_count += svds[iq][1].size();
                if (!rescale) {
                    for (int i = 0; i < szm; i++)
                        if (svds[iq][1][i] > cutoff)
                            s_kept++;
                        else
                            break;
                    if (max_bond_dim > 1)
                        s_kept = min(s_kept, max_bond_dim);
                    svds[iq][1].resize(s_kept);
                } else
                    s_kept = szm;
            }
            if (s_kept != 0)
                info_r[SZ::from_q(mq.first)] = s_kept;
            s_kept_total += s_kept;
            nr_total += szr;
        }
        if (rescale) {
            s_kept_total = 0;
            assert(max_bond_dim == -1);
            res_factor = res_s_sum / res_s_count;
            // keep only 1 significant digit
            uint64_t rrepr =
                (uint64_t &)res_factor & ~(((uint64_t)1 << 52) - 1);
            res_factor = (double &)rrepr;
            if (res_factor == 0)
                res_factor = 1;
            for (auto &mq : q_map) {
                int s_kept = 0;
                int iq = mq.second;
                auto &nm = nms[iq];
                int szl = nm.first, szr = nm.second;
                int szm = min(szl, szr);
                if (pholder_term != -1 && iq == 0)
                    szm = min(szl - 1, szr) + 1;
                for (int i = 0; i < szm; i++)
                    svds[iq][1][i] /= res_factor;
                for (int i = 0; i < szm; i++)
                    if (svds[iq][1][i] > cutoff)
                        s_kept++;
                    else
                        break;
                svds[iq][1].resize(s_kept);
                if (s_kept != 0)
                    info_r[SZ::from_q(mq.first)] = s_kept;
                s_kept_total += s_kept;
            }
        }

        // [optional optimization] ip can be inner loop
        // [optional optimization] better set basis as input; cannot remove
        // orb_sym currently just construct basis from orb_sym use skelton and
        // info_l; info_r to build skelton; physical indices at the end
        // skeleton: +-+-; dq = 0
        // skeleton guarentees that right indices are contiguous
        vector<map_uint_uint<SZ>> infos = {
            (map_uint_uint<SZ> &)info_l, (map_uint_uint<SZ> &)info_r,
            (map_uint_uint<SZ> &)basis, (map_uint_uint<SZ> &)basis};
        auto skl = flat_sparse_tensor_skeleton<SZ>(infos, "+-+-",
                                                   SZ::from_q(SZ(0, 0, 0)));
        // separate odd and even
        int n_odd = 0, n_total = get<0>(skl).shape()[0];
        ssize_t size_odd = 0;
        vector<bool> skf(n_total, false);
        const uint32_t *psklqs = get<0>(skl).data();
        const uint32_t *psklshs = get<1>(skl).data();
        const uint64_t *psklis = get<2>(skl).data();
        const ssize_t sklqi = get<0>(skl).strides()[0] / sizeof(uint32_t),
                      sklqj = get<0>(skl).strides()[1] / sizeof(uint32_t);
        for (int i = 0; i < n_total; i++)
            if (SZ::to_q(psklqs[i * sklqi + 2 * sklqj]).is_fermion() !=
                SZ::to_q(psklqs[i * sklqi + 3 * sklqj]).is_fermion())
                n_odd++, skf[i] = true, size_odd += psklis[i + 1] - psklis[i];
        int n_even = n_total - n_odd;

        ssize_t size_even = psklis[n_total] - size_odd;
        auto &rodd = rr[ii * 2], &reven = rr[ii * 2 + 1];
        auto &oqs = get<0>(rodd), &oshs = get<1>(rodd);
        auto &oi = get<3>(rodd);
        auto &eqs = get<0>(reven), &eshs = get<1>(reven);
        auto &ei = get<3>(reven);
        oqs = py::array_t<uint32_t>(vector<ssize_t>{n_odd, 4});
        oshs = py::array_t<uint32_t>(vector<ssize_t>{n_odd, 4});
        oi = py::array_t<uint64_t>(vector<ssize_t>{n_odd + 1});
        eqs = py::array_t<uint32_t>(vector<ssize_t>{n_even, 4});
        eshs = py::array_t<uint32_t>(vector<ssize_t>{n_even, 4});
        auto odata = py::array_t<double>(vector<ssize_t>{size_odd});
        auto edata = py::array_t<double>(vector<ssize_t>{size_even});
        ei = py::array_t<uint64_t>(vector<ssize_t>{n_even + 1});
        uint32_t *poqs = oqs.mutable_data(), *poshs = oshs.mutable_data();
        uint64_t *poi = oi.mutable_data();
        uint32_t *peqs = eqs.mutable_data(), *peshs = eshs.mutable_data();
        uint64_t *pei = ei.mutable_data();
        double *po = odata.mutable_data(), *pe = edata.mutable_data();
        memset(po, 0, sizeof(double) * size_odd);
        memset(pe, 0, sizeof(double) * size_even);
        poi[0] = pei[0] = 0;
        // map<uint64_t (uint32_t << 32 + uint32_t), data index>
        unordered_map<uint64_t, size_t> rdt_map;
        for (int i = 0, iodd = 0, ieven = 0; i < n_total; i++)
            if (skf[i]) {
                for (int j = 0; j < 4; j++) {
                    poqs[iodd * 4 + j] = psklqs[i * sklqi + j * sklqj];
                    poshs[iodd * 4 + j] = psklshs[i * sklqi + j * sklqj];
                }
                poi[iodd + 1] = poi[iodd] + psklis[i + 1] - psklis[i];
                uint64_t pk =
                    ((uint64_t)poqs[iodd * 4 + 0] << 32) | poqs[iodd * 4 + 1];
                if (rdt_map.count(pk) == 0)
                    rdt_map[pk] = poi[iodd];
                iodd++;
            } else {
                for (int j = 0; j < 4; j++) {
                    peqs[ieven * 4 + j] = psklqs[i * sklqi + j * sklqj];
                    peshs[ieven * 4 + j] = psklshs[i * sklqi + j * sklqj];
                }
                pei[ieven + 1] = pei[ieven] + psklis[i + 1] - psklis[i];
                uint64_t pk =
                    ((uint64_t)peqs[ieven * 4 + 0] << 32) | peqs[ieven * 4 + 1];
                if (rdt_map.count(pk) == 0)
                    rdt_map[pk] = pei[ieven];
                ieven++;
            }
        // single term matrix multiplication better use operator mul plan
        // (create one) adding matrices; multiply data; add to operator matrix
        // must be matrices of the same shape been added
        // so first get matrix for all single term -> a vector of length il
        // then just sum data (dgemm)
        // prepare on-site operators
        unordered_map<uint32_t, op_skeleton> sk_map;
        vector<map_uint_uint<SZ>> op_infos = {(map_uint_uint<SZ> &)basis,
                                              (map_uint_uint<SZ> &)basis};
        vector<uint32_t> sk_qs = {
            SZ::from_q(SZ(0, 0, 0)), SZ::from_q(SZ(1, 1, porb[ii])),
            SZ::from_q(SZ(1, -1, porb[ii])), SZ::from_q(SZ(-1, -1, porb[ii])),
            SZ::from_q(SZ(-1, 1, porb[ii]))};
        for (auto &k : sk_qs)
            sk_map[k] = flat_sparse_tensor_skeleton<SZ>(op_infos, "+-", k);
        // data for on-site operators
        vector<ssize_t> op_sh(1, 2), op_ish(1, 4);
        unordered_map<uint32_t, vector<double>> dt_map;
        dt_map[sk_qs[0]].resize(4);
        for (int i = 1; i <= 4; i++)
            dt_map[sk_qs[i]].resize(2);
        double *pi = dt_map.at(sk_qs[0]).data();
        double *pca = dt_map.at(sk_qs[1]).data(),
               *pcb = dt_map.at(sk_qs[2]).data();
        double *pda = dt_map.at(sk_qs[3]).data(),
               *pdb = dt_map.at(sk_qs[4]).data();
        pi[0] = pi[1] = pi[2] = pi[3] = 1.0;
        pca[0] = pcb[0] = pca[1] = pda[0] = pdb[0] = pda[1] = 1.0;
        pcb[1] = -1.0, pdb[1] = -1.0;
        const int incx = 1;
        vector<pair<int, int>> ip_idx(cur_values.size());
        ip_idx[0] = make_pair(0, 0);
        for (int ip = 1; ip < (int)cur_values.size(); ip++) {
            if (left_q[ip] == left_q[ip - 1])
                ip_idx[ip] =
                    make_pair(ip_idx[ip - 1].first, ip_idx[ip - 1].second + 1);
            else
                ip_idx[ip] = make_pair(ip_idx[ip - 1].first + 1, 0);
        }
        // sum and multiplication
        for (auto &mq : q_map) {
            int iq = mq.second;
            SZ q = qs[iq];
            auto &matvs = mats[iq];
            auto &mpl = map_ls[iq];
            auto &nm = nms[iq];
            int szl = nm.first, szr = nm.second, szm;
            if (max_bond_dim == -2) // NC
                szm = szl;
            else if (max_bond_dim == -3) // CN
                szm = szr;
            else if (max_bond_dim == -4) // bipartitie (-4/-5)
                szm = (int)mvcs[iq][0].size() + (int)mvcs[iq][1].size();
            else { // SVD (>= -1)
                szm = min(szl, szr);
                if (pholder_term != -1 && iq == 0)
                    szm = min(szl - 1, szr) + 1;
            }
            vector<vector<double>> reprs(szl);
            vector<uint32_t> repr_q(szl);
            for (auto &vls : mpl)
                for (auto &vl : vls.second) {
                    int il = vl.second, ip = vl.first.first,
                        it = vl.first.second;
                    int itt = it * term_len;
                    int ik = term_i[it], k = cur_term_i[it];
                    if (ik == k) {
                        reprs[il].resize(4);
                        memcpy(reprs[il].data(), pi, sizeof(double) * 4);
                        repr_q[il] = sk_qs[0];
                    } else {
                        SZ qi =
                            from_op(term_sorted[itt + ik], porb, m_site, m_op);
                        vector<double> p = dt_map.at(SZ::from_q(qi));
                        for (int i = ik + 1; i < k; i++) {
                            SZ qx = from_op(term_sorted[itt + i], porb, m_site,
                                            m_op);
                            uint32_t fqk = SZ::from_q(qi + qx),
                                     fqx = SZ::from_q(qx), fqi = SZ::from_q(qi);
                            if (sk_map.count(fqk) == 0)
                                sk_map[fqk] = flat_sparse_tensor_skeleton<SZ>(
                                    op_infos, "+-", fqk);
                            auto &skt = sk_map.at(fqk);
                            vector<double> pp(
                                get<2>(skt).data()[get<2>(skt).size() - 1], 0);
                            op_matmul(sk_map.at(fqi), sk_map.at(fqx), skt,
                                      p.data(), dt_map.at(fqx).data(),
                                      pp.data());
                            p = pp;
                            qi = qi + qx;
                        }
                        reprs[il] = p;
                        repr_q[il] = SZ::from_q(qi);
                    }
                }
            if (max_bond_dim == -4) { // bipartite
                vector<int> lip(szl), lix(szl, -1), rix(szr, -1);
                int ixln = (int)mvcs[iq][0].size(),
                    ixrn = (int)mvcs[iq][1].size();
                int szm = ixln + ixrn;
                for (auto &vls : mpl)
                    for (auto &vl : vls.second) {
                        int il = vl.second, ip = vl.first.first;
                        lip[il] = ip;
                    }
                for (int ixl = 0; ixl < ixln; ixl++)
                    lix[mvcs[iq][0][ixl]] = ixl;
                for (int ixr = 0; ixr < ixrn; ixr++)
                    rix[mvcs[iq][1][ixr]] = ixr + ixln;
                for (auto &lrv : matvs) {
                    int il = lrv.first.first, ir = lrv.first.second, irx;
                    double factor = 1;
                    if (lix[il] == -2)
                        continue;
                    else if (lix[il] != -1)
                        irx = lix[il], lix[il] = -2;
                    else
                        irx = rix[ir], factor = lrv.second;
                    int ip = lip[il];
                    int ipp = ip_idx[ip].first, ipr = ip_idx[ip].second;
                    uint64_t ql = SZ::from_q(left_q[ip]), qr = SZ::from_q(q);
                    int npr = (int)info_l.at(ql);
                    double *pr =
                        left_q[ip].is_fermion() == q.is_fermion() ? pe : po;
                    double *term_data = reprs[il].data();
                    int term_size = (int)reprs[il].size();
                    if (term_size == 0)
                        continue;
                    size_t pir = rdt_map.at((ql << 32) | qr);
                    op_skeleton &sk_repr = sk_map.at(repr_q[il]);
                    const int n_blocks = get<0>(sk_repr).shape()[0];
                    const uint64_t *pb = get<2>(sk_repr).data();
                    for (int ib = 0; ib < n_blocks; ib++) {
                        int nb = pb[ib + 1] - pb[ib];
                        daxpy(&nb, &factor, term_data + pb[ib], &incx,
                              pr + pir + (size_t)pb[ib] * szm * npr +
                                  nb * ((size_t)szm * ipr + irx),
                              &incx);
                    }
                }
            } else {
                int rszm = (int)svds[iq][1].size();
                for (int ir = 0; ir < rszm; ir++) {
                    for (auto &vls : mpl)
                        for (auto &vl : vls.second) {
                            int il = vl.second, ip = vl.first.first;
                            int ipp = ip_idx[ip].first, ipr = ip_idx[ip].second;
                            uint64_t ql = SZ::from_q(left_q[ip]),
                                     qr = SZ::from_q(q);
                            int npr = (int)info_l.at(ql);
                            double *pr =
                                left_q[ip].is_fermion() == q.is_fermion() ? pe
                                                                          : po;
                            double *term_data = reprs[il].data();
                            int term_size = (int)reprs[il].size();
                            if (term_size == 0)
                                continue;
                            size_t pir = rdt_map.at((ql << 32) | qr);
                            op_skeleton &sk_repr = sk_map.at(repr_q[il]);
                            double factor =
                                svds[iq][0][il * szm + ir] * res_factor;
                            if (ii == n_sites - 1)
                                factor *= svds[iq][1][ir];
                            const int n_blocks = get<0>(sk_repr).shape()[0];
                            const uint64_t *pb = get<2>(sk_repr).data();
                            for (int ib = 0; ib < n_blocks; ib++) {
                                int nb = pb[ib + 1] - pb[ib];
                                daxpy(&nb, &factor, term_data + pb[ib], &incx,
                                      pr + pir + (size_t)pb[ib] * rszm * npr +
                                          nb * ((size_t)rszm * ipr + ir),
                                      &incx);
                            }
                        }
                }
            }
        }
        // transpose
        auto &todata = get<2>(rodd), &tedata = get<2>(reven);
        todata = py::array_t<double>(vector<ssize_t>{size_odd});
        tedata = py::array_t<double>(vector<ssize_t>{size_even});
        flat_sparse_tensor_transpose<SZ>(oshs, odata, oi, perm, todata);
        flat_sparse_tensor_transpose<SZ>(eshs, edata, ei, perm, tedata);
        for (int i = 0, iodd = 0, ieven = 0; i < n_total; i++)
            if (skf[i]) {
                for (int j = 0; j < 4; j++) {
                    poqs[iodd * 4 + j] = psklqs[i * sklqi + pperm[j] * sklqj];
                    poshs[iodd * 4 + j] = psklshs[i * sklqi + pperm[j] * sklqj];
                }
                iodd++;
            } else {
                for (int j = 0; j < 4; j++) {
                    peqs[ieven * 4 + j] = psklqs[i * sklqi + pperm[j] * sklqj];
                    peshs[ieven * 4 + j] =
                        psklshs[i * sklqi + pperm[j] * sklqj];
                }
                ieven++;
            }
        // info_l = info_r;
        // assign cur_values
        // assign cur_terms and update term_i
        info_l = info_r;
        vector<vector<double>> new_cur_values(s_kept_total);
        vector<vector<long long int>> new_cur_terms(s_kept_total);
        int isk = 0;
        left_q.resize(s_kept_total);
        for (int iq = 0; iq < (int)qs.size(); iq++) {
            SZ q = qs[iq];
            auto &mpr = map_rs[iq];
            auto &nm = nms[iq];
            int szr = nm.second, szl = nm.first;
            vector<long long int> vct(szr);
            for (auto &vrs : mpr)
                for (auto &vr : vrs.second) {
                    vct[vr.second] = vr.first;
                    term_i[vr.first] = cur_term_i[vr.first];
                }
            int rszm;
            if (max_bond_dim == -4) { // bipartite
                auto &matvs = mats[iq];
                vector<int> lix(szl, -1), rix(szr, -1);
                int ixln = (int)mvcs[iq][0].size(),
                    ixrn = (int)mvcs[iq][1].size();
                rszm = ixln + ixrn;
                for (int ixl = 0; ixl < ixln; ixl++)
                    lix[mvcs[iq][0][ixl]] = ixl;
                // add right vertices in MVC
                for (int ixr = 0; ixr < ixrn; ixr++) {
                    int ir = mvcs[iq][1][ixr];
                    rix[ir] = ixr + ixln;
                    new_cur_terms[rix[ir] + isk].push_back(vct[ir]);
                    new_cur_values[rix[ir] + isk].push_back(1);
                }
                for (int ir = 0; ir < rszm; ir++)
                    left_q[ir + isk] = q;
                // add edges with right vertex not in MVC
                // and edges with both left and right vertices in MVC
                for (auto &lrv : matvs) {
                    int il = lrv.first.first, ir = lrv.first.second;
                    if (rix[ir] != -1 && lix[il] == -1)
                        continue;
                    assert(lix[il] != -1);
                    // placeholder for O(K^4) terms
                    if (iq == 0 && vct[ir] == pholder_term)
                        continue;
                    new_cur_terms[lix[il] + isk].push_back(vct[ir]);
                    new_cur_values[lix[il] + isk].push_back(lrv.second);
                }
            } else {
                rszm = (int)svds[iq][1].size();
                if (iq == 0 && pholder_term != -1)
                    assert(rszm != 0);
                bool has_pf = false;
                double pf_factor = 0;
                for (int j = 0; j < rszm; j++) {
                    left_q[j + isk] = q;
                    for (int ir = 0; ir < szr; ir++) {
                        // singular values multiplies to right
                        double val =
                            ii == n_sites - 1
                                ? svds[iq][2][j * szr + ir]
                                : svds[iq][2][j * szr + ir] * svds[iq][1][j];
                        if (iq == 0 && vct[ir] == pholder_term) {
                            pf_factor += val;
                            has_pf = true;
                            continue;
                        }
                        if (abs(svds[iq][2][j * szr + ir]) < cutoff)
                            continue;
                        new_cur_terms[j + isk].push_back(vct[ir]);
                        new_cur_values[j + isk].push_back(val);
                    }
                }
                if (has_pf)
                    rsc_factor = pf_factor / prefix_values[pholder_term];
            }
            isk += rszm;
        }
        assert(isk == s_kept_total);
        cur_terms = new_cur_terms;
        cur_values = new_cur_values;
        if (cur_terms.size() == 0) {
            cur_terms.emplace_back();
            cur_values.emplace_back();
        }
    }
    if (n_terms != 0) {
        // end of loop; check last term is identity with cur_values = 1
        assert(cur_values.size() == 1 && cur_values[0].size() == 1);
        assert(cur_values[0][0] == 1.0);
        assert(term_i[cur_terms[0][0]] == term_l[cur_terms[0][0]]);
    }
    return rr;
}
