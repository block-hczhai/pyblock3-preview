
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
#include "hamiltonian.hpp"
#include "max_flow.hpp"
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <limits>
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

inline string print_op(int32_t op) {
    const int m_site = 2, m_op = 16384;
    stringstream ss;
    ss << (op / m_op ? "D" : "C");
    ss << (int)((op % m_op) / m_site);
    ss << ((op % m_site) ? "b" : "a");
    return ss.str();
}

inline string print_string(const vector<int32_t> &term_sorted, int idx, int ii,
                           int jj) {
    stringstream ss;
    for (int i = ii; i < jj; i++)
        if (term_sorted[idx * 4 + i] == -1)
            break;
        else
            ss << print_op(term_sorted[idx * 4 + i]) << " ";
    string xx = ss.str();
    if (xx.length() == 0)
        return "I";
    return xx;
}

struct PrefixTreeNode {
    vector<long long int> children;
    long long int size;
    int depth;
    PrefixTreeNode(int depth) : size(0), depth(depth) {}
};

struct PrefixTree {
    vector<PrefixTreeNode> data;
    vector<long long int> pdata;
    unordered_map<int32_t, int> op_map;
    int max_depth;
    PrefixTree(int max_depth) : max_depth(max_depth) {
        for (int d = 1; d <= max_depth; d++) {
            data.push_back(PrefixTreeNode(d));
            pdata.push_back(-1);
        }
    }
    void build(const vector<int32_t> &term_sorted, vector<double> &values,
               long long int n_terms, int term_len) {
        assert(n_terms * term_len == term_sorted.size());
        vector<long long int> ixx(term_len);
        for (long long int it = 0; it < n_terms; it++) {
            int d = term_len - 1;
            for (; d >= 0 && term_sorted[it * term_len + d] == -1; d--)
                ;
            long long int ix = d;
            if (pdata[ix] == -1)
                pdata[ix] = it;
            for (int k = 0; k <= d; k++) {
                int32_t term = term_sorted[it * term_len + k];
                if (op_map.count(term) == 0)
                    op_map[term] = (int)op_map.size();
                if (data[ix].children.size() < op_map.size())
                    data[ix].children.resize(op_map.size());
                int ic = op_map[term];
                if (data[ix].children[ic] == 0) {
                    data[ix].children[ic] = data.size();
                    data.push_back(PrefixTreeNode(d - k));
                    pdata.push_back(it);
                }
                ixx[k] = ix;
                ix = data[ix].children[ic];
            }
            if (data[ix].size == 0) {
                data[ix].size++;
                for (int k = 0; k <= d; k++)
                    data[ixx[k]].size++;
            } else
                values[pdata[ix]] += values[it];
        }
    }
};

vector<tuple<py::array_t<uint32_t>, py::array_t<uint32_t>, py::array_t<double>,
             py::array_t<uint64_t>>>
build_mpo_ptree(py::array_t<int32_t> orb_sym, py::array_t<double> h_values,
                py::array_t<int32_t> h_terms) {
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
    vector<double> term_coefs(n_terms);
    // length of each term; starting index of each term
    // at the beginning, term_i is all zero
    vector<int> term_l(n_terms), term_i(n_terms, 0);
    // index of current terms
    vector<vector<long long int>> cur_terms(1);
    vector<vector<double>> cur_values(1);
    const int32_t *pt = h_terms.data(), *porb = orb_sym.data();
    const double *pv = h_values.data();
    const ssize_t hsi = h_terms.strides()[0] / sizeof(uint32_t),
                  hsj = h_terms.strides()[1] / sizeof(uint32_t);
    vector<int> term_site(term_len);
    memset(term_sorted.data(), -1, (n_terms * term_len) * sizeof(int32_t));
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
        int ffactor = 1;
        for (int i = 0; i < ix; i++)
            for (int j = i + 1; j < ix; j++)
                if (term_site[i] > term_site[j])
                    ffactor = -ffactor;
        term_coefs[it] = pv[it] * ffactor;
        stable_sort(term_sorted.data() + it * term_len,
                    term_sorted.data() + it * term_len + ix,
                    [m_op, m_site](int32_t i, int32_t j) {
                        return (i % m_op) / m_site < (j % m_op) / m_site;
                    });
    }
    PrefixTree pftr(term_len);
    pftr.build(term_sorted, term_coefs, n_terms, term_len);
    for (int i = 1; i <= term_len; i++)
        if (pftr.data[i - 1].size != 0) {
            cur_terms[0].push_back(i - 1);
            cur_values[0].push_back(term_coefs[pftr.pdata[i - 1]]);
        }
    // result mpo
    vector<tuple<py::array_t<uint32_t>, py::array_t<uint32_t>,
                 py::array_t<double>, py::array_t<uint64_t>>>
        rr(n_sites * 2);
    // do svd from left to right
    // time complexity: O(NL)
    // K: n_sites, D: max_bond_dim, L: term_len, N: n_terms
    // using block-structure according to left q number
    // this is the map from left q number to its block index
    unordered_map<SZ, int> q_map;
    // for each iq block, a map from hashed repr of string of op in left block
    // to (mpo index, term index, left block string of op index)
    vector<unordered_map<size_t, vector<pair<pair<int, long long int>, int>>>>
        map_ls;
    // for each iq block, a map from hashed repr of string of op in right block
    // to (tree index, right block string of op index)
    vector<unordered_map<size_t, vector<pair<long long int, int>>>> map_rs;
    // tree nodes for each right block
    vector<vector<vector<pair<long long int, int>>>> rt_map;
    // weight for each right block
    vector<vector<int>> rt_weight;
    // sparse repr of the connection (edge) matrix for each block
    vector<vector<pair<pair<int, int>, double>>> mats;
    // for each block, the nrow and ncol of the block
    vector<pair<long long int, long long int>> nms;
    vector<int> cur_term_i(n_terms, -1);
    py::array_t<int32_t> perm(vector<ssize_t>{4});
    perm.mutable_data()[0] = 0, perm.mutable_data()[1] = 2,
    perm.mutable_data()[2] = 3, perm.mutable_data()[3] = 1;
    const int32_t *pperm = perm.data();
    vector<int> que(pftr.data.size());
    vector<double> mque(pftr.data.size());
    for (int ii = 0; ii < n_sites; ii++) {
        cout << "MPO site" << setw(4) << ii << " / " << n_sites << endl;
        q_map.clear();
        map_ls.clear();
        map_rs.clear();
        mats.clear();
        rt_map.clear();
        rt_weight.clear();
        nms.clear();
        info_r.clear();
        // site basis
        unordered_map<uint32_t, uint32_t> basis;
        basis[SZ::from_q(SZ(0, 0, 0))] = 1;
        basis[SZ::from_q(SZ(1, 1, porb[ii]))] = 1;
        basis[SZ::from_q(SZ(1, -1, porb[ii]))] = 1;
        basis[SZ::from_q(SZ(2, 0, 0))] = 1;
        // iter over all left legs
        for (int ip = 0; ip < (int)cur_terms.size(); ip++) {
            SZ qll = left_q[ip];
            size_t h = 0, l = 0;
            for (size_t ipt = 0; ipt < cur_terms[ip].size(); ipt++)
                que[h + l] = cur_terms[ip][ipt],
                        mque[h + l++] = cur_values[ip][ipt];
            // iter over tree nodes
            while (l != 0) {
                long long int itr = que[h];
                double ival = mque[h];
                l--, h++;
                long long int it = pftr.pdata[itr];
                int kmax = term_l[it];
                int kmin = term_i[it], k = kmin;
                int ktree = kmax - pftr.data[itr].depth;
                long long int itt = it * term_len;
                if ((ktree == kmin || pftr.data[itr].size == 1) &&
                    pftr.data[itr].children.size() != 0)
                    // unpack
                    for (auto isr : pftr.data[itr].children) {
                        if (isr == 0)
                            continue;
                        term_i[pftr.pdata[isr]] = term_i[it];
                        que[h + l] = isr;
                        mque[h + l++] = pftr.data[itr].size == 1
                                            ? ival
                                            : term_coefs[pftr.pdata[isr]];
                    }
                else {
                    assert(ktree > kmin || kmin == kmax);
                    // separate the current product into two parts
                    // (left block part and right block part)
                    for (; k < kmax &&
                           (term_sorted[itt + k] % m_op) / m_site <= ii;
                         k++)
                        ;
                    if (k >= ktree && ktree != kmax)
                        // unpack
                        for (auto isr : pftr.data[itr].children) {
                            if (isr == 0)
                                continue;
                            term_i[pftr.pdata[isr]] = term_i[it];
                            que[h + l] = isr;
                            mque[h + l++] = pftr.data[itr].size == 1
                                                ? ival
                                                : term_coefs[pftr.pdata[isr]];
                        }
                    else {
                        // first right site position
                        cur_term_i[it] = k;
                        size_t hl = op_hash(term_sorted.data() + itt + kmin,
                                            k - kmin, ip);
                        size_t hr =
                            op_hash(term_sorted.data() + itt + k, ktree - k);
                        SZ ql = qll;
                        for (int i = kmin; i < k; i++)
                            ql = ql + from_op(term_sorted[itt + i], porb,
                                              m_site, m_op);
                        if (q_map.count(ql) == 0) {
                            q_map[ql] = (int)q_map.size();
                            map_ls.emplace_back();
                            map_rs.emplace_back();
                            rt_map.emplace_back();
                            rt_weight.emplace_back();
                            mats.emplace_back();
                            nms.push_back(make_pair(0, 0));
                        }
                        int iq = q_map.at(ql), il = -1, ir = -1;
                        long long int &nml = nms[iq].first,
                                      &nmr = nms[iq].second;
                        auto &mpl = map_ls[iq];
                        auto &mpr = map_rs[iq];
                        if (mpl.count(hl)) {
                            int iqq = 0;
                            auto &vq = mpl.at(hl);
                            for (; iqq < vq.size(); iqq++) {
                                int vip = vq[iqq].first.first;
                                long long int vit = vq[iqq].first.second;
                                long long int vitt = vit * term_len;
                                int vik = term_i[vit], vk = cur_term_i[vit];
                                if (vip == ip && vk - vik == k - kmin &&
                                    equal(term_sorted.data() + vitt + vik,
                                          term_sorted.data() + vitt + vk,
                                          term_sorted.data() + itt + kmin))
                                    break;
                            }
                            if (iqq == (int)vq.size())
                                vq.push_back(
                                    make_pair(make_pair(ip, it), il = nml++));
                            else
                                il = vq[iqq].second;
                        } else
                            mpl[hl].push_back(
                                make_pair(make_pair(ip, it), il = nml++));
                        if (mpr.count(hr)) {
                            int iqq = 0;
                            auto &vq = mpr.at(hr);
                            for (; iqq < vq.size(); iqq++) {
                                int vir = vq[iqq].first;
                                int vit = pftr.pdata[itr];
                                int vitt = vit * term_len;
                                int vktree = term_l[vit] - pftr.data[vir].depth;
                                int vkmax = term_l[vit];
                                int vk = cur_term_i[vit];
                                if (vktree - vk == ktree - k &&
                                    vkmax - vk == kmax - k &&
                                    equal(term_sorted.data() + vitt + vk,
                                          term_sorted.data() + vitt + vktree,
                                          term_sorted.data() + itt + k))
                                    break;
                            }
                            if (iqq == (int)vq.size()) {
                                vq.push_back(make_pair(itr, ir = nmr++));
                                rt_map[iq].emplace_back();
                                rt_weight[iq].push_back(
                                    (int)pftr.data[itr].size);
                            } else {
                                ir = vq[iqq].second;
                                rt_weight[iq][ir] =
                                    min(rt_weight[iq][ir],
                                        (int)pftr.data[itr].size);
                            }
                        } else {
                            mpr[hr].push_back(make_pair(itr, ir = nmr++));
                            rt_map[iq].emplace_back();
                            rt_weight[iq].push_back((int)pftr.data[itr].size);
                        }
                        assert(pftr.data[itr].size <
                               (long long int)numeric_limits<int>::max());
                        rt_map[iq][ir].push_back(make_pair(itr, il));
                        mats[iq].push_back(make_pair(make_pair(il, ir), ival));
                    }
                }
            }
        }
        vector<array<vector<int>, 2>> mvcs(q_map.size());
        vector<SZ> qs(q_map.size());
        int s_kept_total = 0;
        for (auto &mq : q_map) {
            int iq = mq.second;
            qs[iq] = mq.first;
            int s_kept = 0;
            while (true) {
                auto &matvs = mats[iq];
                auto &nm = nms[iq];
                int szl = nm.first, szr = nm.second;
                FLOW flow(szl + szr);
                for (auto &lrv : matvs)
                    flow.resi[lrv.first.first][lrv.first.second + szl] +=
                        rt_weight[iq][lrv.first.second];
                for (int i = 0; i < szl; i++)
                    flow.resi[szl + szr][i] = 1;
                for (int i = 0; i < szr; i++)
                    flow.resi[szl + i][szl + szr + 1] = rt_weight[iq][i];
                mvcs[iq][0].clear();
                mvcs[iq][1].clear();
                if (szr == 1 && rt_weight[iq][0] == 1 && ii == n_sites - 1)
                    mvcs[iq][1].push_back(0);
                else
                    flow.MVC(0, szl, szl, szr, mvcs[iq][0], mvcs[iq][1]);
                bool need_unpack = false;
                for (auto ir : mvcs[iq][1])
                    if (rt_weight[iq][ir] != 1)
                        need_unpack = true;
                // unpack selected right vertices
                // need to generate full connection list
                // can simply revise matvs
                // first add all in matvs where ir not in mvcs or ir size = 1
                // but change ir index
                // then for ir size > 1 add unpack terms to matvs
                if (need_unpack) {
                    vector<pair<pair<int, int>, double>> xmatvs;
                    vector<int> mvcsr, mvcsrx;
                    unordered_map<size_t, vector<pair<long long int, int>>>
                        xmap_rs;
                    vector<vector<pair<long long int, int>>> xrt_map;
                    vector<int> xrt_weight;
                    int xir = 0;
                    vector<int> rix(szr, -1);
                    for (int ixr = 0; ixr < (int)mvcs[iq][1].size(); ixr++)
                        rix[mvcs[iq][1][ixr]] = -2;
                    for (int ixr = 0; ixr < szr; ixr++) {
                        if (rt_weight[iq][ixr] == 1 && rix[ixr] == -2)
                            mvcsr.push_back(xir);
                        if (rt_weight[iq][ixr] == 1 || rix[ixr] == -1) {
                            xrt_weight.push_back(rt_weight[iq][ixr]);
                            rix[ixr] = xir++;
                        }
                    }
                    for (auto &lrv : matvs)
                        if (rix[lrv.first.second] >= 0)
                            xmatvs.push_back(
                                make_pair(make_pair(lrv.first.first,
                                                    rix[lrv.first.second]),
                                          lrv.second));
                    int nmr = xir;
                    vector<int> rrix(xir, 0);
                    for (auto &mc : mvcsr)
                        rrix[mc] = 1;
                    mvcsr.clear();
                    xmap_rs.reserve(map_rs[iq].size());
                    xrt_map.resize(xir);
                    for (auto &mr : map_rs[iq])
                        for (auto &mmr : mr.second)
                            if (rix[mmr.second] >= 0)
                                xmap_rs[mr.first].push_back(
                                    make_pair(mmr.first, rix[mmr.second]));
                    for (int ixr = 0; ixr < szr; ixr++)
                        if (rix[ixr] >= 0)
                            xrt_map[rix[ixr]] = rt_map[iq][ixr];
                    for (int ixr = 0; ixr < szr; ixr++)
                        if (rix[ixr] < 0) {
                            int pnmr = nmr;
                            for (auto &mr : rt_map[iq][ixr]) {
                                int il = mr.second;
                                long long int ritr = mr.first;
                                size_t h = 0, l = 0;
                                que[h + l++] = ritr;
                                // iter over tree nodes
                                while (l != 0) {
                                    long long int itr = que[h];
                                    l--, h++;
                                    long long int it = pftr.pdata[itr];
                                    int kmax = term_l[it], k = cur_term_i[it];
                                    int ktree = kmax - pftr.data[itr].depth;
                                    long long int itt = it * term_len;
                                    if (pftr.data[itr].children.size() != 0)
                                        for (auto isr :
                                             pftr.data[itr].children) {
                                            if (isr == 0)
                                                continue;
                                            term_i[pftr.pdata[isr]] =
                                                term_i[it];
                                            cur_term_i[pftr.pdata[isr]] =
                                                cur_term_i[it];
                                            que[h + l++] = isr;
                                        }
                                    else {
                                        assert(ktree == kmax);
                                        size_t hr = op_hash(term_sorted.data() +
                                                                itt + k,
                                                            ktree - k);
                                        auto &mpr = xmap_rs;
                                        int ir = -1;
                                        if (mpr.count(hr)) {
                                            int iqq = 0;
                                            auto &vq = mpr.at(hr);
                                            for (; iqq < vq.size(); iqq++) {
                                                int vir = vq[iqq].first;
                                                int vit = pftr.pdata[itr];
                                                int vitt = vit * term_len;
                                                int vktree =
                                                    term_l[vit] -
                                                    pftr.data[vir].depth;
                                                int vkmax = term_l[vit];
                                                int vk = cur_term_i[vit];
                                                if (vktree - vk == ktree - k &&
                                                    vkmax - vk == kmax - k &&
                                                    equal(term_sorted.data() +
                                                              vitt + vk,
                                                          term_sorted.data() +
                                                              vitt + vktree,
                                                          term_sorted.data() +
                                                              itt + k))
                                                    break;
                                            }
                                            if (iqq == (int)vq.size()) {
                                                vq.push_back(
                                                    make_pair(itr, ir = nmr++));
                                                xrt_map.emplace_back();
                                            } else
                                                ir = vq[iqq].second;
                                        } else {
                                            mpr[hr].push_back(
                                                make_pair(itr, ir = nmr++));
                                            xrt_map.emplace_back();
                                        }
                                        if (ir < xir)
                                            rrix[ir] = 1;
                                        xrt_map[ir].push_back(
                                            make_pair(itr, il));
                                        xmatvs.push_back(make_pair(
                                            make_pair(il, ir), term_coefs[it]));
                                    }
                                }
                            }
                            if (rix[ixr] == -2)
                                for (int ixxr = pnmr; ixxr < nmr; ixxr++)
                                    mvcsrx.push_back(ixxr);
                        }
                    for (int ixxr = 0; ixxr < xir; ixxr++)
                        if (rrix[ixxr] == 1)
                            mvcsr.push_back(ixxr);
                    mvcsr.insert(mvcsr.end(), mvcsrx.begin(), mvcsrx.end());
                    xrt_weight.resize(nmr, 1);
                    mvcs[iq][1] = mvcsr;
                    map_rs[iq] = xmap_rs;
                    mats[iq] = xmatvs;
                    rt_map[iq] = xrt_map;
                    nms[iq].second = nmr;
                    rt_weight[iq] = xrt_weight;
                } else
                    break;
            }
            s_kept = (int)mvcs[iq][0].size() + (int)mvcs[iq][1].size();
            if (s_kept != 0)
                info_r[SZ::from_q(mq.first)] = s_kept;
            s_kept_total += s_kept;
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
        vector<pair<int, int>> ip_idx(cur_terms.size());
        ip_idx[0] = make_pair(0, 0);
        for (int ip = 1; ip < (int)cur_terms.size(); ip++) {
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
            auto &mpr = map_rs[iq];
            auto &nm = nms[iq];
            int szl = nm.first, szr = nm.second;
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
            // bipartite
            vector<int> lip(szl), lix(szl, -1), rix(szr, -1);
            int ixln = (int)mvcs[iq][0].size(), ixrn = (int)mvcs[iq][1].size();
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
            vector<long long int> vct(szr);
            for (auto &vrs : mpr)
                for (auto &vr : vrs.second) {
                    long long int it = pftr.pdata[vr.first];
                    vct[vr.second] = vr.first;
                    term_i[it] = cur_term_i[it];
                }
            for (auto &lrv : matvs) {
                int il = lrv.first.first, ir = lrv.first.second, irx;
                double factor = 1;
                // add left in MVC right not in MVC -> left normal factor = 1
                // add left not in MVC right in MVC -> right normal factor = lrv
                if (lix[il] == -2)
                    continue;
                else if (lix[il] != -1)
                    irx = lix[il], lix[il] = -2;
                else {
                    irx = rix[ir], factor = lrv.second;
                    assert(pftr.data[vct[ir]].size == 1);
                }
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
        // assign cur_terms and update term_i
        info_l = info_r;
        vector<vector<long long int>> new_cur_terms(s_kept_total);
        vector<vector<double>> new_cur_values(s_kept_total);
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
                    long long int it = pftr.pdata[vr.first];
                    vct[vr.second] = vr.first;
                    term_i[it] = cur_term_i[it];
                }
            auto &matvs = mats[iq];
            vector<int> lix(szl, -1), rix(szr, -1);
            int ixln = (int)mvcs[iq][0].size(), ixrn = (int)mvcs[iq][1].size();
            int rszm = ixln + ixrn;
            for (int ixl = 0; ixl < ixln; ixl++)
                lix[mvcs[iq][0][ixl]] = ixl;
            // add right vertices in MVC
            for (int ixr = 0; ixr < ixrn; ixr++) {
                int ir = mvcs[iq][1][ixr];
                rix[ir] = ixr + ixln;
                new_cur_terms[rix[ir] + isk].push_back(vct[ir]);
                new_cur_values[rix[ir] + isk].push_back(1);
                assert(pftr.data[vct[ir]].size == 1);
            }
            for (int ir = 0; ir < rszm; ir++)
                left_q[ir + isk] = q;
            // add edges with left in MVC and right vertex not in MVC
            // and edges with both left and right vertices in MVC
            auto &rt_mapx = rt_map[iq];
            vector<int> rtm_idx(szr, 0);
            for (auto &lrv : matvs) {
                int il = lrv.first.first, ir = lrv.first.second;
                if (rix[ir] != -1 && lix[il] == -1)
                    continue;
                assert(lix[il] != -1);
                for (; rtm_idx[ir] < rt_mapx[ir].size() &&
                       rt_mapx[ir][rtm_idx[ir]].second != il;
                     rtm_idx[ir]++)
                    ;
                assert(rtm_idx[ir] < rt_mapx[ir].size() &&
                       rt_mapx[ir][rtm_idx[ir]].second == il);
                long long int itr = rt_mapx[ir][rtm_idx[ir]].first;
                new_cur_terms[lix[il] + isk].push_back(itr);
                new_cur_values[lix[il] + isk].push_back(lrv.second);
                long long int it = pftr.pdata[itr];
                term_i[it] = cur_term_i[it];
            }
            isk += rszm;
        }
        assert(isk == s_kept_total);
        cur_terms = new_cur_terms;
        cur_values = new_cur_values;
        if (cur_terms.size() == 0)
            cur_terms.emplace_back();
    }
    if (n_terms != 0) {
        // end of loop; check last term is identity
        assert(cur_values.size() == 1 && cur_values[0].size() == 1);
        assert(cur_values[0][0] == 1);
        assert(term_i[pftr.pdata[cur_terms[0][0]]] ==
               term_l[pftr.pdata[cur_terms[0][0]]]);
    }
    return rr;
}
