
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

#include "bond_info.hpp"
#include <cassert>
#include <cmath>

void bond_info_trans_to_sz(
    const vector<unordered_map<uint32_t, uint32_t>> &infos,
    const string &pattern, vector<vector<pair<SZ, uint32_t>>> &infox,
    bool sorted) {
    int ndim = (int)infos.size();
    infox.resize(ndim);
    for (int i = 0; i < ndim; i++) {
        infox[i].resize(infos[i].size());
        int j = 0;
        for (auto &mr : infos[i]) {
            infox[i][j].first = to_sz(mr.first);
            infox[i][j].second = mr.second;
            j++;
        }
        if (sorted)
            sort(infox[i].begin(), infox[i].end(), less_psz);
        if (pattern[i] == '-')
            for (j = 0; j < (int)infox[i].size(); j++)
                infox[i][j].first = -infox[i][j].first;
    }
}

map_fusing
bond_info_fusing_product(const vector<unordered_map<uint32_t, uint32_t>> &infos,
                         const string &pattern) {
    int ndim = (int)infos.size();
    size_t nx = 1;
    for (int i = 0; i < ndim; i++)
        nx *= infos[i].size();
    vector<vector<pair<SZ, uint32_t>>> infox;
    bond_info_trans_to_sz(infos, pattern, infox, true);
    map_fusing r;
    vector<uint32_t> qk(ndim), shk(ndim);
    for (size_t x = 0; x < nx; x++) {
        uint32_t sz = 1;
        size_t xp = x;
        SZ xq;
        for (int i = ndim - 1; i >= 0; xp /= infox[i].size(), i--) {
            auto &r = infox[i][xp % infox[i].size()];
            xq = xq + r.first;
            qk[i] = pattern[i] == '+' ? from_sz(r.first) : from_sz(-r.first);
            shk[i] = r.second;
            sz *= r.second;
        }
        auto &rr = r[from_sz(xq)];
        rr.second[qk] = make_pair(rr.first, shk);
        rr.first += sz;
    }
    return r;
}

inline unordered_map<uint32_t, double>
tensor_product_no_collect(const unordered_map<uint32_t, double> &ma,
                          const unordered_map<uint32_t, double> &mb,
                          const unordered_map<uint32_t, uint32_t> &mcref) {
    unordered_map<uint32_t, double> mc;
    for (auto &a : ma)
        for (auto &b : mb) {
            uint32_t q = from_sz(to_sz(a.first) + to_sz(b.first));
            if (mcref.count(q))
                mc[q] += a.second * b.second;
        }
    return mc;
}

unordered_map<uint32_t, uint32_t>
tensor_product_ref(const unordered_map<uint32_t, uint32_t> &ma,
                   const unordered_map<uint32_t, uint32_t> &mb,
                   const unordered_map<uint32_t, uint32_t> &mcref) {
    unordered_map<uint32_t, uint32_t> mc;
    for (auto &a : ma)
        for (auto &b : mb) {
            uint32_t q = from_sz(to_sz(a.first) + to_sz(b.first));
            if (mcref.count(q))
                mc[q] = min(a.second * b.second + mc[q], 65535U);
        }
    return mc;
}

pair<vector<unordered_map<uint32_t, uint32_t>>,
     vector<unordered_map<uint32_t, uint32_t>>>
bond_info_set_bond_dimension_occ(
    const vector<unordered_map<uint32_t, uint32_t>> &basis,
    vector<unordered_map<uint32_t, uint32_t>> &left_dims,
    vector<unordered_map<uint32_t, uint32_t>> &right_dims, uint32_t vacuum,
    uint32_t target, int m, const vector<double> &occ, double bias) {
    // site state probabilities
    int n_sites = basis.size();
    vector<unordered_map<uint32_t, double>> site_probs(n_sites);
    for (int i = 0; i < n_sites; i++) {
        double alpha_occ = occ[i];
        if (bias != 1.0) {
            if (alpha_occ > 1)
                alpha_occ = 1 + pow(alpha_occ - 1, bias);
            else if (alpha_occ < 1)
                alpha_occ = 1 - pow(1 - alpha_occ, bias);
        }
        alpha_occ /= 2;
        assert(0 <= alpha_occ && alpha_occ <= 1);
        vector<double> probs = {(1 - alpha_occ) * (1 - alpha_occ),
                                (1 - alpha_occ) * alpha_occ,
                                alpha_occ * alpha_occ};
        for (auto &p : basis[i])
            site_probs[i][p.first] = probs[to_sz(p.first).n()];
    }
    vector<unordered_map<uint32_t, uint32_t>> inv_basis(n_sites);
    for (int i = 0; i < n_sites; i++)
        for (auto &p : basis[i])
            inv_basis[i][from_sz(-to_sz(p.first))] = p.second;
    vector<unordered_map<uint32_t, double>> inv_site_probs(n_sites);
    for (int i = 0; i < n_sites; i++)
        for (auto &p : site_probs[i])
            inv_site_probs[i][from_sz(-to_sz(p.first))] = p.second;
    // left and right block probabilities
    vector<unordered_map<uint32_t, double>> left_probs(n_sites + 1);
    vector<unordered_map<uint32_t, double>> right_probs(n_sites + 1);
    left_probs[0][vacuum] = 1;
    for (int i = 0; i < n_sites; i++)
        left_probs[i + 1] = tensor_product_no_collect(
            left_probs[i], site_probs[i], left_dims[i + 1]);
    right_probs[n_sites][target] = 1;
    for (int i = n_sites - 1; i >= 0; i--)
        right_probs[i] = tensor_product_no_collect(
            inv_site_probs[i], right_probs[i + 1], right_dims[i]);
    // conditional probabilities
    for (int i = 0; i <= n_sites; i++) {
        unordered_map<uint32_t, double> lprobs = left_probs[i];
        unordered_map<uint32_t, double> rprobs = right_probs[i];
        if (i > 0)
            for (auto &p : left_probs[i])
                p.second *= rprobs.count(p.first) ? rprobs[p.first] : 0;
        if (i < n_sites)
            for (auto &p : right_probs[i])
                p.second *= lprobs.count(p.first) ? lprobs[p.first] : 0;
    }
    // adjusted temparary fci dims
    vector<unordered_map<uint32_t, uint32_t>> left_dims_fci_t = left_dims;
    vector<unordered_map<uint32_t, uint32_t>> right_dims_fci_t = right_dims;
    // left and right block dims
    for (int i = 1; i <= n_sites; i++) {
        left_dims[i].clear();
        double prob_sum = 0.0;
        for (auto &p : left_probs[i])
            prob_sum += p.second;
        for (auto &p : left_probs[i]) {

            uint32_t v = min((uint32_t)round(p.second / prob_sum * m),
                             left_dims_fci_t[i].at(p.first));
            if (v != 0)
                left_dims[i][p.first] = v;
        }
        if (i != n_sites) {
            auto tmp = tensor_product_ref(left_dims[i], basis[i],
                                          left_dims_fci_t[i + 1]);
            for (auto &p : left_dims_fci_t[i + 1])
                if (tmp.count(p.first))
                    p.second = min(tmp.at(p.first), p.second);
            for (auto &p : left_probs[i + 1])
                if (!tmp.count(p.first))
                    p.second = 0;
        }
    }
    for (int i = n_sites - 1; i >= 0; i--) {
        right_dims[i].clear();
        double prob_sum = 0.0;
        for (auto &p : right_probs[i])
            prob_sum += p.second;
        for (auto &p : right_probs[i]) {
            uint32_t v = min((uint32_t)round(p.second / prob_sum * m),
                             right_dims_fci_t[i].at(p.first));
            if (v != 0)
                right_dims[i][p.first] = v;
        }
        if (i != 0) {
            auto tmp = tensor_product_ref(inv_basis[i - 1], right_dims[i],
                                          right_dims_fci_t[i - 1]);
            for (auto &p : right_dims_fci_t[i - 1])
                if (tmp.count(p.first))
                    p.second = min(tmp.at(p.first), p.second);
            for (auto &p : right_probs[i - 1])
                if (!tmp.count(p.first))
                    p.second = 0;
        }
    }
    return std::make_pair(left_dims, right_dims);
}
