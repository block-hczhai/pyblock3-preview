
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

#pragma once

#include "sz.hpp"
#include <algorithm>
#include <unordered_map>
#include <vector>

using namespace std;

template <typename Q> struct map_uint_uint : unordered_map<uint32_t, uint32_t> {
    map_uint_uint() : unordered_map<uint32_t, uint32_t>() {}
    map_uint_uint(const vector<pair<uint32_t, uint32_t>>::iterator &a,
                  const vector<pair<uint32_t, uint32_t>>::iterator &b)
        : unordered_map<uint32_t, uint32_t>(a, b) {}
};

typedef unordered_map<
    uint32_t, pair<uint32_t, unordered_map<vector<uint32_t>,
                                           pair<uint32_t, vector<uint32_t>>>>>
    map_fusing;

template <typename Q>
void bond_info_trans(const vector<map_uint_uint<Q>> &infos,
                     const string &pattern,
                     vector<vector<pair<Q, uint32_t>>> &infox,
                     bool sorted = false);

template <typename Q>
map_fusing bond_info_fusing_product(const vector<map_uint_uint<Q>> &infos,
                                    const string &pattern);

template <typename Q>
pair<vector<map_uint_uint<Q>>, vector<map_uint_uint<Q>>>
bond_info_set_bond_dimension_occ(const vector<map_uint_uint<Q>> &basis,
                                 vector<map_uint_uint<Q>> &left_dims,
                                 vector<map_uint_uint<Q>> &right_dims,
                                 uint32_t vacuum, uint32_t target, int m,
                                 const vector<double> &occ, double bias);

template <>
pair<vector<map_uint_uint<SZ>>, vector<map_uint_uint<SZ>>>
bond_info_set_bond_dimension_occ<SZ>(const vector<map_uint_uint<SZ>> &basis,
                                     vector<map_uint_uint<SZ>> &left_dims,
                                     vector<map_uint_uint<SZ>> &right_dims,
                                     uint32_t vacuum, uint32_t target, int m,
                                     const vector<double> &occ, double bias);

template <typename Q>
map_uint_uint<Q> tensor_product_ref(const map_uint_uint<Q> &ma,
                                    const map_uint_uint<Q> &mb,
                                    const map_uint_uint<Q> &mcref);

#define TMPL_EXTERN extern
#define TMPL_NAME bond_info
#include "symmetry_tmpl.hpp"
#undef TMPL_NAME
#undef TMPL_EXTERN

inline size_t q_labels_hash(const uint32_t *qs, int nctr, const int *idxs,
                            const int inc) noexcept {
    size_t h = 0;
    for (int i = 0; i < nctr; i++)
        h ^= (size_t)qs[idxs[i] * inc] + 0x9E3779B9 + (h << 6) + (h >> 2);
    return h;
}

inline size_t q_labels_hash(const uint32_t *qs, int nctr,
                            const int inc) noexcept {
    size_t h = 0;
    for (int i = 0; i < nctr; i++)
        h ^= (size_t)qs[i * inc] + 0x9E3779B9 + (h << 6) + (h >> 2);
    return h;
}

namespace std {

template <> struct hash<vector<uint32_t>> {
    size_t operator()(const vector<uint32_t> &s) const noexcept {
        return q_labels_hash(s.data(), s.size(), 1);
    }
};

} // namespace std

template <typename Q>
inline bool less_psz(const pair<Q, uint32_t> &x,
                     const pair<Q, uint32_t> &y) noexcept {
    return x.first < y.first;
}

template <typename Q>
inline bool less_vsz(const vector<Q> &x, const vector<Q> &y) noexcept {
    for (size_t i = 0; i < x.size(); i++)
        if (x[i] != y[i])
            return x[i] < y[i];
    return false;
}

template <typename Q, typename T>
inline bool less_pvsz(const pair<vector<Q>, T> &x,
                      const pair<vector<Q>, T> &y) noexcept {
    return less_vsz(x.first, y.first);
}

inline bool is_shape_one(const uint32_t *shs, int n, int nfree, const int inci,
                         const int incj) noexcept {
    for (int j = 0; j < nfree * incj; j += incj)
        for (int i = 0; i < n * inci; i += inci)
            if (shs[i + j] != 1)
                return false;
    return true;
}
