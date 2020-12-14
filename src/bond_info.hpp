
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

using namespace std;

typedef unordered_map<
    uint32_t, pair<uint32_t, unordered_map<vector<uint32_t>,
                                           pair<uint32_t, vector<uint32_t>>>>>
    map_fusing;

void bond_info_trans_to_sz(
    const vector<unordered_map<uint32_t, uint32_t>> &infos,
    const string &pattern, vector<vector<pair<SZ, uint32_t>>> &infox,
    bool sorted = false);

map_fusing
bond_info_fusing_product(const vector<unordered_map<uint32_t, uint32_t>> &infos,
                         const string &pattern);

pair<vector<unordered_map<uint32_t, uint32_t>>,
     vector<unordered_map<uint32_t, uint32_t>>>
bond_info_set_bond_dimension_occ(
    const vector<unordered_map<uint32_t, uint32_t>> &basis,
    vector<unordered_map<uint32_t, uint32_t>> &left_dims,
    vector<unordered_map<uint32_t, uint32_t>> &right_dims, uint32_t vacuum,
    uint32_t target, int m, const vector<double> &occ, double bias);

unordered_map<uint32_t, uint32_t>
tensor_product_ref(const unordered_map<uint32_t, uint32_t> &ma,
                   const unordered_map<uint32_t, uint32_t> &mb,
                   const unordered_map<uint32_t, uint32_t> &mcref);
