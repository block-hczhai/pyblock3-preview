
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

// explicit template instantiation

#ifndef TMPL_EXTERN
#define TMPL_EXTERN
#endif

#ifdef TMPL_Q

TMPL_EXTERN template
void bond_info_trans<TMPL_Q>(
    const vector<unordered_map<uint32_t, uint32_t>> &infos,
    const string &pattern, vector<vector<pair<TMPL_Q, uint32_t>>> &infox,
    bool sorted = false);

TMPL_EXTERN template
map_fusing
bond_info_fusing_product<TMPL_Q>(const vector<unordered_map<uint32_t, uint32_t>> &infos,
                         const string &pattern);

TMPL_EXTERN template
pair<vector<unordered_map<uint32_t, uint32_t>>,
     vector<unordered_map<uint32_t, uint32_t>>>
bond_info_set_bond_dimension_occ<TMPL_Q>(
    const vector<unordered_map<uint32_t, uint32_t>> &basis,
    vector<unordered_map<uint32_t, uint32_t>> &left_dims,
    vector<unordered_map<uint32_t, uint32_t>> &right_dims, uint32_t vacuum,
    uint32_t target, int m, const vector<double> &occ, double bias);

TMPL_EXTERN template
unordered_map<uint32_t, uint32_t>
tensor_product_ref<TMPL_Q>(const unordered_map<uint32_t, uint32_t> &ma,
                   const unordered_map<uint32_t, uint32_t> &mb,
                   const unordered_map<uint32_t, uint32_t> &mcref);

#endif
