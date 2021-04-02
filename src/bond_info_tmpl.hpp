
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

TMPL_EXTERN template void bond_info_trans<TMPL_Q>(
    const vector<map_uint_uint<TMPL_Q>> &infos, const string &pattern,
    vector<vector<pair<TMPL_Q, uint32_t>>> &infox, bool sorted = false);

TMPL_EXTERN template map_fusing
bond_info_fusing_product<TMPL_Q>(const vector<map_uint_uint<TMPL_Q>> &infos,
                                 const string &pattern);

TMPL_EXTERN template pair<vector<map_uint_uint<TMPL_Q>>,
                          vector<map_uint_uint<TMPL_Q>>>
bond_info_set_bond_dimension_occ<TMPL_Q>(
    const vector<map_uint_uint<TMPL_Q>> &basis,
    vector<map_uint_uint<TMPL_Q>> &left_dims,
    vector<map_uint_uint<TMPL_Q>> &right_dims, uint32_t vacuum, uint32_t target,
    int m, const vector<double> &occ, double bias);

TMPL_EXTERN template map_uint_uint<TMPL_Q>
tensor_product_ref<TMPL_Q>(const map_uint_uint<TMPL_Q> &ma,
                           const map_uint_uint<TMPL_Q> &mb,
                           const map_uint_uint<TMPL_Q> &mcref);

#endif
