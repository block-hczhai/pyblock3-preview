
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

#ifndef TMPL_EXTERN
#define TMPL_EXTERN
#endif

#ifdef TMPL_Q

#ifdef TMPL_FL

TMPL_EXTERN template void flat_fermion_tensor_transpose<TMPL_Q, TMPL_FL>(
    const py::array_t<uint32_t> &aqs, const py::array_t<uint32_t> &ashs,
    const py::array_t<TMPL_FL> &adata, const py::array_t<uint64_t> &aidxs,
    const py::array_t<int32_t> &perm, py::array_t<TMPL_FL> &cdata);

TMPL_EXTERN template tuple<py::array_t<uint32_t>, py::array_t<uint32_t>,
                           py::array_t<TMPL_FL>, py::array_t<uint64_t>>
flat_fermion_tensor_tensordot<TMPL_Q, TMPL_FL>(
    const py::array_t<uint32_t> &aqs, const py::array_t<uint32_t> &ashs,
    const py::array_t<TMPL_FL> &adata, const py::array_t<uint64_t> &aidxs,
    const py::array_t<uint32_t> &bqs, const py::array_t<uint32_t> &bshs,
    const py::array_t<TMPL_FL> &bdata, const py::array_t<uint64_t> &bidxs,
    const py::array_t<int> &idxa, const py::array_t<int> &idxb);

TMPL_EXTERN template tuple<py::array_t<uint32_t>, py::array_t<uint32_t>,
                           py::array_t<TMPL_FL>, py::array_t<uint64_t>,
                           py::array_t<uint32_t>, py::array_t<uint32_t>,
                           py::array_t<TMPL_FL>, py::array_t<uint64_t>>
flat_fermion_tensor_qr<TMPL_Q, TMPL_FL>(const py::array_t<uint32_t> &aqs,
                                        const py::array_t<uint32_t> &ashs,
                                        const py::array_t<TMPL_FL> &adata,
                                        const py::array_t<uint64_t> &aidxs,
                                        int idx, const string &pattern,
                                        bool is_qr);

#else

TMPL_EXTERN template tuple<py::array_t<uint32_t>, py::array_t<uint32_t>,
                           py::array_t<uint64_t>>
flat_fermion_tensor_skeleton<TMPL_Q>(const vector<map_uint_uint<TMPL_Q>> &infos,
                                     uint32_t fdq);

#endif

#endif
