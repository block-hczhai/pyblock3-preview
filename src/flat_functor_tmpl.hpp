
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

#ifdef TMPL_FL

TMPL_EXTERN template size_t flat_sparse_tensor_matmul<TMPL_Q, TMPL_FL>(
    const py::array_t<int32_t> &plan, const py::array_t<TMPL_FL> &adata,
    const py::array_t<TMPL_FL> &bdata, py::array_t<TMPL_FL> &cdata);

#else

TMPL_EXTERN template tuple<py::array_t<uint32_t>, py::array_t<uint32_t>,
                           py::array_t<double>, py::array_t<uint64_t>>
flat_sparse_tensor_diag<TMPL_Q>(const py::array_t<uint32_t> &aqs,
                                const py::array_t<uint32_t> &ashs,
                                const py::array_t<double> &adata,
                                const py::array_t<uint64_t> &aidxs,
                                const py::array_t<int> &idxa,
                                const py::array_t<int> &idxb);

TMPL_EXTERN template tuple<py::array_t<uint32_t>, py::array_t<uint32_t>,
                           py::array_t<uint64_t>>
flat_sparse_tensor_tensordot_skeleton<TMPL_Q>(const py::array_t<uint32_t> &aqs,
                                              const py::array_t<uint32_t> &ashs,
                                              const py::array_t<uint32_t> &bqs,
                                              const py::array_t<uint32_t> &bshs,
                                              const py::array_t<int> &idxa,
                                              const py::array_t<int> &idxb);

TMPL_EXTERN template tuple<int, int, vector<map_uint_uint<TMPL_Q>>,
                           vector<map_uint_uint<TMPL_Q>>>
flat_sparse_tensor_matmul_init<TMPL_Q>(
    const py::array_t<uint32_t> &loqs, const py::array_t<uint32_t> &loshs,
    const py::array_t<uint32_t> &leqs, const py::array_t<uint32_t> &leshs,
    const py::array_t<uint32_t> &roqs, const py::array_t<uint32_t> &roshs,
    const py::array_t<uint32_t> &reqs, const py::array_t<uint32_t> &reshs);

TMPL_EXTERN template py::array_t<int32_t>
flat_sparse_tensor_matmul_plan<TMPL_Q>(
    const py::array_t<uint32_t> &aqs, const py::array_t<uint32_t> &ashs,
    const py::array_t<uint64_t> &aidxs, const py::array_t<uint32_t> &bqs,
    const py::array_t<uint32_t> &bshs, const py::array_t<uint64_t> &bidxs,
    const py::array_t<int> &idxa, const py::array_t<int> &idxb,
    const py::array_t<uint32_t> &cqs, const py::array_t<uint64_t> &cidxs,
    bool ferm_op);

#endif

#endif
