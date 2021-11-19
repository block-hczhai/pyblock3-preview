
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

#include "bond_info.hpp"
#include "tensor_impl.hpp"
#ifdef I
#undef I
#endif
#include <pybind11/numpy.h>
#include <tuple>

namespace py = pybind11;
using namespace std;

template <typename Q, typename FL>
void flat_fermion_tensor_transpose(const py::array_t<uint32_t> &aqs,
                                   const py::array_t<uint32_t> &ashs,
                                   const py::array_t<FL> &adata,
                                   const py::array_t<uint64_t> &aidxs,
                                   const py::array_t<int32_t> &perm,
                                   py::array_t<FL> &cdata);

template <typename Q, typename FL>
tuple<py::array_t<uint32_t>, py::array_t<uint32_t>, py::array_t<FL>,
      py::array_t<uint64_t>>
flat_fermion_tensor_tensordot(
    const py::array_t<uint32_t> &aqs, const py::array_t<uint32_t> &ashs,
    const py::array_t<FL> &adata, const py::array_t<uint64_t> &aidxs,
    const py::array_t<uint32_t> &bqs, const py::array_t<uint32_t> &bshs,
    const py::array_t<FL> &bdata, const py::array_t<uint64_t> &bidxs,
    const py::array_t<int> &idxa, const py::array_t<int> &idxb);

template <typename Q>
tuple<py::array_t<uint32_t>, py::array_t<uint32_t>, py::array_t<uint64_t>>
flat_fermion_tensor_skeleton(const vector<map_uint_uint<Q>> &infos,
                             uint32_t fdq);

template <typename Q, typename FL>
tuple<py::array_t<uint32_t>, py::array_t<uint32_t>, py::array_t<FL>,
      py::array_t<uint64_t>, py::array_t<uint32_t>, py::array_t<uint32_t>,
      py::array_t<FL>, py::array_t<uint64_t>>
flat_fermion_tensor_qr(const py::array_t<uint32_t> &aqs,
                       const py::array_t<uint32_t> &ashs,
                       const py::array_t<FL> &adata,
                       const py::array_t<uint64_t> &aidxs, int idx,
                       const string &pattern, bool is_qr);

#define TMPL_EXTERN extern
#define TMPL_NAME flat_fermion

#include "symmetry_tmpl.hpp"
#define TMPL_FL double
#include "symmetry_tmpl.hpp"
#undef TMPL_FL
#define TMPL_FL complex<double>
#include "symmetry_tmpl.hpp"
#undef TMPL_FL

#undef TMPL_NAME
#undef TMPL_EXTERN
