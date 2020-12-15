
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

template <typename FL>
void flat_fermion_tensor_transpose(const py::array_t<uint32_t> &aqs,
                                   const py::array_t<uint32_t> &ashs,
                                   const py::array_t<FL> &adata,
                                   const py::array_t<uint32_t> &aidxs,
                                   const py::array_t<int32_t> &perm,
                                   py::array_t<FL> &cdata);

template <typename FL>
tuple<py::array_t<uint32_t>, py::array_t<uint32_t>, py::array_t<FL>,
      py::array_t<uint32_t>>
flat_fermion_tensor_tensordot(
    const py::array_t<uint32_t> &aqs, const py::array_t<uint32_t> &ashs,
    const py::array_t<FL> &adata, const py::array_t<uint32_t> &aidxs,
    const py::array_t<uint32_t> &bqs, const py::array_t<uint32_t> &bshs,
    const py::array_t<FL> &bdata, const py::array_t<uint32_t> &bidxs,
    const py::array_t<int> &idxa, const py::array_t<int> &idxb);

tuple<py::array_t<uint32_t>, py::array_t<uint32_t>, py::array_t<uint32_t>>
flat_fermion_tensor_skeleton(
    const vector<unordered_map<uint32_t, uint32_t>> &infos, uint32_t fdq);

// explicit template instantiation
extern template void flat_fermion_tensor_transpose<double>(
    const py::array_t<uint32_t> &aqs, const py::array_t<uint32_t> &ashs,
    const py::array_t<double> &adata, const py::array_t<uint32_t> &aidxs,
    const py::array_t<int32_t> &perm, py::array_t<double> &cdata);
extern template void flat_fermion_tensor_transpose<complex<double>>(
    const py::array_t<uint32_t> &aqs, const py::array_t<uint32_t> &ashs,
    const py::array_t<complex<double>> &adata,
    const py::array_t<uint32_t> &aidxs, const py::array_t<int32_t> &perm,
    py::array_t<complex<double>> &cdata);
extern template tuple<py::array_t<uint32_t>, py::array_t<uint32_t>,
                      py::array_t<double>, py::array_t<uint32_t>>
flat_fermion_tensor_tensordot<double>(
    const py::array_t<uint32_t> &aqs, const py::array_t<uint32_t> &ashs,
    const py::array_t<double> &adata, const py::array_t<uint32_t> &aidxs,
    const py::array_t<uint32_t> &bqs, const py::array_t<uint32_t> &bshs,
    const py::array_t<double> &bdata, const py::array_t<uint32_t> &bidxs,
    const py::array_t<int> &idxa, const py::array_t<int> &idxb);
extern template tuple<py::array_t<uint32_t>, py::array_t<uint32_t>,
                      py::array_t<complex<double>>, py::array_t<uint32_t>>
flat_fermion_tensor_tensordot<complex<double>>(
    const py::array_t<uint32_t> &aqs, const py::array_t<uint32_t> &ashs,
    const py::array_t<complex<double>> &adata,
    const py::array_t<uint32_t> &aidxs, const py::array_t<uint32_t> &bqs,
    const py::array_t<uint32_t> &bshs,
    const py::array_t<complex<double>> &bdata,
    const py::array_t<uint32_t> &bidxs, const py::array_t<int> &idxa,
    const py::array_t<int> &idxb);
