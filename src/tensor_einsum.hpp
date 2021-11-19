
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

#include "tensor_impl.hpp"
#include <pybind11/numpy.h>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace std;

template <typename FL>
py::array_t<FL> tensor_einsum(const string &script,
                              const vector<py::array_t<FL>> &arrs);

extern template py::array_t<double>
tensor_einsum(const string &script, const vector<py::array_t<double>> &arrs);

extern template py::array_t<complex<double>>
tensor_einsum(const string &script,
              const vector<py::array_t<complex<double>>> &arrs);
