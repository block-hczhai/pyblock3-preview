
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

// Dense tensor api with py::array_t interface

#include "tensor.hpp"

template <typename FL>
py::array_t<FL> tensor_transpose(const py::array_t<FL> &x,
                                 const py::array_t<int> &perm, const FL alpha,
                                 const FL beta) {
    int ndim = (int)x.ndim();
    vector<int> shape(ndim);
    for (int i = 0; i < ndim; i++)
        shape[i] = (int)x.shape()[i];
    vector<ssize_t> shape_y(ndim);
    for (int i = 0; i < ndim; i++)
        shape_y[i] = x.shape()[perm.data()[i]];
    py::array_t<FL> c(shape_y);
    tensor_transpose_impl<FL>(ndim, x.size(), perm.data(), shape.data(),
                              x.data(), c.mutable_data(), alpha, beta);
    return c;
}

template <typename FL>
py::array_t<FL>
tensor_tensordot(const py::array_t<FL> &a, const py::array_t<FL> &b,
                 const py::array_t<int> &idxa, const py::array_t<int> &idxb,
                 FL alpha, FL beta) {
    int ndima = (int)a.ndim(), ndimb = (int)b.ndim(), nctr = (int)idxa.size();
    vector<ssize_t> shapec;
    shapec.reserve(ndima + ndimb);
    shapec.insert(shapec.end(), a.shape(), a.shape() + ndima);
    shapec.insert(shapec.end(), b.shape(), b.shape() + ndima);
    for (int i = 0; i < nctr; i++)
        shapec[idxa.data()[i]] = -1, shapec[idxb.data()[i] + ndima] = -1;
    shapec.resize(
        distance(shapec.begin(), remove(shapec.begin(), shapec.end(), -1)));
    py::array_t<FL> c(shapec);
    tensordot_impl<FL>(a.data(), ndima, a.shape(), b.data(), ndimb, b.shape(),
                       nctr, idxa.data(), idxb.data(), c.mutable_data(), alpha,
                       beta);
    return c;
}

// explicit template instantiation
template py::array_t<double>
tensor_transpose<double>(const py::array_t<double> &x,
                         const py::array_t<int> &perm, const double alpha,
                         const double beta);
template py::array_t<complex<double>> tensor_transpose<complex<double>>(
    const py::array_t<complex<double>> &x, const py::array_t<int> &perm,
    const complex<double> alpha, const complex<double> beta);
template py::array_t<double>
tensor_tensordot(const py::array_t<double> &a, const py::array_t<double> &b,
                 const py::array_t<int> &idxa, const py::array_t<int> &idxb,
                 double alpha, double beta);
template py::array_t<complex<double>>
tensor_tensordot(const py::array_t<complex<double>> &a,
                 const py::array_t<complex<double>> &b,
                 const py::array_t<int> &idxa, const py::array_t<int> &idxb,
                 complex<double> alpha, complex<double> beta);
