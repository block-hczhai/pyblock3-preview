
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

template <typename FL>
pair<py::array_t<FL>, py::array_t<FL>> tensor_qr(const py::array_t<FL> &x,
                                                 bool is_qr) {
    int ndimx = (int)x.ndim();
    assert(ndimx == 2);
    int ml = x.shape()[0], mr = x.shape()[1], mm = min(ml, mr);
    size_t l_size = ml * mm, r_size = mm * mr;
    py::array_t<FL> ldata(vector<ssize_t>{(ssize_t)ml, (ssize_t)mm});
    py::array_t<FL> rdata(vector<ssize_t>{(ssize_t)mm, (ssize_t)mr});
    FL *pl = ldata.mutable_data(), *pr = rdata.mutable_data();

    int lwork = (is_qr ? mr : ml) * 34, info = 0;
    vector<FL> tau, work, tmpl, tmpr;
    work.reserve(lwork);
    tau.reserve(mm);
    tmpl.reserve(ml * mr);
    tmpr.reserve(ml * mr);
    if (is_qr) {
        memset(pr, 0, sizeof(FL) * r_size);
        memcpy(tmpr.data(), x.data(), sizeof(FL) * ml * mr);
        int qlwork = -1;
        xgelqf<FL>(&mr, &ml, tmpr.data(), &mr, tau.data(), work.data(), &qlwork,
                   &info);
        if (lwork < (int)abs(work[0]))
            lwork = (int)abs(work[0]), work.reserve(lwork);
        xgelqf<FL>(&mr, &ml, tmpr.data(), &mr, tau.data(), work.data(), &lwork,
                   &info);
        assert(info == 0);
        memcpy(tmpl.data(), tmpr.data(), sizeof(FL) * ml * mr);
        xunglq<FL>(&mm, &ml, &mm, tmpl.data(), &mr, tau.data(), work.data(),
                   &lwork, &info);
        assert(info == 0);
        xlacpy<FL>("N", &mm, &ml, tmpl.data(), &mr, pl, &mm);
        for (int j = 0; j < min(mm, mr); j++)
            memcpy(pr + j * mr + j, tmpr.data() + j + j * mr,
                   sizeof(FL) * (mr - j));
    } else {
        memset(pl, 0, sizeof(FL) * l_size);
        memcpy(tmpl.data(), x.data(), sizeof(FL) * ml * mr);
        int qlwork = -1;
        xgeqrf<FL>(&mr, &ml, tmpl.data(), &mr, tau.data(), work.data(), &qlwork,
                   &info);
        if (lwork < (int)abs(work[0]))
            lwork = (int)abs(work[0]), work.reserve(lwork);
        xgeqrf<FL>(&mr, &ml, tmpl.data(), &mr, tau.data(), work.data(), &lwork,
                   &info);
        assert(info == 0);
        memcpy(tmpr.data(), tmpl.data(), sizeof(FL) * ml * mr);
        xungqr<FL>(&mr, &mm, &mm, tmpr.data(), &mr, tau.data(), work.data(),
                   &lwork, &info);
        assert(info == 0);
        for (int j = 0; j < ml; j++)
            memcpy(pl + j * mm, tmpl.data() + j * mr,
                   sizeof(FL) * min(mm, j + 1));
        xlacpy<FL>("N", &mr, &mm, tmpr.data(), &mr, pr, &mr);
    }
    return std::make_pair(ldata, rdata);
}

// explicit template instantiation
template py::array_t<double>
tensor_transpose<double>(const py::array_t<double> &x,
                         const py::array_t<int> &perm, const double alpha,
                         const double beta);
template py::array_t<float>
tensor_transpose<float>(const py::array_t<float> &x,
                        const py::array_t<int> &perm, const float alpha,
                        const float beta);
template py::array_t<complex<double>> tensor_transpose<complex<double>>(
    const py::array_t<complex<double>> &x, const py::array_t<int> &perm,
    const complex<double> alpha, const complex<double> beta);
template py::array_t<double> tensor_tensordot(const py::array_t<double> &a,
                                              const py::array_t<double> &b,
                                              const py::array_t<int> &idxa,
                                              const py::array_t<int> &idxb,
                                              double alpha, double beta);
template py::array_t<float> tensor_tensordot(const py::array_t<float> &a,
                                             const py::array_t<float> &b,
                                             const py::array_t<int> &idxa,
                                             const py::array_t<int> &idxb,
                                             float alpha, float beta);
template py::array_t<complex<double>>
tensor_tensordot(const py::array_t<complex<double>> &a,
                 const py::array_t<complex<double>> &b,
                 const py::array_t<int> &idxa, const py::array_t<int> &idxb,
                 complex<double> alpha, complex<double> beta);

template pair<py::array_t<double>, py::array_t<double>>
tensor_qr(const py::array_t<double> &x, bool is_qr);
template pair<py::array_t<float>, py::array_t<float>>
tensor_qr(const py::array_t<float> &x, bool is_qr);
template pair<py::array_t<complex<double>>, py::array_t<complex<double>>>
tensor_qr(const py::array_t<complex<double>> &x, bool is_qr);
