
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

#include "fermion_symmetry.hpp"
#include "flat_fermion.hpp"
#include "flat_functor.hpp"
#include "flat_sparse.hpp"
#include "hamiltonian.hpp"
#include "hamiltonian_ptree.hpp"
#include "qc_hamiltonian.hpp"
#include "sz.hpp"
#include "tensor.hpp"
#include "tensor_einsum.hpp"
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

PYBIND11_MAKE_OPAQUE(vector<uint32_t>);
PYBIND11_MAKE_OPAQUE(vector<uint64_t>);
PYBIND11_MAKE_OPAQUE(
    unordered_map<vector<uint32_t>, pair<uint32_t, vector<uint32_t>>>);
PYBIND11_MAKE_OPAQUE(map_fusing);
PYBIND11_MAKE_OPAQUE(vector<unordered_map<uint32_t, uint32_t>>);
PYBIND11_MAKE_OPAQUE(unordered_map<uint32_t, uint32_t>);
PYBIND11_MAKE_OPAQUE(
    vector<std::tuple<py::array_t<uint32_t>, py::array_t<uint32_t>,
                      py::array_t<double>, py::array_t<uint64_t>>>);
PYBIND11_MAKE_OPAQUE(
    vector<std::tuple<py::array_t<uint32_t>, py::array_t<uint32_t>,
                      py::array_t<complex<double>>, py::array_t<uint64_t>>>);

template <typename Q>
void bind_sparse_tensor(py::module &m, py::module &pm, string name) {

    py::bind_map<map_uint_uint<Q>>(m, "MapUIntUInt")
        .def_property_readonly("n_bonds",
                               [](map_uint_uint<Q> *self) {
                                   uint32_t r = 0;
                                   for (auto &kv : *self)
                                       r += kv.second;
                                   return r;
                               })
        .def("__neg__",
             [](map_uint_uint<Q> *self) {
                 map_uint_uint<Q> r;
                 for (auto &kv : *self)
                     r[Q::from_q(-Q::to_q(kv.first))] = kv.second;
                 return r;
             })
        .def("__add__",
             [](map_uint_uint<Q> *self, map_uint_uint<Q> *other) {
                 map_uint_uint<Q> r;
                 r.insert(self->begin(), self->end());
                 for (auto &kv : *other)
                     r[kv.first] += kv.second;
                 return r;
             })
        .def("__mul__",
             [](map_uint_uint<Q> *self, map_uint_uint<Q> *other) {
                 map_uint_uint<Q> r;
                 for (auto &a : *self)
                     for (auto &b : *other) {
                         uint32_t q =
                             Q::from_q(Q::to_q(a.first) + Q::to_q(b.first));
                         r[q] = min(a.second * b.second + r[q], 65535U);
                     }
                 return r;
             })
        .def("__and__",
             [](map_uint_uint<Q> *self, map_uint_uint<Q> *other) {
                 map_uint_uint<Q> r;
                 for (auto &a : *self)
                     if (other->count(a.first))
                         r[a.first] = a.second;
                 return r;
             })
        .def("__or__",
             [](map_uint_uint<Q> *self, map_uint_uint<Q> *other) {
                 map_uint_uint<Q> r;
                 r.insert(self->begin(), self->end());
                 r.insert(other->begin(), other->end());
                 return r;
             })
        .def("__xor__",
             [](map_uint_uint<Q> *self, map_uint_uint<Q> *other) {
                 return move(bond_info_fusing_product<Q>(
                     vector<map_uint_uint<Q>>{*self, *other}, "++"));
             })
        .def("filter",
             [](map_uint_uint<Q> *self, map_uint_uint<Q> *other) {
                 map_uint_uint<Q> r;
                 for (auto &a : *self)
                     if (other->count(a.first))
                         r[a.first] = min(a.second, other->at(a.first));
                 return r;
             })
        .def("truncate",
             [](map_uint_uint<Q> *self, int bond_dim,
                map_uint_uint<Q> *ref = nullptr) {
                 size_t n_total = 0;
                 for (auto &kv : *self)
                     n_total += kv.second;
                 if (n_total > bond_dim) {
                     for (auto &kv : *self) {
                         kv.second = (uint32_t)ceil(
                             (double)kv.second * bond_dim / n_total + 0.1);
                         if (ref != nullptr)
                             kv.second = ref->count(kv.first)
                                             ? min(kv.second, ref->at(kv.first))
                                             : 0;
                     }
                 }
             })
        .def("keep_maximal",
             [](map_uint_uint<Q> *self) {
                 vector<Q> qs;
                 map_uint_uint<Q> r;
                 for (auto &a : *self)
                     qs.push_back(Q::to_q(a.first));
                 Q q = *max_element(qs.begin(), qs.end());
                 r[Q::from_q(q)] = (*self)[Q::from_q(q)];
                 return r;
             })
        .def(py::pickle(
            [](map_uint_uint<Q> *self) {
                py::array_t<uint32_t> data{
                    vector<ssize_t>{(ssize_t)self->size() * 2}};
                vector<pair<uint32_t, uint32_t>> vpu(self->begin(),
                                                     self->end());
                for (size_t i = 0; i < vpu.size(); i++) {
                    data.mutable_data()[i * 2] = vpu[i].first;
                    data.mutable_data()[i * 2 + 1] = vpu[i].second;
                }
                return py::make_tuple(data);
            },
            [](py::tuple t) {
                py::array_t<uint32_t> data = t[0].cast<py::array_t<uint32_t>>();
                vector<pair<uint32_t, uint32_t>> vpu(data.shape()[0] / 2);
                for (size_t i = 0; i < vpu.size(); i++) {
                    vpu[i].first = data.data()[i * 2];
                    vpu[i].second = data.data()[i * 2 + 1];
                }
                return map_uint_uint<Q>(vpu.begin(), vpu.end());
            }))
        .def_static(
            "set_bond_dimension_occ",
            [](const vector<map_uint_uint<Q>> &basis,
               vector<map_uint_uint<Q>> &left_dims,
               vector<map_uint_uint<Q>> &right_dims, uint32_t vacuum,
               uint32_t target, int m, py::array_t<double> &occ, double bias) {
                vector<double> vocc(occ.data(), occ.data() + occ.size());
                return bond_info_set_bond_dimension_occ<Q>(
                    basis, left_dims, right_dims, vacuum, target, m, vocc,
                    bias);
            })
        .def_static("tensor_product", &tensor_product_ref<Q>, py::arg("a"),
                    py::arg("b"), py::arg("ref"));

    py::bind_vector<vector<map_uint_uint<Q>>>(m, "VectorMapUIntUInt");

    m.attr("VectorUInt") = pm.attr("VectorUInt");
    m.attr("MapVUIntPUV") = pm.attr("MapVUIntPUV");
    m.attr("MapFusing") = pm.attr("MapFusing");
    m.attr("VectorFlat") = pm.attr("VectorFlat");
    m.attr("VectorComplexFlat") = pm.attr("VectorComplexFlat");

    py::module flat_sparse_tensor =
        m.def_submodule("flat_sparse_tensor", "FlatSparseTensor");
    flat_sparse_tensor.def("fix_pattern", &flat_sparse_tensor_fix_pattern<Q>,
                           py::arg("aqs"), py::arg("pattern"), py::arg("dq"));
    flat_sparse_tensor.def("skeleton", &flat_sparse_tensor_skeleton<Q>,
                           py::arg("infos"), py::arg("pattern"), py::arg("dq"));
    flat_sparse_tensor.def("get_infos", &flat_sparse_tensor_get_infos<Q>,
                           py::arg("aqs"), py::arg("ashs"));
    flat_sparse_tensor.def("kron_sum_info",
                           &flat_sparse_tensor_kron_sum_info<Q>, py::arg("aqs"),
                           py::arg("ashs"), py::arg("pattern"));
    flat_sparse_tensor.def("kron_product_info", &bond_info_fusing_product<Q>,
                           py::arg("infos"), py::arg("pattern"));
    flat_sparse_tensor.def("tensordot_skeleton",
                           &flat_sparse_tensor_tensordot_skeleton<Q>,
                           py::arg("aqs"), py::arg("ashs"), py::arg("bqs"),
                           py::arg("bshs"), py::arg("idxa"), py::arg("idxb"));

    // double
    flat_sparse_tensor.def(
        "transpose",
        [](const py::object &ashs, const py::array_t<double> &adata,
           const py::object &aidxs, const py::object &perm,
           py::array_t<double> &cdata) {
            return flat_sparse_tensor_transpose<Q, double>(ashs, adata, aidxs,
                                                           perm, cdata);
        },
        py::arg("ashs"), py::arg("adata"), py::arg("aidxs"), py::arg("perm"),
        py::arg("cdata"));
    flat_sparse_tensor.def(
        "tensordot",
        [](const py::object &aqs, const py::object &ashs,
           const py::array_t<double> &adata, const py::object &aidxs,
           const py::object &bqs, const py::object &bshs,
           const py::array_t<double> &bdata, const py::object &bidxs,
           const py::object &idxa, const py::object &idxb) {
            return flat_sparse_tensor_tensordot<Q, double>(
                aqs, ashs, adata, aidxs, bqs, bshs, bdata, bidxs, idxa, idxb);
        },
        py::arg("aqs"), py::arg("ashs"), py::arg("adata"), py::arg("aidxs"),
        py::arg("bqs"), py::arg("bshs"), py::arg("bdata"), py::arg("bidxs"),
        py::arg("idxa"), py::arg("idxb"));
    flat_sparse_tensor.def(
        "add",
        [](const py::object &aqs, const py::object &ashs,
           const py::array_t<double> &adata, const py::object &aidxs,
           const py::object &bqs, const py::object &bshs,
           const py::array_t<double> &bdata, const py::object &bidxs) {
            return flat_sparse_tensor_add<Q, double>(aqs, ashs, adata, aidxs,
                                                     bqs, bshs, bdata, bidxs);
        },
        py::arg("aqs"), py::arg("ashs"), py::arg("adata"), py::arg("aidxs"),
        py::arg("bqs"), py::arg("bshs"), py::arg("bdata"), py::arg("bidxs"));
    flat_sparse_tensor.def(
        "kron_add",
        [](const py::object &aqs, const py::object &ashs,
           const py::array_t<double> &adata, const py::object &aidxs,
           const py::object &bqs, const py::object &bshs,
           const py::array_t<double> &bdata, const py::object &bidxs,
           const map_uint_uint<Q> &infol, const map_uint_uint<Q> &infor) {
            return flat_sparse_tensor_kron_add<Q, double>(
                aqs, ashs, adata, aidxs, bqs, bshs, bdata, bidxs, infol, infor);
        },
        py::arg("aqs"), py::arg("ashs"), py::arg("adata"), py::arg("aidxs"),
        py::arg("bqs"), py::arg("bshs"), py::arg("bdata"), py::arg("bidxs"),
        py::arg("infol"), py::arg("infor"));
    flat_sparse_tensor.def(
        "fuse",
        [](const py::object &aqs, const py::object &ashs,
           const py::array_t<double> &adata, const py::object &aidxs,
           const py::object &idxs, const map_fusing &info,
           const string &pattern) {
            return flat_sparse_tensor_fuse<Q, double>(aqs, ashs, adata, aidxs,
                                                      idxs, info, pattern);
        },
        py::arg("aqs"), py::arg("ashs"), py::arg("adata"), py::arg("aidxs"),
        py::arg("idxs"), py::arg("info"), py::arg("pattern"));
    flat_sparse_tensor.def(
        "left_canonicalize",
        [](const py::object &aqs, const py::object &ashs,
           const py::array_t<double> &adata, const py::object &aidxs) {
            return flat_sparse_left_canonicalize<Q, double>(aqs, ashs, adata,
                                                            aidxs);
        },
        py::arg("aqs"), py::arg("ashs"), py::arg("adata"), py::arg("aidxs"));
    flat_sparse_tensor.def(
        "right_canonicalize",
        [](const py::object &aqs, const py::object &ashs,
           const py::array_t<double> &adata, const py::object &aidxs) {
            return flat_sparse_right_canonicalize<Q, double>(aqs, ashs, adata,
                                                             aidxs);
        },
        py::arg("aqs"), py::arg("ashs"), py::arg("adata"), py::arg("aidxs"));
    flat_sparse_tensor.def(
        "left_canonicalize_indexed",
        [](const py::object &aqs, const py::object &ashs,
           const py::array_t<double> &adata, const py::object &aidxs) {
            return flat_sparse_left_canonicalize_indexed<Q, double>(
                aqs, ashs, adata, aidxs);
        },
        py::arg("aqs"), py::arg("ashs"), py::arg("adata"), py::arg("aidxs"));
    flat_sparse_tensor.def(
        "right_canonicalize_indexed",
        [](const py::object &aqs, const py::object &ashs,
           const py::array_t<double> &adata, const py::object &aidxs) {
            return flat_sparse_right_canonicalize_indexed<Q, double>(
                aqs, ashs, adata, aidxs);
        },
        py::arg("aqs"), py::arg("ashs"), py::arg("adata"), py::arg("aidxs"));
    flat_sparse_tensor.def("left_svd", 
        [](const py::object &aqs, const py::object &ashs,
           const py::array_t<double> &adata, const py::object &aidxs) {
            return flat_sparse_left_svd<Q, double>(aqs, ashs, adata,
                                                            aidxs);
        },
        py::arg("aqs"), py::arg("ashs"), py::arg("adata"), py::arg("aidxs"));
    flat_sparse_tensor.def("right_svd", 
        [](const py::object &aqs, const py::object &ashs,
           const py::array_t<double> &adata, const py::object &aidxs) {
            return flat_sparse_right_svd<Q, double>(aqs, ashs, adata,
                                                             aidxs);
        },
        py::arg("aqs"), py::arg("ashs"), py::arg("adata"), py::arg("aidxs"));
    flat_sparse_tensor.def("left_svd_indexed",
        [](const py::object &aqs, const py::object &ashs,
           const py::array_t<double> &adata, const py::object &aidxs) {
            return flat_sparse_left_svd_indexed<Q, double>(
                aqs, ashs, adata, aidxs);
        },
        py::arg("aqs"), py::arg("ashs"), py::arg("adata"), py::arg("aidxs"));
    flat_sparse_tensor.def("right_svd_indexed",
        [](const py::object &aqs, const py::object &ashs,
           const py::array_t<double> &adata, const py::object &aidxs) {
            return flat_sparse_right_svd_indexed<Q, double>(
                aqs, ashs, adata, aidxs);
        },
        py::arg("aqs"), py::arg("ashs"), py::arg("adata"), py::arg("aidxs"));
    flat_sparse_tensor.def(
        "tensor_svd",
        [](const py::object &aqs, const py::object &ashs,
           const py::array_t<double> &adata, const py::object &aidxs, int idx,
           const map_fusing &linfo, const map_fusing &rinfo,
           const string &pattern) {
            return flat_sparse_tensor_svd<Q, double>(
                aqs, ashs, adata, aidxs, idx, linfo, rinfo, pattern);
        },
        py::arg("aqs"), py::arg("ashs"), py::arg("adata"), py::arg("aidxs"),
        py::arg("idx"), py::arg("linfo"), py::arg("rinfo"), py::arg("pattern"));
    flat_sparse_tensor.def(
        "truncate_svd",
        [](const py::object &lqs, const py::object &lshs,
           const py::array_t<double> &ldata, const py::object &lidxs,
           const py::object &sqs, const py::object &sshs,
           const py::array_t<double> &sdata, const py::object &sidxs,
           const py::object &rqs, const py::object &rshs,
           const py::array_t<double> &rdata, const py::object &ridxs,
           py::object max_bond_dim, double cutoff, double max_dw,
           double norm_cutoff, bool eigen_values) {
            return flat_sparse_truncate_svd<Q, double>(
                lqs, lshs, ldata, lidxs, sqs, sshs, sdata, sidxs, rqs, rshs,
                rdata, ridxs, max_bond_dim.cast<int>(), cutoff, max_dw,
                norm_cutoff, eigen_values);
        },
        py::arg("lqs"), py::arg("lshs"), py::arg("ldata"), py::arg("lidxs"),
        py::arg("sqs"), py::arg("sshs"), py::arg("sdata"), py::arg("sidxs"),
        py::arg("rqs"), py::arg("rshs"), py::arg("rdata"), py::arg("ridxs"),
        py::arg("max_bond_dim") = -1, py::arg("cutoff") = 0.0,
        py::arg("max_dw") = 0.0, py::arg("norm_cutoff") = 0.0,
        py::arg("eigen_values") = false);

    // complex double
    flat_sparse_tensor.def(
        "transpose",
        [](const py::object &ashs, const py::array_t<complex<double>> &adata,
           const py::object &aidxs, const py::object &perm,
           py::array_t<complex<double>> &cdata) {
            return flat_sparse_tensor_transpose<Q, complex<double>>(
                ashs, adata, aidxs, perm, cdata);
        },
        py::arg("ashs"), py::arg("adata"), py::arg("aidxs"), py::arg("perm"),
        py::arg("cdata"));
    flat_sparse_tensor.def(
        "tensordot",
        [](const py::object &aqs, const py::object &ashs,
           const py::array_t<complex<double>> &adata, const py::object &aidxs,
           const py::object &bqs, const py::object &bshs,
           const py::array_t<complex<double>> &bdata, const py::object &bidxs,
           const py::object &idxa, const py::object &idxb) {
            return flat_sparse_tensor_tensordot<Q, complex<double>>(
                aqs, ashs, adata, aidxs, bqs, bshs, bdata, bidxs, idxa, idxb);
        },
        py::arg("aqs"), py::arg("ashs"), py::arg("adata"), py::arg("aidxs"),
        py::arg("bqs"), py::arg("bshs"), py::arg("bdata"), py::arg("bidxs"),
        py::arg("idxa"), py::arg("idxb"));
    flat_sparse_tensor.def(
        "add",
        [](const py::object &aqs, const py::object &ashs,
           const py::array_t<complex<double>> &adata, const py::object &aidxs,
           const py::object &bqs, const py::object &bshs,
           const py::array_t<complex<double>> &bdata, const py::object &bidxs) {
            return flat_sparse_tensor_add<Q, complex<double>>(
                aqs, ashs, adata, aidxs, bqs, bshs, bdata, bidxs);
        },
        py::arg("aqs"), py::arg("ashs"), py::arg("adata"), py::arg("aidxs"),
        py::arg("bqs"), py::arg("bshs"), py::arg("bdata"), py::arg("bidxs"));
    flat_sparse_tensor.def(
        "kron_add",
        [](const py::object &aqs, const py::object &ashs,
           const py::array_t<complex<double>> &adata, const py::object &aidxs,
           const py::object &bqs, const py::object &bshs,
           const py::array_t<complex<double>> &bdata, const py::object &bidxs,
           const map_uint_uint<Q> &infol, const map_uint_uint<Q> &infor) {
            return flat_sparse_tensor_kron_add<Q, complex<double>>(
                aqs, ashs, adata, aidxs, bqs, bshs, bdata, bidxs, infol, infor);
        },
        py::arg("aqs"), py::arg("ashs"), py::arg("adata"), py::arg("aidxs"),
        py::arg("bqs"), py::arg("bshs"), py::arg("bdata"), py::arg("bidxs"),
        py::arg("infol"), py::arg("infor"));
    flat_sparse_tensor.def(
        "fuse",
        [](const py::object &aqs, const py::object &ashs,
           const py::array_t<complex<double>> &adata, const py::object &aidxs,
           const py::object &idxs, const map_fusing &info,
           const string &pattern) {
            return flat_sparse_tensor_fuse<Q, complex<double>>(
                aqs, ashs, adata, aidxs, idxs, info, pattern);
        },
        py::arg("aqs"), py::arg("ashs"), py::arg("adata"), py::arg("aidxs"),
        py::arg("idxs"), py::arg("info"), py::arg("pattern"));
    flat_sparse_tensor.def(
        "left_canonicalize",
        [](const py::object &aqs, const py::object &ashs,
           const py::array_t<complex<double>> &adata, const py::object &aidxs) {
            return flat_sparse_left_canonicalize<Q, complex<double>>(
                aqs, ashs, adata, aidxs);
        },
        py::arg("aqs"), py::arg("ashs"), py::arg("adata"), py::arg("aidxs"));
    flat_sparse_tensor.def(
        "right_canonicalize",
        [](const py::object &aqs, const py::object &ashs,
           const py::array_t<complex<double>> &adata, const py::object &aidxs) {
            return flat_sparse_right_canonicalize<Q, complex<double>>(
                aqs, ashs, adata, aidxs);
        },
        py::arg("aqs"), py::arg("ashs"), py::arg("adata"), py::arg("aidxs"));
    flat_sparse_tensor.def(
        "left_canonicalize_indexed",
        [](const py::object &aqs, const py::object &ashs,
           const py::array_t<complex<double>> &adata, const py::object &aidxs) {
            return flat_sparse_left_canonicalize_indexed<Q, complex<double>>(
                aqs, ashs, adata, aidxs);
        },
        py::arg("aqs"), py::arg("ashs"), py::arg("adata"), py::arg("aidxs"));
    flat_sparse_tensor.def(
        "right_canonicalize_indexed",
        [](const py::object &aqs, const py::object &ashs,
           const py::array_t<complex<double>> &adata, const py::object &aidxs) {
            return flat_sparse_right_canonicalize_indexed<Q, complex<double>>(
                aqs, ashs, adata, aidxs);
        },
        py::arg("aqs"), py::arg("ashs"), py::arg("adata"), py::arg("aidxs"));
     flat_sparse_tensor.def("left_svd", 
        [](const py::object &aqs, const py::object &ashs,
           const py::array_t<complex<double>> &adata, const py::object &aidxs) {
            return flat_sparse_left_svd<Q, complex<double>>(aqs, ashs, adata,
                                                            aidxs);
        },
        py::arg("aqs"), py::arg("ashs"), py::arg("adata"), py::arg("aidxs"));
    flat_sparse_tensor.def("right_svd", 
        [](const py::object &aqs, const py::object &ashs,
           const py::array_t<complex<double>> &adata, const py::object &aidxs) {
            return flat_sparse_right_svd<Q, complex<double>>(aqs, ashs, adata,
                                                             aidxs);
        },
        py::arg("aqs"), py::arg("ashs"), py::arg("adata"), py::arg("aidxs"));
    flat_sparse_tensor.def("left_svd_indexed",
        [](const py::object &aqs, const py::object &ashs,
           const py::array_t<complex<double>> &adata, const py::object &aidxs) {
            return flat_sparse_left_svd_indexed<Q, complex<double>>(
                aqs, ashs, adata, aidxs);
        },
        py::arg("aqs"), py::arg("ashs"), py::arg("adata"), py::arg("aidxs"));
    flat_sparse_tensor.def("right_svd_indexed",
        [](const py::object &aqs, const py::object &ashs,
           const py::array_t<complex<double>> &adata, const py::object &aidxs) {
            return flat_sparse_right_svd_indexed<Q, complex<double>>(
                aqs, ashs, adata, aidxs);
        },
        py::arg("aqs"), py::arg("ashs"), py::arg("adata"), py::arg("aidxs"));
    flat_sparse_tensor.def(
        "tensor_svd",
        [](const py::object &aqs, const py::object &ashs,
           const py::array_t<complex<double>> &adata, const py::object &aidxs,
           int idx, const map_fusing &linfo, const map_fusing &rinfo,
           const string &pattern) {
            return flat_sparse_tensor_svd<Q, complex<double>>(
                aqs, ashs, adata, aidxs, idx, linfo, rinfo, pattern);
        },
        py::arg("aqs"), py::arg("ashs"), py::arg("adata"), py::arg("aidxs"),
        py::arg("idx"), py::arg("linfo"), py::arg("rinfo"), py::arg("pattern"));
    flat_sparse_tensor.def(
        "truncate_svd",
        [](const py::object &lqs, const py::object &lshs,
           const py::array_t<complex<double>> &ldata, const py::object &lidxs,
           const py::object &sqs, const py::object &sshs,
           const py::array_t<double> &sdata, const py::object &sidxs,
           const py::object &rqs, const py::object &rshs,
           const py::array_t<complex<double>> &rdata, const py::object &ridxs,
           py::object max_bond_dim, double cutoff, double max_dw,
           double norm_cutoff, bool eigen_values) {
            return flat_sparse_truncate_svd<Q, complex<double>>(
                lqs, lshs, ldata, lidxs, sqs, sshs, sdata, sidxs, rqs, rshs,
                rdata, ridxs, max_bond_dim.cast<int>(), cutoff, max_dw,
                norm_cutoff, eigen_values);
        },
        py::arg("lqs"), py::arg("lshs"), py::arg("ldata"), py::arg("lidxs"),
        py::arg("sqs"), py::arg("sshs"), py::arg("sdata"), py::arg("sidxs"),
        py::arg("rqs"), py::arg("rshs"), py::arg("rdata"), py::arg("ridxs"),
        py::arg("max_bond_dim") = -1, py::arg("cutoff") = 0.0,
        py::arg("max_dw") = 0.0, py::arg("norm_cutoff") = 0.0,
        py::arg("eigen_values") = false);

    // mixed C x D
    flat_sparse_tensor.def(
        "tensordot",
        [](const py::object &aqs, const py::object &ashs,
           const py::array_t<complex<double>> &adata, const py::object &aidxs,
           const py::object &bqs, const py::object &bshs,
           const py::array_t<double> &bdata, const py::object &bidxs,
           const py::object &idxa, const py::object &idxb) {
            return flat_sparse_tensor_tensordot<Q, complex<double>>(
                aqs, ashs, adata, aidxs, bqs, bshs, bdata, bidxs, idxa, idxb);
        },
        py::arg("aqs"), py::arg("ashs"), py::arg("adata"), py::arg("aidxs"),
        py::arg("bqs"), py::arg("bshs"), py::arg("bdata"), py::arg("bidxs"),
        py::arg("idxa"), py::arg("idxb"));
    flat_sparse_tensor.def(
        "add",
        [](const py::object &aqs, const py::object &ashs,
           const py::array_t<complex<double>> &adata, const py::object &aidxs,
           const py::object &bqs, const py::object &bshs,
           const py::array_t<double> &bdata, const py::object &bidxs) {
            return flat_sparse_tensor_add<Q, complex<double>>(
                aqs, ashs, adata, aidxs, bqs, bshs, bdata, bidxs);
        },
        py::arg("aqs"), py::arg("ashs"), py::arg("adata"), py::arg("aidxs"),
        py::arg("bqs"), py::arg("bshs"), py::arg("bdata"), py::arg("bidxs"));
    flat_sparse_tensor.def(
        "kron_add",
        [](const py::object &aqs, const py::object &ashs,
           const py::array_t<complex<double>> &adata, const py::object &aidxs,
           const py::object &bqs, const py::object &bshs,
           const py::array_t<double> &bdata, const py::object &bidxs,
           const map_uint_uint<Q> &infol, const map_uint_uint<Q> &infor) {
            return flat_sparse_tensor_kron_add<Q, complex<double>>(
                aqs, ashs, adata, aidxs, bqs, bshs, bdata, bidxs, infol, infor);
        },
        py::arg("aqs"), py::arg("ashs"), py::arg("adata"), py::arg("aidxs"),
        py::arg("bqs"), py::arg("bshs"), py::arg("bdata"), py::arg("bidxs"),
        py::arg("infol"), py::arg("infor"));

    // mixed D x C
    flat_sparse_tensor.def(
        "tensordot",
        [](const py::object &aqs, const py::object &ashs,
           const py::array_t<double> &adata, const py::object &aidxs,
           const py::object &bqs, const py::object &bshs,
           const py::array_t<complex<double>> &bdata, const py::object &bidxs,
           const py::object &idxa, const py::object &idxb) {
            return flat_sparse_tensor_tensordot<Q, complex<double>>(
                aqs, ashs, adata, aidxs, bqs, bshs, bdata, bidxs, idxa, idxb);
        },
        py::arg("aqs"), py::arg("ashs"), py::arg("adata"), py::arg("aidxs"),
        py::arg("bqs"), py::arg("bshs"), py::arg("bdata"), py::arg("bidxs"),
        py::arg("idxa"), py::arg("idxb"));
    flat_sparse_tensor.def(
        "add",
        [](const py::object &aqs, const py::object &ashs,
           const py::array_t<double> &adata, const py::object &aidxs,
           const py::object &bqs, const py::object &bshs,
           const py::array_t<complex<double>> &bdata, const py::object &bidxs) {
            return flat_sparse_tensor_add<Q, complex<double>>(
                aqs, ashs, adata, aidxs, bqs, bshs, bdata, bidxs);
        },
        py::arg("aqs"), py::arg("ashs"), py::arg("adata"), py::arg("aidxs"),
        py::arg("bqs"), py::arg("bshs"), py::arg("bdata"), py::arg("bidxs"));
    flat_sparse_tensor.def(
        "kron_add",
        [](const py::object &aqs, const py::object &ashs,
           const py::array_t<double> &adata, const py::object &aidxs,
           const py::object &bqs, const py::object &bshs,
           const py::array_t<complex<double>> &bdata, const py::object &bidxs,
           const map_uint_uint<Q> &infol, const map_uint_uint<Q> &infor) {
            return flat_sparse_tensor_kron_add<Q, complex<double>>(
                aqs, ashs, adata, aidxs, bqs, bshs, bdata, bidxs, infol, infor);
        },
        py::arg("aqs"), py::arg("ashs"), py::arg("adata"), py::arg("aidxs"),
        py::arg("bqs"), py::arg("bshs"), py::arg("bdata"), py::arg("bidxs"),
        py::arg("infol"), py::arg("infor"));

    flat_sparse_tensor.def("diag", &flat_sparse_tensor_diag<Q>, py::arg("aqs"),
                           py::arg("ashs"), py::arg("adata"), py::arg("aidxs"),
                           py::arg("idxa"), py::arg("idxb"));
    flat_sparse_tensor.def("matmul_init", &flat_sparse_tensor_matmul_init<Q>,
                           py::arg("loqs"), py::arg("loshs"), py::arg("leqs"),
                           py::arg("leshs"), py::arg("roqs"), py::arg("roshs"),
                           py::arg("reqs"), py::arg("reshs"));
    flat_sparse_tensor.def("matmul_plan", &flat_sparse_tensor_matmul_plan<Q>,
                           py::arg("aqs"), py::arg("ashs"), py::arg("aidxs"),
                           py::arg("bqs"), py::arg("bshs"), py::arg("bidxs"),
                           py::arg("idxa"), py::arg("idxb"), py::arg("cqs"),
                           py::arg("cidxs"), py::arg("ferm_op"));

    flat_sparse_tensor.def(
        "matmul",
        [](const py::object &plan, const py::array_t<double> &adata,
           const py::array_t<double> &bdata, py::array_t<double> &cdata) {
            return flat_sparse_tensor_matmul<Q, double>(plan, adata, bdata,
                                                        cdata);
        },
        py::arg("plan"), py::arg("adata"), py::arg("bdata"), py::arg("cdata"));
    flat_sparse_tensor.def(
        "matmul",
        [](const py::object &plan, const py::array_t<complex<double>> &adata,
           const py::array_t<complex<double>> &bdata,
           py::array_t<complex<double>> &cdata) {
            return flat_sparse_tensor_matmul<Q, complex<double>>(plan, adata,
                                                                 bdata, cdata);
        },
        py::arg("plan"), py::arg("adata"), py::arg("bdata"), py::arg("cdata"));
    flat_sparse_tensor.def(
        "matmul",
        [](const py::object &plan, const py::array_t<double> &adata,
           const py::array_t<double> &bdata,
           py::array_t<complex<double>> &cdata) {
            return flat_sparse_tensor_matmul<Q, complex<double>>(plan, adata,
                                                                 bdata, cdata);
        },
        py::arg("plan"), py::arg("adata"), py::arg("bdata"), py::arg("cdata"));
    flat_sparse_tensor.def(
        "matmul",
        [](const py::object &plan, const py::array_t<double> &adata,
           const py::array_t<complex<double>> &bdata,
           py::array_t<complex<double>> &cdata) {
            return flat_sparse_tensor_matmul<Q, complex<double>>(plan, adata,
                                                                 bdata, cdata);
        },
        py::arg("plan"), py::arg("adata"), py::arg("bdata"), py::arg("cdata"));

    py::module flat_fermion_tensor = m.def_submodule("flat_fermion_tensor");
    flat_fermion_tensor.def("skeleton", &flat_fermion_tensor_skeleton<Q>,
                            py::arg("infos"), py::arg("dq"));

    // double
    flat_fermion_tensor.def(
        "transpose",
        [](const py::object &aqs, const py::object &ashs,
           const py::array_t<double> &adata, const py::object &aidxs,
           const py::object &perm, py::array_t<double> &cdata) {
            return flat_fermion_tensor_transpose<Q, double>(aqs, ashs, adata,
                                                            aidxs, perm, cdata);
        },
        py::arg("aqs"), py::arg("ashs"), py::arg("adata"), py::arg("aidxs"),
        py::arg("perm"), py::arg("cdata"));
    flat_fermion_tensor.def(
        "tensordot",
        [](const py::object &aqs, const py::object &ashs,
           const py::array_t<double> &adata, const py::object &aidxs,
           const py::object &bqs, const py::object &bshs,
           const py::array_t<double> &bdata, const py::object &bidxs,
           const py::object &idxa, const py::object &idxb) {
            return flat_fermion_tensor_tensordot<Q, double>(
                aqs, ashs, adata, aidxs, bqs, bshs, bdata, bidxs, idxa, idxb);
        },
        py::arg("aqs"), py::arg("ashs"), py::arg("adata"), py::arg("aidxs"),
        py::arg("bqs"), py::arg("bshs"), py::arg("bdata"), py::arg("bidxs"),
        py::arg("idxa"), py::arg("idxb"));
    flat_fermion_tensor.def(
        "tensor_qr",
        [](const py::object &aqs, const py::object &ashs,
           const py::array_t<double> &adata, const py::object &aidxs,
           const py::object idx, const string &pattern, bool is_qr) {
            return flat_fermion_tensor_qr<Q, double>(
                aqs, ashs, adata, aidxs, idx.cast<int>(), pattern, is_qr);
        },
        py::arg("aqs"), py::arg("ashs"), py::arg("adata"), py::arg("aidxs"),
        py::arg("idx"), py::arg("pattern"), py::arg("is_qr"));

    // complex double
    flat_fermion_tensor.def(
        "transpose",
        [](const py::object &aqs, const py::object &ashs,
           const py::array_t<complex<double>> &adata, const py::object &aidxs,
           const py::object &perm, py::array_t<complex<double>> &cdata) {
            return flat_fermion_tensor_transpose<Q, complex<double>>(
                aqs, ashs, adata, aidxs, perm, cdata);
        },
        py::arg("aqs"), py::arg("ashs"), py::arg("adata"), py::arg("aidxs"),
        py::arg("perm"), py::arg("cdata"));
    flat_fermion_tensor.def(
        "tensordot",
        [](const py::object &aqs, const py::object &ashs,
           const py::array_t<complex<double>> &adata, const py::object &aidxs,
           const py::object &bqs, const py::object &bshs,
           const py::array_t<complex<double>> &bdata, const py::object &bidxs,
           const py::object &idxa, const py::object &idxb) {
            return flat_fermion_tensor_tensordot<Q, complex<double>>(
                aqs, ashs, adata, aidxs, bqs, bshs, bdata, bidxs, idxa, idxb);
        },
        py::arg("aqs"), py::arg("ashs"), py::arg("adata"), py::arg("aidxs"),
        py::arg("bqs"), py::arg("bshs"), py::arg("bdata"), py::arg("bidxs"),
        py::arg("idxa"), py::arg("idxb"));
    flat_fermion_tensor.def(
        "tensor_qr",
        [](const py::object &aqs, const py::object &ashs,
           const py::array_t<complex<double>> &adata, const py::object &aidxs,
           const py::object idx, const string &pattern, bool is_qr) {
            return flat_fermion_tensor_qr<Q, complex<double>>(
                aqs, ashs, adata, aidxs, idx.cast<int>(), pattern, is_qr);
        },
        py::arg("aqs"), py::arg("ashs"), py::arg("adata"), py::arg("aidxs"),
        py::arg("idx"), py::arg("pattern"), py::arg("is_qr"));

    // mixed C x D
    flat_fermion_tensor.def(
        "tensordot",
        [](const py::object &aqs, const py::object &ashs,
           const py::array_t<complex<double>> &adata, const py::object &aidxs,
           const py::object &bqs, const py::object &bshs,
           const py::array_t<double> &bdata, const py::object &bidxs,
           const py::object &idxa, const py::object &idxb) {
            return flat_fermion_tensor_tensordot<Q, complex<double>>(
                aqs, ashs, adata, aidxs, bqs, bshs, bdata, bidxs, idxa, idxb);
        },
        py::arg("aqs"), py::arg("ashs"), py::arg("adata"), py::arg("aidxs"),
        py::arg("bqs"), py::arg("bshs"), py::arg("bdata"), py::arg("bidxs"),
        py::arg("idxa"), py::arg("idxb"));

    // mixed D x C
    flat_fermion_tensor.def(
        "tensordot",
        [](const py::object &aqs, const py::object &ashs,
           const py::array_t<double> &adata, const py::object &aidxs,
           const py::object &bqs, const py::object &bshs,
           const py::array_t<complex<double>> &bdata, const py::object &bidxs,
           const py::object &idxa, const py::object &idxb) {
            return flat_fermion_tensor_tensordot<Q, complex<double>>(
                aqs, ashs, adata, aidxs, bqs, bshs, bdata, bidxs, idxa, idxb);
        },
        py::arg("aqs"), py::arg("ashs"), py::arg("adata"), py::arg("aidxs"),
        py::arg("bqs"), py::arg("bshs"), py::arg("bdata"), py::arg("bidxs"),
        py::arg("idxa"), py::arg("idxb"));
}

template <typename Q = void>
void bind_hamiltonian(py::module &m, const string &name) {

    py::module hamiltonian = m.def_submodule("hamiltonian", "Hamiltonian");
    hamiltonian.def("build_mpo", &build_mpo, py::arg("orb_sym"),
                    py::arg("h_values"), py::arg("h_terms"), py::arg("cutoff"),
                    py::arg("max_bond_dim"));
    hamiltonian.def("build_mpo_ptree", &build_mpo_ptree, py::arg("orb_sym"),
                    py::arg("h_values"), py::arg("h_terms"));
    hamiltonian.def("build_qc_mpo", &build_qc_mpo, py::arg("orb_sym"),
                    py::arg("t"), py::arg("v"));
}

PYBIND11_MODULE(block3, m) {

    m.doc() = "python extension part for pyblock3.";

    m.def("set_num_threads", [](int n) {
#ifdef _HAS_INTEL_MKL
        mkl_set_num_threads(n);
        mkl_set_dynamic(0);
#endif
        hptt_num_threads = n;
    });

    py::bind_vector<vector<uint32_t>>(m, "VectorUInt");
    py::bind_vector<vector<uint64_t>>(m, "VectorUInt64");
    py::bind_map<
        unordered_map<vector<uint32_t>, pair<uint32_t, vector<uint32_t>>>>(
        m, "MapVUIntPUV");
    py::bind_map<map_fusing>(m, "MapFusing")
        .def("__and__", [](map_fusing *self, map_fusing *other) {
            map_fusing r;
            for (auto &a : *self)
                if (other->count(a.first))
                    r[a.first] = a.second;
            return r;
        });

    py::bind_vector<vector<tuple<py::array_t<uint32_t>, py::array_t<uint32_t>,
                                 py::array_t<double>, py::array_t<uint64_t>>>>(
        m, "VectorFlat");
    py::bind_vector<
        vector<tuple<py::array_t<uint32_t>, py::array_t<uint32_t>,
                     py::array_t<complex<double>>, py::array_t<uint64_t>>>>(
        m, "VectorComplexFlat");

    py::module m_sz = m.def_submodule(
        "sz", "Non-spin-adapted symmetry class for quantum chemistry.");
    bind_sparse_tensor<SZ>(m_sz, m, "SZ");
    bind_hamiltonian<>(m_sz, "SZ");

    py::module m_u11 = m.def_submodule("u11", "U11 symmetry");
    bind_sparse_tensor<U11>(m_u11, m, "U11");

    py::module m_u1 = m.def_submodule("u1", "U1 symmetry");
    bind_sparse_tensor<U1>(m_u1, m, "U1");

    py::module m_z2 = m.def_submodule("z2", "Z2 symmetry");
    bind_sparse_tensor<Z2>(m_z2, m, "Z2");

    py::module m_z4 = m.def_submodule("z4", "Z4 symmetry");
    bind_sparse_tensor<Z4>(m_z4, m, "Z4");

    py::module m_z22 = m.def_submodule("z22", "Z22 symmetry");
    bind_sparse_tensor<Z22>(m_z22, m, "Z22");

    // bind extra symmetry here ...
    // py::module m_qpn = m.def_submodule("qpn", "General other symmetry.");
    // bind_sparse_tensor<QPN>(m_qpn, m, "QPN");

    py::module tensor = m.def_submodule("tensor", "Tensor");

    tensor.def("einsum", [](const string &script, py::args &args) {
        bool has_complex = false;
        for (int ia = 0; ia < args.size(); ia++)
            if (py::isinstance<py::array_t<complex<double>>>(args[ia]))
                has_complex = true;
        if (has_complex) {
            vector<py::array_t<complex<double>>> arrs(args.size());
            for (int ia = 0; ia < args.size(); ia++)
                arrs[ia] = args[ia].cast<py::array_t<complex<double>>>();
            return tensor_einsum<complex<double>>(script, arrs)
                .cast<py::object>();
        } else {
            vector<py::array_t<double>> arrs(args.size());
            for (int ia = 0; ia < args.size(); ia++)
                arrs[ia] = args[ia].cast<py::array_t<double>>();
            return tensor_einsum<double>(script, arrs).cast<py::object>();
        }
    });

    // double
    tensor.def("transpose", &tensor_transpose<double>, py::arg("x"),
               py::arg("perm"), py::arg("alpha") = 1.0, py::arg("beta") = 0.0);
    tensor.def("tensordot", &tensor_tensordot<double>, py::arg("a"),
               py::arg("b"), py::arg("idxa"), py::arg("idxb"),
               py::arg("alpha") = 1.0, py::arg("beta") = 0.0);
    // complex double
    tensor.def("transpose", &tensor_transpose<complex<double>>, py::arg("x"),
               py::arg("perm"), py::arg("alpha") = 1.0, py::arg("beta") = 0.0);
    tensor.def("tensordot", &tensor_tensordot<complex<double>>, py::arg("a"),
               py::arg("b"), py::arg("idxa"), py::arg("idxb"),
               py::arg("alpha") = 1.0, py::arg("beta") = 0.0);
    // mixed
    tensor.def(
        "tensordot",
        [](const py::array_t<double> &a, const py::array_t<complex<double>> &b,
           const py::object &idxa, const py::object &idxb,
           complex<double> alpha, complex<double> beta) {
            return tensor_tensordot<complex<double>>(a, b, idxa, idxb, alpha,
                                                     beta);
        },
        py::arg("a"), py::arg("b"), py::arg("idxa"), py::arg("idxb"),
        py::arg("alpha") = 1.0, py::arg("beta") = 0.0);
    tensor.def(
        "tensordot",
        [](const py::array_t<complex<double>> &a, const py::array_t<double> &b,
           const py::object &idxa, const py::object &idxb,
           complex<double> alpha, complex<double> beta) {
            return tensor_tensordot<complex<double>>(a, b, idxa, idxb, alpha,
                                                     beta);
        },
        py::arg("a"), py::arg("b"), py::arg("idxa"), py::arg("idxb"),
        py::arg("alpha") = 1.0, py::arg("beta") = 0.0);
}
