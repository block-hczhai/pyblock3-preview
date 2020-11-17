
#include "flat_fermion.hpp"
#include "flat_functor.hpp"
#include "flat_sparse.hpp"
#include "hamiltonian.hpp"
#include "qc_hamiltonian.hpp"
#include "sz.hpp"
#include "tensor.hpp"
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

PYBIND11_MAKE_OPAQUE(vector<uint32_t>);
PYBIND11_MAKE_OPAQUE(
    unordered_map<vector<uint32_t>, pair<uint32_t, vector<uint32_t>>>);
PYBIND11_MAKE_OPAQUE(map_fusing);
PYBIND11_MAKE_OPAQUE(vector<unordered_map<uint32_t, uint32_t>>);
PYBIND11_MAKE_OPAQUE(unordered_map<uint32_t, uint32_t>);
PYBIND11_MAKE_OPAQUE(
    vector<std::tuple<py::array_t<uint32_t>, py::array_t<uint32_t>,
                      py::array_t<double>, py::array_t<uint32_t>>>);

PYBIND11_MODULE(block3, m) {

    m.doc() = "python extension part for pyblock3.";

    m.def("set_num_threads", [](int n) {
#ifdef _HAS_INTEL_MKL
        mkl_set_num_threads(n);
        mkl_set_dynamic(0);
#endif
        hptt_num_threads = n;
    });

    py::bind_map<unordered_map<uint32_t, uint32_t>>(m, "MapUIntUInt")
        .def_property_readonly("n_bonds",
                               [](unordered_map<uint32_t, uint32_t> *self) {
                                   uint32_t r = 0;
                                   for (auto &kv : *self)
                                       r += kv.second;
                                   return r;
                               })
        .def("__neg__",
             [](unordered_map<uint32_t, uint32_t> *self) {
                 unordered_map<uint32_t, uint32_t> r;
                 for (auto &kv : *self)
                     r[from_sz(-to_sz(kv.first))] = kv.second;
                 return r;
             })
        .def("__add__",
             [](unordered_map<uint32_t, uint32_t> *self,
                unordered_map<uint32_t, uint32_t> *other) {
                 unordered_map<uint32_t, uint32_t> r;
                 r.insert(self->begin(), self->end());
                 for (auto &kv : *other)
                     r[kv.first] += kv.second;
                 return r;
             })
        .def("__mul__",
             [](unordered_map<uint32_t, uint32_t> *self,
                unordered_map<uint32_t, uint32_t> *other) {
                 unordered_map<uint32_t, uint32_t> r;
                 for (auto &a : *self)
                     for (auto &b : *other) {
                         uint32_t q = from_sz(to_sz(a.first) + to_sz(b.first));
                         r[q] = min(a.second * b.second + r[q], 65535U);
                     }
                 return r;
             })
        .def("__or__",
             [](unordered_map<uint32_t, uint32_t> *self,
                unordered_map<uint32_t, uint32_t> *other) {
                 unordered_map<uint32_t, uint32_t> r;
                 r.insert(self->begin(), self->end());
                 r.insert(other->begin(), other->end());
                 return r;
             })
        .def("__xor__",
             [](unordered_map<uint32_t, uint32_t> *self,
                unordered_map<uint32_t, uint32_t> *other) {
                 return move(bond_info_fusing_product(
                     vector<unordered_map<uint32_t, uint32_t>>{*self, *other},
                     "++"));
             })
        .def("filter",
             [](unordered_map<uint32_t, uint32_t> *self,
                unordered_map<uint32_t, uint32_t> *other) {
                 unordered_map<uint32_t, uint32_t> r;
                 for (auto &a : *self)
                     if (other->count(a.first))
                         r[a.first] = min(a.second, other->at(a.first));
                 return r;
             })
        .def("truncate",
             [](unordered_map<uint32_t, uint32_t> *self, int bond_dim,
                unordered_map<uint32_t, uint32_t> *ref = nullptr) {
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
             [](unordered_map<uint32_t, uint32_t> *self) {
                 vector<SZ> qs;
                 unordered_map<uint32_t, uint32_t> r;
                 for (auto &a : *self)
                     qs.push_back(to_sz(a.first));
                 SZ q = *max_element(qs.begin(), qs.end());
                 r[from_sz(q)] = (*self)[from_sz(q)];
                 return r;
             })
        .def_static(
            "set_bond_dimension_occ",
            [](const vector<unordered_map<uint32_t, uint32_t>> &basis,
               vector<unordered_map<uint32_t, uint32_t>> &left_dims,
               vector<unordered_map<uint32_t, uint32_t>> &right_dims,
               uint32_t vacuum, uint32_t target, int m,
               py::array_t<double> &occ, double bias) {
                vector<double> vocc(occ.data(), occ.data() + occ.size());
                return bond_info_set_bond_dimension_occ(basis, left_dims,
                                                        right_dims, vacuum,
                                                        target, m, vocc, bias);
            })
        .def_static("tensor_product", &tensor_product_ref, py::arg("a"),
                    py::arg("b"), py::arg("ref"));

    py::bind_vector<vector<unordered_map<uint32_t, uint32_t>>>(
        m, "VectorMapUIntUInt");
    py::bind_vector<vector<uint32_t>>(m, "VectorUInt");
    py::bind_map<
        unordered_map<vector<uint32_t>, pair<uint32_t, vector<uint32_t>>>>(
        m, "MapVUIntPUV");
    py::bind_map<map_fusing>(m, "MapFusing");

    py::bind_vector<vector<tuple<py::array_t<uint32_t>, py::array_t<uint32_t>,
                                 py::array_t<double>, py::array_t<uint32_t>>>>(
        m, "VectorFlat");

    py::module tensor = m.def_submodule("tensor", "Tensor");
    tensor.def("transpose", &tensor_transpose, py::arg("x"), py::arg("perm"),
               py::arg("alpha") = 1.0, py::arg("beta") = 0.0);
    tensor.def("tensordot", &tensor_tensordot, py::arg("a"), py::arg("b"),
               py::arg("idxa"), py::arg("idxb"), py::arg("alpha") = 1.0,
               py::arg("beta") = 0.0);

    py::module flat_sparse_tensor =
        m.def_submodule("flat_sparse_tensor", "FlatSparseTensor");
    flat_sparse_tensor.def("skeleton", &flat_sparse_tensor_skeleton,
                           py::arg("infos"), py::arg("pattern"), py::arg("dq"));
    flat_sparse_tensor.def("left_canonicalize", &flat_sparse_left_canonicalize,
                           py::arg("aqs"), py::arg("ashs"), py::arg("adata"),
                           py::arg("aidxs"));
    flat_sparse_tensor.def("right_canonicalize",
                           &flat_sparse_right_canonicalize, py::arg("aqs"),
                           py::arg("ashs"), py::arg("adata"), py::arg("aidxs"));
    flat_sparse_tensor.def("left_svd", &flat_sparse_left_svd, py::arg("aqs"),
                           py::arg("ashs"), py::arg("adata"), py::arg("aidxs"));
    flat_sparse_tensor.def("right_svd", &flat_sparse_right_svd, py::arg("aqs"),
                           py::arg("ashs"), py::arg("adata"), py::arg("aidxs"));
    flat_sparse_tensor.def(
        "left_canonicalize_indexed", &flat_sparse_left_canonicalize_indexed,
        py::arg("aqs"), py::arg("ashs"), py::arg("adata"), py::arg("aidxs"));
    flat_sparse_tensor.def(
        "right_canonicalize_indexed", &flat_sparse_right_canonicalize_indexed,
        py::arg("aqs"), py::arg("ashs"), py::arg("adata"), py::arg("aidxs"));
    flat_sparse_tensor.def("left_svd_indexed", &flat_sparse_left_svd_indexed,
                           py::arg("aqs"), py::arg("ashs"), py::arg("adata"),
                           py::arg("aidxs"));
    flat_sparse_tensor.def("right_svd_indexed", &flat_sparse_right_svd_indexed,
                           py::arg("aqs"), py::arg("ashs"), py::arg("adata"),
                           py::arg("aidxs"));
    flat_sparse_tensor.def("tensor_svd", &flat_sparse_tensor_svd,
                           py::arg("aqs"), py::arg("ashs"), py::arg("adata"),
                           py::arg("aidxs"), py::arg("idx"), py::arg("linfo"),
                           py::arg("rinfo"), py::arg("pattern"));
    flat_sparse_tensor.def(
        "truncate_svd", &flat_sparse_truncate_svd, py::arg("lqs"),
        py::arg("lshs"), py::arg("ldata"), py::arg("lidxs"), py::arg("sqs"),
        py::arg("sshs"), py::arg("sdata"), py::arg("sidxs"), py::arg("rqs"),
        py::arg("rshs"), py::arg("rdata"), py::arg("ridxs"),
        py::arg("max_bond_dim") = -1, py::arg("cutoff") = 0.0,
        py::arg("max_dw") = 0.0, py::arg("norm_cutoff") = 0.0,
        py::arg("eigen_values") = false);
    flat_sparse_tensor.def("get_infos", &flat_sparse_tensor_get_infos,
                           py::arg("aqs"), py::arg("ashs"));
    flat_sparse_tensor.def("kron_sum_info", &flat_sparse_tensor_kron_sum_info,
                           py::arg("aqs"), py::arg("ashs"), py::arg("pattern"));
    flat_sparse_tensor.def("kron_product_info", &bond_info_fusing_product,
                           py::arg("infos"), py::arg("pattern"));
    flat_sparse_tensor.def("transpose", &flat_sparse_tensor_transpose,
                           py::arg("ashs"), py::arg("adata"), py::arg("aidxs"),
                           py::arg("perm"), py::arg("cdata"));
    flat_sparse_tensor.def("tensordot", &flat_sparse_tensor_tensordot,
                           py::arg("aqs"), py::arg("ashs"), py::arg("adata"),
                           py::arg("aidxs"), py::arg("bqs"), py::arg("bshs"),
                           py::arg("bdata"), py::arg("bidxs"), py::arg("idxa"),
                           py::arg("idxb"));
    flat_sparse_tensor.def("tensordot_skeleton",
                           &flat_sparse_tensor_tensordot_skeleton,
                           py::arg("aqs"), py::arg("ashs"), py::arg("bqs"),
                           py::arg("bshs"), py::arg("idxa"), py::arg("idxb"));
    flat_sparse_tensor.def("add", &flat_sparse_tensor_add, py::arg("aqs"),
                           py::arg("ashs"), py::arg("adata"), py::arg("aidxs"),
                           py::arg("bqs"), py::arg("bshs"), py::arg("bdata"),
                           py::arg("bidxs"));
    flat_sparse_tensor.def("kron_add", &flat_sparse_tensor_kron_add,
                           py::arg("aqs"), py::arg("ashs"), py::arg("adata"),
                           py::arg("aidxs"), py::arg("bqs"), py::arg("bshs"),
                           py::arg("bdata"), py::arg("bidxs"), py::arg("infol"),
                           py::arg("infor"));
    flat_sparse_tensor.def("fuse", &flat_sparse_tensor_fuse, py::arg("aqs"),
                           py::arg("ashs"), py::arg("adata"), py::arg("aidxs"),
                           py::arg("idxs"), py::arg("info"),
                           py::arg("pattern"));
    flat_sparse_tensor.def("diag", &flat_sparse_tensor_diag, py::arg("aqs"),
                           py::arg("ashs"), py::arg("adata"), py::arg("aidxs"),
                           py::arg("idxa"), py::arg("idxb"));
    flat_sparse_tensor.def("matmul", &flat_sparse_tensor_matmul,
                           py::arg("plan"), py::arg("adata"), py::arg("bdata"),
                           py::arg("cdata"));
    flat_sparse_tensor.def("matmul_init", &flat_sparse_tensor_matmul_init,
                           py::arg("loqs"), py::arg("loshs"), py::arg("leqs"),
                           py::arg("leshs"), py::arg("roqs"), py::arg("roshs"),
                           py::arg("reqs"), py::arg("reshs"));
    flat_sparse_tensor.def("matmul_plan", &flat_sparse_tensor_matmul_plan,
                           py::arg("aqs"), py::arg("ashs"), py::arg("aidxs"),
                           py::arg("bqs"), py::arg("bshs"), py::arg("bidxs"),
                           py::arg("idxa"), py::arg("idxb"), py::arg("cqs"),
                           py::arg("cidxs"), py::arg("ferm_op"));

    py::module flat_fermion_tensor = m.def_submodule("flat_fermion_tensor");
    flat_fermion_tensor.def("transpose", &flat_fermion_tensor_transpose,
                            py::arg("aqs"), py::arg("ashs"), py::arg("adata"),
                            py::arg("aidxs"), py::arg("perm"),
                            py::arg("cdata"));
    flat_fermion_tensor.def("tensordot", &flat_fermion_tensor_tensordot,
                            py::arg("aqs"), py::arg("ashs"), py::arg("adata"),
                            py::arg("aidxs"), py::arg("bqs"), py::arg("bshs"),
                            py::arg("bdata"), py::arg("bidxs"), py::arg("idxa"),
                            py::arg("idxb"));

    py::module hamiltonian = m.def_submodule("hamiltonian", "Hamiltonian");
    hamiltonian.def("build_mpo", &build_mpo, py::arg("orb_sym"),
                    py::arg("h_values"), py::arg("h_terms"), py::arg("cutoff"),
                    py::arg("max_bond_dim"));
    hamiltonian.def("build_qc_mpo", &build_qc_mpo, py::arg("orb_sym"),
                    py::arg("t"), py::arg("v"));

    py::class_<SZ>(m, "SZ")
        .def(py::init<>())
        .def(py::init<int, int, int>())
        .def_property("n", &SZ::n, &SZ::set_n)
        .def_property("twos", &SZ::twos, &SZ::set_twos)
        .def_property("pg", &SZ::pg, &SZ::set_pg)
        .def_property_readonly("multiplicity", &SZ::multiplicity)
        .def_property_readonly("is_fermion", &SZ::is_fermion)
        .def_property_readonly("count", &SZ::count)
        .def("__getitem__", &SZ::operator[])
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::self < py::self)
        .def(-py::self)
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def("__hash__", &SZ::hash)
        .def("__repr__", &SZ::to_str);
}
