
#include "flat_functor.hpp"
#include "flat_sparse.hpp"
#include "fermion_sparse.hpp"
#include "hamiltonian.hpp"
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
PYBIND11_MAKE_OPAQUE(vector<std::tuple<py::array_t<uint32_t>, py::array_t<uint32_t>,
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
                     for (auto &b : *other)
                         r[from_sz(to_sz(a.first) + to_sz(b.first))] +=
                             a.second * b.second;
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
                 uint32_t n_total = 0;
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
             });
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
        py::arg("max_dw") = 0.0, py::arg("norm_cutoff") = 0.0);
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

    py::module fermion_sparse_tensor =
        m.def_submodule("fermion_sparse_tensor");
    fermion_sparse_tensor.def("transpose", &fermion_sparse_tensor_transpose, py::arg("aqs"),
                              py::arg("ashs"), py::arg("adata"), py::arg("aidxs"),
                              py::arg("perm"), py::arg("cdata"));
    fermion_sparse_tensor.def("tensordot", &fermion_sparse_tensor_tensordot,
                              py::arg("aqs"), py::arg("ashs"), py::arg("adata"),
                              py::arg("aidxs"), py::arg("bqs"), py::arg("bshs"),
                              py::arg("bdata"), py::arg("bidxs"), py::arg("idxa"),
                              py::arg("idxb"));

    py::module hamiltonian = m.def_submodule("hamiltonian", "Hamiltonian");
    hamiltonian.def("build_mpo", &build_mpo, py::arg("orb_sym"),
                    py::arg("h_values"), py::arg("h_terms"), py::arg("cutoff"),
                    py::arg("max_bond_dim"));

    py::class_<SZLong>(m, "SZ")
        .def(py::init<>())
        .def(py::init<uint32_t>())
        .def(py::init<int, int, int>())
        .def_readwrite("data", &SZLong::data)
        .def_property("n", &SZLong::n, &SZLong::set_n)
        .def_property("twos", &SZLong::twos, &SZLong::set_twos)
        .def_property("pg", &SZLong::pg, &SZLong::set_pg)
        .def_property_readonly("multiplicity", &SZLong::multiplicity)
        .def_property_readonly("is_fermion", &SZLong::is_fermion)
        .def_property_readonly("count", &SZLong::count)
        .def("combine", &SZLong::combine)
        .def("__getitem__", &SZLong::operator[])
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::self < py::self)
        .def(-py::self)
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def("get_ket", &SZLong::get_ket)
        .def("get_bra", &SZLong::get_bra, py::arg("dq"))
        .def("__hash__", &SZLong::hash)
        .def("__repr__", &SZLong::to_str);
}
