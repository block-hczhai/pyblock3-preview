
#include "flat_functor.hpp"
#include "flat_sparse.hpp"
#include "sz.hpp"
#include "tensor.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

PYBIND11_MAKE_OPAQUE(vector<uint32_t>);
PYBIND11_MAKE_OPAQUE(
    unordered_map<vector<uint32_t>, pair<uint32_t, vector<uint32_t>>>);
PYBIND11_MAKE_OPAQUE(
    unordered_map<
        uint32_t,
        pair<uint32_t, unordered_map<vector<uint32_t>,
                                     pair<uint32_t, vector<uint32_t>>>>>);
PYBIND11_MAKE_OPAQUE(vector<unordered_map<uint32_t, uint32_t>>);
PYBIND11_MAKE_OPAQUE(unordered_map<uint32_t, uint32_t>);

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
        .def("__add__",
             [](unordered_map<uint32_t, uint32_t> *self,
                unordered_map<uint32_t, uint32_t> *other) {
                 unordered_map<uint32_t, uint32_t> r;
                 r.insert(self->begin(), self->end());
                 for (auto &kv : *other)
                     r[kv.first] += kv.second;
                 return move(r);
             })
        .def("__or__",
             [](unordered_map<uint32_t, uint32_t> *self,
                unordered_map<uint32_t, uint32_t> *other) {
                 unordered_map<uint32_t, uint32_t> r;
                 r.insert(self->begin(), self->end());
                 r.insert(other->begin(), other->end());
                 return move(r);
             })
        .def("__xor__", [](unordered_map<uint32_t, uint32_t> *self,
                           unordered_map<uint32_t, uint32_t> *other) {
            return move(bond_info_fusing_product(
                vector<unordered_map<uint32_t, uint32_t>>{*self, *other},
                "++"));
        });
    py::bind_vector<vector<unordered_map<uint32_t, uint32_t>>>(
        m, "VectorMapUIntUInt");
    py::bind_vector<vector<uint32_t>>(m, "VectorUInt");
    py::bind_map<
        unordered_map<vector<uint32_t>, pair<uint32_t, vector<uint32_t>>>>(
        m, "MapVUIntPUV");
    py::bind_map<unordered_map<
        uint32_t,
        pair<uint32_t, unordered_map<vector<uint32_t>,
                                     pair<uint32_t, vector<uint32_t>>>>>>(
        m, "MapFusing");

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
    flat_sparse_tensor.def("get_infos", &flat_sparse_tensor_get_infos,
                           py::arg("aqs"), py::arg("ashs"));
    flat_sparse_tensor.def("tensordot", &flat_sparse_tensor_tensordot,
                           py::arg("aqs"), py::arg("ashs"), py::arg("adata"),
                           py::arg("aidxs"), py::arg("bqs"), py::arg("bshs"),
                           py::arg("bdata"), py::arg("bidxs"), py::arg("idxa"),
                           py::arg("idxb"));
    flat_sparse_tensor.def("add", &flat_sparse_tensor_add, py::arg("aqs"),
                           py::arg("ashs"), py::arg("adata"), py::arg("aidxs"),
                           py::arg("bqs"), py::arg("bshs"), py::arg("bdata"),
                           py::arg("bidxs"));
    flat_sparse_tensor.def("diag", &flat_sparse_tensor_diag, py::arg("aqs"),
                           py::arg("ashs"), py::arg("adata"), py::arg("aidxs"),
                           py::arg("idxa"), py::arg("idxb"));
    flat_sparse_tensor.def("transpose", &flat_sparse_tensor_transpose,
                           py::arg("ashs"), py::arg("adata"), py::arg("aidxs"),
                           py::arg("perm"), py::arg("cdata"));
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
