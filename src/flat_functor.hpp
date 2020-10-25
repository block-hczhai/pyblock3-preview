
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

tuple<py::array_t<uint32_t>, py::array_t<uint32_t>, py::array_t<double>,
      py::array_t<uint32_t>>
flat_sparse_tensor_diag(const py::array_t<uint32_t> &aqs,
                        const py::array_t<uint32_t> &ashs,
                        const py::array_t<double> &adata,
                        const py::array_t<uint32_t> &aidxs,
                        const py::array_t<int> &idxa,
                        const py::array_t<int> &idxb);

tuple<py::array_t<uint32_t>, py::array_t<uint32_t>, py::array_t<uint32_t>>
flat_sparse_tensor_tensordot_skeleton(const py::array_t<uint32_t> &aqs,
                                      const py::array_t<uint32_t> &ashs,
                                      const py::array_t<uint32_t> &bqs,
                                      const py::array_t<uint32_t> &bshs,
                                      const py::array_t<int> &idxa,
                                      const py::array_t<int> &idxb);

void flat_sparse_tensor_matmul(const py::array_t<int32_t> &plan,
                               const py::array_t<double> &adata,
                               const py::array_t<double> &bdata,
                               py::array_t<double> &cdata);

tuple<int, int, vector<unordered_map<uint32_t, uint32_t>>>
flat_sparse_tensor_matmul_init(
    const py::array_t<uint32_t> &loqs, const py::array_t<uint32_t> &loshs,
    const py::array_t<uint32_t> &leqs, const py::array_t<uint32_t> &leshs,
    const py::array_t<uint32_t> &roqs, const py::array_t<uint32_t> &roshs,
    const py::array_t<uint32_t> &reqs, const py::array_t<uint32_t> &reshs);

py::array_t<int32_t> flat_sparse_tensor_matmul_plan(
    const py::array_t<uint32_t> &aqs, const py::array_t<uint32_t> &ashs,
    const py::array_t<uint32_t> &aidxs, const py::array_t<uint32_t> &bqs,
    const py::array_t<uint32_t> &bshs, const py::array_t<uint32_t> &bidxs,
    const py::array_t<int> &idxa, const py::array_t<int> &idxb,
    const py::array_t<uint32_t> &cqs, const py::array_t<uint32_t> &cidxs,
    bool ferm_op);
