
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

void flat_fermion_tensor_transpose(const py::array_t<uint32_t> &aqs,
                                  const py::array_t<uint32_t> &ashs,
                                  const py::array_t<double> &adata,
                                  const py::array_t<uint32_t> &aidxs,
                                  const py::array_t<int32_t> &perm,
                                  py::array_t<double> &cdata);

tuple<py::array_t<uint32_t>, py::array_t<uint32_t>, py::array_t<double>,
      py::array_t<uint32_t>>
flat_fermion_tensor_tensordot(
    const py::array_t<uint32_t> &aqs, const py::array_t<uint32_t> &ashs,
    const py::array_t<double> &adata, const py::array_t<uint32_t> &aidxs,
    const py::array_t<uint32_t> &bqs, const py::array_t<uint32_t> &bshs,
    const py::array_t<double> &bdata, const py::array_t<uint32_t> &bidxs,
    const py::array_t<int> &idxa, const py::array_t<int> &idxb);
