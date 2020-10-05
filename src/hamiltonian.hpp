
#pragma once

#include "bond_info.hpp"
#include "tensor_impl.hpp"
#ifdef I
#undef I
#endif
#include <pybind11/numpy.h>
#include <tuple>
#include <vector>

namespace py = pybind11;
using namespace std;

vector<tuple<py::array_t<uint32_t>, py::array_t<uint32_t>, py::array_t<double>,
             py::array_t<uint32_t>>>
build_mpo(py::array_t<int32_t> orb_sym, py::array_t<double> h_values,
          py::array_t<int32_t> h_terms, double cutoff = 1E-20, int max_bond_dim = -1);
