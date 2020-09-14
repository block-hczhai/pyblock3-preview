
#pragma once

#include <pybind11/numpy.h>
#include "tensor_impl.hpp"

namespace py = pybind11;
using namespace std;

py::array_t<double> tensor_transpose(const py::array_t<double> &x,
                                     const py::array_t<int> &perm,
                                     const double alpha, const double beta);

py::array_t<double> tensor_tensordot(const py::array_t<double> &a,
                                     const py::array_t<double> &b,
                                     const py::array_t<int> &idxa,
                                     const py::array_t<int> &idxb, double alpha,
                                     double beta);
