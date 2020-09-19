
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

void flat_sparse_tensor_transpose(const py::array_t<uint32_t> &ashs,
                                  const py::array_t<double> &adata,
                                  const py::array_t<uint32_t> &aidxs,
                                  const py::array_t<int32_t> &perm,
                                  py::array_t<double> &cdata);

tuple<py::array_t<uint32_t>, py::array_t<uint32_t>, py::array_t<double>,
      py::array_t<uint32_t>>
flat_sparse_tensor_tensordot(
    const py::array_t<uint32_t> &aqs, const py::array_t<uint32_t> &ashs,
    const py::array_t<double> &adata, const py::array_t<uint32_t> &aidxs,
    const py::array_t<uint32_t> &bqs, const py::array_t<uint32_t> &bshs,
    const py::array_t<double> &bdata, const py::array_t<uint32_t> &bidxs,
    const py::array_t<int> &idxa, const py::array_t<int> &idxb);

tuple<py::array_t<uint32_t>, py::array_t<uint32_t>, py::array_t<double>,
      py::array_t<uint32_t>>
flat_sparse_tensor_add(
    const py::array_t<uint32_t> &aqs, const py::array_t<uint32_t> &ashs,
    const py::array_t<double> &adata, const py::array_t<uint32_t> &aidxs,
    const py::array_t<uint32_t> &bqs, const py::array_t<uint32_t> &bshs,
    const py::array_t<double> &bdata, const py::array_t<uint32_t> &bidxs);

tuple<py::array_t<uint32_t>, py::array_t<uint32_t>, py::array_t<double>,
      py::array_t<uint32_t>>
flat_sparse_tensor_kron_add(
    const py::array_t<uint32_t> &aqs, const py::array_t<uint32_t> &ashs,
    const py::array_t<double> &adata, const py::array_t<uint32_t> &aidxs,
    const py::array_t<uint32_t> &bqs, const py::array_t<uint32_t> &bshs,
    const py::array_t<double> &bdata, const py::array_t<uint32_t> &bidxs,
    const unordered_map<uint32_t, uint32_t> &infol,
    const unordered_map<uint32_t, uint32_t> &infor);

tuple<py::array_t<uint32_t>, py::array_t<uint32_t>, py::array_t<double>,
      py::array_t<uint32_t>>
flat_sparse_tensor_fuse(
    const py::array_t<uint32_t> &aqs, const py::array_t<uint32_t> &ashs,
    const py::array_t<double> &adata, const py::array_t<uint32_t> &aidxs,
    const py::array_t<int32_t> &idxs,
    const unordered_map<
        uint32_t,
        pair<uint32_t,
             unordered_map<vector<uint32_t>, pair<uint32_t, vector<uint32_t>>>>>
        &info,
    const string &pattern);

unordered_map<uint32_t,
              pair<uint32_t, unordered_map<vector<uint32_t>,
                                           pair<uint32_t, vector<uint32_t>>>>>
flat_sparse_tensor_kron_sum_info(const py::array_t<uint32_t> &aqs,
                                 const py::array_t<uint32_t> &ashs,
                                 const string &pattern);

tuple<py::array_t<uint32_t>, py::array_t<uint32_t>, py::array_t<uint32_t>>
flat_sparse_tensor_skeleton(
    const vector<unordered_map<uint32_t, uint32_t>> &infos,
    const string &pattern, uint32_t fdq);

vector<unordered_map<uint32_t, uint32_t>>
flat_sparse_tensor_get_infos(const py::array_t<uint32_t> &aqs,
                             const py::array_t<uint32_t> &ashs);

enum DIRECTION { LEFT = 1, RIGHT = 0 };

template <DIRECTION L>
tuple<py::array_t<uint32_t>, py::array_t<uint32_t>, py::array_t<double>,
      py::array_t<uint32_t>, py::array_t<uint32_t>, py::array_t<uint32_t>,
      py::array_t<double>, py::array_t<uint32_t>>
flat_sparse_canonicalize(const py::array_t<uint32_t> &aqs,
                         const py::array_t<uint32_t> &ashs,
                         const py::array_t<double> &adata,
                         const py::array_t<uint32_t> &aidxs);

template <DIRECTION L>
tuple<py::array_t<uint32_t>, py::array_t<uint32_t>, py::array_t<double>,
      py::array_t<uint32_t>, py::array_t<uint32_t>, py::array_t<uint32_t>,
      py::array_t<double>, py::array_t<uint32_t>, py::array_t<uint32_t>,
      py::array_t<uint32_t>, py::array_t<double>, py::array_t<uint32_t>>
flat_sparse_svd(const py::array_t<uint32_t> &aqs,
                const py::array_t<uint32_t> &ashs,
                const py::array_t<double> &adata,
                const py::array_t<uint32_t> &aidxs);

extern template tuple<py::array_t<uint32_t>, py::array_t<uint32_t>,
                      py::array_t<double>, py::array_t<uint32_t>,
                      py::array_t<uint32_t>, py::array_t<uint32_t>,
                      py::array_t<double>, py::array_t<uint32_t>>
flat_sparse_canonicalize<LEFT>(const py::array_t<uint32_t> &aqs,
                               const py::array_t<uint32_t> &ashs,
                               const py::array_t<double> &adata,
                               const py::array_t<uint32_t> &aidxs);

extern template tuple<py::array_t<uint32_t>, py::array_t<uint32_t>,
                      py::array_t<double>, py::array_t<uint32_t>,
                      py::array_t<uint32_t>, py::array_t<uint32_t>,
                      py::array_t<double>, py::array_t<uint32_t>>
flat_sparse_canonicalize<RIGHT>(const py::array_t<uint32_t> &aqs,
                                const py::array_t<uint32_t> &ashs,
                                const py::array_t<double> &adata,
                                const py::array_t<uint32_t> &aidxs);
extern template tuple<
    py::array_t<uint32_t>, py::array_t<uint32_t>, py::array_t<double>,
    py::array_t<uint32_t>, py::array_t<uint32_t>, py::array_t<uint32_t>,
    py::array_t<double>, py::array_t<uint32_t>, py::array_t<uint32_t>,
    py::array_t<uint32_t>, py::array_t<double>, py::array_t<uint32_t>>
flat_sparse_svd<LEFT>(const py::array_t<uint32_t> &aqs,
                      const py::array_t<uint32_t> &ashs,
                      const py::array_t<double> &adata,
                      const py::array_t<uint32_t> &aidxs);

extern template tuple<
    py::array_t<uint32_t>, py::array_t<uint32_t>, py::array_t<double>,
    py::array_t<uint32_t>, py::array_t<uint32_t>, py::array_t<uint32_t>,
    py::array_t<double>, py::array_t<uint32_t>, py::array_t<uint32_t>,
    py::array_t<uint32_t>, py::array_t<double>, py::array_t<uint32_t>>
flat_sparse_svd<RIGHT>(const py::array_t<uint32_t> &aqs,
                       const py::array_t<uint32_t> &ashs,
                       const py::array_t<double> &adata,
                       const py::array_t<uint32_t> &aidxs);
