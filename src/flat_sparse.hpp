
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

template <typename Q, typename FL>
void flat_sparse_tensor_transpose(const py::array_t<uint32_t> &ashs,
                                  const py::array_t<FL> &adata,
                                  const py::array_t<uint64_t> &aidxs,
                                  const py::array_t<int32_t> &perm,
                                  py::array_t<FL> &cdata);

template <typename Q, typename FL>
tuple<py::array_t<uint32_t>, py::array_t<uint32_t>, py::array_t<FL>,
      py::array_t<uint64_t>>
flat_sparse_tensor_tensordot(
    const py::array_t<uint32_t> &aqs, const py::array_t<uint32_t> &ashs,
    const py::array_t<FL> &adata, const py::array_t<uint64_t> &aidxs,
    const py::array_t<uint32_t> &bqs, const py::array_t<uint32_t> &bshs,
    const py::array_t<FL> &bdata, const py::array_t<uint64_t> &bidxs,
    const py::array_t<int> &idxa, const py::array_t<int> &idxb);

template <typename Q, typename FL>
tuple<py::array_t<uint32_t>, py::array_t<uint32_t>, py::array_t<FL>,
      py::array_t<uint64_t>>
flat_sparse_tensor_add(
    const py::array_t<uint32_t> &aqs, const py::array_t<uint32_t> &ashs,
    const py::array_t<FL> &adata, const py::array_t<uint64_t> &aidxs,
    const py::array_t<uint32_t> &bqs, const py::array_t<uint32_t> &bshs,
    const py::array_t<FL> &bdata, const py::array_t<uint64_t> &bidxs);

template <typename Q, typename FL>
tuple<py::array_t<uint32_t>, py::array_t<uint32_t>, py::array_t<FL>,
      py::array_t<uint64_t>>
flat_sparse_tensor_kron_add(
    const py::array_t<uint32_t> &aqs, const py::array_t<uint32_t> &ashs,
    const py::array_t<FL> &adata, const py::array_t<uint64_t> &aidxs,
    const py::array_t<uint32_t> &bqs, const py::array_t<uint32_t> &bshs,
    const py::array_t<FL> &bdata, const py::array_t<uint64_t> &bidxs,
    const map_uint_uint<Q> &infol, const map_uint_uint<Q> &infor);

template <typename Q, typename FL>
tuple<py::array_t<uint32_t>, py::array_t<uint32_t>, py::array_t<FL>,
      py::array_t<uint64_t>>
flat_sparse_tensor_fuse(const py::array_t<uint32_t> &aqs,
                        const py::array_t<uint32_t> &ashs,
                        const py::array_t<FL> &adata,
                        const py::array_t<uint64_t> &aidxs,
                        const py::array_t<int32_t> &idxs,
                        const map_fusing &info, const string &pattern);

template <typename Q>
map_fusing flat_sparse_tensor_kron_sum_info(const py::array_t<uint32_t> &aqs,
                                            const py::array_t<uint32_t> &ashs,
                                            const string &pattern);

template <typename Q>
tuple<py::array_t<uint32_t>, uint32_t>
flat_sparse_tensor_fix_pattern(py::array_t<uint32_t> aqs, const string &pattern,
                               uint32_t fdq);

template <typename Q>
tuple<py::array_t<uint32_t>, py::array_t<uint32_t>, py::array_t<uint64_t>>
flat_sparse_tensor_skeleton(const vector<map_uint_uint<Q>> &infos,
                            const string &pattern, uint32_t fdq);

template <typename Q>
vector<map_uint_uint<Q>>
flat_sparse_tensor_get_infos(const py::array_t<uint32_t> &aqs,
                             const py::array_t<uint32_t> &ashs);

enum DIRECTION { LEFT = 1, RIGHT = 0 };

template <typename Q, typename FL, DIRECTION L>
tuple<py::array_t<uint32_t>, py::array_t<uint32_t>, py::array_t<FL>,
      py::array_t<uint64_t>, py::array_t<uint32_t>, py::array_t<uint32_t>,
      py::array_t<FL>, py::array_t<uint64_t>>
flat_sparse_canonicalize(const py::array_t<uint32_t> &aqs,
                         const py::array_t<uint32_t> &ashs,
                         const py::array_t<FL> &adata,
                         const py::array_t<uint64_t> &aidxs, uint32_t *pxidx);

template <typename Q, typename FL, DIRECTION L>
tuple<py::array_t<uint32_t>, py::array_t<uint32_t>, py::array_t<FL>,
      py::array_t<uint64_t>, py::array_t<uint32_t>, py::array_t<uint32_t>,
      py::array_t<double>, py::array_t<uint64_t>, py::array_t<uint32_t>,
      py::array_t<uint32_t>, py::array_t<FL>, py::array_t<uint64_t>>
flat_sparse_svd(const py::array_t<uint32_t> &aqs,
                const py::array_t<uint32_t> &ashs, const py::array_t<FL> &adata,
                const py::array_t<uint64_t> &aidxs, uint32_t *pxidx);

template <typename Q, typename FL>
tuple<py::array_t<uint32_t>, py::array_t<uint32_t>, py::array_t<FL>,
      py::array_t<uint64_t>, py::array_t<uint32_t>, py::array_t<uint32_t>,
      py::array_t<FL>, py::array_t<uint64_t>>
flat_sparse_left_canonicalize(const py::array_t<uint32_t> &aqs,
                              const py::array_t<uint32_t> &ashs,
                              const py::array_t<FL> &adata,
                              const py::array_t<uint64_t> &aidxs);

template <typename Q, typename FL>
tuple<py::array_t<uint32_t>, py::array_t<uint32_t>, py::array_t<FL>,
      py::array_t<uint64_t>, py::array_t<uint32_t>, py::array_t<uint32_t>,
      py::array_t<FL>, py::array_t<uint64_t>>
flat_sparse_right_canonicalize(const py::array_t<uint32_t> &aqs,
                               const py::array_t<uint32_t> &ashs,
                               const py::array_t<FL> &adata,
                               const py::array_t<uint64_t> &aidxs);

template <typename Q, typename FL>
tuple<py::array_t<uint32_t>, py::array_t<uint32_t>, py::array_t<FL>,
      py::array_t<uint64_t>, py::array_t<uint32_t>, py::array_t<uint32_t>,
      py::array_t<double>, py::array_t<uint64_t>, py::array_t<uint32_t>,
      py::array_t<uint32_t>, py::array_t<FL>, py::array_t<uint64_t>>
flat_sparse_left_svd(const py::array_t<uint32_t> &aqs,
                     const py::array_t<uint32_t> &ashs,
                     const py::array_t<FL> &adata,
                     const py::array_t<uint64_t> &aidxs);

template <typename Q, typename FL>
tuple<py::array_t<uint32_t>, py::array_t<uint32_t>, py::array_t<FL>,
      py::array_t<uint64_t>, py::array_t<uint32_t>, py::array_t<uint32_t>,
      py::array_t<double>, py::array_t<uint64_t>, py::array_t<uint32_t>,
      py::array_t<uint32_t>, py::array_t<FL>, py::array_t<uint64_t>>
flat_sparse_right_svd(const py::array_t<uint32_t> &aqs,
                      const py::array_t<uint32_t> &ashs,
                      const py::array_t<FL> &adata,
                      const py::array_t<uint64_t> &aidxs);

template <typename Q, typename FL>
pair<tuple<py::array_t<uint32_t>, py::array_t<uint32_t>, py::array_t<FL>,
           py::array_t<uint64_t>, py::array_t<uint32_t>, py::array_t<uint32_t>,
           py::array_t<FL>, py::array_t<uint64_t>>,
     py::array_t<uint32_t>>
flat_sparse_left_canonicalize_indexed(const py::array_t<uint32_t> &aqs,
                                      const py::array_t<uint32_t> &ashs,
                                      const py::array_t<FL> &adata,
                                      const py::array_t<uint64_t> &aidxs);

template <typename Q, typename FL>
pair<tuple<py::array_t<uint32_t>, py::array_t<uint32_t>, py::array_t<FL>,
           py::array_t<uint64_t>, py::array_t<uint32_t>, py::array_t<uint32_t>,
           py::array_t<FL>, py::array_t<uint64_t>>,
     py::array_t<uint32_t>>
flat_sparse_right_canonicalize_indexed(const py::array_t<uint32_t> &aqs,
                                       const py::array_t<uint32_t> &ashs,
                                       const py::array_t<FL> &adata,
                                       const py::array_t<uint64_t> &aidxs);

template <typename Q, typename FL>
pair<tuple<py::array_t<uint32_t>, py::array_t<uint32_t>, py::array_t<FL>,
           py::array_t<uint64_t>, py::array_t<uint32_t>, py::array_t<uint32_t>,
           py::array_t<double>, py::array_t<uint64_t>, py::array_t<uint32_t>,
           py::array_t<uint32_t>, py::array_t<FL>, py::array_t<uint64_t>>,
     py::array_t<uint32_t>>
flat_sparse_left_svd_indexed(const py::array_t<uint32_t> &aqs,
                             const py::array_t<uint32_t> &ashs,
                             const py::array_t<FL> &adata,
                             const py::array_t<uint64_t> &aidxs);

template <typename Q, typename FL>
pair<tuple<py::array_t<uint32_t>, py::array_t<uint32_t>, py::array_t<FL>,
           py::array_t<uint64_t>, py::array_t<uint32_t>, py::array_t<uint32_t>,
           py::array_t<double>, py::array_t<uint64_t>, py::array_t<uint32_t>,
           py::array_t<uint32_t>, py::array_t<FL>, py::array_t<uint64_t>>,
     py::array_t<uint32_t>>
flat_sparse_right_svd_indexed(const py::array_t<uint32_t> &aqs,
                              const py::array_t<uint32_t> &ashs,
                              const py::array_t<FL> &adata,
                              const py::array_t<uint64_t> &aidxs);

template <typename Q, typename FL>
tuple<py::array_t<uint32_t>, py::array_t<uint32_t>, py::array_t<FL>,
      py::array_t<uint64_t>, py::array_t<uint32_t>, py::array_t<uint32_t>,
      py::array_t<double>, py::array_t<uint64_t>, py::array_t<uint32_t>,
      py::array_t<uint32_t>, py::array_t<FL>, py::array_t<uint64_t>>
flat_sparse_tensor_svd(const py::array_t<uint32_t> &aqs,
                       const py::array_t<uint32_t> &ashs,
                       const py::array_t<FL> &adata,
                       const py::array_t<uint64_t> &aidxs, int idx,
                       const map_fusing &linfo, const map_fusing &rinfo,
                       const string &pattern);

template <typename Q, typename FL>
tuple<py::array_t<uint32_t>, py::array_t<uint32_t>, py::array_t<FL>,
      py::array_t<uint64_t>, py::array_t<uint32_t>, py::array_t<uint32_t>,
      py::array_t<double>, py::array_t<uint64_t>, py::array_t<uint32_t>,
      py::array_t<uint32_t>, py::array_t<FL>, py::array_t<uint64_t>, double>
flat_sparse_truncate_svd(
    const py::array_t<uint32_t> &lqs, const py::array_t<uint32_t> &lshs,
    const py::array_t<FL> &ldata, const py::array_t<uint64_t> &lidxs,
    const py::array_t<uint32_t> &sqs, const py::array_t<uint32_t> &sshs,
    const py::array_t<double> &sdata, const py::array_t<uint64_t> &sidxs,
    const py::array_t<uint32_t> &rqs, const py::array_t<uint32_t> &rshs,
    const py::array_t<FL> &rdata, const py::array_t<uint64_t> &ridxs,
    int max_bond_dim, double cutoff, double max_dw, double norm_cutoff,
    bool eigen_values);

#define TMPL_EXTERN extern
#define TMPL_NAME flat_sparse

#include "symmetry_tmpl.hpp"
#define TMPL_FL double
#include "symmetry_tmpl.hpp"
#undef TMPL_FL
#define TMPL_FL complex<double>
#include "symmetry_tmpl.hpp"
#undef TMPL_FL

#undef TMPL_NAME
#undef TMPL_EXTERN
