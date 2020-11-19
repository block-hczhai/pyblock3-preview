
/*
 * pyblock3: An Efficient python MPS/DMRG Library
 * Copyright (C) 2020 The pyblock3 developers. All Rights Reserved.
 * 
 * block2: Efficient MPO implementation of quantum chemistry DMRG
 * Copyright (C) 2020 Huanchen Zhai <hczhai@caltech.edu>
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

#include "qc_symbolic.hpp"
#include <iomanip>
#include <map>
#include <memory>
#include <sstream>
#include <vector>

using namespace std;

namespace block2 {

// Matrix/Vector of symbols representation a tensor in MPO or contracted MPO
template <typename S> struct OperatorTensor {
    // Symbolic tensor for left blocking and right blocking
    // For normal MPO, lmat and rmat are the same
    shared_ptr<Symbolic<S>> lmat, rmat;
    // SparseMatrix representation of symbols
    map<shared_ptr<OpExpr<S>>, vector<double>, op_expr_less<S>> ops;
    OperatorTensor() : lmat(nullptr), rmat(nullptr) {}
    virtual ~OperatorTensor() = default;
};

// Symbolic Matrix Product Operator
template <typename S> struct MPO {
    vector<shared_ptr<OperatorTensor<S>>> tensors;
    vector<shared_ptr<Symbolic<S>>> left_operator_names;
    vector<shared_ptr<Symbolic<S>>> right_operator_names;
    // Number of sites
    int n_sites;
    MPO(int n_sites) : n_sites(n_sites) {}
    virtual ~MPO() = default;
};

struct FCIDUMP {
    FCIDUMP() {}
    virtual ~FCIDUMP() = default;
    virtual double v(uint8_t sl, uint8_t sr, uint16_t i, uint16_t j, uint16_t k,
                     uint16_t l) const {
        return 0.0;
    }
    virtual double t(uint8_t s, uint16_t i, uint16_t j) const { return 0.0; }
};

// Quantum chemistry MPO (non-spin-adapted)
template <typename S> struct MPOQC : MPO<S> {
    using MPO<S>::n_sites;
    bool symmetrized_p;
    MPOQC(vector<uint8_t> orb_sym, shared_ptr<FCIDUMP> fd,
          bool symmetrized_p = true)
        : MPO<S>(orb_sym.size()), symmetrized_p(symmetrized_p) {
        S vacuum;
        shared_ptr<OpExpr<S>> h_op =
            make_shared<OpElement<S>>(OpNames::H, SiteIndex(), vacuum);
        shared_ptr<OpExpr<S>> i_op =
            make_shared<OpElement<S>>(OpNames::I, SiteIndex(), vacuum);
        shared_ptr<OpExpr<S>> c_op[n_sites][2], d_op[n_sites][2];
        shared_ptr<OpExpr<S>> mc_op[n_sites][2], md_op[n_sites][2];
        shared_ptr<OpExpr<S>> rd_op[n_sites][2], r_op[n_sites][2];
        shared_ptr<OpExpr<S>> mrd_op[n_sites][2], mr_op[n_sites][2];
        shared_ptr<OpExpr<S>> a_op[n_sites][n_sites][4];
        shared_ptr<OpExpr<S>> ad_op[n_sites][n_sites][4];
        shared_ptr<OpExpr<S>> b_op[n_sites][n_sites][4];
        shared_ptr<OpExpr<S>> p_op[n_sites][n_sites][4];
        shared_ptr<OpExpr<S>> pd_op[n_sites][n_sites][4];
        shared_ptr<OpExpr<S>> q_op[n_sites][n_sites][4];
        const int sz[2] = {1, -1};
        const int sz_plus[4] = {2, 0, 0, -2}, sz_minus[4] = {0, -2, 2, 0};
        for (uint16_t m = 0; m < n_sites; m++)
            for (uint8_t s = 0; s < 2; s++) {
                c_op[m][s] = make_shared<OpElement<S>>(
                    OpNames::C, SiteIndex({m}, {s}), S(1, sz[s], orb_sym[m]));
                d_op[m][s] = make_shared<OpElement<S>>(
                    OpNames::D, SiteIndex({m}, {s}), S(-1, -sz[s], orb_sym[m]));
                mc_op[m][s] =
                    make_shared<OpElement<S>>(OpNames::C, SiteIndex({m}, {s}),
                                              S(1, sz[s], orb_sym[m]), -1.0);
                md_op[m][s] =
                    make_shared<OpElement<S>>(OpNames::D, SiteIndex({m}, {s}),
                                              S(-1, -sz[s], orb_sym[m]), -1.0);
                rd_op[m][s] = make_shared<OpElement<S>>(
                    OpNames::RD, SiteIndex({m}, {s}), S(1, sz[s], orb_sym[m]));
                r_op[m][s] = make_shared<OpElement<S>>(
                    OpNames::R, SiteIndex({m}, {s}), S(-1, -sz[s], orb_sym[m]));
                mrd_op[m][s] =
                    make_shared<OpElement<S>>(OpNames::RD, SiteIndex({m}, {s}),
                                              S(1, sz[s], orb_sym[m]), -1.0);
                mr_op[m][s] =
                    make_shared<OpElement<S>>(OpNames::R, SiteIndex({m}, {s}),
                                              S(-1, -sz[s], orb_sym[m]), -1.0);
            }
        for (uint16_t i = 0; i < n_sites; i++)
            for (uint16_t j = 0; j < n_sites; j++)
                for (uint8_t s = 0; s < 4; s++) {
                    SiteIndex sidx({i, j},
                                   {(uint8_t)(s & 1), (uint8_t)(s >> 1)});
                    a_op[i][j][s] = make_shared<OpElement<S>>(
                        OpNames::A, sidx,
                        S(2, sz_plus[s], orb_sym[i] ^ orb_sym[j]));
                    ad_op[i][j][s] = make_shared<OpElement<S>>(
                        OpNames::AD, sidx,
                        S(-2, -sz_plus[s], orb_sym[i] ^ orb_sym[j]));
                    b_op[i][j][s] = make_shared<OpElement<S>>(
                        OpNames::B, sidx,
                        S(0, sz_minus[s], orb_sym[i] ^ orb_sym[j]));
                    p_op[i][j][s] = make_shared<OpElement<S>>(
                        OpNames::P, sidx,
                        S(-2, -sz_plus[s], orb_sym[i] ^ orb_sym[j]));
                    pd_op[i][j][s] = make_shared<OpElement<S>>(
                        OpNames::PD, sidx,
                        S(2, sz_plus[s], orb_sym[i] ^ orb_sym[j]));
                    q_op[i][j][s] = make_shared<OpElement<S>>(
                        OpNames::Q, sidx,
                        S(0, -sz_minus[s], orb_sym[i] ^ orb_sym[j]));
                }
        int p;
        for (uint16_t m = 0; m < n_sites; m++) {
            shared_ptr<Symbolic<S>> pmat;
            int lshape, rshape;
            lshape = 2 + 4 * n_sites + 12 * m * m;
            rshape = 2 + 4 * n_sites + 12 * (m + 1) * (m + 1);
            if (m == 0)
                pmat = make_shared<SymbolicRowVector<S>>(rshape);
            else if (m == n_sites - 1)
                pmat = make_shared<SymbolicColumnVector<S>>(lshape);
            else
                pmat = make_shared<SymbolicMatrix<S>>(lshape, rshape);
            Symbolic<S> &mat = *pmat;
            if (m == 0) {
                mat[{0, 0}] = h_op;
                mat[{0, 1}] = i_op;
                mat[{0, 2}] = c_op[m][0];
                mat[{0, 3}] = c_op[m][1];
                mat[{0, 4}] = d_op[m][0];
                mat[{0, 5}] = d_op[m][1];
                p = 6;
                for (uint8_t s = 0; s < 2; s++) {
                    for (uint16_t j = m + 1; j < n_sites; j++)
                        mat[{0, p + j - m - 1}] = rd_op[j][s];
                    p += n_sites - (m + 1);
                }
                for (uint8_t s = 0; s < 2; s++) {
                    for (uint16_t j = m + 1; j < n_sites; j++)
                        mat[{0, p + j - m - 1}] = mr_op[j][s];
                    p += n_sites - (m + 1);
                }
            } else if (m == n_sites - 1) {
                mat[{0, 0}] = i_op;
                mat[{1, 0}] = h_op;
                p = 2;
                for (uint8_t s = 0; s < 2; s++) {
                    for (uint16_t j = 0; j < m; j++)
                        mat[{p + j, 0}] = r_op[j][s];
                    p += m;
                }
                for (uint8_t s = 0; s < 2; s++) {
                    for (uint16_t j = 0; j < m; j++)
                        mat[{p + j, 0}] = mrd_op[j][s];
                    p += m;
                }
                for (uint8_t s = 0; s < 2; s++) {
                    mat[{p, 0}] = d_op[m][s];
                    p += n_sites - m;
                }
                for (uint8_t s = 0; s < 2; s++) {
                    mat[{p, 0}] = c_op[m][s];
                    p += n_sites - m;
                }
            }
            if (m == 0) {
                for (uint8_t s = 0; s < 4; s++)
                    mat[{0, p + s}] = a_op[m][m][s];
                p += 4;
                for (uint8_t s = 0; s < 4; s++)
                    mat[{0, p + s}] = ad_op[m][m][s];
                p += 4;
                for (uint8_t s = 0; s < 4; s++)
                    mat[{0, p + s}] = b_op[m][m][s];
                p += 4;
                assert(p == mat.n);
            } else {
                if (m != n_sites - 1) {
                    mat[{0, 0}] = i_op;
                    mat[{1, 0}] = h_op;
                    p = 2;
                    for (uint8_t s = 0; s < 2; s++) {
                        for (uint16_t j = 0; j < m; j++)
                            mat[{p + j, 0}] = r_op[j][s];
                        p += m;
                    }
                    for (uint8_t s = 0; s < 2; s++) {
                        for (uint16_t j = 0; j < m; j++)
                            mat[{p + j, 0}] = mrd_op[j][s];
                        p += m;
                    }
                    for (uint8_t s = 0; s < 2; s++) {
                        mat[{p, 0}] = d_op[m][s];
                        p += n_sites - m;
                    }
                    for (uint8_t s = 0; s < 2; s++) {
                        mat[{p, 0}] = c_op[m][s];
                        p += n_sites - m;
                    }
                }
                for (uint8_t s = 0; s < 4; s++)
                    for (uint16_t j = 0; j < m; j++) {
                        for (uint16_t k = 0; k < m; k++)
                            mat[{p + k, 0}] = 0.5 * p_op[j][k][s];
                        p += m;
                    }
                for (uint8_t s = 0; s < 4; s++)
                    for (uint16_t j = 0; j < m; j++) {
                        for (uint16_t k = 0; k < m; k++)
                            mat[{p + k, 0}] = 0.5 * pd_op[j][k][s];
                        p += m;
                    }
                for (uint8_t s = 0; s < 4; s++)
                    for (uint16_t j = 0; j < m; j++) {
                        for (uint16_t k = 0; k < m; k++)
                            mat[{p + k, 0}] = q_op[j][k][s];
                        p += m;
                    }
                assert(p == mat.m);
            }
            if (m != 0 && m != n_sites - 1) {
                mat[{1, 1}] = i_op;
                p = 2;
                // pointers
                int pi = 1;
                int pc[2] = {2, 2 + m};
                int pd[2] = {2 + m * 2, 2 + m * 3};
                int prd[2] = {2 + m * 4 - m, 2 + m * 3 + n_sites - m};
                int pr[2] = {2 + m * 2 + n_sites * 2 - m,
                             2 + m + n_sites * 3 - m};
                int pa[4] = {
                    2 + n_sites * 4 + m * m * 0, 2 + n_sites * 4 + m * m * 1,
                    2 + n_sites * 4 + m * m * 2, 2 + n_sites * 4 + m * m * 3};
                int pad[4] = {
                    2 + n_sites * 4 + m * m * 4, 2 + n_sites * 4 + m * m * 5,
                    2 + n_sites * 4 + m * m * 6, 2 + n_sites * 4 + m * m * 7};
                int pb[4] = {
                    2 + n_sites * 4 + m * m * 8, 2 + n_sites * 4 + m * m * 9,
                    2 + n_sites * 4 + m * m * 10, 2 + n_sites * 4 + m * m * 11};
                // C
                for (uint8_t s = 0; s < 2; s++) {
                    for (uint16_t j = 0; j < m; j++)
                        mat[{pc[s] + j, p + j}] = i_op;
                    mat[{pi, p + m}] = c_op[m][s];
                    p += m + 1;
                }
                // D
                for (uint8_t s = 0; s < 2; s++) {
                    for (uint16_t j = 0; j < m; j++)
                        mat[{pd[s] + j, p + j}] = i_op;
                    mat[{pi, p + m}] = d_op[m][s];
                    p += m + 1;
                }
                // RD
                for (uint8_t s = 0; s < 2; s++) {
                    for (uint16_t i = m + 1; i < n_sites; i++) {
                        mat[{prd[s] + i, p + i - (m + 1)}] = i_op;
                        mat[{pi, p + i - (m + 1)}] = rd_op[i][s];
                        for (uint8_t sp = 0; sp < 2; sp++)
                            for (uint16_t k = 0; k < m; k++) {
                                mat[{pd[sp] + k, p + i - (m + 1)}] =
                                    -1.0 * pd_op[k][i][sp | (s << 1)];
                                mat[{pc[sp] + k, p + i - (m + 1)}] =
                                    q_op[k][i][sp | (s << 1)];
                            }
                        if (!symmetrized_p)
                            for (uint8_t sp = 0; sp < 2; sp++)
                                for (uint16_t j = 0; j < m; j++)
                                    for (uint16_t l = 0; l < m; l++) {
                                        double f = fd->v(s, sp, i, j, m, l);
                                        mat[{pa[s | (sp << 1)] + j * m + l,
                                             p + i - (m + 1)}] =
                                            f * d_op[m][sp];
                                    }
                        else
                            for (uint8_t sp = 0; sp < 2; sp++)
                                for (uint16_t j = 0; j < m; j++)
                                    for (uint16_t l = 0; l < m; l++) {
                                        double f0 = 0.5 *
                                                    fd->v(s, sp, i, j, m, l),
                                               f1 = -0.5 *
                                                    fd->v(s, sp, i, l, m, j);
                                        mat[{pa[s | (sp << 1)] + j * m + l,
                                             p + i - (m + 1)}] +=
                                            f0 * d_op[m][sp];
                                        mat[{pa[sp | (s << 1)] + j * m + l,
                                             p + i - (m + 1)}] +=
                                            f1 * d_op[m][sp];
                                    }
                        for (uint8_t sp = 0; sp < 2; sp++)
                            for (uint16_t k = 0; k < m; k++)
                                for (uint16_t l = 0; l < m; l++) {
                                    double f = fd->v(s, sp, i, m, k, l);
                                    mat[{pb[sp | (sp << 1)] + l * m + k,
                                         p + i - (m + 1)}] = f * c_op[m][s];
                                }
                        for (uint8_t sp = 0; sp < 2; sp++)
                            for (uint16_t j = 0; j < m; j++)
                                for (uint16_t k = 0; k < m; k++) {
                                    double f = -1.0 * fd->v(s, sp, i, j, k, m);
                                    mat[{pb[s | (sp << 1)] + j * m + k,
                                         p + i - (m + 1)}] += f * c_op[m][sp];
                                }
                    }
                    p += n_sites - (m + 1);
                }
                // R
                for (uint8_t s = 0; s < 2; s++) {
                    for (uint16_t i = m + 1; i < n_sites; i++) {
                        mat[{pr[s] + i, p + i - (m + 1)}] = i_op;
                        mat[{pi, p + i - (m + 1)}] = mr_op[i][s];
                        for (uint8_t sp = 0; sp < 2; sp++)
                            for (uint16_t k = 0; k < m; k++) {
                                mat[{pc[sp] + k, p + i - (m + 1)}] =
                                    p_op[k][i][sp | (s << 1)];
                                mat[{pd[sp] + k, p + i - (m + 1)}] =
                                    -1.0 * q_op[i][k][s | (sp << 1)];
                            }
                        if (!symmetrized_p)
                            for (uint8_t sp = 0; sp < 2; sp++)
                                for (uint16_t j = 0; j < m; j++)
                                    for (uint16_t l = 0; l < m; l++) {
                                        double f =
                                            -1.0 * fd->v(s, sp, i, j, m, l);
                                        mat[{pad[s | (sp << 1)] + j * m + l,
                                             p + i - (m + 1)}] =
                                            f * c_op[m][sp];
                                    }
                        else
                            for (uint8_t sp = 0; sp < 2; sp++)
                                for (uint16_t j = 0; j < m; j++)
                                    for (uint16_t l = 0; l < m; l++) {
                                        double f0 = -0.5 *
                                                    fd->v(s, sp, i, j, m, l),
                                               f1 = 0.5 *
                                                    fd->v(s, sp, i, l, m, j);
                                        mat[{pad[s | (sp << 1)] + j * m + l,
                                             p + i - (m + 1)}] +=
                                            f0 * c_op[m][sp];
                                        mat[{pad[sp | (s << 1)] + j * m + l,
                                             p + i - (m + 1)}] +=
                                            f1 * c_op[m][sp];
                                    }
                        for (uint8_t sp = 0; sp < 2; sp++)
                            for (uint16_t k = 0; k < m; k++)
                                for (uint16_t l = 0; l < m; l++) {
                                    double f = -1.0 * fd->v(s, sp, i, m, k, l);
                                    mat[{pb[sp | (sp << 1)] + k * m + l,
                                         p + i - (m + 1)}] = f * d_op[m][s];
                                }
                        for (uint8_t sp = 0; sp < 2; sp++)
                            for (uint16_t j = 0; j < m; j++)
                                for (uint16_t k = 0; k < m; k++) {
                                    double f = (-1.0) * (-1.0) *
                                               fd->v(s, sp, i, j, k, m);
                                    mat[{pb[sp | (s << 1)] + k * m + j,
                                         p + i - (m + 1)}] = f * d_op[m][sp];
                                }
                    }
                    p += n_sites - (m + 1);
                }
                // A
                for (uint8_t s = 0; s < 4; s++) {
                    for (uint16_t i = 0; i < m; i++)
                        for (uint16_t j = 0; j < m; j++)
                            mat[{pa[s] + i * m + j, p + i * (m + 1) + j}] =
                                i_op;
                    for (uint16_t i = 0; i < m; i++) {
                        mat[{pc[s & 1] + i, p + i * (m + 1) + m}] =
                            c_op[m][s >> 1];
                        mat[{pc[s >> 1] + i, p + m * (m + 1) + i}] =
                            mc_op[m][s & 1];
                    }
                    mat[{pi, p + m * (m + 1) + m}] = a_op[m][m][s];
                    p += (m + 1) * (m + 1);
                }
                // AD
                for (uint8_t s = 0; s < 4; s++) {
                    for (uint16_t i = 0; i < m; i++)
                        for (uint16_t j = 0; j < m; j++)
                            mat[{pad[s] + i * m + j, p + i * (m + 1) + j}] =
                                i_op;
                    for (uint16_t i = 0; i < m; i++) {
                        mat[{pd[s & 1] + i, p + i * (m + 1) + m}] =
                            md_op[m][s >> 1];
                        mat[{pd[s >> 1] + i, p + m * (m + 1) + i}] =
                            d_op[m][s & 1];
                    }
                    mat[{pi, p + m * (m + 1) + m}] = ad_op[m][m][s];
                    p += (m + 1) * (m + 1);
                }
                // B
                for (uint8_t s = 0; s < 4; s++) {
                    for (uint16_t i = 0; i < m; i++)
                        for (uint16_t j = 0; j < m; j++)
                            mat[{pb[s] + i * m + j, p + i * (m + 1) + j}] =
                                i_op;
                    for (uint16_t i = 0; i < m; i++) {
                        mat[{pc[s & 1] + i, p + i * (m + 1) + m}] =
                            d_op[m][s >> 1];
                        mat[{pd[s >> 1] + i, p + m * (m + 1) + i}] =
                            mc_op[m][s & 1];
                    }
                    mat[{pi, p + m * (m + 1) + m}] = b_op[m][m][s];
                    p += (m + 1) * (m + 1);
                }
                assert(p == mat.n);
            }
            shared_ptr<OperatorTensor<S>> opt =
                make_shared<OperatorTensor<S>>();
            opt->lmat = opt->rmat = pmat;
            // operator names
            shared_ptr<SymbolicRowVector<S>> plop;
            if (m == n_sites - 1)
                plop = make_shared<SymbolicRowVector<S>>(1);
            else
                plop = make_shared<SymbolicRowVector<S>>(rshape);
            SymbolicRowVector<S> &lop = *plop;
            lop[0] = h_op;
            if (m != n_sites - 1) {
                lop[1] = i_op;
                p = 2;
                for (uint8_t s = 0; s < 2; s++) {
                    for (uint16_t j = 0; j < m + 1; j++)
                        lop[p + j] = c_op[j][s];
                    p += m + 1;
                }
                for (uint8_t s = 0; s < 2; s++) {
                    for (uint16_t j = 0; j < m + 1; j++)
                        lop[p + j] = d_op[j][s];
                    p += m + 1;
                }
                for (uint8_t s = 0; s < 2; s++) {
                    for (uint16_t j = m + 1; j < n_sites; j++)
                        lop[p + j - (m + 1)] = rd_op[j][s];
                    p += n_sites - (m + 1);
                }
                for (uint8_t s = 0; s < 2; s++) {
                    for (uint16_t j = m + 1; j < n_sites; j++)
                        lop[p + j - (m + 1)] = mr_op[j][s];
                    p += n_sites - (m + 1);
                }
                for (uint8_t s = 0; s < 4; s++)
                    for (uint16_t j = 0; j < m + 1; j++) {
                        for (uint16_t k = 0; k < m + 1; k++)
                            lop[p + k] = a_op[j][k][s];
                        p += m + 1;
                    }
                for (uint8_t s = 0; s < 4; s++)
                    for (uint16_t j = 0; j < m + 1; j++) {
                        for (uint16_t k = 0; k < m + 1; k++)
                            lop[p + k] = ad_op[j][k][s];
                        p += m + 1;
                    }
                for (uint8_t s = 0; s < 4; s++)
                    for (uint16_t j = 0; j < m + 1; j++) {
                        for (uint16_t k = 0; k < m + 1; k++)
                            lop[p + k] = b_op[j][k][s];
                        p += m + 1;
                    }
                assert(p == rshape);
            }
            this->left_operator_names.push_back(plop);
            shared_ptr<SymbolicColumnVector<S>> prop;
            if (m == 0)
                prop = make_shared<SymbolicColumnVector<S>>(1);
            else
                prop = make_shared<SymbolicColumnVector<S>>(lshape);
            SymbolicColumnVector<S> &rop = *prop;
            if (m == 0)
                rop[0] = h_op;
            else {
                rop[0] = i_op;
                rop[1] = h_op;
                p = 2;
                for (uint8_t s = 0; s < 2; s++) {
                    for (uint16_t j = 0; j < m; j++)
                        rop[p + j] = r_op[j][s];
                    p += m;
                }
                for (uint8_t s = 0; s < 2; s++) {
                    for (uint16_t j = 0; j < m; j++)
                        rop[p + j] = mrd_op[j][s];
                    p += m;
                }
                for (uint8_t s = 0; s < 2; s++) {
                    for (uint16_t j = m; j < n_sites; j++)
                        rop[p + j - m] = d_op[j][s];
                    p += n_sites - m;
                }
                for (uint8_t s = 0; s < 2; s++) {
                    for (uint16_t j = m; j < n_sites; j++)
                        rop[p + j - m] = c_op[j][s];
                    p += n_sites - m;
                }
                for (uint8_t s = 0; s < 4; s++)
                    for (uint16_t j = 0; j < m; j++) {
                        for (uint16_t k = 0; k < m; k++)
                            rop[p + k] = 0.5 * p_op[j][k][s];
                        p += m;
                    }
                for (uint8_t s = 0; s < 4; s++)
                    for (uint16_t j = 0; j < m; j++) {
                        for (uint16_t k = 0; k < m; k++)
                            rop[p + k] = 0.5 * pd_op[j][k][s];
                        p += m;
                    }
                for (uint8_t s = 0; s < 4; s++)
                    for (uint16_t j = 0; j < m; j++) {
                        for (uint16_t k = 0; k < m; k++)
                            rop[p + k] = q_op[j][k][s];
                        p += m;
                    }
                assert(p == lshape);
            }
            this->right_operator_names.push_back(prop);
            for (auto &x : opt->lmat->data) {
                switch (x->get_type()) {
                case OpTypes::Zero:
                    break;
                case OpTypes::Elem:
                    opt->ops[abs_value(x)] = vector<double>();
                    break;
                case OpTypes::Sum:
                    for (auto &r : dynamic_pointer_cast<OpSum<S>>(x)->strings)
                        opt->ops[abs_value(
                            (shared_ptr<OpExpr<S>>)r->get_op())] =
                            vector<double>();
                    break;
                default:
                    assert(false);
                }
            }
            const shared_ptr<OpElement<S>> i_op =
                make_shared<OpElement<S>>(OpNames::I, SiteIndex(), vacuum);
            opt->ops[i_op] = vector<double>();
            this->tensors.push_back(opt);
        }
    }
};

} // namespace block2
