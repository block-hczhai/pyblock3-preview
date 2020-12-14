
#  pyblock3: An Efficient python MPS/DMRG Library
#  Copyright (C) 2020 The pyblock3 developers. All Rights Reserved.
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program. If not, see <https://www.gnu.org/licenses/>.
#
#

"""Symbolic MPO construction."""

from .expr import OpElement, OpNames
from ..algebra.symmetry import SZ
from ..algebra.mps import MPS
from .symbolic import SymbolicMatrix, SymbolicRowVector, SymbolicColumnVector, SymbolicSparseTensor
import numpy as np


class IdentityMPO(MPS):
    """Symbolic MPO for identity operator."""

    def __init__(self, hamil, opts=None):
        n_sites = hamil.n_sites
        vac = SZ(0, 0, 0)
        tensors = [None] * n_sites
        i_op = OpElement(OpNames.I, (), q_label=vac)
        for m in range(n_sites):
            if m == 0:
                mat = SymbolicRowVector(1)
            elif m == n_sites - 1:
                mat = SymbolicColumnVector(1)
            else:
                mat = SymbolicMatrix(1, 1)
            mat[0, 0] = i_op
            lop = SymbolicColumnVector(1, data=np.array([i_op]))
            rop = SymbolicRowVector(1, data=np.array([i_op]))
            ops = hamil.get_site_ops(m, mat.get_symbols())
            tensors[m] = SymbolicSparseTensor(mat, ops, lop, rop).deflate()
        super().__init__(tensors=tensors, const=0.0, opts=opts)


class SiteMPO(MPS):
    """Symbolic MPO for site operator (at site k)."""

    def __init__(self, hamil, op, k=-1, opts=None):
        n_sites = hamil.n_sites
        vac = SZ(0, 0, 0)
        tensors = [None] * n_sites
        i_op = OpElement(OpNames.I, (), q_label=vac)
        if k == -1:
            assert len(op.site_index) >= 1
            k = op.site_index[0]
        for m in range(n_sites):
            if m == 0:
                mat = SymbolicRowVector(1)
            elif m == n_sites - 1:
                mat = SymbolicColumnVector(1)
            else:
                mat = SymbolicMatrix(1, 1)
            mat[0, 0] = op if m == k else i_op
            lop = SymbolicColumnVector(
                1, data=np.array([op if m <= k else i_op]))
            rop = SymbolicRowVector(1, data=np.array([op if m >= k else i_op]))
            ops = hamil.get_site_ops(m, mat.get_symbols())
            tensors[m] = SymbolicSparseTensor(mat, ops, lop, rop).deflate()
        super().__init__(tensors=tensors, const=0.0, opts=opts, dq=op.q_label)


class QCSymbolicMPO(MPS):
    """Quantum chemistry symbolic Matrix Product Operator for Hamiltonain"""

    def __init__(self, hamil, symmetrized_p=True, opts=None):
        n_sites = hamil.n_sites
        vac = SZ(0, 0, 0)
        sz = [1, -1]

        tensors = [None] * n_sites

        v = hamil.fcidump.v

        h_op = OpElement(OpNames.H, (), q_label=vac)
        i_op = OpElement(OpNames.I, (), q_label=vac)
        c_op = np.zeros((n_sites, 2), dtype=OpElement)
        d_op = np.zeros((n_sites, 2), dtype=OpElement)
        mc_op = np.zeros((n_sites, 2), dtype=OpElement)
        md_op = np.zeros((n_sites, 2), dtype=OpElement)
        rd_op = np.zeros((n_sites, 2), dtype=OpElement)
        r_op = np.zeros((n_sites, 2), dtype=OpElement)
        mrd_op = np.zeros((n_sites, 2), dtype=OpElement)
        mr_op = np.zeros((n_sites, 2), dtype=OpElement)
        a_op = np.zeros((n_sites, n_sites, 2, 2), dtype=OpElement)
        ad_op = np.zeros((n_sites, n_sites, 2, 2), dtype=OpElement)
        b_op = np.zeros((n_sites, n_sites, 2, 2), dtype=OpElement)
        p_op = np.zeros((n_sites, n_sites, 2, 2), dtype=OpElement)
        pd_op = np.zeros((n_sites, n_sites, 2, 2), dtype=OpElement)
        q_op = np.zeros((n_sites, n_sites, 2, 2), dtype=OpElement)

        for m in range(n_sites):
            for s in range(2):
                sid = (m, s)
                qa = SZ(1, sz[s], hamil.orb_sym[m])
                qb = SZ(-1, -sz[s], hamil.orb_sym[m])
                c_op[m, s] = OpElement(OpNames.C, sid, q_label=qa)
                d_op[m, s] = OpElement(OpNames.D, sid, q_label=qb)
                mc_op[m, s] = -c_op[m, s]
                md_op[m, s] = -d_op[m, s]
                rd_op[m, s] = OpElement(OpNames.RD, sid, q_label=qa)
                r_op[m, s] = OpElement(OpNames.R, sid, q_label=qb)
                mrd_op[m, s] = -rd_op[m, s]
                mr_op[m, s] = -r_op[m, s]

        for i in range(n_sites):
            for j in range(n_sites):
                for si in range(2):  # low
                    for sj in range(2):  # high
                        sid = (i, j, si, sj)
                        qp = SZ(2, sz[si] + sz[sj],
                                hamil.orb_sym[i] ^ hamil.orb_sym[j])
                        mqp = -qp
                        qm = SZ(0, sz[si] - sz[sj],
                                hamil.orb_sym[i] ^ hamil.orb_sym[j])
                        mqm = -qm
                        a_op[i, j, si, sj] = OpElement(
                            OpNames.A, sid, q_label=qp)
                        ad_op[i, j, si, sj] = OpElement(
                            OpNames.AD, sid, q_label=mqp)
                        b_op[i, j, si, sj] = OpElement(
                            OpNames.B, sid, q_label=qm)
                        p_op[i, j, si, sj] = OpElement(
                            OpNames.P, sid, q_label=mqp)
                        pd_op[i, j, si, sj] = OpElement(
                            OpNames.PD, sid, q_label=qp)
                        q_op[i, j, si, sj] = OpElement(
                            OpNames.Q, sid, q_label=mqm)

        for m in range(n_sites):
            print('MPO site %4d / %4d' % (m, n_sites))
            lshape = 2 + 4 * n_sites + 12 * m * m
            rshape = 2 + 4 * n_sites + 12 * (m + 1) * (m + 1)
            if m == 0:
                mat = SymbolicRowVector(rshape)
            elif m == n_sites - 1:
                mat = SymbolicColumnVector(lshape)
            else:
                mat = SymbolicMatrix(lshape, rshape)

            if m == 0:
                mat[0, 0] = h_op
                mat[0, 1] = i_op
                mat[0, 2] = c_op[m, 0]
                mat[0, 3] = c_op[m, 1]
                mat[0, 4] = d_op[m, 0]
                mat[0, 5] = d_op[m, 1]
                p = 6
                for s in range(2):
                    for j in range(m + 1, n_sites):
                        mat[0, p + j - m - 1] = rd_op[j, s]
                    p += n_sites - (m + 1)
                for s in range(2):
                    for j in range(m + 1, n_sites):
                        mat[0, p + j - m - 1] = mr_op[j, s]
                    p += n_sites - (m + 1)
            elif m == n_sites - 1:
                mat[0, 0] = i_op
                mat[1, 0] = h_op
                p = 2
                for s in range(2):
                    for j in range(m):
                        mat[p + j, 0] = r_op[j, s]
                    p += m
                for s in range(2):
                    for j in range(m):
                        mat[p + j, 0] = mrd_op[j, s]
                    p += m
                for s in range(2):
                    mat[p, 0] = d_op[m, s]
                    p += n_sites - m
                for s in range(2):
                    mat[p, 0] = c_op[m, s]
                    p += n_sites - m
            if m == 0:
                for sr in range(2):
                    for sl in range(2):
                        mat[0, p] = a_op[m, m, sl, sr]
                        p += 1
                for sr in range(2):
                    for sl in range(2):
                        mat[0, p] = ad_op[m, m, sl, sr]
                        p += 1
                for sr in range(2):
                    for sl in range(2):
                        mat[0, p] = b_op[m, m, sl, sr]
                        p += 1
                assert p == mat.n_cols
            else:
                if m != n_sites - 1:
                    mat[0, 0] = i_op
                    mat[1, 0] = h_op
                    p = 2
                    for s in range(2):
                        for j in range(m):
                            mat[p + j, 0] = r_op[j, s]
                        p += m
                    for s in range(2):
                        for j in range(m):
                            mat[p + j, 0] = mrd_op[j, s]
                        p += m
                    for s in range(2):
                        mat[p, 0] = d_op[m, s]
                        p += n_sites - m
                    for s in range(2):
                        mat[p, 0] = c_op[m, s]
                        p += n_sites - m
                for sr in range(2):
                    for sl in range(2):
                        for j in range(m):
                            for k in range(m):
                                mat[p + k, 0] = 0.5 * p_op[j, k, sl, sr]
                            p += m
                for sr in range(2):
                    for sl in range(2):
                        for j in range(m):
                            for k in range(m):
                                mat[p + k, 0] = 0.5 * pd_op[j, k, sl, sr]
                            p += m
                for sr in range(2):
                    for sl in range(2):
                        for j in range(m):
                            for k in range(m):
                                mat[p + k, 0] = q_op[j, k, sl, sr]
                            p += m
                assert p == mat.n_rows
            if m != 0 and m != n_sites - 1:
                mat[1, 1] = i_op
                p = 2
                # pointers
                pi = 1
                pc = [2, 2 + m]
                pd = np.array([2 + m * 2, 2 + m * 3], dtype=int)
                prd = np.array(
                    [2 + m * 4 - m, 2 + m * 3 + n_sites - m], dtype=int)
                pr = np.array([2 + m * 2 + n_sites * 2 - m,
                               2 + m + n_sites * 3 - m], dtype=int)
                pa = np.array([[2 + n_sites * 4 + m * m * 0,
                                2 + n_sites * 4 + m * m * 2],
                               [2 + n_sites * 4 + m * m * 1,
                                2 + n_sites * 4 + m * m * 3]], dtype=int)
                pad = np.array([[2 + n_sites * 4 + m * m * 4,
                                 2 + n_sites * 4 + m * m * 6],
                                [2 + n_sites * 4 + m * m * 5,
                                 2 + n_sites * 4 + m * m * 7]], dtype=int)
                pb = np.array([[2 + n_sites * 4 + m * m * 8,
                                2 + n_sites * 4 + m * m * 10],
                               [2 + n_sites * 4 + m * m * 9,
                                2 + n_sites * 4 + m * m * 11]], dtype=int)
                # C
                for s in range(2):
                    for j in range(m):
                        mat[pc[s] + j, p + j] = i_op
                    mat[pi, p + m] = c_op[m, s]
                    p += m + 1
                for s in range(2):
                    for j in range(m):
                        mat[pd[s] + j, p + j] = i_op
                    mat[pi, p + m] = d_op[m, s]
                    p += m + 1
                # RD
                for s in range(2):
                    for i in range(m + 1, n_sites):
                        mat[prd[s] + i, p + i - (m + 1)] = i_op
                        mat[pi, p + i - (m + 1)] = rd_op[i, s]
                        for sp in range(2):
                            for k in range(m):
                                mat[pd[sp] + k, p + i -
                                    (m + 1)] = pd_op[i, k, s, sp]
                                mat[pc[sp] + k, p + i -
                                    (m + 1)] = q_op[k, i, sp, s]
                        if not symmetrized_p:
                            for sp in range(2):
                                for j in range(m):
                                    for l in range(m):
                                        f = v(s, sp, i, j, m, l)
                                        mat[pa[s, sp] + j * m + l, p +
                                            i - (m + 1)] = f * d_op[m, sp]
                        else:
                            for sp in range(2):
                                for j in range(m):
                                    for l in range(m):
                                        f0 = 0.5 * v(s, sp, i, j, m, l)
                                        f1 = -0.5 * v(s, sp, i, l, m, j)
                                        mat[pa[s, sp] + j * m + l, p +
                                            i - (m + 1)] = f0 * d_op[m, sp]
                                        mat[pa[sp, s] + j * m + l, p +
                                            i - (m + 1)] = f1 * d_op[m, sp]
                        for sp in range(2):
                            for k in range(m):
                                for l in range(m):
                                    f = v(s, sp, i, m, k, l)
                                    mat[pb[sp, sp] + l * m + k, p +
                                        i - (m + 1)] = f * c_op[m, s]
                        for sp in range(2):
                            for j in range(m):
                                for k in range(m):
                                    f = -1.0 * v(s, sp, i, j, k, m)
                                    mat[pb[s, sp] + j * m + k, p +
                                        i - (m + 1)] = f * c_op[m, sp]
                    p += n_sites - (m + 1)
                # R
                for s in range(2):
                    for i in range(m + 1, n_sites):
                        mat[pr[s] + i, p + i - (m + 1)] = i_op
                        mat[pi, p + i - (m + 1)] = mr_op[i, s]
                        for sp in range(2):
                            for k in range(m):
                                mat[pc[sp] + k, p + i -
                                    (m + 1)] = -1.0 * p_op[i, k, s, sp]
                                mat[pd[sp] + k, p + i -
                                    (m + 1)] = -1.0 * q_op[i, k, s, sp]
                        if not symmetrized_p:
                            for sp in range(2):
                                for j in range(m):
                                    for l in range(m):
                                        f = -1.0 * v(s, sp, i, j, m, l)
                                        mat[pad[s, sp] + j * m + l, p +
                                            i - (m + 1)] = f * c_op[m, sp]
                        else:
                            for sp in range(2):
                                for j in range(m):
                                    for l in range(m):
                                        f0 = -0.5 * v(s, sp, i, j, m, l)
                                        f1 = 0.5 * v(s, sp, i, l, m, j)
                                        mat[pad[s, sp] + j * m + l, p +
                                            i - (m + 1)] = f0 * c_op[m, sp]
                                        mat[pad[sp, s] + j * m + l, p +
                                            i - (m + 1)] = f1 * c_op[m, sp]
                        for sp in range(2):
                            for k in range(m):
                                for l in range(m):
                                    f = -1.0 * v(s, sp, i, m, k, l)
                                    mat[pb[sp, sp] + k * m + l, p +
                                        i - (m + 1)] = f * d_op[m, s]
                        for sp in range(2):
                            for j in range(m):
                                for k in range(m):
                                    f = (-1.0) * (-1.0) * v(s, sp, i, j, k, m)
                                    mat[pb[sp, s] + k * m + j, p +
                                        i - (m + 1)] = f * d_op[m, sp]
                    p += n_sites - (m + 1)
                # A
                for sr in range(2):
                    for sl in range(2):
                        for i in range(m):
                            for j in range(m):
                                mat[pa[sl, sr] + i * m + j,
                                    p + i * (m + 1) + j] = i_op
                        for i in range(m):
                            mat[pc[sl] + i, p + i * (m + 1) + m] = c_op[m, sr]
                            mat[pc[sr] + i, p + m * (m + 1) + i] = mc_op[m, sl]
                        mat[pi, p + m * (m + 1) + m] = a_op[m, m, sl, sr]
                        p += (m + 1) * (m + 1)
                # AD
                for sr in range(2):
                    for sl in range(2):
                        for i in range(m):
                            for j in range(m):
                                mat[pad[sl, sr] + i * m + j,
                                    p + i * (m + 1) + j] = i_op
                        for i in range(m):
                            mat[pd[sl] + i, p + i * (m + 1) + m] = md_op[m, sr]
                            mat[pd[sr] + i, p + m * (m + 1) + i] = d_op[m, sl]
                        mat[pi, p + m * (m + 1) + m] = ad_op[m, m, sl, sr]
                        p += (m + 1) * (m + 1)
                # B
                for sr in range(2):
                    for sl in range(2):
                        for i in range(m):
                            for j in range(m):
                                mat[pb[sl, sr] + i * m + j,
                                    p + i * (m + 1) + j] = i_op
                        for i in range(m):
                            mat[pc[sl] + i, p + i * (m + 1) + m] = d_op[m, sr]
                            mat[pd[sr] + i, p + m * (m + 1) + i] = mc_op[m, sl]
                        mat[pi, p + m * (m + 1) + m] = b_op[m, m, sl, sr]
                        p += (m + 1) * (m + 1)
                assert p == mat.n_cols

            # left virtual dimension operator names
            lop = SymbolicColumnVector(1 if m == 0 else lshape)
            if m == 0:
                lop[0] = h_op
            else:
                lop[0] = i_op
                lop[1] = h_op
                p = 2
                for s in range(2):
                    for j in range(m):
                        lop[p + j] = r_op[j, s]
                    p += m
                for s in range(2):
                    for j in range(m):
                        lop[p + j] = mrd_op[j, s]
                    p += m
                for s in range(2):
                    for j in range(m, n_sites):
                        lop[p + j - m] = d_op[j, s]
                    p += n_sites - m
                for s in range(2):
                    for j in range(m, n_sites):
                        lop[p + j - m] = c_op[j, s]
                    p += n_sites - m
                for sr in range(2):
                    for sl in range(2):
                        for j in range(m):
                            for k in range(m):
                                lop[p + k] = 0.5 * p_op[j, k, sl, sr]
                            p += m
                for sr in range(2):
                    for sl in range(2):
                        for j in range(m):
                            for k in range(m):
                                lop[p + k] = 0.5 * pd_op[j, k, sl, sr]
                            p += m
                for sr in range(2):
                    for sl in range(2):
                        for j in range(m):
                            for k in range(m):
                                lop[p + k] = q_op[j, k, sl, sr]
                            p += m
                assert p == lshape

            # right virtual dimension operator names
            rop = SymbolicRowVector(1 if m == n_sites - 1 else rshape)
            rop[0] = h_op
            if m != n_sites - 1:
                rop[1] = i_op
                p = 2
                for s in range(2):
                    for j in range(m + 1):
                        rop[p + j] = c_op[j, s]
                    p += m + 1
                for s in range(2):
                    for j in range(m + 1):
                        rop[p + j] = d_op[j, s]
                    p += m + 1
                for s in range(2):
                    for j in range(m + 1, n_sites):
                        rop[p + j - (m + 1)] = rd_op[j, s]
                    p += n_sites - (m + 1)
                for s in range(2):
                    for j in range(m + 1, n_sites):
                        rop[p + j - (m + 1)] = mr_op[j, s]
                    p += n_sites - (m + 1)
                for sr in range(2):
                    for sl in range(2):
                        for j in range(m + 1):
                            for k in range(m + 1):
                                rop[p + k] = a_op[j, k, sl, sr]
                            p += m + 1
                for sr in range(2):
                    for sl in range(2):
                        for j in range(m + 1):
                            for k in range(m + 1):
                                rop[p + k] = ad_op[j, k, sl, sr]
                            p += m + 1
                for sr in range(2):
                    for sl in range(2):
                        for j in range(m + 1):
                            for k in range(m + 1):
                                rop[p + k] = b_op[j, k, sl, sr]
                            p += m + 1
                assert p == rshape

            ops = hamil.get_site_ops(m, mat.get_symbols())
            tensors[m] = SymbolicSparseTensor(mat, ops, lop, rop).deflate()

        super().__init__(tensors=tensors, const=hamil.fcidump.const_e, opts=opts)
