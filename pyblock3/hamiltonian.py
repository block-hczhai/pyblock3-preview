
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

"""
Quantum chemistry/general Hamiltonian object.
For construction of MPS/MPO.
"""

from .algebra.symmetry import SZ, BondInfo
from .algebra.core import FermionTensor
from .symbolic.expr import OpNames, OpElement, OpString, OpSum
from .symbolic.symbolic import SymbolicSparseTensor, SymbolicColumnVector, SymbolicRowVector
from .algebra.mps import MPS, MPSInfo
from functools import lru_cache, reduce
import numpy as np


class Hamiltonian:
    """
    Quantum chemistry/general Hamiltonian.
    For construction of MPS/MPO

    Attributes:
        basis : list(BondInfo)
            BondInfo in each site
        orb_sym : list(int)
            Point group symmetry for each site
        n_sites : int
            Number of sites
        n_syms : int
            Number of Point group symmetries
        fcidump : FCIDUMP
            one-electron and two-electron file
        vacuum : SZ
            vacuum state
        target : SZ
            target state
    """

    def __init__(self, fcidump, flat=False):
        self.orb_sym = fcidump.orb_sym
        self.n_syms = max(self.orb_sym) + 1
        self.fcidump = fcidump
        self.n_sites = fcidump.n_sites
        self.basis = [None] * self.n_sites
        self.vacuum = SZ(0, 0, 0)
        self.target = SZ(fcidump.n_elec, fcidump.twos, fcidump.ipg)
        for i, ipg in enumerate(self.orb_sym):
            self.basis[i] = BondInfo()
            self.basis[i][SZ(0, 0, 0)] = 1
            self.basis[i][SZ(1, 1, ipg)] = 1
            self.basis[i][SZ(1, -1, ipg)] = 1
            self.basis[i][SZ(2, 0, 0)] = 1
        if flat:
            self.vacuum = self.vacuum.to_flat()
            self.target = self.target.to_flat()
            import block3.sz as block3
            from .algebra.flat import FlatFermionTensor
            for i, b in enumerate(self.basis):
                dt = block3.MapUIntUInt()
                for k, v in b.items():
                    dt[k.to_flat()] = v
                self.basis[i] = dt
            self.basis = block3.VectorMapUIntUInt(self.basis)
            self.FT = FlatFermionTensor
        else:
            self.FT = FermionTensor
        self.flat = flat

    def build_mps(self, bond_dim, target=None, occ=None, bias=1):
        if target is None:
            target = self.target
        mps_info = MPSInfo(self.n_sites, self.vacuum, target, self.basis)
        if occ is None:
            mps_info.set_bond_dimension(bond_dim)
        else:
            assert len(occ) == len(self.basis)
            mps_info.set_bond_dimension_occ(bond_dim, occ, bias)
        return MPS.random(mps_info)
    
    def build_ancilla_mps(self, target=None):
        if target is None:
            target = SZ(self.n_sites * 2, 0, 0)
            if self.flat:
                target = target.to_flat()
        basis = [self.basis[i // 2] for i in range(self.n_sites * 2)]
        mps_info = MPSInfo(self.n_sites * 2, self.vacuum, target, basis)
        mps_info.set_bond_dimension_thermal_limit()
        mps = MPS.ones(mps_info)
        return mps / np.linalg.norm(mps)

    def build_identity_mpo(self):
        from .symbolic.symbolic_mpo import IdentityMPO
        impo = IdentityMPO(self).to_sparse()
        if self.flat:
            impo = impo.to_flat()
        return impo

    def build_site_mpo(self, op, k=-1):
        from .symbolic.symbolic_mpo import SiteMPO
        impo = SiteMPO(self, op, k=k).to_sparse()
        if self.flat:
            impo = impo.to_flat()
        return impo
    
    def build_ancilla_mpo(self, mpo, left=False):
        tensors = []
        for m in range(self.n_sites):
            infos = mpo.tensors[m].infos
            if not left:
                a = np.diag(mpo.tensors[m].__class__.ones((infos[-1], )))
                b = np.diag(mpo.tensors[m].__class__.ones((self.basis[m], )))
                tensors.append(mpo.tensors[m])
                tensors.append(np.transpose(np.tensordot(a, b, axes=0), axes=(0, 2, 3, 1)))
            else:
                a = np.diag(mpo.tensors[m].__class__.ones((infos[0], )))
                b = np.diag(mpo.tensors[m].__class__.ones((self.basis[m], )))
                tensors.append(np.transpose(np.tensordot(a, b, axes=0), axes=(0, 2, 3, 1)))
                tensors.append(mpo.tensors[m])
        return MPS(tensors=tensors, const=mpo.const, opts=mpo.opts)

    def build_qc_mpo(self):
        if self.flat:
            import block3.sz.hamiltonian as hm
            from .algebra.flat import FlatFermionTensor, FlatSparseTensor
            ts = hm.build_qc_mpo(self.fcidump.orb_sym,
                                 self.fcidump.h1e, self.fcidump.g2e)
            tensors = [None] * self.n_sites
            for i in range(0, self.n_sites):
                tensors[i] = FlatFermionTensor(
                    odd=FlatSparseTensor(*ts[i * 2 + 0]),
                    even=FlatSparseTensor(*ts[i * 2 + 1]))
            return MPS(tensors=tensors, const=self.fcidump.const_e)
        else:
            from .symbolic.symbolic_mpo import QCSymbolicMPO
            return QCSymbolicMPO(self).to_sparse()

    def build_mpo(self, gen, cutoff=1E-12, max_bond_dim=-1):

        if self.flat and isinstance(gen, tuple):
            import block3.sz.hamiltonian as hm
            from .algebra.flat import FlatFermionTensor, FlatSparseTensor
            orb_sym = np.array(self.orb_sym, dtype=int)
            if max_bond_dim == -9:
                ts = hm.build_mpo_ptree(orb_sym, *gen)
            else:
                ts = hm.build_mpo(orb_sym, *gen, cutoff, max_bond_dim)
            tensors = [None] * self.n_sites
            for i in range(0, self.n_sites):
                tensors[i] = FlatFermionTensor(
                    odd=FlatSparseTensor(*ts[i * 2 + 0]),
                    even=FlatSparseTensor(*ts[i * 2 + 1]))
            return MPS(tensors=tensors, const=self.fcidump.const_e)

        vac = SZ(0, 0, 0)
        i_op = OpElement(OpNames.I, (), q_label=vac)
        h_op = OpElement(OpNames.H, (), q_label=vac)
        c_op = np.zeros((self.n_sites, 2), dtype=OpElement)
        d_op = np.zeros((self.n_sites, 2), dtype=OpElement)
        for m in range(self.n_sites):
            for s in range(2):
                qa = SZ(1, [1, -1][s], self.orb_sym[m])
                qb = SZ(-1, -[1, -1][s], self.orb_sym[m])
                c_op[m, s] = OpElement(OpNames.C, (m, s), q_label=qa)
                d_op[m, s] = OpElement(OpNames.D, (m, s), q_label=qb)
        if isinstance(gen, tuple):
            SPIN, SITE, OP = 1, 2, 16384
            terms = [None] * len(gen[0])
            for ii, (x, t) in enumerate(zip(*gen)):
                terms[ii] = OpString(
                    [[c_op, d_op][i // OP][(i % OP) // SITE, (i % SITE) // SPIN] for i in t if i != -1], x)
        elif isinstance(gen, OpSum):
            terms = gen.strings
        elif isinstance(gen, OpString):
            terms = [gen]
        else:
            terms = list(gen(self.n_sites, c_op, d_op))
        h_expr = OpSum(terms)

        h_expr.sort(fermion=True)

        mpots = SymbolicSparseTensor(
            mat=SymbolicColumnVector(1, data=np.array([h_op], dtype=object)),
            ops={h_op: h_expr},
            lop=SymbolicColumnVector(1, data=np.array([h_op], dtype=object)),
            rop=SymbolicRowVector(1, data=np.array([i_op], dtype=object))
        )

        tensors = [None] * self.n_sites
        tensors[0] = mpots
        for i in range(0, self.n_sites - 1):
            print('MPO site %4d / %4d' % (i, self.n_sites))
            tensors[i], tensors[i + 1] = tensors[i].right_svd(i, cutoff=cutoff)

        for i in range(0, self.n_sites):
            tensors[i].ops = self.get_site_ops(
                i, tensors[i].ops.items(), cutoff=cutoff)
        mpo = MPS(tensors=tensors, const=self.fcidump.const_e)
        if self.flat:
            mpo = mpo.to_sparse().to_flat()
        return mpo

    def get_site_ops(self, m, op_names, cutoff=1E-20):
        """Get dict for matrix representation of site operators in mat
        (used for SparseSymbolicTensor.ops)"""
        op_reprs = {}
        basis = self.basis[m]
        ipg = self.orb_sym[m]
        sz = [1, -1]

        t = self.fcidump.t
        v = self.fcidump.v

        @lru_cache(maxsize=None)
        def i_operator():
            repr = self.FT.zeros(bond_infos=(basis, basis), dq=SZ(0, 0, 0))
            repr.even[SZ(0, 0, 0)][0, 0] = 1
            repr.even[SZ(1, -1, ipg)][0, 0] = 1
            repr.even[SZ(1, 1, ipg)][0, 0] = 1
            repr.even[SZ(2, 0, 0)][0, 0] = 1
            return repr

        @lru_cache(maxsize=None)
        def h_operator():
            repr = self.FT.zeros(bond_infos=(basis, basis), dq=SZ(0, 0, 0))
            repr.even[SZ(0, 0, 0)][0, 0] = 0
            repr.even[SZ(1, -1, ipg)][0, 0] = t(1, m, m)
            repr.even[SZ(1, 1, ipg)][0, 0] = t(0, m, m)
            repr.even[SZ(2, 0, 0)][0, 0] = t(0, m, m) + t(1, m, m) \
                + 0.5 * (v(0, 1, m, m, m, m) + v(1, 0, m, m, m, m))
            return repr

        @lru_cache(maxsize=None)
        def c_operator(s):
            repr = self.FT.zeros(bond_infos=(
                basis, basis), dq=SZ(1, sz[s], ipg))
            repr.odd[SZ(0, 0, 0)][0, 0] = 1
            repr.odd[SZ(1, -sz[s], ipg)][0, 0] = -1 if s else 1
            return repr

        @lru_cache(maxsize=None)
        def d_operator(s):
            repr = self.FT.zeros(bond_infos=(basis, basis),
                                 dq=SZ(-1, -sz[s], ipg))
            repr.odd[SZ(1, sz[s], ipg)][0, 0] = 1
            repr.odd[SZ(2, 0, 0)][0, 0] = -1 if s else 1
            return repr

        @lru_cache(maxsize=None)
        def a_operator(sl, sr):
            return c_operator(sl) @ c_operator(sr)

        @lru_cache(maxsize=None)
        def ad_operator(sl, sr):
            return d_operator(sr) @ d_operator(sl)

        @lru_cache(maxsize=None)
        def b_operator(sl, sr):
            return c_operator(sl) @ d_operator(sr)

        def get_op(op):
            assert isinstance(op, OpElement)
            if op.name == OpNames.I:
                return i_operator()
            elif op.name == OpNames.H:
                return h_operator()
            elif op.name == OpNames.C:
                return c_operator(op.site_index[1])
            elif op.name == OpNames.D:
                return d_operator(op.site_index[1])
            elif op.name == OpNames.A:
                return a_operator(op.site_index[2], op.site_index[3])
            elif op.name == OpNames.AD:
                return ad_operator(op.site_index[2], op.site_index[3])
            elif op.name == OpNames.B:
                return b_operator(op.site_index[2], op.site_index[3])
            elif op.name == OpNames.R:
                i, s = op.site_index
                if self.orb_sym[i] != self.orb_sym[m] or (
                    abs(t(s, i, m)) < cutoff and abs(v(s, 0, i, m, m, m)) < cutoff and
                        abs(v(s, 1, i, m, m, m)) < cutoff):
                    return 0
                else:
                    return (0.5 * t(s, i, m)) * d_operator(s) + \
                        (v(s, 0, i, m, m, m) * b_operator(0, 0)
                         + v(s, 1, i, m, m, m) * b_operator(1, 1)) @ d_operator(s)
            elif op.name == OpNames.RD:
                i, s = op.site_index
                if self.orb_sym[i] != self.orb_sym[m] or (
                    abs(t(s, i, m)) < cutoff and abs(v(s, 0, i, m, m, m)) < cutoff and
                        abs(v(s, 1, i, m, m, m)) < cutoff):
                    return 0
                else:
                    return (0.5 * t(s, i, m)) * c_operator(s) + \
                        c_operator(s) @ (v(s, 0, i, m, m, m) * b_operator(0, 0)
                                         + v(s, 1, i, m, m, m) * b_operator(1, 1))
            elif op.name == OpNames.P:
                i, k, si, sk = op.site_index
                if abs(v(si, sk, i, m, k, m)) < cutoff:
                    return 0
                else:
                    return v(si, sk, i, m, k, m) * ad_operator(si, sk)
            elif op.name == OpNames.PD:
                i, k, si, sk = op.site_index
                if abs(v(si, sk, i, m, k, m)) < cutoff:
                    return 0
                else:
                    return v(si, sk, i, m, k, m) * a_operator(si, sk)
            elif op.name == OpNames.Q:
                i, j, si, sj = op.site_index
                if si == sj:
                    if abs(v(si, sj, i, m, m, j)) < cutoff and abs(v(si, 0, i, j, m, m)) < cutoff \
                            and abs(v(si, 1, i, j, m, m)) < cutoff:
                        return 0
                    else:
                        return (-v(si, sj, i, m, m, j)) * b_operator(sj, si) \
                            + (b_operator(0, 0) * v(si, 0, i, j, m, m)
                               + b_operator(1, 1) * v(si, 1, i, j, m, m))
                else:
                    if abs(v(si, sj, i, m, m, j)) < cutoff:
                        return 0
                    else:
                        return (-v(si, sj, i, m, m, j)
                                ) * b_operator(sj, si)
            else:
                raise RuntimeError('Operator name not supported %s' % op.name)

        for op in op_names:
            if isinstance(op, OpElement):
                op_reprs[op] = get_op(op)
            else:
                xop, expr = op
                if isinstance(expr, OpElement):
                    op_reprs[xop] = get_op(expr)
                elif isinstance(expr, OpString):
                    op_reprs[xop] = expr.factor * \
                        reduce(np.matmul, [get_op(x) for x in expr.ops])
                elif isinstance(expr, OpSum):
                    op_reprs[xop] = 0
                    for sop in expr.strings:
                        op_reprs[xop] = op_reprs[xop] + sop.factor * \
                            reduce(np.matmul, [get_op(x) for x in sop.ops])
                else:
                    assert False

        return op_reprs
