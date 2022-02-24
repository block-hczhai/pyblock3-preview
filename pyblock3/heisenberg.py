
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
S=any-half-integer Heisenberg Hamiltonian object.
For construction of MPS/MPO.
"""

from .algebra.symmetry import SZ, BondInfo
from .algebra.core import FermionTensor
from .symbolic.expr import OpNames, OpElement, OpString, OpSum
from .symbolic.symbolic import SymbolicSparseTensor, SymbolicColumnVector, SymbolicRowVector
from functools import lru_cache, reduce
from .algebra.mps import MPS, MPSInfo
import numpy as np

class Heisenberg:
    """
    S=any-half-integer Heisenberg Hamiltonian.
    For construction of MPS/MPO

    Attributes:
        basis : list(BondInfo)
            BondInfo in each site
        n_sites : int
            Number of sites
        topology : [(int, int, float)]
            list of connected sites and the coupling
        vacuum : SZ
            vacuum state
        target : SZ
            target state
    """

    def __init__(self, twos, n_sites, topology, flat=False):
        self.twos = twos
        self.topology = topology
        self.n_sites = n_sites
        self.basis = [None] * self.n_sites
        self.vacuum = SZ(0, 0, 0)
        self.target = SZ(0, 0, 0)
        for i in range(self.n_sites):
            self.basis[i] = BondInfo()
            for j in range(-self.twos, self.twos + 1, 2):
                self.basis[i][SZ(0, j, 0)] = 1
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

    def build_mps(self, bond_dim, target=None, dtype=float):
        if target is None:
            target = self.target
        mps_info = MPSInfo(self.n_sites, self.vacuum, target, self.basis)
        mps_info.set_bond_dimension(bond_dim)
        return MPS.random(mps_info, dtype=dtype)

    def build_identity_mpo(self):
        from .symbolic.symbolic_mpo import IdentityMPO
        impo = IdentityMPO(self).to_sparse()
        if self.flat:
            impo = impo.to_flat()
        return impo

    def build_mpo(self, cutoff=1E-12, max_bond_dim=-1, const=0):

        vac = SZ(0, 0, 0)
        i_op = OpElement(OpNames.I, (), q_label=vac)
        h_op = OpElement(OpNames.H, (), q_label=vac)
        sp_op = np.zeros((self.n_sites, ), dtype=OpElement)
        sm_op = np.zeros((self.n_sites, ), dtype=OpElement)
        sz_op = np.zeros((self.n_sites, ), dtype=OpElement)
        for m in range(self.n_sites):
            sp_op[m] = OpElement(OpNames.SP, (m, ), q_label=SZ(0, 2, 0))
            sm_op[m] = OpElement(OpNames.SM, (m, ), q_label=SZ(0, -2, 0))
            sz_op[m] = OpElement(OpNames.SZ, (m, ), q_label=SZ(0, 0, 0))

        terms = []
        for (i, j, v) in self.topology:
            assert i != j
            terms.append(0.5 * v * sp_op[i] * sm_op[j])
            terms.append(0.5 * v * sm_op[i] * sp_op[j])
            terms.append(v * sz_op[i] * sz_op[j])

        h_expr = OpSum(terms)
        h_expr.sort(fermion=False)

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
        mpo = MPS(tensors=tensors, const=const)
        if self.flat:
            mpo = mpo.to_sparse().to_flat()
        return mpo

    def get_site_ops(self, m, op_names, cutoff=1E-20):
        """Get dict for matrix representation of site operators in mat
        (used for SparseSymbolicTensor.ops)"""
        op_reprs = {}
        basis = self.basis[m]
        tj = self.twos

        @lru_cache(maxsize=None)
        def i_operator():
            repr = self.FT.zeros(bond_infos=(basis, basis), dq=SZ(0, 0, 0))
            for k, _ in basis.items():
                repr.even[k][0, 0] = 1
            return repr

        @lru_cache(maxsize=None)
        def h_operator():
            repr = self.FT.zeros(bond_infos=(basis, basis), dq=SZ(0, 0, 0))
            return repr

        @lru_cache(maxsize=None)
        def sp_operator():
            repr = self.FT.zeros(bond_infos=(basis, basis), dq=SZ(0, 2, 0))
            for tm in range(-tj, tj - 1, 2):
                repr.even[SZ(0, tm, 0)][0, 0] = np.sqrt((tj - tm) * (tj + tm + 2) // 4)
            return repr

        @lru_cache(maxsize=None)
        def sm_operator():
            repr = self.FT.zeros(bond_infos=(basis, basis), dq=SZ(0, -2, 0))
            for tm in range(-tj + 2, tj + 1, 2):
                repr.even[SZ(0, tm, 0)][0, 0] = np.sqrt((tj + tm) * (tj - tm + 2) // 4)
            return repr

        @lru_cache(maxsize=None)
        def sz_operator():
            repr = self.FT.zeros(bond_infos=(basis, basis), dq=SZ(0, 0, 0))
            for tm in range(-tj, tj + 1, 2):
                repr.even[SZ(0, tm, 0)][0, 0] = tm / 2.0
            return repr

        def get_op(op):
            assert isinstance(op, OpElement)
            if op.name == OpNames.I:
                return i_operator()
            elif op.name == OpNames.H:
                return h_operator()
            elif op.name == OpNames.SP:
                return sp_operator()
            elif op.name == OpNames.SM:
                return sm_operator()
            elif op.name == OpNames.SZ:
                return sz_operator()
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
