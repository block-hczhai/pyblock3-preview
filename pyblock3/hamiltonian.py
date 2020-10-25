
from .algebra.symmetry import SZ, BondInfo
from .algebra.core import FermionTensor
from .symbolic.expr import OpNames, OpElement, OpString, OpSum
from .symbolic.symbolic import SymbolicSparseTensor, SymbolicColumnVector, SymbolicRowVector
from .algebra.mps import MPS
from functools import lru_cache, reduce
import numpy as np


class Hamiltonian:
    """
    Quantum chemistry/general Hamiltonian.
    For construction of MPO
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
            from block3 import MapUIntUInt
            from .algebra.flat import FlatFermionTensor
            for i, b in enumerate(self.basis):
                dt = MapUIntUInt()
                for k, v in b.items():
                    dt[k.to_flat()] = v
                self.basis[i] = dt
            self.FT = FlatFermionTensor
        else:
            self.FT = FermionTensor
        self.flat = flat
    
    def build_qc_mpo(self):
        if self.flat:
            from block3 import hamiltonian as hm
            from .algebra.flat import FlatFermionTensor, FlatSparseTensor
            ts = hm.build_qc_mpo(self.fcidump.orb_sym, self.fcidump.h1e, self.fcidump.g2e)
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

        if self.flat:
            assert isinstance(gen, tuple)
            from block3 import hamiltonian as hm
            from .algebra.flat import FlatFermionTensor, FlatSparseTensor
            orb_sym = np.array(self.orb_sym, dtype=int)
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
        return MPS(tensors=tensors, const=self.fcidump.const_e)

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
                    abs(t(s, i, m)) < cutoff and abs(v(s, 0, i, m, m, m)) < cutoff
                        and abs(v(s, 1, i, m, m, m)) < cutoff):
                    return 0
                else:
                    return (0.5 * t(s, i, m)) * d_operator(s) + \
                        (v(s, 0, i, m, m, m) * b_operator(0, 0) +
                         v(s, 1, i, m, m, m) * b_operator(1, 1)) @ d_operator(s)
            elif op.name == OpNames.RD:
                i, s = op.site_index
                if self.orb_sym[i] != self.orb_sym[m] or (
                    abs(t(s, i, m)) < cutoff and abs(v(s, 0, i, m, m, m)) < cutoff
                        and abs(v(s, 1, i, m, m, m)) < cutoff):
                    return 0
                else:
                    return (0.5 * t(s, i, m)) * c_operator(s) + \
                        c_operator(s) @ (v(s, 0, i, m, m, m) * b_operator(0, 0) +
                                         v(s, 1, i, m, m, m) * b_operator(1, 1))
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
                            + (b_operator(0, 0) * v(si, 0, i, j, m, m) +
                               b_operator(1, 1) * v(si, 1, i, j, m, m))
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
