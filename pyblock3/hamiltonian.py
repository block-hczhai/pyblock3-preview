
from .algebra.symmetry import SZ, StateInfo
from .algebra.core import FermionTensor
from .symbolic.expr import OpNames, OpElement
from functools import lru_cache
import numpy as np


class QCHamiltonian:
    """
    Quantum chemistry Hamiltonian.
    For construction of QCMPO
    Attributes:
        basis : list(StateInfo)
            StateInfo in each site
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

    def __init__(self, fcidump):
        self.orb_sym = fcidump.orb_sym
        self.n_syms = max(self.orb_sym) + 1
        self.fcidump = fcidump
        self.n_sites = fcidump.n_sites
        self.vacuum = SZ(0, 0, 0)
        self.target = SZ(fcidump.n_elec, fcidump.twos, fcidump.ipg)
        self.basis = [None] * self.n_sites
        for i, ipg in enumerate(self.orb_sym):
            self.basis[i] = StateInfo()
            self.basis[i].quanta[SZ(0, 0, 0)] = 1
            self.basis[i].quanta[SZ(1, 1, ipg)] = 1
            self.basis[i].quanta[SZ(1, -1, ipg)] = 1
            self.basis[i].quanta[SZ(2, 0, 0)] = 1

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
            repr = FermionTensor.zeros(SZ(0, 0, 0), basis, basis)
            repr.even[SZ(0, 0, 0)][0, 0] = 1
            repr.even[SZ(1, -1, ipg)][0, 0] = 1
            repr.even[SZ(1, 1, ipg)][0, 0] = 1
            repr.even[SZ(2, 0, 0)][0, 0] = 1
            return repr

        @lru_cache(maxsize=None)
        def h_operator():
            repr = FermionTensor.zeros(SZ(0, 0, 0), basis, basis)
            repr.even[SZ(0, 0, 0)][0, 0] = 0
            repr.even[SZ(1, -1, ipg)][0, 0] = t(1, m, m)
            repr.even[SZ(1, 1, ipg)][0, 0] = t(0, m, m)
            repr.even[SZ(2, 0, 0)][0, 0] = t(0, m, m) + t(1, m, m) \
                + 0.5 * (v(0, 1, m, m, m, m) + v(1, 0, m, m, m, m))
            return repr

        @lru_cache(maxsize=None)
        def c_operator(s):
            repr = FermionTensor.zeros(SZ(1, sz[s], ipg), basis, basis)
            repr.odd[SZ(0, 0, 0)][0, 0] = 1
            repr.odd[SZ(1, -sz[s], ipg)][0, 0] = -1 if s else 1
            return repr

        @lru_cache(maxsize=None)
        def d_operator(s):
            repr = FermionTensor.zeros(SZ(-1, -sz[s], ipg), basis, basis)
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

        for op in op_names:
            assert isinstance(op, OpElement)
            if op.name == OpNames.I:
                op_reprs[op] = i_operator()
            elif op.name == OpNames.H:
                op_reprs[op] = h_operator()
            elif op.name == OpNames.C:
                op_reprs[op] = c_operator(op.site_index[1])
            elif op.name == OpNames.D:
                op_reprs[op] = d_operator(op.site_index[1])
            elif op.name == OpNames.A:
                op_reprs[op] = a_operator(op.site_index[2], op.site_index[3])
            elif op.name == OpNames.AD:
                op_reprs[op] = ad_operator(op.site_index[2], op.site_index[3])
            elif op.name == OpNames.B:
                op_reprs[op] = b_operator(op.site_index[2], op.site_index[3])
            elif op.name == OpNames.R:
                i, s = op.site_index
                if self.orb_sym[i] != self.orb_sym[m] or (
                    abs(t(s, i, m)) < cutoff and abs(v(s, 0, i, m, m, m)) < cutoff and
                        abs(v(s, 1, i, m, m, m)) < cutoff):
                    op_reprs[op] = 0
                else:
                    op_reprs[op] = (0.5 * t(s, i, m)) * d_operator(s) + \
                        (v(s, 0, i, m, m, m) * b_operator(0, 0)
                         + v(s, 1, i, m, m, m) * b_operator(1, 1)) @ d_operator(s)
            elif op.name == OpNames.RD:
                i, s = op.site_index
                if self.orb_sym[i] != self.orb_sym[m] or (
                    abs(t(s, i, m)) < cutoff and abs(v(s, 0, i, m, m, m)) < cutoff and
                        abs(v(s, 1, i, m, m, m)) < cutoff):
                    op_reprs[op] = 0
                else:
                    op_reprs[op] = (0.5 * t(s, i, m)) * c_operator(s) + \
                        c_operator(s) @ (v(s, 0, i, m, m, m) * b_operator(0, 0)
                                         + v(s, 1, i, m, m, m) * b_operator(1, 1))
            elif op.name == OpNames.P:
                i, k, si, sk = op.site_index
                if abs(v(si, sk, i, m, k, m)) < cutoff:
                    op_reprs[op] = 0
                else:
                    op_reprs[op] = v(si, sk, i, m, k, m) * ad_operator(si, sk)
            elif op.name == OpNames.PD:
                i, k, si, sk = op.site_index
                if abs(v(si, sk, i, m, k, m)) < cutoff:
                    op_reprs[op] = 0
                else:
                    op_reprs[op] = v(si, sk, i, m, k, m) * a_operator(si, sk)
            elif op.name == OpNames.Q:
                i, j, si, sj = op.site_index
                if si == sj:
                    if abs(v(si, sj, i, m, m, j)) < cutoff and abs(v(si, 0, i, j, m, m)) < cutoff \
                            and abs(v(si, 1, i, j, m, m)) < cutoff:
                        op_reprs[op] = 0
                    else:
                        op_reprs[op] = (-v(si, sj, i, m, m, j)) * b_operator(sj, si) \
                            + (b_operator(0, 0) * v(si, 0, i, j, m, m)
                               + b_operator(1, 1) * v(si, 1, i, j, m, m))
                else:
                    if abs(v(si, sj, i, m, m, j)) < cutoff:
                        op_reprs[op] = 0
                    else:
                        op_reprs[op] = (-v(si, sj, i, m, m, j)
                                        ) * b_operator(sj, si)
            else:
                raise RuntimeError('Operator name not supported %s' % op.name)
        return op_reprs
