
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

"""
Tests for general Fermionic sparse tensor.
Author: Yang Gao
"""

import unittest
import numpy as np
from pyblock3.algebra.symmetry import BondInfo, QPN
from pyblock3.algebra.fermion import SparseFermionTensor
from pyblock3.algebra.core import SubTensor
from pyblock3.algebra import fermion_operators as ops


np.random.seed(3)
q0 = QPN(0,0)
q1 = QPN(1,1)
q2 = QPN(1,-1)
q3 = QPN(2,0)

phys_bond = BondInfo({q0: 1, q1: 1,
                  q2: 1, q3: 1})
vir_bond = BondInfo({q0: 3, q1: 2,
                  q2: 2, q3: 5})

Tsa = SparseFermionTensor.random((phys_bond,vir_bond), pattern="+-", dq=QPN(1,1))
Tsb = SparseFermionTensor.random((vir_bond,phys_bond), pattern="++", dq=QPN(2,0))

ket = np.tensordot(Tsa, Tsb, axes=((-1,),(0,)))
bra = ket.dagger
norm = np.tensordot(bra, ket, axes=((1,0), (0,1)))
norm = np.asarray(norm.blocks[0])

class KnownValues(unittest.TestCase):

    def test_nops(self):
        nop = ops.count_n()
        state1 = np.tensordot(nop, ket, axes=((-1,),(0,)))
        state2 = np.tensordot(nop, ket, axes=((-1,),(1,))).transpose((1,0))
        state = state1 + state2
        n_expec = np.tensordot(bra, state, axes=((1,0), (0,1)))
        n_expec = np.asarray(n_expec.blocks[0]) / norm
        self.assertAlmostEqual(n_expec, 3, 8)

    def test_sz(self):
        zop = ops.measure_sz()
        state1 = np.tensordot(zop, ket, axes=((-1,),(0,)))
        state2 = np.tensordot(zop, ket, axes=((-1,),(1,))).transpose((1,0))
        state = state1+state2

        sz_expec = np.tensordot(bra, state, axes=((1,0), (0,1)))
        sz_expec = np.asarray(sz_expec.blocks[0]) / norm
        self.assertAlmostEqual(sz_expec, .5, 8)

    def test_hop(self):
        #psi = |0+> - |+0>, eigenstate of hopping(eigval = t)
        t = 2.0
        blocks =[]
        blocks.append(SubTensor(reduced=np.eye(1), q_labels=(QPN(1,1),QPN(0,0))))
        blocks.append(SubTensor(reduced=-np.eye(1), q_labels=(QPN(0,0),QPN(1,1))))
        psi = SparseFermionTensor(blocks=blocks, pattern="++")
        psi_bra = psi.dagger
        norm1 = np.tensordot(psi_bra, psi, axes=((1,0), (0,1)))
        norm1 = np.asarray(norm1.blocks[0])
        hop = ops.hopping(t)

        psi2 = np.tensordot(hop, psi, axes=((2,3),(0,1)))
        expec = np.tensordot(psi_bra, psi2, axes=((1,0), (0,1)))
        expec = np.asarray(expec.blocks[0]) / norm1
        self.assertAlmostEqual(expec, t, 8)

if __name__ == "__main__":
    print("Full Tests for Fermionic Operators")
    unittest.main()
