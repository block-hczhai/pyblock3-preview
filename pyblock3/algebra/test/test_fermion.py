
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

def finger(T):
    if isinstance(T, SparseFermionTensor):
        out = 0
        for tsr in T:
            data = np.asarray(tsr).ravel()
            out += np.sum(np.sin(data) * np.arange(data.size))
    else:
        out = np.sum(np.sin(T.data) * np.arange(T.data.size))
    return out

np.random.seed(3)
q0 = QPN(0,0)
q1 = QPN(1,1)
q2 = QPN(1,-1)
q3 = QPN(2,0)

infox = BondInfo({q0: 2, q1: 3,
                  q2: 3, q3: 2})
infoy = BondInfo({q0: 3, q1: 2,
                  q2: 2, q3: 5})

Tsa = SparseFermionTensor.random((infox,infoy), pattern="+-")
Tsb = SparseFermionTensor.random((infox,infoy,infox), pattern="-++", dq=QPN(2,0))
Tfa = Tsa.to_flat()
Tfb = Tsb.to_flat()

class KnownValues(unittest.TestCase):

    def test_flip(self):
        Tsa1 = Tsa.copy()
        Tsa1._local_flip(1)
        self.assertAlmostEqual(finger(Tsa1), 17.64635702584, 8)

        Tfa1 = Tfa.copy()
        Tfa1._local_flip(1)
        self.assertAlmostEqual(finger(Tsa1-Tfa1.to_sparse()), 0, 8)

        Tsb1 = Tsb.copy()
        Tsb1._global_flip()
        self.assertAlmostEqual(finger(Tsb), -finger(Tsb1), 8)

        Tfb1 = Tfb.copy()
        Tfb1._global_flip()
        self.assertAlmostEqual(finger(Tsb1-Tfb1.to_sparse()), 0, 8)

        Tsb2 = Tsb.copy()
        Tsb1._local_flip(1)
        Tsb2._local_flip([0,2])
        self.assertAlmostEqual(finger(Tsb1), -finger(Tsb2), 8)

    def test_transpose(self):
        Tsa1 = np.transpose(Tsa, (1,0))
        self.assertAlmostEqual(finger(Tsa1), 16.65082083209277, 8)

        Tfa1 = np.transpose(Tfa, (1,0))
        self.assertAlmostEqual(finger(Tsa1-Tfa1.to_sparse()), 0, 8)

        Tsb1 = np.transpose(Tsb, (2,0,1))
        self.assertAlmostEqual(finger(Tsb1), -740.2582350554802, 8)

        Tfb1 = np.transpose(Tfb, (2,0,1))
        self.assertAlmostEqual(finger(Tsb1-Tfb1.to_sparse()), 0, 8)

        Tsb2 = np.transpose(Tsb1, (1,2,0))
        self.assertAlmostEqual(finger(Tsb), finger(Tsb2), 8)

        Tfb2 = np.transpose(Tfb1, (1,2,0))
        self.assertAlmostEqual(finger(Tfb2-Tfb), 0, 8)

    def test_tensordot(self):
        Tsc = np.tensordot(Tsa, Tsb, axes=((0,),(2,)))
        Tsa1 = np.transpose(Tsa, (1,0))
        Tsb1 = np.transpose(Tsb, (2,0,1))
        Tsc1 = np.tensordot(Tsa1, Tsb1, axes=((-1,),(0,)))
        self.assertAlmostEqual(finger(Tsc-Tsc1), 0.0, 8)

        Tfc = np.tensordot(Tfa, Tfb, axes=((0,),(2,)))
        Tfa1 = np.transpose(Tfa, (1,0))
        Tfb1 = np.transpose(Tfb, (2,0,1))
        Tfc1 = np.tensordot(Tfa1, Tfb1, axes=((-1,),(0,)))
        self.assertAlmostEqual(finger(Tfc-Tfc1), 0.0, 8)

        self.assertAlmostEqual(finger(Tsc.to_flat()-Tfc), 0.0, 8)

if __name__ == "__main__":
    print("Full Tests for Fermionic Tensors")
    unittest.main()
