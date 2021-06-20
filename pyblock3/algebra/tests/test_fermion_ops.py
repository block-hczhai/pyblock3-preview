import numpy as np
import unittest
from pyblock3.algebra.fermion import SparseFermionTensor
from pyblock3.algebra import fermion_ops
from pyblock3.algebra.fermion_symmetry import U11, U1, Z2, Z22
from pyblock3.algebra.symmetry import BondInfo
from pyblock3.algebra.core import SubTensor
from pyblock3.algebra import fermion_setting

fermion_setting.set_options(flat=False, fermion=True)

class TestU11(unittest.TestCase):
    def setUp(self):
        self.t = 2
        self.U = 4
        self.tau = 0.01
        self.mu = 0.2
        self.symmetry = U11
        states = np.ones([1, 1]) * .5 ** .5
        blocks = [SubTensor(reduced=states, q_labels=(U11(0), U11(1, 1))),  # 0+
                  SubTensor(reduced=states, q_labels=(U11(1, 1), U11(0)))]  # +0, eigenstate of hopping
        self.hop_psi = SparseFermionTensor(blocks=blocks, pattern="++")

        states = np.ones([1, 1]) * .5
        blocks = [SubTensor(reduced=states, q_labels=(U11(2), U11(0))),
                  SubTensor(reduced=states, q_labels=(U11(0), U11(2))),
                  SubTensor(reduced=-states, q_labels=(U11(1, 1), U11(1, -1))),
                  SubTensor(reduced=states, q_labels=(U11(1, -1), U11(1, 1)))]
        self.hop_exp_psi = SparseFermionTensor(blocks=blocks, pattern="++")

        psi0 = SparseFermionTensor(blocks=[SubTensor(reduced=np.ones([1]), q_labels=(U11(0,0),))], pattern="+")
        psi1 = SparseFermionTensor(blocks=[SubTensor(reduced=np.ones([1]), q_labels=(U11(1,1),))], pattern="+")
        psi2 = SparseFermionTensor(blocks=[SubTensor(reduced=np.ones([1]), q_labels=(U11(1,-1),))], pattern="+")
        psi3 = SparseFermionTensor(blocks=[SubTensor(reduced=np.ones([1]), q_labels=(U11(2,0),))], pattern="+")
        self.psi_array = [psi0, psi1, psi2, psi3]
        self.sz_array = [0,1,-1,0]
        self.pn_array = [0,1,1,2]

        bond = BondInfo({U11(0):1, U11(1,1):1, U11(1,-1):1, U11(2,0):1})
        self.psi = SparseFermionTensor.random((bond,)*2, pattern="++", dq=U11(2,0))
        self.fac = (0.5, 0.3)

    def test_hopping(self):
        t = self.t
        hop = fermion_ops.H1(-t, symmetry=self.symmetry)
        ket = self.hop_psi
        ket1 = np.tensordot(hop, ket, axes=((2, 3), (0, 1)))
        bra = ket.dagger
        expec = np.tensordot(bra, ket1, axes=((1, 0), (0, 1)))
        self.assertAlmostEqual(expec, -t, 8)

    def test_hopping_exponential(self):
        t = self.t
        tau = self.tau
        hop = fermion_ops.H1(-t, symmetry=self.symmetry)
        hop_exp = hop.to_exponential(-tau)
        ket = self.hop_exp_psi
        bra = ket.dagger
        ket1 = np.tensordot(hop, ket, axes=((2, 3), (0, 1)))
        expec = np.tensordot(bra, ket1, axes=((1, 0), (0, 1)))
        self.assertAlmostEqual(expec, 2*t, 8)

        ket1 = np.tensordot(hop_exp, ket, axes=((2, 3), (0, 1)))
        expec = np.tensordot(bra, ket1, axes=((1, 0), (0, 1)))
        self.assertAlmostEqual(expec, np.e**(-2 * t * tau), 8)

    def test_onsite_u(self):
        uop = fermion_ops.onsite_U(u=self.U, symmetry=self.symmetry)
        for pn, psi in zip(self.pn_array, self.psi_array):
            ket = np.tensordot(uop, psi, axes=((1,), (0,)))
            expec = np.tensordot(psi.dagger, ket, axes=((0,),(0,)))
            self.assertAlmostEqual(expec, (pn==2)*self.U, 8)

    def test_onsite_u_exponential(self):
        uop = fermion_ops.onsite_U(u=self.U, symmetry=self.symmetry)
        uop_exp = uop.to_exponential(-self.tau)
        for pn, psi in zip(self.pn_array, self.psi_array):
            ket = np.tensordot(uop_exp, psi, axes=((1,), (0,)))
            expec = np.tensordot(psi.dagger, ket, axes=((0,),(0,)))
            self.assertAlmostEqual(expec, np.e**(-self.tau * self.U*(pn==2)), 8)

    def test_sz(self):
        szop = fermion_ops.measure_SZ(symmetry=self.symmetry)
        for sz, psi in zip(self.sz_array, self.psi_array):
            ket = np.tensordot(szop, psi, axes=((1,), (0,)))
            expec = np.tensordot(psi.dagger, ket, axes=((0,),(0,)))
            self.assertAlmostEqual(expec, .5*sz, 8)

    def test_n(self):
        nop = fermion_ops.ParticleNumber(symmetry=self.symmetry)
        for pn, psi in zip(self.pn_array, self.psi_array):
            ket = np.tensordot(nop, psi, axes=((1,), (0,)))
            expec = np.tensordot(psi.dagger, ket, axes=((0,),(0,)))
            self.assertAlmostEqual(expec, pn, 8)

    def test_hubbard(self):
        hop = fermion_ops.H1(-self.t, symmetry=self.symmetry)
        uop = fermion_ops.onsite_U(u=self.U, symmetry=self.symmetry)
        nop = fermion_ops.ParticleNumber(symmetry=self.symmetry)
        ket = self.psi
        bra = ket.dagger
        mu = self.mu
        faca, facb = self.fac
        hub = fermion_ops.Hubbard(self.t, self.U, mu=mu, fac=self.fac,
                                  symmetry=self.symmetry)

        ket1 = np.tensordot(hop, ket, axes=((2, 3), (0, 1)))
        ket1 = ket1 + faca * np.tensordot(uop, ket, axes=((-1,), (0,)))
        ket1 = ket1 + facb * \
            np.tensordot(uop, ket, axes=((-1,), (1,))).transpose([1, 0])
        ket1 = ket1 + faca * mu * np.tensordot(nop, ket, axes=((-1,), (0,)))
        ket1 = ket1 + facb * mu * \
            np.tensordot(nop, ket, axes=((-1,), (1,))).transpose([1, 0])
        expec = np.tensordot(bra, ket1, axes=((1, 0), (0, 1)))

        ket1 = np.tensordot(hub, ket, axes=((2, 3), (0, 1)))
        expec1 = np.tensordot(bra, ket1, axes=((1, 0), (0, 1)))

        self.assertAlmostEqual(expec, expec1, 8)


class TestZ22(TestU11):
    def setUp(self):
        self.t = 2
        self.U = 4
        self.tau = 0.1
        self.mu = 0.2
        self.symmetry = Z22
        states = np.ones([1, 1]) * .5 ** .5
        #states[0, 0] = .5 ** .5
        blocks = [SubTensor(reduced=states, q_labels=(Z22(0), Z22(1, 0))),  # 0+
                  SubTensor(reduced=states, q_labels=(Z22(1), Z22(0)))]  # +0, eigenstate of hopping
        self.hop_psi = SparseFermionTensor(blocks=blocks, pattern="++")

        blocks = []
        states = np.ones([1, 1]) * .5
        blocks = [SubTensor(reduced=states, q_labels=(Z22(0, 1), Z22(0))),
                  SubTensor(reduced=states, q_labels=(Z22(0), Z22(0, 1))),
                  SubTensor(reduced=-states, q_labels=(Z22(1, 0), Z22(1, 1))),
                  SubTensor(reduced=states, q_labels=(Z22(1, 1), Z22(1, 0)))]
        self.hop_exp_psi = SparseFermionTensor(blocks=blocks, pattern="++")

        psi0 = SparseFermionTensor(blocks=[SubTensor(reduced=np.ones([1]), q_labels=(Z22(0,0),))], pattern="+")
        psi1 = SparseFermionTensor(blocks=[SubTensor(reduced=np.ones([1]), q_labels=(Z22(1,0),))], pattern="+")
        psi2 = SparseFermionTensor(blocks=[SubTensor(reduced=np.ones([1]), q_labels=(Z22(1,1),))], pattern="+")
        psi3 = SparseFermionTensor(blocks=[SubTensor(reduced=np.ones([1]), q_labels=(Z22(0,1),))], pattern="+")
        self.psi_array = [psi0, psi1, psi2, psi3]
        self.sz_array = [0,1,-1,0]
        self.pn_array = [0,1,1,2]

        bond = BondInfo({Z22(0):1, Z22(0,1):1, Z22(1,0):1, Z22(1,1):1})
        self.psi = SparseFermionTensor.random((bond,)*2, pattern="++", dq=Z22(0,1))
        self.fac = (0.5, 0.3)

class TestU1(TestU11):
    def setUp(self):
        self.t = 2
        self.U = 4
        self.tau = 0.1
        self.mu = 0.2
        self.symmetry = U1
        states = np.zeros([1, 2])
        states[0, 0] = .5 ** .5
        blocks = [SubTensor(reduced=states, q_labels=(U1(0), U1(1))),  # 0+
                  SubTensor(reduced=states.T, q_labels=(U1(1), U1(0)))]  # +0, eigenstate of hopping
        self.hop_psi = SparseFermionTensor(blocks=blocks, pattern="++")

        blocks = []
        states = np.zeros([2, 2])
        states[0, 1] = -.5
        states[1, 0] = .5
        blocks = [SubTensor(reduced=np.ones([1, 1]) * .5, q_labels=(U1(2), U1(0))),
                  SubTensor(reduced=np.ones([1, 1]) * .5, q_labels=(U1(0), U1(2))),
                  SubTensor(reduced=states, q_labels=(U1(1), U1(1)))]
        self.hop_exp_psi = SparseFermionTensor(blocks=blocks, pattern="++")

        psi0 = SparseFermionTensor(blocks=[SubTensor(reduced=np.ones([1]), q_labels=(U1(0),))], pattern="+")
        psi1 = SparseFermionTensor(blocks=[SubTensor(reduced=np.asarray([1,0]), q_labels=(U1(1),))], pattern="+")
        psi2 = SparseFermionTensor(blocks=[SubTensor(reduced=np.asarray([0,1]), q_labels=(U1(1),))], pattern="+")
        psi3 = SparseFermionTensor(blocks=[SubTensor(reduced=np.ones([1]), q_labels=(U1(2),))], pattern="+")
        self.psi_array = [psi0, psi1, psi2, psi3]
        self.sz_array = [0,1,-1,0]
        self.pn_array = [0,1,1,2]

        bond = BondInfo({U1(0):1, U1(1):2, U1(2):1})
        self.psi = SparseFermionTensor.random((bond,)*2, pattern="++", dq=U1(2))
        self.fac = (0.5, 0.3)

class TestZ2(TestU11):
    def setUp(self):
        self.t = 2
        self.U = 4
        self.tau = 0.1
        self.mu = 0.2
        self.symmetry = Z2
        states = np.zeros([2, 2])
        states[0, 0] = .5 ** .5
        blocks = [SubTensor(reduced=states, q_labels=(Z2(0), Z2(1))),  # 0+
                  SubTensor(reduced=states, q_labels=(Z2(1), Z2(0)))]  # +0, eigenstate of hopping
        self.hop_psi = SparseFermionTensor(blocks=blocks, pattern="++")

        blocks = []
        states = np.zeros([2, 2])
        states[1, 0] = .5
        blocks = [SubTensor(reduced=states, q_labels=(Z2(0), Z2(0))),
                  SubTensor(reduced=states.T, q_labels=(Z2(0), Z2(0))),
                  SubTensor(reduced=-states.T, q_labels=(Z2(1), Z2(1))),
                  SubTensor(reduced=states, q_labels=(Z2(1), Z2(1)))]

        self.hop_exp_psi = SparseFermionTensor(blocks=blocks, pattern="++")

        psi0 = SparseFermionTensor(blocks=[SubTensor(reduced=np.asarray([1,0]), q_labels=(Z2(0),))], pattern="+")
        psi1 = SparseFermionTensor(blocks=[SubTensor(reduced=np.asarray([1,0]), q_labels=(Z2(1),))], pattern="+")
        psi2 = SparseFermionTensor(blocks=[SubTensor(reduced=np.asarray([0,1]), q_labels=(Z2(1),))], pattern="+")
        psi3 = SparseFermionTensor(blocks=[SubTensor(reduced=np.asarray([0,1]), q_labels=(Z2(0),))], pattern="+")
        self.psi_array = [psi0, psi1, psi2, psi3]
        self.sz_array = [0,1,-1,0]
        self.pn_array = [0,1,1,2]

        bond = BondInfo({Z2(0):2, Z2(1):2})
        self.psi = SparseFermionTensor.random((bond,)*2, pattern="++", dq=Z2(0))
        self.fac = (0.5, 0.3)

if __name__ == "__main__":
    print("Full Tests for fermion operators")
    unittest.main()
