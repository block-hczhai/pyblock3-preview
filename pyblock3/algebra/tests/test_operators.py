import numpy as np
from pyblock3.algebra.fermion_symmetry import U1, Z2, Z4, _blocks_to_tensor
from pyblock3.algebra.core import SubTensor
import unittest

class TestU1(unittest.TestCase):
    def setUp(self):
        self.cls = U1
        onsite_u_blocks = [SubTensor(reduced=np.ones([1,1]), q_labels=(U1(0),U1(2))),
                           SubTensor(reduced=np.ones([1,1])*2, q_labels=(U1(2),U1(0)))]
        self.onsite_u_psi = _blocks_to_tensor(onsite_u_blocks, "++", FLAT=False)

        sz_blocks = [SubTensor(reduced=np.ones([1,1]), q_labels=(U1(1,1),U1(2))),
                     SubTensor(reduced=np.ones([1,1])*2, q_labels=(U1(2),U1(1,1)))]
        self.sz_psi = _blocks_to_tensor(sz_blocks, "++", FLAT=False)

        pn_blocks = [SubTensor(reduced=np.ones([1,1]), q_labels=(U1(1,1),U1(1,-1))),
                     SubTensor(reduced=np.ones([1,1])*2, q_labels=(U1(2),U1(0)))]
        self.pn_psi = _blocks_to_tensor(pn_blocks, "++", FLAT=False)

        h1_blocks = [SubTensor(reduced=np.ones([1,1]), q_labels=(U1(1,1),U1(0))),
                     SubTensor(reduced=-np.ones([1,1]), q_labels=(U1(0),U1(1,1)))]
        self.h1_psi = _blocks_to_tensor(h1_blocks, "++", FLAT=False)
        self.tau = 0.1

    def test_onsite_U(self):
        Uop = self.cls.onsite_U(U=4, FLAT=False)
        psi = self.onsite_u_psi
        outpsi1 = np.tensordot(Uop, psi, axes=((-1,),(0,)))
        outpsi2 = np.tensordot(Uop, psi, axes=((-1,),(1,))).transpose((1,0))
        delta = (outpsi1+outpsi2-4*psi).norm()
        self.assertAlmostEqual(delta, 0, 8)

        bra = (outpsi1+outpsi2).dagger
        norm = np.tensordot(psi.dagger, psi, axes=((1,0),(0,1))).blocks[0]
        expec = np.tensordot(bra, psi, axes=((1,0),(0,1))).blocks[0]
        expec = np.asarray(expec/norm)
        self.assertAlmostEqual(expec, 4, 8)

    def test_measure_sz(self):
        szop = self.cls.Measure_SZ(FLAT=False)
        psi = self.sz_psi
        outpsi1 = np.tensordot(szop, psi, axes=((-1,),(0,)))
        outpsi2 = np.tensordot(szop, psi, axes=((-1,),(1,))).transpose((1,0))
        delta = (outpsi1+outpsi2-0.5*psi).norm()
        self.assertAlmostEqual(delta, 0, 8)

        bra = (outpsi1+outpsi2).dagger
        norm = np.tensordot(psi.dagger, psi, axes=((1,0),(0,1))).blocks[0]
        expec = np.tensordot(bra, psi, axes=((1,0),(0,1))).blocks[0]
        expec = np.asarray(expec/norm)
        self.assertAlmostEqual(expec, 0.5, 8)

    def test_particle_number(self):
        nop = self.cls.ParticleNumber(FLAT=False)
        psi = self.pn_psi
        outpsi1 = np.tensordot(nop, psi, axes=((-1,),(0,)))
        outpsi2 = np.tensordot(nop, psi, axes=((-1,),(1,))).transpose((1,0))
        delta = (outpsi1+outpsi2-2*psi).norm()
        self.assertAlmostEqual(delta, 0, 8)

        bra = (outpsi1+outpsi2).dagger
        norm = np.tensordot(psi.dagger, psi, axes=((1,0),(0,1))).blocks[0]
        expec = np.tensordot(bra, psi, axes=((1,0),(0,1))).blocks[0]
        expec = np.asarray(expec/norm)
        self.assertAlmostEqual(expec, 2, 8)

    def test_h1(self):
        hop = self.cls.H1(h=2, FLAT=False)
        psi = self.h1_psi
        outpsi = np.tensordot(hop, psi, axes=((-2,-1),(0,1)))
        delta = (outpsi+2*psi).norm()
        self.assertAlmostEqual(delta, 0, 8)
        bra = outpsi.dagger
        norm = np.tensordot(psi.dagger, psi, axes=((1,0),(0,1))).blocks[0]
        expec = np.tensordot(bra, psi, axes=((1,0),(0,1))).blocks[0]
        expec = np.asarray(expec/norm)
        self.assertAlmostEqual(expec, -2, 8)

class TestZ4(TestU1):
    def setUp(self):
        self.cls = Z4
        onsite_u_blocks = [SubTensor(reduced=np.ones([1,1]), q_labels=(Z4(0),Z4(2))),
                           SubTensor(reduced=np.ones([1,1])*2, q_labels=(Z4(2),Z4(0)))]
        self.onsite_u_psi = _blocks_to_tensor(onsite_u_blocks, "++", FLAT=False)

        sz_blocks = [SubTensor(reduced=np.ones([1,1]), q_labels=(Z4(1),Z4(2))),
                     SubTensor(reduced=np.ones([1,1])*2, q_labels=(Z4(2),Z4(1)))]
        self.sz_psi = _blocks_to_tensor(sz_blocks, "++", FLAT=False)

        pn_blocks = [SubTensor(reduced=np.ones([1,1]), q_labels=(Z4(1),Z4(3))),
                     SubTensor(reduced=np.ones([1,1])*2, q_labels=(Z4(2),Z4(0)))]
        self.pn_psi = _blocks_to_tensor(pn_blocks, "++", FLAT=False)

        h1_blocks = [SubTensor(reduced=np.ones([1,1]), q_labels=(Z4(1),Z4(0))),
                     SubTensor(reduced=-np.ones([1,1]), q_labels=(Z4(0),Z4(1)))]
        self.h1_psi = _blocks_to_tensor(h1_blocks, "++", FLAT=False)
        self.tau = 0.1

class TestZ2(TestU1):
    def setUp(self):
        self.cls = Z2
        onsite_u_blocks = [SubTensor(reduced=np.ones([1,1]), q_labels=(Z2(0),Z2(0))),
                           SubTensor(reduced=np.ones([1,1])*2, q_labels=(Z2(2),Z2(0)))]
        data = np.zeros([2,2])
        data[0,1] = 1
        data[1,0] = 2
        onsite_u_blocks = [SubTensor(reduced=data, q_labels=(Z2(0),Z2(0)))]
        self.onsite_u_psi = _blocks_to_tensor(onsite_u_blocks, "++", FLAT=False)
        data = np.zeros([2,2])
        data[0,1] = 1
        sz_blocks = [SubTensor(reduced=data, q_labels=(Z2(1),Z2(0))),
                     SubTensor(reduced=data.T*2, q_labels=(Z2(0),Z2(1)))]
        self.sz_psi = _blocks_to_tensor(sz_blocks, "++", FLAT=False)

        pn_blocks = [SubTensor(reduced=data, q_labels=(Z2(1),Z2(3))),
                     SubTensor(reduced=data.T*2, q_labels=(Z2(0),Z2(0)))]
        self.pn_psi = _blocks_to_tensor(pn_blocks, "++", FLAT=False)
        data = np.zeros([2,2])
        data[0,0] = 1
        h1_blocks = [SubTensor(reduced=data, q_labels=(Z2(1),Z2(0))),
                     SubTensor(reduced=-data, q_labels=(Z2(0),Z2(1)))]
        self.h1_psi = _blocks_to_tensor(h1_blocks, "++", FLAT=False)
        self.tau = 0.1

class TestU1_exp(unittest.TestCase):
    def setUp(self):
        self.cls = U1
        onsite_u_blocks = [SubTensor(reduced=np.ones([1,1])*2, q_labels=(U1(2),U1(0)))]
        self.onsite_u_psi = _blocks_to_tensor(onsite_u_blocks, "++", FLAT=False)

        sz_blocks = [SubTensor(reduced=np.ones([1,1]), q_labels=(U1(1,1),U1(2)))]
        self.sz_psi = _blocks_to_tensor(sz_blocks, "++", FLAT=False)

        pn_blocks = [SubTensor(reduced=np.ones([1,1]), q_labels=(U1(1,1),U1(1,-1)))]
        self.pn_psi = _blocks_to_tensor(pn_blocks, "++", FLAT=False)

        h1_blocks = [SubTensor(reduced=np.ones([1,1]), q_labels=(U1(1,1),U1(0))),
                     SubTensor(reduced=-np.ones([1,1]), q_labels=(U1(0),U1(1,1)))]
        self.h1_psi = _blocks_to_tensor(h1_blocks, "++", FLAT=False)
        self.tau = 0.1

    def test_onsite_U(self):
        tau = self.tau
        Uop = self.cls.onsite_U(U=4, FLAT=False)
        psi = self.onsite_u_psi
        #Uop = get_sparse_exponential(Uop, -tau)
        Uop = Uop.to_exponential(-tau)
        outpsi = np.tensordot(Uop, psi, axes=((-1,),(0,)))
        delta = (outpsi-np.exp(4*-tau)*psi).norm()
        self.assertAlmostEqual(delta, 0, 8)
        bra = outpsi.dagger
        norm = np.tensordot(psi.dagger, psi, axes=((1,0),(0,1))).blocks[0]
        expec = np.tensordot(bra, psi, axes=((1,0),(0,1))).blocks[0]
        expec = np.asarray(expec/norm)
        self.assertAlmostEqual(expec, np.exp(4*-tau), 8)

    def test_measure_sz(self):
        tau = self.tau
        szop = self.cls.Measure_SZ(FLAT=False)
        psi = self.sz_psi
        #szop = get_sparse_exponential(szop, -tau)
        szop = szop.to_exponential(-tau)
        outpsi = np.tensordot(szop, psi, axes=((-1,),(0,)))
        delta = (outpsi-np.exp(0.5*-tau)*psi).norm()
        self.assertAlmostEqual(delta, 0, 8)

        bra = outpsi.dagger
        norm = np.tensordot(psi.dagger, psi, axes=((1,0),(0,1))).blocks[0]
        expec = np.tensordot(bra, psi, axes=((1,0),(0,1))).blocks[0]
        expec = np.asarray(expec/norm)
        self.assertAlmostEqual(expec, np.exp(0.5*-tau), 8)

    def test_particle_number(self):
        tau = self.tau
        nop = self.cls.ParticleNumber(FLAT=False)
        psi = self.pn_psi
        #nop = get_sparse_exponential(nop, -tau)
        nop = nop.to_exponential(-tau)
        outpsi = np.tensordot(nop, psi, axes=((-1,),(0,)))
        delta = (outpsi-np.exp(1*-tau)*psi).norm()
        self.assertAlmostEqual(delta, 0, 8)

        bra = outpsi.dagger
        norm = np.tensordot(psi.dagger, psi, axes=((1,0),(0,1))).blocks[0]
        expec = np.tensordot(bra, psi, axes=((1,0),(0,1))).blocks[0]
        expec = np.asarray(expec/norm)
        self.assertAlmostEqual(expec, np.exp(1*-tau), 8)

    def test_h1(self):
        tau = self.tau
        hop = self.cls.H1(h=2, FLAT=False)
        psi = self.h1_psi
        #hop = get_sparse_exponential(hop, -tau)
        hop = hop.to_exponential(-tau)
        outpsi = np.tensordot(hop, psi, axes=((-2,-1),(0,1)))
        delta = (outpsi-np.exp(-2*-tau)*psi).norm()
        self.assertAlmostEqual(delta, 0, 8)
        bra = outpsi.dagger
        norm = np.tensordot(psi.dagger, psi, axes=((1,0),(0,1))).blocks[0]
        expec = np.tensordot(bra, psi, axes=((1,0),(0,1))).blocks[0]
        expec = np.asarray(expec/norm)
        self.assertAlmostEqual(expec, np.exp(-2*-tau), 8)

    def test_h1_expand(self):
        tau = self.tau
        hop = self.cls.H1(h=2, FLAT=False)
        psi = self.h1_psi
        #hop1 = get_sparse_exponential(hop, -tau)
        hop1 = hop.to_exponential(-tau)
        #identity = get_sparse_exponential(hop, 0)
        identity = hop.to_exponential(0)
        tmp = hop * -tau
        for nit in range(100):
            identity = identity + tmp/ np.math.factorial(nit+1)
            tmp = np.tensordot(hop*-tau, tmp, axes=((2,3),(0,1)))
        err = (identity-hop1).norm()
        self.assertAlmostEqual(err, 0, 8)

        hopf = self.cls.H1(h=2, FLAT=True)
        #hopf1 = get_flat_exponential(hopf, -tau)
        hopf1 = hopf.to_exponential(-tau)
        err = (hopf1-hop1.to_flat()).norm()
        self.assertAlmostEqual(err, 0, 8)

class TestZ4_exp(TestU1_exp):
    def setUp(self):
        self.cls = Z4
        onsite_u_blocks = [SubTensor(reduced=np.ones([1,1])*2, q_labels=(Z4(2),Z4(0)))]
        self.onsite_u_psi = _blocks_to_tensor(onsite_u_blocks, "++", FLAT=False)

        sz_blocks = [SubTensor(reduced=np.ones([1,1]), q_labels=(Z4(1),Z4(2)))]
        self.sz_psi = _blocks_to_tensor(sz_blocks, "++", FLAT=False)

        pn_blocks = [SubTensor(reduced=np.ones([1,1]), q_labels=(Z4(1),Z4(3)))]
        self.pn_psi = _blocks_to_tensor(pn_blocks, "++", FLAT=False)

        h1_blocks = [SubTensor(reduced=np.ones([1,1]), q_labels=(Z4(1),Z4(0))),
                     SubTensor(reduced=-np.ones([1,1]), q_labels=(Z4(0),Z4(1)))]
        self.h1_psi = _blocks_to_tensor(h1_blocks, "++", FLAT=False)
        self.tau = 0.1

class TestZ2_exp(TestU1_exp):
    def setUp(self):
        self.cls = Z2
        data= np.zeros([2,2])
        data[1,0] = 2
        onsite_u_blocks = [SubTensor(reduced=data, q_labels=(Z4(0),Z2(0)))]
        self.onsite_u_psi = _blocks_to_tensor(onsite_u_blocks, "++", FLAT=False)

        data= np.zeros([2,2])
        data[0,1] = 1
        sz_blocks = [SubTensor(reduced=data, q_labels=(Z2(1),Z2(0)))]
        self.sz_psi = _blocks_to_tensor(sz_blocks, "++", FLAT=False)

        data= np.zeros([2,2])
        data[0,1] = 1
        pn_blocks = [SubTensor(reduced=data, q_labels=(Z2(1),Z2(1)))]
        self.pn_psi = _blocks_to_tensor(pn_blocks, "++", FLAT=False)

        data = np.zeros([2,2])
        data[0,0] =1
        h1_blocks = [SubTensor(reduced=data, q_labels=(Z2(1),Z2(0))),
                     SubTensor(reduced=-data.T, q_labels=(Z2(0),Z2(1)))]
        self.h1_psi = _blocks_to_tensor(h1_blocks, "++", FLAT=False)
        self.tau = 0.1

if __name__ == "__main__":
    print("Full Tests for Fermionic Operators")
    unittest.main()
