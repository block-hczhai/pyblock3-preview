import numpy as np
import unittest
from pyblock3.algebra.fermion import SparseFermionTensor
from pyblock3.algebra.fermion_symmetry import U11, U1, Z4, Z2, Z22
from pyblock3.algebra.symmetry import BondInfo, SZ
from pyblock3.algebra.core import SparseTensor, SubTensor
from pyblock3.algebra import fermion_setting

fermion_setting.set_options(flat=True, fermion=True)
rand = SparseFermionTensor.random

class TestU11(unittest.TestCase):
    def setUp(self):
        bond = BondInfo({U11(0):2, U11(1,1):3, U11(1,-1):4,
                        U11(2,0):4})
        self.bond = bond
        self.T = rand((bond,)*4, pattern="++--", dq=U11(0))
        self.Tf = self.T.to_flat()
        self.symmetry = U11
        self.skeleton_test_dq = U11(2)
        self.contract_test_dq = (U11(1,1),U11(1,-1))
        self.svd_dq_iterator = [U11(0), U11(1,1), U11(1,-1),U11(2)]
        self.shift = U11(1,1)
        self.T1 = rand((bond,)*4, pattern="++--", dq=U11(2))
        self.T1f = self.T1.to_flat()
        self.Tt1 = rand((bond,)*4, pattern="++--", dq=U11(0)).to_flat()
        self.Tt2 = rand((bond,)*4, pattern="++--", dq=U11(1,1)).to_flat()
    
    def test_trace(self):
        out1 = self.Tt1.trace((0,1), (3,2))
        out2 = self.Tt1.trace(0,3).trace(0,1)
        out3 = self.Tt1.trace(1,2).trace(0,1)
        self.assertAlmostEqual(out1, out2, 8)
        self.assertAlmostEqual(out1, out3, 8)
        
        out1 = self.Tt2.trace((0,1),(3,2))
        out2 = self.Tt2.trace(0,3).trace(0,1)
        out3 = self.Tt2.trace(1,2).trace(0,1)
        
        assert out1==0
        assert out2==0
        assert out3==0
    
    def test_skeleton(self):
        bond = self.bond
        dq0 = self.skeleton_test_dq
        T = rand((bond,)*4, pattern="++--", dq=dq0).to_flat()
        dqdelta = self.symmetry._compute(T.pattern, T.q_labels) - dq0.to_flat()
        assert np.amax(dqdelta) == 0

    def test_local_flip(self):
        axes = [0,3]
        new_T = self.T.copy()
        new_T._local_flip(axes)
        Tf = self.Tf.copy()
        Tf._local_flip(axes)
        self.assertAlmostEqual((Tf.to_sparse()-new_T).norm(), 0, 8)

    def test_global_flip(self):
        new_Tf = self.Tf.copy()
        new_Tf._global_flip()
        if fermion_setting.DEFAULT_FERMION:
            self.assertAlmostEqual((self.Tf+new_Tf).norm(), 0, 8)
        else:
            self.assertAlmostEqual((self.Tf-new_Tf).norm(), 0, 8)

    def test_transpose(self):
        new_order = (1,3,2,0)
        new_T = self.T.transpose(new_order)
        new_Tf = self.Tf.transpose(new_order)
        self.assertAlmostEqual((new_T.to_flat()-new_Tf).norm(), 0, 8)

    def test_tensor_contract(self):
        bond = self.bond
        dqa, dqb = self.contract_test_dq
        Ta = rand((bond,)*4, pattern="++--", dq=dqa)
        Tb = rand((bond,)*3, pattern="++-", dq=dqb)
        out = np.tensordot(Ta, Tb, axes=((2,3),(1,0)))
        outf = np.tensordot(Ta.to_flat(), Tb.to_flat(), axes=((2,3),(1,0)))
        self.assertAlmostEqual((out.to_flat()-outf).norm(), 0., 8)

        out = np.tensordot(Ta, Tb, axes=((2,3),(0,1)))
        outf = np.tensordot(Ta.to_flat(), Tb.to_flat(), axes=((2,3),(0,1)))
        self.assertAlmostEqual((out.to_flat()-outf).norm(), 0., 8)
        Ta1 = Ta.transpose([0,1,3,2])
        out1 = np.tensordot(Ta1, Tb, axes=((2,3),(1,0)))
        self.assertAlmostEqual((out-out1).norm(), 0., 8)

    def test_tensor_svd(self):
        Tf = self.Tf
        left_idx = [3,1]
        for dq in self.svd_dq_iterator:
            qpn_partition = (dq, Tf.dq-dq)
            for absorb in [0,1,-1,None]:
                u, s, v = Tf.tensor_svd(left_idx=left_idx, absorb=absorb, qpn_partition=qpn_partition)
                if s is None:
                    out = np.tensordot(u, v, axes=((-1,),(0,)))
                else:
                    out = np.tensordot(u, s, axes=((-1,),(0,)))
                    out = np.tensordot(out, v, axes=((-1,),(0,)))
                out = out.transpose((2,1,3,0))
                self.assertAlmostEqual((out-Tf).norm(), 0, 8)
                assert (u.dq==dq) and (v.dq==Tf.dq-dq)

    def test_qr(self):
        bond = self.bond
        left_idx=[3,1]
        for dq in self.svd_dq_iterator:
            T = rand((bond,)*4, pattern="++--", dq=dq).to_flat()
            q, r = T.tensor_qr(left_idx=left_idx, mod="qr")
            self.assertTrue(q.dq==dq)
            self.assertTrue(r.dq==dq.__class__(0))
            out = np.tensordot(q, r, axes=((-1,),(0,))).transpose((2,1,3,0))
            self.assertAlmostEqual((out-T).norm(), 0, 8)
            l, q = T.tensor_qr(left_idx=left_idx, mod="lq")
            self.assertTrue(q.dq==dq)
            self.assertTrue(l.dq==dq.__class__(0))
            out = np.tensordot(l, q, axes=((-1,),(0,))).transpose((2,1,3,0))
            self.assertAlmostEqual((out-T).norm(), 0, 8)

class TestZ22(TestU11):
    def setUp(self):
        bond = BondInfo({Z22(0):2, Z22(0,1):2, Z22(1,0):3, Z22(1,1):5})
        self.bond = bond
        self.T = rand((bond,)*4, pattern="++--", dq=Z22(0))
        self.Tf = self.T.to_flat()
        self.symmetry = Z22
        self.skeleton_test_dq = Z22(1,0)
        self.contract_test_dq = (Z22(1,0),Z22(0,1))
        self.svd_dq_iterator = [Z22(0), Z22(0,1), Z22(1,0),Z22(1,1)]
        self.shift = Z22(1,1)
        self.T1 = rand((bond,)*4, pattern="++--", dq=Z22(0,1))
        self.T1f = self.T1.to_flat()
        self.Tt1 = rand((bond,)*4, pattern="++--", dq=Z22(0)).to_flat()
        self.Tt2 = rand((bond,)*4, pattern="++--", dq=Z22(1,0)).to_flat()

class TestU1(TestU11):
    def setUp(self):
        bond = BondInfo({U1(0):2, U1(1):1, U1(2):4, U1(3):5})
        self.bond = bond
        self.T = rand((bond,)*4, pattern="++--", dq=U1(1))
        self.Tf = self.T.to_flat()
        self.symmetry = U1
        self.skeleton_test_dq = U1(2)
        self.contract_test_dq = (U1(1),U1(3))
        self.svd_dq_iterator = [U1(0), U1(1), U1(2), U1(3)]
        self.shift = U1(1)
        self.T1 = rand((bond,)*4, pattern="++--", dq=U1(2))
        self.T1f = self.T1.to_flat()
        self.Tt1 = rand((bond,)*4, pattern="++--", dq=U1(0)).to_flat()
        self.Tt2 = rand((bond,)*4, pattern="++--", dq=U1(2)).to_flat()

class TestZ4(TestU11):
    def setUp(self):
        bond = BondInfo({Z4(0):2, Z4(1):5, Z4(2):4, Z4(3):5})
        self.bond = bond
        self.T = rand((bond,)*4, pattern="++--", dq=Z4(1))
        self.Tf = self.T.to_flat()
        self.symmetry = Z4
        self.skeleton_test_dq = Z4(2)
        self.contract_test_dq = (Z4(1),Z4(3))
        self.svd_dq_iterator = [Z4(0), Z4(1), Z4(2), Z4(3)]
        self.shift = Z4(1)
        self.T1 = rand((bond,)*4, pattern="++++", dq=Z4(2))
        self.T1f = self.T1.to_flat()
        self.Tt1 = rand((bond,)*4, pattern="++--", dq=Z4(0)).to_flat()
        self.Tt2 = rand((bond,)*4, pattern="++--", dq=Z4(2)).to_flat()

class TestZ2(TestU11):
    def setUp(self):
        bond = BondInfo({Z2(0):5, Z2(1):7})
        self.bond = bond
        self.T = rand((bond,)*4, pattern="++--", dq=Z2(1))
        self.Tf = self.T.to_flat()
        self.symmetry = Z2
        self.skeleton_test_dq = Z2(1)
        self.contract_test_dq = (Z2(0),Z2(1))
        self.svd_dq_iterator = [Z2(0), Z2(1)]
        self.shift = Z2(0)
        self.T1 = rand((bond,)*4, pattern="++--", dq=Z2(0))
        self.T1f = self.T1.to_flat()
        self.Tt1 = rand((bond,)*4, pattern="++--", dq=Z2(0)).to_flat()
        self.Tt2 = rand((bond,)*4, pattern="++--", dq=Z2(1)).to_flat()

if __name__ == "__main__":
    print("Full Tests for fermion CPP backend")
    unittest.main()
