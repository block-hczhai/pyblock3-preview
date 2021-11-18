import numpy as np
import unittest
from pyblock3.algebra.fermion import SparseFermionTensor
from pyblock3.algebra.fermion_symmetry import U11, U1, Z4, Z2, Z22
from pyblock3.algebra.symmetry import BondInfo, SZ
from pyblock3.algebra.core import SparseTensor, SubTensor
from pyblock3.algebra import fermion_setting

fermion_setting.set_options(flat=False, fermion=True)
rand = SparseFermionTensor.random

def _get_phase(q_label):
    if fermion_setting.DEFAULT_FERMION and q_label.parity==1:
        return -1
    return 1

class TestU11(unittest.TestCase):
    def setUp(self):
        bond = BondInfo({U11(0):2, U11(1,1):3, U11(1,-1):4})
        self.bond = bond
        self.T = rand((bond,)*4, pattern="++--", dq=U11(0))
        self.symmetry = U11
        self.skeleton_test_dq = U11(2)
        self.contract_test_dq = (U11(1,1),U11(1,-1))
        self.svd_dq_iterator = [U11(0), U11(1,1), U11(1,-1),U11(2)]
        self.shift = U11(1,1)
        self.T1 = rand((bond,)*4, pattern="++--", dq=U11(2))
        self.Tt1 = rand((bond,)*4, pattern="++--", dq=U11(0))
        self.Tt2 = rand((bond,)*4, pattern="++--", dq=U11(1,1))

    def test_trace(self):
        out1 = self.Tt1.trace((0,1), (3,2))
        out2 = self.Tt1.trace(0,3).trace(0,1)
        out3 = self.Tt1.trace(1,2).trace(0,1)
        self.assertAlmostEqual(out1, out2, 8)
        self.assertAlmostEqual(out1, out3, 8)

        out1 = self.Tt2.trace((0,1), (3,2))
        out2 = self.Tt2.trace(0,3).trace(0,1)
        out3 = self.Tt2.trace(1,2).trace(0,1)
        assert out1==0
        assert out2==0
        assert out3==0

    def test_skeleton(self):
        bond = self.bond
        dq0 = self.skeleton_test_dq
        T = rand((bond,)*4, pattern="++--", dq=dq0)
        for iblk in T.blocks:
            dq = self.symmetry._compute(T.pattern, iblk.q_labels)
            assert dq == dq0

    def test_local_flip(self):
        T = self.T
        axes = [0,3]
        blocks = []
        new_T = T.copy()
        new_T._local_flip(axes)
        pattern = "".join([T.pattern[ix] for ix in axes])
        for iblk, iblk1 in zip(T.blocks, new_T.blocks):
            dq = self.symmetry._compute(pattern, [iblk.q_labels[ix] for ix in axes])
            phase = _get_phase(dq)
            blocks.append(iblk * phase)
        new_T2 = T.__class__(blocks=blocks, pattern=T.pattern)
        self.assertAlmostEqual((new_T-new_T2).norm(), 0, 8)

    def test_global_flip(self):
        new_T = self.T.copy()
        new_T._global_flip()
        self.assertAlmostEqual((self.T+new_T).norm(), 0, 8)

    def test_transpose(self):
        new_order = (1,3,2,0)
        new_T = self.T.transpose(new_order)
        blocks = []
        for iblk in self.T.blocks:
            q_labels = iblk.q_labels
            finished = []
            qphase = 0
            if fermion_setting.DEFAULT_FERMION:
                for ix in new_order:
                    passed = [ax for ax in range(ix) if ax not in finished]
                    if len(passed) != 0:
                        qix = q_labels[ix].parity
                        qpassed = np.sum([q_labels[ax] for ax in passed]).parity
                        qphase += qix * qpassed
                    finished.append(ix)
                qphase = qphase % 2
            if qphase==1:
                blocks.append(-iblk.transpose(new_order))
            else:
                blocks.append(iblk.transpose(new_order))
        new_T2 = new_T.__class__(blocks=blocks, pattern=new_T.pattern)
        self.assertAlmostEqual((new_T-new_T2).norm(), 0, 8)

    def test_tensor_contract(self):
        bond = self.bond
        dqa, dqb = self.contract_test_dq
        Ta = rand((bond,)*4, pattern="++--", dq=dqa)
        Tb = rand((bond,)*3, pattern="++-", dq=dqb)
        out = np.tensordot(Ta, Tb, axes=((2,3),(1,0)))
        blocksa = []
        blocksb = []
        for iblk in Ta.blocks:
            if hasattr(iblk.q_labels[0], 'sz'):
                q_labels = [SZ(iq.n, iq.sz) for iq in iblk.q_labels]
            else:
                q_labels = [SZ(iq.n) for iq in iblk.q_labels]
            blocksa.append(SubTensor(reduced=np.asarray(iblk), q_labels=q_labels))
        for iblk in Tb.blocks:
            if hasattr(iblk.q_labels[0], 'sz'):
                q_labels = [SZ(iq.n, iq.sz) for iq in iblk.q_labels]
            else:
                q_labels = [SZ(iq.n) for iq in iblk.q_labels]
            blocksb.append(SubTensor(reduced=np.asarray(iblk), q_labels=q_labels))
        Xa = SparseTensor(blocks=blocksa)
        Xb = SparseTensor(blocks=blocksb)
        ref = np.tensordot(Xa, Xb, axes=((2,3), (1,0)))

        for iblk1, iblk2 in zip(out.blocks, ref.blocks):
            err = np.asarray(iblk1) - np.asarray(iblk2)
            self.assertAlmostEqual(np.amax(err), 0., 8)

        out = np.tensordot(Ta, Tb, axes=((2,3),(0,1)))
        Ta1 = Ta.transpose([0,1,3,2])
        out1 = np.tensordot(Ta1, Tb, axes=((2,3),(1,0)))
        self.assertAlmostEqual((out-out1).norm(), 0., 8)

    def test_tensor_svd(self):
        T = self.T
        left_idx = [3,1]
        for dq in self.svd_dq_iterator:
            qpn_partition = (dq, T.dq-dq)
            for absorb in [0,1,-1,None]:
                u, s, v = T.tensor_svd(left_idx=left_idx, absorb=absorb, qpn_partition=qpn_partition)
                if s is None:
                    out = np.tensordot(u, v, axes=((-1,),(0,)))
                else:
                    out = np.tensordot(u, s, axes=((-1,),(0,)))
                    out = np.tensordot(out, v, axes=((-1,),(0,)))
                out = out.transpose((2,1,3,0))
                self.assertAlmostEqual((out-T).norm(), 0, 8)
                assert (u.dq==dq) and (v.dq==T.dq-dq)

    def test_qr(self):
        bond = self.bond
        left_idx=[3,1]
        for dq in self.svd_dq_iterator:
            T = rand((bond,)*4, pattern="++--", dq=dq)
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
        bond = BondInfo({Z22(0):2, Z22(0,1):3, Z22(1,0):4, Z22(1,1):5})
        self.bond = bond
        self.T = rand((bond,)*4, pattern="++--", dq=Z22(0))
        self.symmetry = Z22
        self.skeleton_test_dq = Z22(1,0)
        self.contract_test_dq = (Z22(1,0),Z22(0,1))
        self.svd_dq_iterator = [Z22(0), Z22(0,1), Z22(1,0),Z22(1,1)]
        self.shift = Z22(1,1)
        self.T1 = rand((bond,)*4, pattern="++--", dq=Z22(0,1))
        self.Tt1 = rand((bond,)*4, pattern="++--", dq=Z22(0))
        self.Tt2 = rand((bond,)*4, pattern="++--", dq=Z22(1,0))


class TestU1(TestU11):
    def setUp(self):
        bond = BondInfo({U1(0):2, U1(1):1, U1(2):4, U1(3):5})
        self.bond = bond
        self.T = rand((bond,)*4, pattern="++--", dq=U1(1))
        self.symmetry = U1
        self.skeleton_test_dq = U1(2)
        self.contract_test_dq = (U1(1),U1(3))
        self.svd_dq_iterator = [U1(0), U1(1), U1(2), U1(3)]
        self.shift = U1(1)
        self.T1 = rand((bond,)*4, pattern="++--", dq=U1(2))
        self.Tt1 = rand((bond,)*4, pattern="++--", dq=U1(0))
        self.Tt2 = rand((bond,)*4, pattern="++--", dq=U1(1))

class TestZ4(TestU11):
    def setUp(self):
        bond = BondInfo({Z4(0):2, Z4(1):5, Z4(2):4, Z4(3):5})
        self.bond = bond
        self.T = rand((bond,)*4, pattern="++--", dq=Z4(1))
        self.symmetry = Z4
        self.skeleton_test_dq = Z4(2)
        self.contract_test_dq = (Z4(1),Z4(3))
        self.svd_dq_iterator = [Z4(0), Z4(1), Z4(2), Z4(3)]
        self.shift = Z4(1)
        self.T1 = rand((bond,)*4, pattern="++--", dq=Z4(2))
        self.Tt1 = rand((bond,)*4, pattern="++--", dq=Z4(0))
        self.Tt2 = rand((bond,)*4, pattern="++--", dq=Z4(2))

class TestZ2(TestU11):
    def setUp(self):
        bond = BondInfo({Z2(0):5, Z2(1):7})
        self.bond = bond
        self.T = rand((bond,)*4, pattern="++--", dq=Z2(1))
        self.symmetry = Z2
        self.skeleton_test_dq = Z2(1)
        self.contract_test_dq = (Z2(0),Z2(1))
        self.svd_dq_iterator = [Z2(0), Z2(1)]
        self.shift = Z2(0)
        self.T1 = rand((bond,)*4, pattern="++--", dq=Z2(0))
        self.Tt1 = rand((bond,)*4, pattern="++--", dq=Z2(0))
        self.Tt2 = rand((bond,)*4, pattern="++--", dq=Z2(1))

if __name__ == "__main__":
    print("Full Tests for fermion python backend")
    unittest.main()
