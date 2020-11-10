import unittest
import numpy as np
from pyblock3.algebra.symmetry import SZ, BondInfo
from pyblock3.algebra.fermion import SparseFermionTensor, FlatFermionTensor

def finger(tensor):
    if isinstance(tensor, SparseFermionTensor):
        array = FlatFermionTensor.from_sparse(tensor).data
    else:
        array = tensor.data
    return np.dot(np.cos(np.arange(array.size)), array)

def compare_equal(stensor, ftensor):
    nblk, ndim = ftensor.q_labels.shape
    equal_list = []
    for ibk, sblk in enumerate(stensor):
        num = np.asarray([SZ.to_flat(q) for q in sblk.q_labels])[None,]
        delta = abs(num -ftensor.q_labels).sum(axis=1)
        idx = np.where(delta==0)[0][0]
        sarr = np.asarray(sblk).flatten()
        farr = ftensor.data[ftensor.idxs[idx]:ftensor.idxs[idx+1]]
        equal_list.append(np.allclose(sarr, farr))
    return sum(equal_list)==nblk

np.random.seed(3)
x = SZ(0,0,0)
y = SZ(1,0,0)
infox = BondInfo({x:3, y: 2})
infoy = BondInfo({x:2, y: 3})

asp = SparseFermionTensor.random((infox,infoy,infox), dq=y)
bsp = SparseFermionTensor.random((infox,infox,infoy))

af = FlatFermionTensor.from_sparse(asp)
bf = FlatFermionTensor.from_sparse(bsp)

assert(compare_equal(asp, af))
assert(compare_equal(bsp, bf))


class KnownValues(unittest.TestCase):
    def test_flip(self):
        aspx = asp.copy()
        afx = af.copy()

        afx._global_flip()
        aspx._global_flip()
        self.assertTrue(compare_equal(aspx, afx))
        self.assertAlmostEqual(finger(afx)+finger(af), 0, 6)

        afx._local_flip((0,1))
        aspx._local_flip((0,1))
        self.assertTrue(compare_equal(aspx, afx))
        self.assertAlmostEqual(finger(afx), 1.8267976468986804, 6)

    def test_transpose(self):
        axes = (0,2,1)
        aspx = asp.transpose(axes)
        afx = af.transpose(axes)
        self.assertTrue(compare_equal(aspx, afx))

        bspx = bsp.transpose((2,1,0))
        self.assertAlmostEqual(finger(bspx), 0.512065055131486, 6)

    def test_tensordot(self):
        outsp = np.tensordot(asp, bsp, ((0,1), (1,2)))
        outf = np.tensordot(af, bf, ((0,1), (1,2)))

        atmp_sp = asp.transpose((2,1,0))
        btmp_sp = bsp.transpose((1,2,0))
        outtmp_sp = np.tensordot(atmp_sp, btmp_sp, ((2,1), (0,1)))

        atmp_f = af.transpose((2,1,0))
        btmp_f = bf.transpose((1,2,0))
        outtmp_f = np.tensordot(atmp_f, btmp_f, ((2,1), (0,1)))

        self.assertTrue(compare_equal(outsp, outf))
        self.assertTrue(compare_equal(outtmp_sp, outf))
        self.assertTrue(compare_equal(outsp, outtmp_f))

        outsp = np.tensordot(asp, bsp, ((1,), (2,)))
        outf = np.tensordot(af, bf, ((1,), (2,)))

        atmp_sp = asp.transpose((0,2,1))
        btmp_sp = bsp.transpose((2,0,1))
        outtmp_sp = np.tensordot(atmp_sp, btmp_sp, ((2,), (0,)))

        atmp_f = af.transpose((0,2,1))
        btmp_f = bf.transpose((2,0,1))
        outtmp_f = np.tensordot(atmp_f, btmp_f, ((2,), (0,)))

        self.assertTrue(compare_equal(outsp, outf))
        self.assertTrue(compare_equal(outtmp_sp, outf))
        self.assertTrue(compare_equal(outsp, outtmp_f))
        self.assertAlmostEqual(finger(outsp), 0.09654888859793742, 6)

if __name__ == "__main__":
    print("Full Tests for Fermionic Tensors")
    unittest.main()