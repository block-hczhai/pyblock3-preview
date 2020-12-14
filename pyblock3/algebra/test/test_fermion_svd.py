import unittest
import numpy as np
from pyblock3.algebra.symmetry import SZ, BondInfo
from pyblock3.algebra.fermion import (SparseFermionTensor,
                                      FlatFermionTensor,
                                      _run_sparse_fermion_svd,
                                      _run_flat_fermion_svd)

np.random.seed(3)
sx = SZ(0,0,0)
sy = SZ(1,0,0)
infox = BondInfo({sx:8, sy: 11})
tsr = SparseFermionTensor.random((infox,infox,infox,infox,infox), dq=sy)
tsr_f = FlatFermionTensor.from_sparse(tsr)

class KnownValues(unittest.TestCase):
    def test_svd(self):
        u, s, v = _run_sparse_fermion_svd(tsr, ax=2, absorb=None)
        us, ss, vs = _run_flat_fermion_svd(tsr_f, ax=2, absorb=None)
        out = np.tensordot(u,s,axes=((2,),(0,)))
        out = np.tensordot(out,v,axes=((2,),(0,)))
        outs = np.tensordot(us,ss,axes=((2,),(0,)))
        outs = np.tensordot(outs,vs,axes=((2,),(0,)))

        delta = out-tsr
        err = 0.
        for blk in delta:
            err += abs(np.asarray(blk)).sum()

        errs = np.amax((outs - tsr_f).data)
        self.assertAlmostEqual(err, 0., 8)
        self.assertAlmostEqual(errs, 0., 8)

    def test_svd_absorb(self):
        # absorb to U
        u, _, v = _run_sparse_fermion_svd(tsr, ax=2, absorb=-1)
        us, _, vs = _run_flat_fermion_svd(tsr_f, ax=2, absorb=-1)
        out = np.tensordot(u,v,axes=((2,),(0,)))
        outs = np.tensordot(us,vs,axes=((2,),(0,)))

        delta = out-tsr
        err = 0.
        for blk in delta:
            err += abs(np.asarray(blk)).sum()

        errs = np.amax((outs - tsr_f).data)
        self.assertAlmostEqual(err, 0., 8)
        self.assertAlmostEqual(errs, 0., 8)

        # absorb to V
        u, _, v = _run_sparse_fermion_svd(tsr, ax=2, absorb=1)
        us, _, vs = _run_flat_fermion_svd(tsr_f, ax=2, absorb=1)
        out = np.tensordot(u,v,axes=((2,),(0,)))
        outs = np.tensordot(us,vs,axes=((2,),(0,)))

        delta = out-tsr
        err = 0.
        for blk in delta:
            err += abs(np.asarray(blk)).sum()

        errs = np.amax((outs - tsr_f).data)
        self.assertAlmostEqual(err, 0., 8)
        self.assertAlmostEqual(errs, 0., 8)

        # evenly split to U, V
        u, s, v = _run_sparse_fermion_svd(tsr, ax=2)
        us, ss, vs = _run_flat_fermion_svd(tsr_f, ax=2)
        out = np.tensordot(u,v,axes=((2,),(0,)))
        outs = np.tensordot(us,vs,axes=((2,),(0,)))
        delta = out-tsr
        err = 0.
        for blk in delta:
            err += abs(np.asarray(blk)).sum()

        errs = np.amax((outs - tsr_f).data)
        self.assertAlmostEqual(err, 0., 8)
        self.assertAlmostEqual(errs, 0., 8)

    def test_trans_svd(self):
        u, s, v = tsr.tensor_svd((1,3), absorb=None)
        u1, s1, v1 = tsr_f.tensor_svd((1,3), absorb=None)

        out = np.tensordot(u,s,axes=((2,),(0,)))
        out = np.tensordot(out,v,axes=((2,),(0,)))
        out = out.transpose((2,0,3,1,4))

        out1 = np.tensordot(u1,s1,axes=((2,),(0,)))
        out1 = np.tensordot(out1,v1,axes=((2,),(0,)))
        out1 = out1.transpose((2,0,3,1,4))

        delta = out-tsr
        err = 0.
        for blk in delta:
            err += abs(np.asarray(blk)).sum()

        errs = np.amax((out1 - tsr_f).data)
        self.assertAlmostEqual(err, 0., 8)
        self.assertAlmostEqual(errs, 0., 8)

if __name__ == "__main__":
    print("Full Tests for Fermionic Tensor SVD")
    unittest.main()
