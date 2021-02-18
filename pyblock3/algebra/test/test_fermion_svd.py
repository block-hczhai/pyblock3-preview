import unittest
import numpy as np
from pyblock3.algebra.symmetry import QPN, BondInfo
from pyblock3.algebra.fermion import SparseFermionTensor
from pyblock3.algebra.fermion_split import _run_sparse_fermion_svd

np.random.seed(5)
q0 = QPN(0,0)
q1 = QPN(1,1)
q2 = QPN(1,-1)
q3 = QPN(2,0)

infox = BondInfo({q0: 2, q1: 3,
                  q2: 3, q3: 2})
infoy = BondInfo({q0: 3, q1: 2,
                  q2: 2, q3: 5})

T = SparseFermionTensor.random((infox,infox,infox,infox), pattern="++--", dq=QPN(2,0))

u,s,v=_run_sparse_fermion_svd(T, [0,1], absorb=0)
out = np.tensordot(u, v, axes=((-1,),(0,)))
err = out - T
print(err.norm())
print(u.dq, v.dq, u.pattern, v.pattern, out.dq)

u,s,v = _run_sparse_fermion_svd(T, [0,2], absorb=1)
out = np.tensordot(u, v, axes=((-1,),(0,))).transpose([0,2,1,3])
err = out - T
print(err.norm())
print(u.dq, v.dq, u.pattern, v.pattern, out.dq)

u,s,v = _run_sparse_fermion_svd(T, [0,2], absorb=-1)
out = np.tensordot(u, v, axes=((-1,),(0,))).transpose([0,2,1,3])
err = out - T
print(err.norm())
print(u.dq, v.dq, u.pattern, v.pattern, out.dq)

u,s,v = _run_sparse_fermion_svd(T, [0,1], absorb=1, qpn_info=(QPN(1,1), QPN(1,-1)))
out = np.tensordot(u, v, axes=((-1,),(0,)))
err = out - T
print(err.norm())
print(u.dq, v.dq, u.pattern, v.pattern, out.dq)

u,s,v = _run_sparse_fermion_svd(T, [0,2], absorb=-1, qpn_info=(QPN(2,0),QPN(0,0)))
out = np.tensordot(u, v, axes=((-1,),(0,))).transpose([0,2,1,3])
err = out - T
print(err.norm())
print(u.dq, v.dq, u.pattern, v.pattern, out.dq)

if __name__ == "__main__":
    print("Full Tests for Fermionic Tensor SVD")
    unittest.main()
