from pyblock3.algebra.symmetry import SZ, BondInfo
from pyblock3.algebra.core import SparseTensor
from pyblock3.algebra.fermion import FermionSparseTensor, FlatFermionTensor
import numpy as np
np.random.seed(3)

x = SZ(0,0,0)
y = SZ(1,0,0)
infox = BondInfo({x:3, y: 2})
infoy = BondInfo({x:2, y: 3})

a = FermionSparseTensor.random((infox,infoy,infox), dq=y)
b = a.transpose((2,0,1))
for ibk, block in enumerate(a):
    adat = np.asarray(block)
    bdat = np.asarray(b[ibk])
    print(block.q_labels, adat.sum(), bdat.sum())
