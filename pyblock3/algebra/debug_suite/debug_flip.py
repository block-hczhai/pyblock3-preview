from pyblock3.algebra.symmetry import SZ, BondInfo
from pyblock3.algebra.core import SparseTensor
from pyblock3.algebra.fermion import FermionSparseTensor, FlatFermionTensor
import numpy as np
np.random.seed(3)

x = SZ(0,0,0)
y = SZ(1,0,0)
infox = BondInfo({x:3, y: 2})
infoy = BondInfo({x:2, y: 3})

a = FermionSparseTensor.random((infox,infoy,infox))
b = FlatFermionTensor.from_sparse(a)


a._local_flip([1,2])
b._local_flip([1,2])

a._global_flip()
b._global_flip()
print(a.parity_per_block)
print(b.parity_per_block)
print(b.shapes, b.size)

x=b.transpose((1,0,2))
print(x.to_sparse())
print(a)

a = FermionSparseTensor.random((infox,infoy,infox))
b = FlatFermionTensor.from_sparse(a)
