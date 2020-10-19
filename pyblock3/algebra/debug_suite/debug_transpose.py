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
af = FlatFermionTensor.from_sparse(a)

def compute_phase_flat(q_lab_flat, axes):
    nlist = [SZ.from_flat(xq).n for xq in q_lab_flat]
    counted = []
    phase = 1
    for x in axes:
        parity = sum([nlist[i] for i in range(x) if i not in counted]) * nlist[x]
        phase *= (-1) ** parity
        counted.append(x)
    return phase

def transpose_flat(arr, axes):
    nblk, ndim = arr.shapes.shape
    dat=[]
    for ibk in range(nblk):
        phase = compute_phase_flat(af.q_labels[ibk], axes)
        tmp = arr.data[arr.idxs[ibk]:arr.idxs[ibk+1]].reshape(arr.shapes[ibk])
        dat.extend(tmp.transpose(axes).flatten()*phase)
    return np.asarray(dat)

axes = (2,1,0)
b = a.transpose(axes)
for ibk, block in enumerate(a):
    adat = np.asarray(block)
    bdat = np.asarray(b[ibk])
    #print(block.q_labels, adat.sum(), bdat.sum())
bf = FlatFermionTensor.from_sparse(b)
out = transpose_flat(af, axes)
print(bf.data-out)

from block3.flat_fermion_tensor import transpose

data = np.zeros_like(out)
transpose(af.q_labels, af.shapes, af.data, af.idxs, axes, data)
print(data-out)
