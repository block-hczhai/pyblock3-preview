import numpy as np
from itertools import product
from pyblock3.algebra.core import SubTensor
from pyblock3.algebra.fermion import (SparseFermionTensor, FlatFermionTensor)
from pyblock3.algebra.symmetry import SZ, BondInfo

eye = FlatFermionTensor.eye

def gen_h1(h=1.):
    blocks= []
    for i, j in product(range(2), repeat=2):
        qlab = (SZ(i), SZ(j), SZ(1-i), SZ(1-j))
        qlst = [q.n for q in qlab]
        iblk = np.zeros([2,2,2,2])
        blocks.append(SubTensor(reduced=np.zeros([2,2,2,2]), q_labels=(SZ(i), SZ(j), SZ(i), SZ(j))))
        if (i+j)==1:
            iblk[0,0,0,0] = iblk[i,j,j,i] = h
            iblk[1,1,1,1] = iblk[j,i,i,j] = -h
        else:
            if i == 0:
                iblk[0,1,0,1] = iblk[1,0,0,1] = h
                iblk[1,0,1,0] = iblk[0,1,1,0] = -h
            else:
                iblk[0,1,0,1] = iblk[0,1,1,0] = -h
                iblk[1,0,1,0] = iblk[1,0,0,1] = h
        blocks.append(SubTensor(reduced=iblk, q_labels=qlab))
    hop = SparseFermionTensor(blocks=blocks).to_flat()
    return hop

hopping = lambda t=1.0: gen_h1(-t)

def onsite_u(u=1):
    umat0 = np.zeros([2,2])
    umat0[1,1] = u
    umat1 = np.zeros([2,2])
    blocks = [SubTensor(reduced=umat0, q_labels=(SZ(0), SZ(0))),
              SubTensor(reduced=umat1, q_labels=(SZ(1), SZ(1)))]
    umat = SparseFermionTensor(blocks=blocks).to_flat()
    return umat

def hubbard(t, u, fac=None):
    if fac is None:
        fac = (1, 1)
    faca, facb = fac
    ham = hopping(t).to_sparse()
    for iblk in ham:
        qin, qout = iblk.q_labels[:2], iblk.q_labels[2:]
        if qin != qout: continue
        in_pair = [iq.n for iq in qin]
        if in_pair == [0,0]:
            iblk[1,0,1,0] += faca * u
            iblk[0,1,0,1] += facb * u
            iblk[1,1,1,1] += (faca + facb) * u
        elif in_pair == [0,1]:
            iblk[1,:,1,:] += faca * u * np.eye(2)
        elif in_pair == [1,0]:
            iblk[:,1,:,1] += facb * u * np.eye(2)
    return ham.to_flat()

def count_n():
    nmat0 = np.zeros([2,2])
    nmat0[1,1] = 2
    nmat1 = np.eye(2)
    blocks = [SubTensor(reduced=nmat0, q_labels=(SZ(0), SZ(0))),
              SubTensor(reduced=nmat1, q_labels=(SZ(1), SZ(1)))]
    nmat = SparseFermionTensor(blocks=blocks).to_flat()
    return nmat

def measure_sz():
    zmat0 = np.zeros([2,2])
    zmat1 = np.eye(2) * .5
    zmat1[1,1]= -.5
    blocks = [SubTensor(reduced=zmat0, q_labels=(SZ(0), SZ(0))),
              SubTensor(reduced=zmat1, q_labels=(SZ(1), SZ(1)))]
    smat = SparseFermionTensor(blocks=blocks).to_flat()
    return smat