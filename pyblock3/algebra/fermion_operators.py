import numpy as np
from itertools import product
from pyblock3.algebra.core import SubTensor
from pyblock3.algebra.fermion import SparseFermionTensor
from pyblock3.algebra.symmetry import QPN, BondInfo

USE_FLAT = True

def _compute_swap_phase(*qpns):
    cre_map = {QPN(0):"", QPN(1,1):"a", QPN(1,-1):"b", QPN(2,0):"ab"}
    ann_map = {QPN(0):"", QPN(1,1):"a", QPN(1,-1):"b", QPN(2,0):"ba"}
    nops = len(qpns) //2
    cre_string = [cre_map[ikey] for ikey in qpns[:nops]]
    ann_string = [ann_map[ikey] for ikey in qpns[nops:]]
    full_string = cre_string + ann_string
    phase = 1
    for ix, ia in enumerate(cre_string):
        ib = full_string[ix+nops]
        shared = set(ia) & set(ib)
        const_off = sum(len(istring) for istring in full_string[ix+1:ix+nops])
        offset = 0
        for char_a in list(shared):
            inda = ia.index(char_a)
            indb = ib.index(char_a)
            ib = ib.replace(char_a,"")
            ia = ia.replace(char_a,"")
            offset += indb + len(ia)-inda + const_off
        phase *= (-1) ** offset
        full_string[ix] = ia
        full_string[nops+ix] = ib
    return phase

def eye(bond_info, FLAT=USE_FLAT):
    """Create tensor from BondInfo with Identity matrix."""
    blocks = []
    for sh, qs in SparseFermionTensor._skeleton((bond_info, bond_info)):
        blocks.append(SubTensor(reduced=np.eye(sh[0]), q_labels=qs))
    T = SparseFermionTensor(blocks=blocks, pattern="+-")
    if FLAT:
        return T.to_flat()
    else:
        return T

def gen_h1(h=1., FLAT=USE_FLAT):
    '''
    a_i^{\dagger}a_j
    '''
    blocks = []
    qpn_lst = [QPN(0),QPN(1,1),QPN(1,-1),QPN(2,0)]
    for q1, q2 in product(qpn_lst, repeat=2):
        for dq in [QPN(1,1),QPN(1,-1)]:
            if q1 + dq in qpn_lst and q2-dq in qpn_lst:
                phase = _compute_swap_phase(q1, q2, q1+dq, q2-dq)
                blocks.append(SubTensor(reduced=phase*np.ones([1,1,1,1])*h, q_labels=(q1, q2, q1+dq, q2-dq)))
            if q1 - dq in qpn_lst and q2+dq in qpn_lst:
                phase = _compute_swap_phase(q1, q2, q1-dq, q2+dq)
                blocks.append(SubTensor(reduced=phase*np.ones([1,1,1,1])*h, q_labels=(q1, q2, q1-dq, q2+dq)))
        blocks.append(SubTensor(reduced=np.zeros([1,1,1,1]), q_labels=(q1,q2,q1,q2)))
    T = SparseFermionTensor(blocks=blocks, pattern="++--")
    if FLAT:
        return T.to_flat()
    else:
        return T

hopping = lambda t=1.0, FLAT=USE_FLAT: gen_h1(-t, FLAT)

def onsite_u(u=1, FLAT=USE_FLAT):
    '''
    Onsite Coulomb Repulsion
    '''
    blocks = [SubTensor(reduced=np.zeros([1,1]), q_labels=(QPN(0), QPN(0))),
              SubTensor(reduced=np.zeros([1,1]), q_labels=(QPN(1,1), QPN(1,1))),
              SubTensor(reduced=np.zeros([1,1]), q_labels=(QPN(1,-1), QPN(1,-1))),
              SubTensor(reduced=np.eye(1)*u, q_labels=(QPN(2), QPN(2)))]
    T = SparseFermionTensor(blocks=blocks, pattern="+-")
    if FLAT:
        return T.to_flat()
    else:
        return T

def hubbard(t, u, mu=0, fac=None, FLAT=USE_FLAT):
    if fac is None: fac=(1,1)
    faca, facb = fac
    blocks = []
    qpn_lst = [QPN(0),QPN(1,1),QPN(1,-1),QPN(2,0)]
    for q1, q2 in product(qpn_lst, repeat=2):
        for dq in [QPN(1,1),QPN(1,-1)]:
            if q1 + dq in qpn_lst and q2-dq in qpn_lst:
                phase = _compute_swap_phase(q1, q2, q1+dq, q2-dq)
                blocks.append(SubTensor(reduced=phase*np.ones([1,1,1,1])*-t, q_labels=(q1, q2, q1+dq, q2-dq)))
            if q1 - dq in qpn_lst and q2+dq in qpn_lst:
                phase = _compute_swap_phase(q1, q2, q1-dq, q2+dq)
                blocks.append(SubTensor(reduced=phase*np.ones([1,1,1,1])*-t, q_labels=(q1, q2, q1-dq, q2+dq)))
        q_labels = (q1, q2, q1, q2)
        val = (q1==QPN(2)) * faca * u + q1.n * faca * mu +\
              (q2==QPN(2)) * facb * u + q2.n * facb * mu
        phase = _compute_swap_phase(q1, q2, q1, q2)
        blocks.append(SubTensor(reduced=phase*np.ones([1,1,1,1])*val, q_labels=q_labels))
    T = SparseFermionTensor(blocks=blocks, pattern="++--")
    if FLAT:
        return T.to_flat()
    else:
        return T

def count_n(FLAT=USE_FLAT):
    '''
    a_i^{\dagger}a_i
    '''
    blocks = [SubTensor(reduced=np.zeros([1,1]), q_labels=(QPN(0), QPN(0))),
              SubTensor(reduced=np.eye(1), q_labels=(QPN(1,1), QPN(1,1))),
              SubTensor(reduced=np.eye(1), q_labels=(QPN(1,-1), QPN(1,-1))),
              SubTensor(reduced=np.eye(1)*2, q_labels=(QPN(2), QPN(2)))]
    T = SparseFermionTensor(blocks=blocks, pattern="+-")
    if FLAT:
        return T.to_flat()
    else:
        return T

def measure_sz(FLAT=USE_FLAT):
    blocks = [SubTensor(reduced=np.zeros([1,1]), q_labels=(QPN(0), QPN(0))),
              SubTensor(reduced=np.eye(1)*.5, q_labels=(QPN(1,1), QPN(1,1))),
              SubTensor(reduced=np.eye(1)*-.5, q_labels=(QPN(1,-1), QPN(1,-1))),
              SubTensor(reduced=np.zeros([1,1]), q_labels=(QPN(2), QPN(2)))]
    T = SparseFermionTensor(blocks=blocks, pattern="+-")
    if FLAT:
        return T.to_flat()
    else:
        return T
