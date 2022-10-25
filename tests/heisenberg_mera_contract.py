
import quimb.tensor as qtn
import numpy as np
from pyblock3.heisenberg import Heisenberg

import pyblock3.algebra.ad
pyblock3.algebra.ad.ENABLE_JAX = False
pyblock3.algebra.ad.ENABLE_EINSUM = True

from pyblock3.algebra.ad import core
core.ENABLE_FUSED_IMPLS = False

from pyblock3.algebra.ad.core import SparseTensor

import autoray as ar

SparseTensor.shape = property(lambda x: tuple(ix.n_bonds for ix in x.infos))
ar.register_function('pyblock3', 'array', lambda x: x)
ar.register_function('pyblock3', 'to_numpy', lambda x: x)

L = 8
D = 20

np.random.seed(1234)

topology = [(i, i + 1, 1.0) for i in range(L - 1)]
hamil = Heisenberg(twos=1, n_sites=L, topology=topology, flat=False)

class MERAState:

    @staticmethod
    def get_disentanglers_and_isometries(basis, target, low=0, high=1, dtype=float, max_bond=None):
        tensors_uni = []
        tensors_iso = []
        while len(basis) > 1:
            n_sites = len(basis) // 2
            uni_basis = []
            for i in range(n_sites):
                info_high = basis[i * 2] | basis[i * 2 + 1]
                if max_bond is not None:
                    info_high.truncate_no_keep(max_bond)
                ts = SparseTensor.random(
                    (basis[i * 2], basis[i * 2 + 1], info_high, info_high),
                    dtype=dtype, pattern="++--") * (high - low) + low
                tensors_uni.append(ts)
                uni_basis.append(info_high)

            iso_basis = []
            for i in range(n_sites):
                info_high = uni_basis[i] * uni_basis[(i + 1) % n_sites]
                if max_bond is not None:
                    info_high.truncate_no_keep(max_bond)
                if n_sites == 1:
                    info_high = info_high.filter(info_high.__class__({target : 1}))
                ts = SparseTensor.random(
                    (uni_basis[i], uni_basis[(i + 1) % n_sites], info_high),
                    dtype=dtype, pattern="++-") * (high - low) + low
                tensors_iso.append(ts)
                iso_basis.append(info_high)

            basis = iso_basis

        return tensors_uni, tensors_iso

    @staticmethod
    def unitize_(mera):
        for t in mera:
            if t.data.ndim == 3:
                new_data = t.data.left_canonicalize()[0]
            elif t.data.ndim == 4:
                fuse_info = t.data.kron_sum_info(2, 3, pattern=t.data.pattern[2:])
                new_data = t.data.fuse(2, 3, info=fuse_info).left_canonicalize()[0].unfuse(2, info=fuse_info)
                new_data.pattern = t.data.pattern
            else:
                assert False
            t.modify(data=new_data)

    @staticmethod
    def sp_operator(tj, basis, isite, dtype=float):
        Q = basis[isite].symm_class
        repr = SparseTensor.zeros(bond_infos=(basis[isite], basis[isite]), dtype=dtype, dq=Q(0, 2, 0), pattern="+-")
        for tm in range(-tj, tj - 1, 2):
            repr[Q(0, tm, 0)].data[0, 0] = np.sqrt((tj - tm) * (tj + tm + 2) // 4)
        return repr

    @staticmethod
    def sm_operator(tj, basis, isite, dtype=float):
        Q = basis[isite].symm_class
        repr = SparseTensor.zeros(bond_infos=(basis[isite], basis[isite]), dtype=dtype, dq=Q(0, -2, 0), pattern="+-")
        for tm in range(-tj + 2, tj + 1, 2):
            repr[Q(0, tm, 0)].data[0, 0] = np.sqrt((tj + tm) * (tj - tm + 2) // 4)
        return repr

    @staticmethod
    def sz_operator(tj, basis, isite, dtype=float):
        Q = basis[isite].symm_class
        repr = SparseTensor.zeros(bond_infos=(basis[isite], basis[isite]), dtype=dtype, dq=Q(0, 0, 0), pattern="+-")
        for tm in range(-tj, tj + 1, 2):
            repr[Q(0, tm, 0)].data[0, 0] = tm / 2.0
        return repr

    @staticmethod
    def get_hamil_terms(tj, basis, topology, dtype=float):
        terms = []
        sp_op = [None] * len(basis)
        sm_op = [None] * len(basis)
        sz_op = [None] * len(basis)
        for m in range(len(basis)):
            sp_op[m] = MERAState.sp_operator(tj, basis, m, dtype=dtype)
            sm_op[m] = MERAState.sm_operator(tj, basis, m, dtype=dtype)
            sz_op[m] = MERAState.sz_operator(tj, basis, m, dtype=dtype)

        for (i, j, v) in topology:
            assert i != j
            ts = sum([
                np.tensordot(0.5 * v * sp_op[i], sm_op[j], axes=([], [])).transpose((1, 3, 0, 2)),
                np.tensordot(0.5 * v * sm_op[i], sp_op[j], axes=([], [])).transpose((1, 3, 0, 2)),
                np.tensordot(v * sz_op[i], sz_op[j], axes=([], [])).transpose((1, 3, 0, 2)),
                # np.einsum('ij,kl->jlik', 0.5 * v * sp_op[i], sm_op[j]),
                # np.einsum('ij,kl->jlik', 0.5 * v * sm_op[i], sp_op[j]),
                # np.einsum('ij,kl->jlik', v * sz_op[i], sz_op[j])
            ])
            terms.append(((i, j), ts))

        return terms

uni, iso = MERAState.get_disentanglers_and_isometries(hamil.basis, hamil.target, max_bond=D)

mera = qtn.MERA(L, uni, iso, dangle=True)
MERAState.unitize_(mera)

h_terms = MERAState.get_hamil_terms(hamil.twos, hamil.basis, topology)
terms = { k : v for k, v in h_terms }

def norm_fn(mera):
    mera = mera.copy()
    MERAState.unitize_(mera)
    return mera

def local_expectation(mera, terms, where, optimize='auto-hq'):
    tags = [mera.site_tag(coo) for coo in where]
    mera_ij = mera.select(tags, 'any')
    mera_ij_G = mera_ij.gate(terms[where], where)
    mera_ij_ex = (mera_ij_G & mera_ij.H)
    return mera_ij_ex.contract(all, optimize=optimize)

def loss_fn(mera, terms, **kwargs):
    r = sum(
        local_expectation(mera, terms, where, **kwargs)
        for where in terms
    )
    return r

print(loss_fn(norm_fn(mera), terms, optimize='auto-hq'))
