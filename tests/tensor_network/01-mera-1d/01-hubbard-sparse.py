
# 1D NN Hubbard model with 4 sites
# implemented using non-fermionic sparse tensors
# E dmrg = -2.875942808996

import numpy as np
import quimb.tensor as qtn
import warnings
import torch
import time
from pyblock3.hamiltonian import Hamiltonian
from pyblock3.fcidump import FCIDUMP

torch.autograd.set_detect_anomaly(False)
torch.set_num_threads(28)

import pyblock3.algebra.ad
pyblock3.algebra.ad.ENABLE_JAX = True
pyblock3.algebra.ad.ENABLE_AUTORAY = True

from pyblock3.algebra.ad import core
core.ENABLE_FUSED_IMPLS = True

from pyblock3.algebra.ad.core import SparseTensor, FermionTensor

warnings.filterwarnings(action='ignore', category=torch.jit.TracerWarning)

import autoray as ar

SparseTensor.shape = property(lambda x: tuple(ix.n_bonds for ix in x.infos))
FermionTensor.shape = property(lambda x: tuple(ix.n_bonds for ix in x.infos))
ar.register_function('pyblock3', 'array', lambda x: x)
ar.register_function('pyblock3', 'to_numpy', lambda x: x)
ar.register_function('torch', 'conjugate', torch.conj)

def torch_split_wrapper(old_fn):
    def new_fn(ary, indices_or_sections, *args, **kwargs):
        if len(indices_or_sections) == 0:
            return [ary]
        else:
            return old_fn(ary, indices_or_sections, *args, **kwargs)
    return new_fn

ar.register_function('torch', 'split', torch_split_wrapper, wrap=True)

L = 4
D = 20

np.random.seed(1234)

print(core.ENABLE_FUSED_IMPLS)

fd = FCIDUMP(pg='d2h', n_sites=L, n_elec=L, twos=0, ipg=0, orb_sym=[0] * L)
hamil = Hamiltonian(fd, flat=False)

class MERAState:

    @staticmethod
    def get_disentanglers_and_isometries(basis, target, low=0, high=1, dtype=float, max_bond=None):
        tensors_uni = []
        tensors_iso = []
        tensors_cap = []
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
                if n_sites == 1:
                    tensors_cap.append(ts)
                else:
                    tensors_iso.append(ts)
                iso_basis.append(info_high)

            basis = iso_basis

        return tensors_uni, tensors_iso, tensors_cap

    @staticmethod
    def unitize_(mera):
        for t in mera:
            if t.data.ndim - len(t.left_inds) == 1:
                new_data = t.data.left_canonicalize()[0]
            elif t.data.ndim - len(t.left_inds) == 2:
                fuse_info = t.data.kron_sum_info(2, 3, pattern=t.data.pattern[2:])
                new_data = t.data.fuse(2, 3, info=fuse_info).left_canonicalize()[0].unfuse(2, info=fuse_info)
                new_data.pattern = t.data.pattern
            else:
                assert False
            t.modify(data=new_data)
        return mera

    @staticmethod
    def c_operator(orb_sym, sz, basis, isite, dtype=float):
        ipg = orb_sym[isite]
        Q = basis[isite].symm_class
        repr = FermionTensor.zeros(bond_infos=(basis[isite], basis[isite]),
            dtype=dtype, dq=Q(1, sz, ipg), pattern="+-")
        repr.odd[Q(0, 0, 0)].data[0, 0] = 1
        repr.odd[Q(1, -sz, ipg)].data[0, 0] = sz
        return repr

    @staticmethod
    def d_operator(orb_sym, sz, basis, isite, dtype=float):
        ipg = orb_sym[isite]
        Q = basis[isite].symm_class
        repr = FermionTensor.zeros(bond_infos=(basis[isite], basis[isite]),
            dtype=dtype, dq=Q(-1, -sz, ipg), pattern="+-")
        repr.odd[Q(1, sz, ipg)].data[0, 0] = 1
        repr.odd[Q(2, 0, 0)].data[0, 0] = sz
        return repr

    @staticmethod
    def u_operator(basis, isite, dtype=float):
        Q = basis[isite].symm_class
        repr = FermionTensor.zeros(bond_infos=(basis[isite], basis[isite]),
            dtype=dtype, dq=Q(0, 0, 0), pattern="+-")
        repr.even[Q(2, 0, 0)].data[0, 0] = 1
        return repr

    @staticmethod
    def get_hamil_terms(topology, oyb_sym, basis, dtype=float):
        terms = []
        ca_op = [None] * len(basis)
        da_op = [None] * len(basis)
        cb_op = [None] * len(basis)
        db_op = [None] * len(basis)
        u_op = [None] * len(basis)
        for m in range(len(basis)):
            ca_op[m] = MERAState.c_operator(oyb_sym, 1, basis, m, dtype=dtype)
            da_op[m] = MERAState.d_operator(oyb_sym, 1, basis, m, dtype=dtype)
            cb_op[m] = MERAState.c_operator(oyb_sym, -1, basis, m, dtype=dtype)
            db_op[m] = MERAState.d_operator(oyb_sym, -1, basis, m, dtype=dtype)
            u_op[m] = MERAState.u_operator(basis, m, dtype=dtype)

        for (i, j, v) in topology:
            if i != j:
                ts = sum([
                    np.einsum('ij,kl->ikjl', v * ca_op[i], da_op[j]),
                    np.einsum('ij,kl->ikjl', -v * da_op[i], ca_op[j]),
                    np.einsum('ij,kl->ikjl', v * cb_op[i], db_op[j]),
                    np.einsum('ij,kl->ikjl', -v * db_op[i], cb_op[j])
                ])
                terms.append(((i, j), ts))
            else:
                ts = v * u_op[i]
                terms.append(((i, ), ts))

        return terms


class TensorBuilder:
    def __init__(self, hamil, max_bond=None):
        self.uni, self.iso, self.cap = MERAState.get_disentanglers_and_isometries(hamil.basis, hamil.target, max_bond=max_bond)
        self.i_uni = 0
        self.i_iso = 0
        self.i_cap = 0

    @property
    def uni_fill_fn(self):
        def uni_fill_fn(shape):
            uni = self.uni[self.i_uni]
            self.i_uni += 1
            print('uni ', uni.shape, shape)
            assert uni.shape == shape
            return uni
        return uni_fill_fn

    @property
    def iso_fill_fn(self):
        def iso_fill_fn(shape):
            iso = self.iso[self.i_iso]
            self.i_iso += 1
            print('iso ', iso.shape, shape)
            # assert iso.shape == shape
            return iso
        return iso_fill_fn

    @property
    def cap_fill_fn(self):
        def cap_fill_fn(shape):
            cap = self.cap[self.i_cap]
            self.i_cap += 1
            print('cap ', cap.shape, shape)
            # assert cap.shape == shape
            return cap
        return cap_fill_fn

def get_params(self):
    from jax.tree_util import tree_flatten
    params, tree = tree_flatten(self)
    self._tree = tree
    return params

def set_params(self, params):
    from jax.tree_util import tree_unflatten
    x = tree_unflatten(self._tree, params)
    self.blocks = x.blocks
    self.pattern = x.pattern

SparseTensor.get_params = get_params
SparseTensor.set_params = set_params


def f_set_params(self, params):
    from jax.tree_util import tree_unflatten
    x = tree_unflatten(self._tree, params)
    self.odd = x.odd
    self.even = x.even

FermionTensor.get_params = get_params
FermionTensor.set_params = f_set_params

tensor_builder = TensorBuilder(hamil, max_bond=D)

from quimb.experimental.merabuilder import MERA

mera = MERA.from_fill_fn(
    fill_fn=None,
    uni_fill_fn=tensor_builder.uni_fill_fn,
    iso_fill_fn=tensor_builder.iso_fill_fn,
    cap_fill_fn=tensor_builder.cap_fill_fn,
    L=L,         # number of sites, (doesn't have to be a power of block_size)
    D=D,           # max bond dimension
    block_size=2,
    phys_dim=4,
    cyclic=True,  # OBC
)

MERAState.unitize_(mera)

t = 1
u = 2
topology = [(i, i + 1, t) for i in range(L - 1)]
topology += [(i, i, u) for i in range(L)]

h_terms = MERAState.get_hamil_terms(topology, hamil.orb_sym, hamil.basis)
terms = { k : v for k, v in h_terms }

def norm_fn(mera):
    return MERAState.unitize_(mera.copy())

def local_expectation(mera, terms, where, optimize='auto-hq'):
    tags = [mera.site_tag(coo) for coo in where]
    mera_ij = mera.select(tags, 'any')
    mera_ij_G = mera_ij.gate(terms[where], where)
    mera_ij_ex = (mera_ij_G & mera_ij.H)
    return mera_ij_ex.contract(all, optimize=optimize)

def loss_fn(mera, terms, **kwargs):
    with qtn.contraction.contract_backend("numpy"):
        r = sum(
            local_expectation(mera, terms, where, **kwargs)
            for where in terms
        )
        return r

mera = norm_fn(mera)
print(loss_fn(mera, terms, optimize='auto-hq'))

def _tn_callback(self):
    if not hasattr(self, 'tt'):
        self.it = 0
        self.tt = time.perf_counter()
    self.loss_best = min(self.loss_best, self.loss)
    tt = time.perf_counter() - self.tt
    self.tt = time.perf_counter()
    self.it += 1
    msg = f"{self.it:5d} {self.loss:+.12f} [best: {self.loss_best:+.12f}] T = {tt:10.3f}"
    print(msg)

tnopt = qtn.TNOptimizer(
    mera,
    loss_fn=loss_fn,
    norm_fn=norm_fn,
    loss_constants={'terms': terms},
    loss_kwargs={'optimize': 'auto-hq'},
    autodiff_backend='torch',
    callback=_tn_callback,
    progbar=False,
    jit_fn=False,
)

tt = time.perf_counter()
tnopt.optimizer = 'l-bfgs-b'
tnopt.optimize(500)
# tnopt.optimizer = 'adam'
# mera_opt = tnopt.optimize(1000)

print(min(tnopt.losses))
print('time = ', time.perf_counter() - tt)
