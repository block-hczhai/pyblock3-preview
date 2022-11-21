

# 2D NN Heisenberg model with 4x4 sites
# E dmrg = -9.189207065181

import numpy as np
import quimb.tensor as qtn
import warnings
import torch
import time
from pyblock3.heisenberg import Heisenberg
from quimb.experimental.merabuilder import TensorNetworkGenIso

torch.autograd.set_detect_anomaly(False)
torch.set_num_threads(28)

import pyblock3.algebra.ad
pyblock3.algebra.ad.ENABLE_JAX = True
pyblock3.algebra.ad.ENABLE_AUTORAY = True

from pyblock3.algebra.ad import core
core.ENABLE_FUSED_IMPLS = False

from pyblock3.algebra.ad.core import SparseTensor

warnings.filterwarnings(action='ignore', category=torch.jit.TracerWarning)

import autoray as ar

SparseTensor.shape = property(lambda x: tuple(ix.n_bonds for ix in x.infos))
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
D = 12

np.random.seed(1234)

print(core.ENABLE_FUSED_IMPLS)
print('D =', D)

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

class MERAState:

    @staticmethod
    def get_disentanglers(info_low, low=0, high=1, dtype=float):
        return SparseTensor.random(
            (info_low, info_low, info_low, info_low),
            dtype=dtype, pattern="++--") * (high - low) + low
    
    @staticmethod
    def get_basis_next_layer(info_low, max_bond=None):
        info_high = info_low * info_low * info_low * info_low
        if max_bond is not None:
            info_high.truncate_no_keep(max_bond)
        return info_high

    @staticmethod
    def get_isometries(info_low, info_high, low=0, high=1, dtype=float):
        return SparseTensor.random(
            (info_low, info_low, info_low, info_low, info_high),
            dtype=dtype, pattern="++++-") * (high - low) + low

    @staticmethod
    def build_mera(info_phys, target, L, low=0, high=1, dtype=float, max_bond=None):
        sites = [(i, j) for j in range(L) for i in range(L)]
        mera2d = TensorNetworkGenIso.empty(sites)

        dl = 1
        info_low = info_phys
        while L >= 4:
            # horizonatal disentangle
            for where in [((i, j), (i, j + dl)) for i in range(0, L, dl) for j in range(dl, L - dl, dl * 2)]:
                ts = MERAState.get_disentanglers(info_low, low=low, high=high, dtype=dtype)
                mera2d.layer_gate_fill_fn(fill_fn=lambda _: ts, operation='uni', where=where, max_bond=D)
            # vertical disentangle
            for where in [((j, i), (j + dl, i)) for i in range(0, L, dl) for j in range(dl, L - dl, dl * 2)]:
                ts = MERAState.get_disentanglers(info_low, low=low, high=high, dtype=dtype)
                mera2d.layer_gate_fill_fn(fill_fn=lambda _: ts, operation='uni', where=where, max_bond=D)
            info_high = MERAState.get_basis_next_layer(info_low, max_bond=max_bond)
            # group the corners with isometries
            for ix in range(0, L, 2 * dl):
                for iy in range(0, L, 2 * dl):
                    where = ((ix, iy), (ix, iy + dl), (ix + dl, iy), (ix + dl, iy + dl))
                    ts = MERAState.get_isometries(info_low, info_high, low=low, high=high, dtype=dtype)
                    mera2d.layer_gate_fill_fn(fill_fn=lambda _: ts, operation='iso', where=where, max_bond=D)
            L = L // 2
            info_low = info_high
            dl = dl * 2
        
        # cap the 4 remaining open indices with tree
        info_high = MERAState.get_basis_next_layer(info_low, max_bond=max_bond)
        info_high = info_high.filter(info_high.__class__({target : 1}))
        ts = MERAState.get_isometries(info_low, info_high, low=low, high=high, dtype=dtype)

        mera2d.layer_gate_fill_fn(fill_fn=lambda _: ts, operation='cap',
            where=tuple(mera2d._open_upper_sites), max_bond=D)

        return mera2d

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
        sp_op = MERAState.sp_operator(tj, basis, 0, dtype=dtype)
        sm_op = MERAState.sm_operator(tj, basis, 0, dtype=dtype)
        sz_op = MERAState.sz_operator(tj, basis, 0, dtype=dtype)

        for (i, j, v) in topology:
            assert i != j
            ts = sum([
                np.einsum('ij,kl->ikjl', 0.5 * v * sp_op, sm_op),
                np.einsum('ij,kl->ikjl', 0.5 * v * sm_op, sp_op),
                np.einsum('ij,kl->ikjl', v * sz_op, sz_op)
            ])
            terms.append(((i, j), ts))

        return terms

hamil = Heisenberg(twos=1, n_sites=L, topology=None, flat=False)

mera = MERAState.build_mera(hamil.basis[0], hamil.target, L=L, max_bond=D)
topology  = [((i, j), (i + 1, j), 1.0) for i in range(L - 1) for j in range(L)]
topology += [((i, j), (i, j + 1), 1.0) for i in range(L) for j in range(L - 1)]
h_terms = MERAState.get_hamil_terms(hamil.twos, hamil.basis, topology)
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
        return sum(
            local_expectation(mera, terms, where, **kwargs)
            for where in terms
        )

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

mera = norm_fn(mera)
print(loss_fn(mera, terms, optimize='auto-hq'))

tnopt = qtn.TNOptimizer(
    mera, loss_fn,
    norm_fn=norm_fn,
    loss_constants=dict(terms=terms),
    optimizer='adam',
    autodiff_backend='torch',
    # device='cuda',
    callback=_tn_callback,
    progbar=False,
    jit_fn=False,
)

tnopt.optimizer = 'l-bfgs-b'
tnopt.optimize(200)
tnopt.optimizer = 'adam'
tnopt.optimize(200)
