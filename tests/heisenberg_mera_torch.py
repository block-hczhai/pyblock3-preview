

import sys

import numpy as np
import quimb.tensor as qtn
import warnings
import torch
import time
from pyblock3.heisenberg import Heisenberg

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

def p_array(x):
    return x

@property
def p_shape(x):
    return tuple(ix.n_bonds for ix in x.infos)

SparseTensor.shape = p_shape # necessary for printing and contraction

ar.register_function('pyblock3', 'array', p_array)
ar.register_function('pyblock3', 'to_numpy', p_array)
ar.register_function('torch', 'conjugate', torch.conj)

def torch_split_wrapper(old_fn):
    def new_fn(ary, indices_or_sections, *args, **kwargs):
        if len(indices_or_sections) == 0:
            return [ary]
        else:
            return old_fn(ary, indices_or_sections, *args, **kwargs)
    return new_fn

ar.register_function('torch', 'split', torch_split_wrapper, wrap=True)

L = 8
D = 20

np.random.seed(1234)

print(core.ENABLE_FUSED_IMPLS)

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
                np.einsum('ij,kl->jlik', 0.5 * v * sp_op[i], sm_op[j]),
                np.einsum('ij,kl->jlik', 0.5 * v * sm_op[i], sp_op[j]),
                np.einsum('ij,kl->jlik', v * sz_op[i], sz_op[j])
            ])
            terms.append(((i, j), ts))

        return terms

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

class PyTreeVectorizer(qtn.optimize.Vectorizer):

    def __init__(self, arrays):

        from jax.tree_util import tree_flatten
        self.flats, self.tree = tree_flatten(arrays)
        self.shapes = [x.shape for x in self.flats]
        self.iscomplexes = [qtn.array_ops.iscomplex(x) for x in self.flats]
        self.dtypes = [ar.get_dtype_name(x) for x in self.flats]
        self.sizes = [np.prod(s) for s in self.shapes]
        self.d = sum(
            (1 + int(cmplx)) * size
            for size, cmplx in zip(self.sizes, self.iscomplexes)
        )
        self.pack(arrays)

    def pack(self, arrays, name='vector'):

        from jax.tree_util import tree_flatten
        flats = tree_flatten(arrays)[0]

        # scipy's optimization routines require real, double data
        if not hasattr(self, name):
            setattr(self, name, np.empty(self.d, 'float64'))
        x = getattr(self, name)
        i = 0
        for array, size, cmplx in zip(flats, self.sizes, self.iscomplexes):
            if not isinstance(array, np.ndarray):
                array = qtn.optimize.to_numpy(array)
            if not cmplx:
                x[i:i + size] = array.reshape(-1)
                i += size
            else:
                real_view = array.reshape(-1).view(qtn.optimize.equivalent_real_type(array))
                x[i:i + 2 * size] = real_view
                i += 2 * size
        return x

    def unpack(self, vector=None):
        if vector is None:
            vector = self.vector
        i = 0
        flats = []
        for shape, size, cmplx, dtype in zip(self.shapes, self.sizes,
                                             self.iscomplexes, self.dtypes):
            if not cmplx:
                array = vector[i:i + size].reshape(shape)
                i += size
            else:
                array = vector[i:i + 2 * size]
                array = array.view(qtn.optimize.equivalent_complex_type(array))
                array.shape = shape
                i += 2 * size
            if qtn.optimize.get_dtype_name(array) != dtype:
                array = qtn.optimize.astype(array, dtype)
            flats.append(array)
        from jax.tree_util import tree_unflatten
        return tree_unflatten(self.tree, flats)


def _maybe_update_pbar(self):
    if self.progbar:
        self.loss_best = min(self.loss_best, self.loss)
        tt = time.perf_counter() - self.tt
        self.tt = time.perf_counter()
        self.it += 1
        msg = f"{self.it:5d} {self.loss:+.12f} [best: {self.loss_best:+.12f}] T = {tt:10.3f}"
        print(msg)

def _maybe_init_pbar(self, n):
    if self.progbar:
        self._pbar = None
        self.it = 0
        self.tt = time.perf_counter()

qtn.optimize.TNOptimizer._maybe_init_pbar = _maybe_init_pbar
qtn.optimize.TNOptimizer._maybe_update_pbar = _maybe_update_pbar

uni, iso = MERAState.get_disentanglers_and_isometries(hamil.basis, hamil.target, max_bond=D)

mera = qtn.MERA(L, uni, iso, dangle=True)
MERAState.unitize_(mera)

h_terms = MERAState.get_hamil_terms(hamil.twos, hamil.basis, topology)
terms = { k : v for k, v in h_terms }

ptvt = PyTreeVectorizer(terms)
x_terms = ptvt.vector

def norm_fn(mera):
    mera = mera.copy()
    MERAState.unitize_(mera)
    return mera

def local_expectation(mera, terms, where, optimize='auto-hq'):
    # fix tensordot/einsum backend inside einsum
    import collections
    qtn.contraction._TEMP_CONTRACT_BACKENDS = collections.defaultdict(list)
    qtn.contraction._CONTRACT_BACKEND = "numpy"

    tags = [mera.site_tag(coo) for coo in where]
    mera_ij = mera.select(tags, 'any')
    mera_ij_G = mera_ij.gate(terms[where], where)
    mera_ij_ex = (mera_ij_G & mera_ij.H)
    return mera_ij_ex.contract(all, optimize=optimize)

def loss_fn(mera, x_terms, **kwargs):
    terms = ptvt.unpack(x_terms)
    r = sum(
        local_expectation(mera, terms, where, **kwargs)
        for where in terms
    )
    return r

print(loss_fn(norm_fn(mera), x_terms, optimize='auto-hq'))

tnopt = qtn.TNOptimizer(
    mera,
    loss_fn=loss_fn,
    norm_fn=norm_fn,
    loss_constants={'x_terms': x_terms},
    loss_kwargs={'optimize': 'auto-hq'},
    autodiff_backend='torch',
    # device='cuda',
    jit_fn=False,
)

tt = time.perf_counter()
tnopt.optimizer = 'l-bfgs-b'
tnopt.optimize(500)
# tnopt.optimizer = 'adam'
# mera_opt = tnopt.optimize(1000)

print(min(tnopt.losses))
print('time = ', time.perf_counter() - tt)

# 22 -3.374932596931 [best: -3.374932596931] T =      0.703
# -3.3749325969312736
# time =  15.713499780744314