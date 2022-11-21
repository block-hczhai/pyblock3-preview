
import quimb.tensor as qtn
import warnings
import torch
import time
import numpy as np
from pyblock3.heisenberg import Heisenberg

import pyblock3.algebra.ad
pyblock3.algebra.ad.ENABLE_JAX = True

from pyblock3.algebra.ad import core
core.ENABLE_FUSED_IMPLS = False

from pyblock3.algebra.ad.core import SparseTensor

warnings.filterwarnings(action='ignore', category=torch.jit.TracerWarning)

import autoray as ar

SparseTensor.shape = property(lambda x: tuple(ix.n_bonds for ix in x.infos))
ar.register_function('pyblock3', 'array', lambda x: x)
ar.register_function('pyblock3', 'to_numpy', lambda x: x)

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


class PyBlock3Handler:

    def __init__(self, jit_fn=True, device=None):
        self.jit_fn = jit_fn
        self.device = device

    def to_variable(self, x):
        return x

    def to_constant(self, x):
        return x

    def setup_fn(self, fn):
        jax = qtn.optimize.get_jax()
        if self.jit_fn:
            self._backend_fn = jax.jit(fn, backend=self.device)
            self._value_and_grad = jax.jit(
                jax.value_and_grad(fn), backend=self.device)
        else:
            self._backend_fn = fn
            self._value_and_grad = jax.value_and_grad(fn)

        self._setup_hessp(fn)

    def _setup_hessp(self, fn):
        jax = qtn.optimize.get_jax()

        def hvp(primals, tangents):
            return jax.jvp(jax.grad(fn), (primals,), (tangents,))[1]

        if self.jit_fn:
            hvp = jax.jit(hvp, device=self.device)

        self._hvp = hvp

    def value(self, arrays):
        jax_arrays = tuple(map(self.to_constant, arrays))
        return self._backend_fn(jax_arrays)

    def value_and_grad(self, arrays):
        loss, grads = self._value_and_grad(arrays)
        return loss, [x.conj() for x in grads]

    def hessp(self, primals, tangents):
        jax_arrays = self._hvp(primals, tangents)
        return jax_arrays


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


qtn.optimize._BACKEND_HANDLERS['numpy'] = PyBlock3Handler
qtn.optimize.Vectorizer = PyTreeVectorizer
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
    # print(where)
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
    autodiff_backend='numpy',
    # device='cuda',
    jit_fn=False,
)

tt = time.perf_counter()
tnopt.optimizer = 'l-bfgs-b'
tnopt.optimize(50)
# tnopt.optimizer = 'adam'
# mera_opt = tnopt.optimize(1000)

print(min(tnopt.losses))
print('time = ', time.perf_counter() - tt)

# 38 -3.374934911728 [best: -3.374934911728] T =     10.484
# -3.3749349117279053
# time =  412.568538736552
