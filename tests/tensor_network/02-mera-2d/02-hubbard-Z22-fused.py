
# 2D NN Hubbard model (Z2 symmetry)
# implemented using fermionic sparse tensors
# E dmrg (2x2) =  -2.828427124743
# E dmrg (4x4) = -18.114785996935 (approx)

import numpy as np
import quimb.tensor as qtn
import warnings
import torch
import time
from pyblock3.hamiltonian import Hamiltonian
from pyblock3.fcidump import FCIDUMP
from quimb.experimental.merabuilder import TensorNetworkGenIso

import pyblock3.algebra.ad
pyblock3.algebra.ad.ENABLE_JAX = True
pyblock3.algebra.ad.ENABLE_AUTORAY = True

from pyblock3.algebra.fermion_symmetry import U11, Z22
from pyblock3.algebra.symmetry import BondInfo, BondFusingInfo
from pyblock3.algebra import fermion_setting as setting
setting.set_ad(True)

from quimb.tensor.fermion.block_interface import set_options
set_options(symmetry='z22', use_cpp=False)

torch.autograd.set_detect_anomaly(False)
torch.set_num_threads(28)

from pyblock3.algebra.ad import core
core.ENABLE_FUSED_IMPLS = True

from pyblock3.algebra.ad.fermion import SparseFermionTensor

warnings.filterwarnings(action='ignore', category=torch.jit.TracerWarning)

import autoray as ar

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

from quimb.tensor.fermion.fermion_core import FermionTensor as FT, FermionTensorNetwork
from quimb.tensor.fermion.fermion_arbgeom import FermionTensorNetworkGenVector
from quimb.tensor import Tensor

class IsoFermionTensor(FT):
    """A ``Tensor`` subclass which keeps its ``left_inds`` by default even
    when its data is changed.
    """

    __slots__ = ('_data', '_inds', '_tags', '_left_inds', '_owners')

    def modify(self, **kwargs):
        kwargs.setdefault("left_inds", self.left_inds)
        super().modify(**kwargs)

    def fuse(self, *args, inplace=False, **kwargs):
        t = self if inplace else self.copy()
        t.left_inds = None
        return Tensor.fuse(t, *args, inplace=True, **kwargs)


L = 4
D = 5

np.random.seed(1234)

print(core.ENABLE_FUSED_IMPLS)
print('D =', D)

class MERAState:

    @staticmethod
    def get_disentanglers(info_low, low=0, high=1, dtype=float, symm_map=None):
        r = SparseFermionTensor.random(
            (info_low, info_low, info_low, info_low),
            dtype=dtype, pattern="++--") * (high - low) + low
        if symm_map is None:
            return r
        else:
            finfos = [BondFusingInfo.get_symmetry_fusing_info(i, symm_map)
                      for i in (info_low, info_low, info_low, info_low)]
            return r.symmetry_fuse(finfos, symm_map)
    
    @staticmethod
    def get_basis_next_layer(info_low, max_bond=None):
        info_high = info_low * info_low * info_low * info_low
        if max_bond is not None:
            info_high.truncate_no_keep(max_bond)
        return info_high

    @staticmethod
    def get_isometries(info_low, info_high, low=0, high=1, dtype=float, symm_map=None):
        r = SparseFermionTensor.random(
            (info_low, info_low, info_low, info_low, info_high),
            dtype=dtype, pattern="++++-") * (high - low) + low
        if symm_map is None:
            return r
        else:
            finfos = [BondFusingInfo.get_symmetry_fusing_info(i, symm_map)
                      for i in (info_low, info_low, info_low, info_low, info_high)]
            return r.symmetry_fuse(finfos, symm_map)

    @staticmethod
    def build_mera(info_phys, target, L, low=0, high=1, dtype=float, max_bond=None, symm_map=None):
        sites = [(i, j) for j in range(L) for i in range(L)]
        mera2d = TensorNetworkGenIso.empty(sites)

        dl = 1
        info_low = info_phys
        while L >= 4:
            # horizonatal disentangle
            for where in [((i, j), (i, j + dl)) for i in range(0, L, dl) for j in range(dl, L - dl, dl * 2)]:
                ts = MERAState.get_disentanglers(info_low, low=low, high=high, dtype=dtype, symm_map=symm_map)
                mera2d.layer_gate_fill_fn(fill_fn=lambda _: ts, operation='uni', where=where, max_bond=D)
            # vertical disentangle
            for where in [((j, i), (j + dl, i)) for i in range(0, L, dl) for j in range(dl, L - dl, dl * 2)]:
                ts = MERAState.get_disentanglers(info_low, low=low, high=high, dtype=dtype, symm_map=symm_map)
                mera2d.layer_gate_fill_fn(fill_fn=lambda _: ts, operation='uni', where=where, max_bond=D)
            info_high = MERAState.get_basis_next_layer(info_low, max_bond=max_bond)
            print(L, info_high)
            # group the corners with isometries
            for ix in range(0, L, 2 * dl):
                for iy in range(0, L, 2 * dl):
                    where = ((ix, iy), (ix, iy + dl), (ix + dl, iy), (ix + dl, iy + dl))
                    ts = MERAState.get_isometries(info_low, info_high, low=low, high=high, dtype=dtype, symm_map=symm_map)
                    mera2d.layer_gate_fill_fn(fill_fn=lambda _: ts, operation='iso', where=where, max_bond=D)
            L = L // 2
            info_low = info_high
            dl = dl * 2
        
        # cap the 4 remaining open indices with tree
        info_high = MERAState.get_basis_next_layer(info_low, max_bond=max_bond)
        print(L, info_high)
        info_high = info_high.filter(info_high.__class__({target : 1}))
        ts = MERAState.get_isometries(info_low, info_high, low=low, high=high, dtype=dtype, symm_map=symm_map)

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
    def c_operator(sz, basis, isite, dtype=float, symm_map=None):
        Q = basis[isite].symm_class
        repr = SparseFermionTensor.zeros(bond_infos=(basis[isite], basis[isite]),
            dtype=dtype, dq=Q(1, sz), pattern="+-")
        repr[Q(0, 0)].data[0, 0] = 1
        repr[Q(1, -sz)].data[0, 0] = sz
        if symm_map is not None:
            finfos = [BondFusingInfo.get_symmetry_fusing_info(i, symm_map)
                      for i in (basis[isite], basis[isite])]
            repr = repr.symmetry_fuse(finfos, symm_map)
        return repr

    @staticmethod
    def d_operator(sz, basis, isite, dtype=float, symm_map=None):
        Q = basis[isite].symm_class
        repr = SparseFermionTensor.zeros(bond_infos=(basis[isite], basis[isite]),
            dtype=dtype, dq=Q(-1, -sz), pattern="+-")
        repr[Q(1, sz)].data[0, 0] = 1
        repr[Q(2, 0)].data[0, 0] = sz
        if symm_map is not None:
            finfos = [BondFusingInfo.get_symmetry_fusing_info(i, symm_map)
                      for i in (basis[isite], basis[isite])]
            repr = repr.symmetry_fuse(finfos, symm_map)
        return repr

    @staticmethod
    def u_operator(basis, isite, dtype=float, symm_map=None):
        Q = basis[isite].symm_class
        repr = SparseFermionTensor.zeros(bond_infos=(basis[isite], basis[isite]),
            dtype=dtype, dq=Q(0, 0), pattern="+-")
        repr[Q(2, 0)].data[0, 0] = 1
        if symm_map is not None:
            finfos = [BondFusingInfo.get_symmetry_fusing_info(i, symm_map)
                      for i in (basis[isite], basis[isite])]
            repr = repr.symmetry_fuse(finfos, symm_map)
        return repr

    @staticmethod
    def get_hamil_terms(topology, basis, dtype=float, symm_map=None):
        terms = []
        ca_op = MERAState.c_operator(1, basis, 0, dtype=dtype, symm_map=symm_map)
        da_op = MERAState.d_operator(1, basis, 0, dtype=dtype, symm_map=symm_map)
        cb_op = MERAState.c_operator(-1, basis, 0, dtype=dtype, symm_map=symm_map)
        db_op = MERAState.d_operator(-1, basis, 0, dtype=dtype, symm_map=symm_map)
        u_op = MERAState.u_operator(basis, 0, dtype=dtype, symm_map=symm_map)

        for (i, j, v) in topology:
            if i != j:
                ts = sum([
                    np.tensordot(v * ca_op, da_op, axes=([], [])).transpose((0, 2, 1, 3)),
                    np.tensordot(-v * da_op, ca_op, axes=([], [])).transpose((0, 2, 1, 3)),
                    np.tensordot(v * cb_op, db_op, axes=([], [])).transpose((0, 2, 1, 3)),
                    np.tensordot(-v * db_op, cb_op, axes=([], [])).transpose((0, 2, 1, 3))
                ])
                terms.append(((i, j), ts))
            else:
                ts = v * u_op
                terms.append(((i, ), ts))

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
    self._pattern = x.pattern
    self._shape = x.shape

SparseFermionTensor.get_params = get_params
SparseFermionTensor.set_params = set_params

fd = FCIDUMP(pg='d2h', n_sites=L, n_elec=L ** 2, twos=0, ipg=0, orb_sym=[0] * L)
hamil = Hamiltonian(fd, flat=False)

from pyblock3.algebra.symmetry import BondInfo
hamil.basis[0] = BondInfo()
hamil.basis[0][U11(0, 0)] = 1
hamil.basis[0][U11(1, 1)] = 1
hamil.basis[0][U11(1, -1)] = 1
hamil.basis[0][U11(2, 0)] = 1

hamil.target = U11(hamil.target.n, hamil.target.twos)

def symm_map_u11_to_z22(x):
    return Z22(x.n, x.sz)

symm_map = symm_map_u11_to_z22
# symm_map = None

print("symm_map = ", symm_map)

mera = MERAState.build_mera(hamil.basis[0], hamil.target, L=L, max_bond=D, symm_map=symm_map)

fmera = FermionTensorNetwork([])
for ts in mera:
    new_ts = IsoFermionTensor(data=ts.data, inds=ts.inds, tags=ts.tags, left_inds=ts.left_inds)
    fmera.add_tensor(new_ts, virtual=False)

fmera.view_as_(FermionTensorNetworkGenVector, like=mera)
mera = fmera

t = 1
u = 2

topology  = [((i, j), (i + 1, j), t) for i in range(L - 1) for j in range(L)]
topology += [((i, j), (i, j + 1), t) for i in range(L) for j in range(L - 1)]
topology += [((i, j), (i, j), u) for i in range(L) for j in range(L)]
h_terms = MERAState.get_hamil_terms(topology, hamil.basis, symm_map=symm_map)
terms = { k : v for k, v in h_terms }

def norm_fn(mera):
    return MERAState.unitize_(mera.copy())

def local_expectation(mera, terms, where, optimize='auto-hq'):
    tags = [mera.site_tag(coo) for coo in where]
    mera_ij = mera.select(tags, 'any')
    fsites = list(mera_ij.filled_sites) + list(set(mera.filled_sites) - set(mera_ij.filled_sites))
    tid_map = { mera.fermion_space.get_tid_from_site(x) : ix for ix, x in enumerate(fsites) }
    mera_x = mera._reorder_from_tid(tid_map)
    mera_ij = mera_x.select(tags, 'any')
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
tnopt.optimize(200)
tnopt.optimizer = 'adam'
tnopt.optimize(200)

print(min(tnopt.losses))
print('time = ', time.perf_counter() - tt)
