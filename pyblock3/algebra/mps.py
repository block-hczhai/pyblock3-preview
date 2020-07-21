

import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin
import numbers
from collections import Counter
from functools import reduce

from .symmetry import StateInfo, StateFusingInfo
from .core import SparseTensor, SubTensor, SliceableTensor


def implements(np_func):
    global _numpy_func_impls
    return lambda f: (_numpy_func_impls.update({np_func: f}), f)[1]


class MPSInfo:
    """
    StateInfo in every site in MPS
    (a) For constrution of initial MPS.
    (b) For limiting active space in DMRG.
    Attributes:
        n_sites : int
            Number of sites
        vacuum : SZ
            vacuum state
        target : SZ
            target state
        basis : list(StateInfo)
            StateInfo in each site
        left_dims : list(StateInfo)
            Truncated states for left block
        right_dims : list(StateInfo)
            Truncated states for right block
    """

    def __init__(self, n_sites, vacuum, target, basis):
        self.n_sites = n_sites
        self.vacuum = vacuum
        self.target = target
        self.basis = basis
        self.left_dims = None
        self.right_dims = None

    def set_bond_dimension_fci(self, call_back=None):
        """FCI bond dimensions"""

        self.left_dims = [None] * (self.n_sites + 1)
        self.left_dims[0] = StateInfo(quanta=Counter({self.vacuum: 1}))
        for d in range(0, self.n_sites):
            self.left_dims[d + 1] = StateInfo.tensor_product(
                self.left_dims[d], self.basis[d])

        self.right_dims = [None] * (self.n_sites + 1)
        self.right_dims[-1] = StateInfo(quanta=Counter({self.target: 1}))
        for d in range(self.n_sites - 1, -1, -1):
            self.right_dims[d] = StateInfo.tensor_product(
                -self.basis[d], self.right_dims[d + 1])

        if call_back is not None:
            call_back(self)

        for d in range(0, self.n_sites + 1):
            self.left_dims[d].filter(self.right_dims[d])
            self.right_dims[d].filter(self.left_dims[d])

    def set_bond_dimension_occ(self, bond_dim, occ, bias=1):
        """bond dimensions from occupation numbers"""
        return NotImplemented

    def set_bond_dimension(self, bond_dim, call_back=None):
        """Truncated bond dimension based on FCI quantum numbers
            each FCI quantum number has at least one state kept"""

        if self.left_dims is None:
            self.set_bond_dimension_fci(call_back=call_back)

        for d in range(0, self.n_sites):
            ref = StateInfo.tensor_product(self.left_dims[d], self.basis[d])
            self.left_dims[d + 1].truncate(bond_dim, ref)
        for d in range(self.n_sites - 1, -1, -1):
            ref = StateInfo.tensor_product(
                -self.basis[d], self.right_dims[d + 1])
            self.right_dims[d].truncate(bond_dim, ref)


_mps_numpy_func_impls = {}
_numpy_func_impls = _mps_numpy_func_impls


class MPS(NDArrayOperatorsMixin):
    """
    Matrix Product State / Matrix Product Operator.
    Attributes:
        tensors : list(SparseTensor/FermionTensor)
            A list of block-sparse tensors.
        n_sites : int
            Number of sites.
        const : float
            Constant term.
        opts : dict or None
            options indicating how bond dimension trunctation
            should be done after MPO @ MPS, etc.
            Possible options are: max_bond_dim, cutoff, max_dw
    """

    def __init__(self, tensors, const=0, opts=None):
        self.tensors = tensors
        self.opts = opts if opts is not None else {}
        self.const = const

    @property
    def n_sites(self):
        """Number of sites"""
        return len(self.tensors)

    @property
    def dtype(self):
        return self.tensors[0].dtype if self.n_sites != 0 else float

    def __len__(self):
        return len(self.tensors)

    @staticmethod
    def zeros(info, opts=None):
        """Construct unfused MPS from MPSInfo, with zero matrix elements."""
        tensors = [None] * info.n_sites
        for i in range(info.n_sites):
            tensors[i] = SparseTensor.zeros(
                info.left_dims[i], info.basis[i], info.left_dims[i + 1])
        return MPS(tensors=tensors, opts=opts)

    @staticmethod
    def random(info, low=0, high=1, opts=None):
        """Construct unfused MPS from MPSInfo, with random matrix elements."""
        tensors = [None] * info.n_sites
        for i in range(info.n_sites):
            tensors[i] = SparseTensor.random(
                info.left_dims[i], info.basis[i], info.left_dims[i + 1]) * (high - low) + low
        return MPS(tensors=tensors, opts=opts)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if ufunc in _mps_numpy_func_impls:
            types = tuple(
                x.__class__ for x in inputs if not isinstance(x, numbers.Number))
            return self.__array_function__(ufunc, types, inputs, kwargs)
        out = kwargs.pop("out", None)
        if out is not None and not all(isinstance(x, MPS) for x in out):
            return NotImplemented
        if method == "__call__":
            if ufunc.__name__ in ["add", "subtract"]:
                a, b = inputs
                if isinstance(a, numbers.Number):
                    const = b.const + a
                    tensors = b.tensors
                elif isinstance(b, numbers.Number):
                    const = a.const + b
                    tensors = a.tensors
                else:
                    const = a.const + b.const
                    tensors = MPS._add_or_sub(
                        a, b, getattr(ufunc, method)).tensors
                    if len(a.opts) != 0:
                        tensors = MPS.compress(
                            MPS(tensors=tensors), left=True, **a.opts)[0].tensors
            elif ufunc.__name__ in ["multiply", "divide"]:
                a, b = inputs
                if isinstance(a, numbers.Number):
                    const = b.const * a
                    tensors = [getattr(ufunc, method)(
                        a, b.tensors[0])] + b.tensors[1:]
                elif isinstance(b, numbers.Number):
                    const = a.const * b
                    tensors = [getattr(ufunc, method)(
                        a.tensors[0], b)] + a.tensors[1:]
                else:
                    return NotImplemented
            elif len(inputs) == 1:
                const = getattr(ufunc, method)(inputs[0].const)
                tensors = [getattr(ufunc, method)(
                    inputs[0].tensors[0])] + inputs[0].tensors[1:]
            else:
                return NotImplemented
        else:
            return NotImplemented
        if out is not None:
            out.tensors = tensors
            out.const = const
            out.opts = self.opts
        return MPS(tensors=tensors, const=const, opts=self.opts)

    def __array_function__(self, func, types, args, kwargs):
        if func not in _mps_numpy_func_impls:
            return NotImplemented
        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented
        return _mps_numpy_func_impls[func](*args, **kwargs)

    def canonicalize(self, center):
        """
        MPS canonicalization.

        Args:
            center : int
                Site index of canonicalization center.
        """
        tensors = [ts for ts in self.tensors]
        assert 0 <= center < self.n_sites
        for i in range(0, center):
            q, r = tensors[i].left_canonicalize()
            tensors[i] = q
            tensors[i + 1] = np.tensordot(r, tensors[i + 1], axes=1)
        for i in range(self.n_sites - 1, center, -1):
            l, q = tensors[i].right_canonicalize()
            tensors[i] = q
            tensors[i - 1] = np.tensordot(tensors[i - 1], l, axes=1)
        return MPS(tensors=tensors, opts=self.opts, const=self.const)

    def compress(self, left=True, **opts):
        """
        MPS bond dimension compression.
        Args:
            max_bond_dim : int
                Maximal total bond dimension.
                If `k == -1`, no restriction in total bond dimension.
            cutoff : double
                Minimal kept singluar value.
            max_dw : double
                Maximal sum of square of discarded singluar values.
        """
        merror = 0.0
        if left:
            tensors = self.canonicalize(self.n_sites - 1).tensors
            for i in range(self.n_sites - 1, 0, -1):
                l, s, r = tensors[i].right_svd(full_matrices=False)
                l, s, r, err = r.__class__.truncate_svd(l, s, r, **opts)
                ls = np.tensordot(l, s.diag(), axes=1)
                tensors[i] = r
                tensors[i - 1] = np.tensordot(tensors[i - 1], ls, axes=1)
                merror = max(merror, err)
        else:
            tensors = self.canonicalize(0).tensors
            for i in range(0, self.n_sites - 1, 1):
                l, s, r = tensors[i].left_svd(full_matrices=False)
                l, s, r, err = l.__class__.truncate_svd(l, s, r, **opts)
                rs = np.tensordot(s.diag(), r, axes=1)
                tensors[i] = l
                tensors[i + 1] = np.tensordot(rs, tensors[i + 1], axes=1)
                merror = max(merror, err)
        return MPS(tensors=tensors, const=self.const, opts=self.opts), merror

    @staticmethod
    def _add_or_sub(a, b, func=np.add):
        """Add/subtract two MPS"""
        assert isinstance(a, MPS) and isinstance(b, MPS)
        assert a.n_sites == b.n_sites
        n_sites = a.n_sites

        sum_bonds = []
        sum_bonds.append(a.tensors[0].get_state_info(idx=0))
        for i in range(n_sites - 1):
            x = a.tensors[i +
                          1].get_state_info(idx=0) | a.tensors[i].get_state_info(idx=-1)
            y = b.tensors[i +
                          1].get_state_info(idx=0) | b.tensors[i].get_state_info(idx=-1)
            sum_bonds.append(x + y)
        sum_bonds.append(a.tensors[-1].get_state_info(idx=-1))

        tensors = []
        for i in range(n_sites):
            tensors.append(a.tensors[i].__class__.kron_add(
                a.tensors[i], b.tensors[i], func=func, infos=(sum_bonds[i], sum_bonds[i + 1])))
        return MPS(tensors=tensors, const=a.const + b.const, opts=a.opts)

    def __getitem__(self, i):
        return self.tensors[i]

    def __setitem__(self, i, ts):
        self.tensors[i] = ts

    @staticmethod
    @implements(np.dot)
    def _dot(a, b, out=None):
        if isinstance(a, numbers.Number) or isinstance(b, numbers.Number):
            return np.multiply(a, b, out=out)

        assert isinstance(a, MPS) and isinstance(b, MPS)
        assert a.n_sites == b.n_sites

        left = np.array(0)
        for i in range(a.n_sites):
            assert a.tensors[i].ndim == b.tensors[i].ndim
            if a.tensors[i].n_blocks == 0 or b.tensors[i].n_blocks == 0:
                return 0

            if i != a.n_sites - 1:
                cidx = list(range(0, a.tensors[i].ndim - 1))
            else:
                cidx = list(range(0, a.tensors[i].ndim))

            if i == 0:
                left = np.tensordot(
                    a.tensors[i], b.tensors[i], axes=(cidx, cidx))
            else:
                lbra = np.tensordot(left, a.tensors[i], axes=([0], [0]))
                left = np.tensordot(lbra, b.tensors[i], axes=(cidx, cidx))

        if out is not None:
            out[()] = left.item()

        return left.item() + a.const * b.const

    def dot(self, b, out=None):
        return np.dot(self, b, out=out)

    @staticmethod
    @implements(np.linalg.norm)
    def _norm(x):
        return np.sqrt(x.dot(x))

    def norm(self):
        return np.sqrt(self.dot(self))

    @staticmethod
    @implements(np.matmul)
    def _matmul(a, b, out=None):
        if isinstance(a, numbers.Number) or isinstance(b, numbers.Number):
            return np.multiply(a, b, out=out)

        assert isinstance(a, MPS) and isinstance(b, MPS)
        assert a.n_sites == b.n_sites

        n_sites = a.n_sites
        tensors = [None] * n_sites
        opts = {**a.opts, **b.opts}

        if all(ta.ndim == tb.ndim and ta.ndim % 2 == 1 for ta, tb in zip(a.tensors, b.tensors)):
            return np.dot(a, b, out=out)

        for i in range(n_sites):
            a_ndim = a.tensors[i].ndim - 2
            b_ndim = b.tensors[i].ndim - 2

            # MPS x MPS
            if a_ndim == b_ndim and a_ndim % 2 == 1:
                d = a_ndim
                aidx = list(range(1, d + 1))
                bidx = list(range(1, d + 1))
                tr = (0, 2, 1, 3)
            # MPO x MPO
            elif a_ndim == b_ndim:
                d = a_ndim // 2
                aidx = list(range(d + 1, d + d + 1))
                bidx = list(range(1, d + 1))
                tr = tuple([0, d + 2] + list(range(1, d + 1)) +
                           list(range(d + 3, d + d + 2)) + [d + 1, d + d + 3])
            # MPO x MPS
            elif a_ndim == b_ndim + b_ndim:
                d = b_ndim
                aidx = list(range(d + 1, d + d + 1))
                bidx = list(range(1, d + 1))
                tr = tuple([0, d + 2] + list(range(1, d + 1)) + [d + 1, d + 3])
            # MPS x MPO
            elif a_ndim + a_ndim == b_ndim:
                d = a_ndim
                aidx = list(range(1, d + 1))
                bidx = list(range(1, d + 1))
                tr = tuple([0, 2] + list(range(3, d + 3)) + [1, d + 3])
            else:
                raise RuntimeError(
                    "Cannot matmul tensors with number phyiscal indices: %d x %d" % (a_ndim, b_ndim))

            tensors[i] = np.tensordot(
                a.tensors[i], b.tensors[i], axes=(aidx, bidx))
            tensors[i] = np.transpose(tensors[i], axes=tr)

        # merge virtual dims
        prod_bonds = []
        x = tensors[0].get_state_info(idx=0)
        y = tensors[0].get_state_info(idx=1)
        prod_bonds.append(StateFusingInfo.tensor_product(x, y, x * y))
        for tl, tr in zip(tensors[1:], tensors[:-1]):
            x = tl.get_state_info(idx=0) | tr.get_state_info(idx=-2)
            y = tl.get_state_info(idx=1) | tr.get_state_info(idx=-1)
            prod_bonds.append(StateFusingInfo.tensor_product(x, y, x * y))
        x = tensors[-1].get_state_info(idx=-2)
        y = tensors[-1].get_state_info(idx=-1)
        prod_bonds.append(StateFusingInfo.tensor_product(x, y, x * y))

        for i in range(n_sites):
            tensors[i] = tensors[i].fuse(-2, -1, info=prod_bonds[i + 1]
                                         ).fuse(0, 1, info=prod_bonds[i])

        # const terms
        r = MPS(tensors=tensors)
        if a.const != 0 and b.const == 0:
            r += a.const * b
        elif a.const == 0 and b.const != 0:
            r += a * b.const
        elif a.const != 0 and b.const != 0:
            r += a * b.const + a.const * b
            r.const -= a.const * b.const

        # compression
        if len(opts) != 0:
            r, _ = MPS.compress(r, left=True, **opts)
            r.opts = opts

        if out is not None:
            out.tensors = r.tensors
            out.const = r.const
            out.opts = r.opts

        return r

    def matmul(self, b, out=None):
        return np.matmul(self, b, out=out)

    def show_bond_dims(self):
        bonds = []
        bonds.append(self.tensors[0].get_state_info(idx=0))
        for i in range(self.n_sites - 1):
            l = self.tensors[i + 1].get_state_info(idx=0)
            r = self.tensors[i].get_state_info(idx=-1)
            bonds.append(l | r)
        bonds.append(self.tensors[-1].get_state_info(idx=-1))
        r = '|'.join([str(x.n_states_total) for x in bonds])
        return r if self.const == 0 else r + " (%+12.5f)" % self.const

    @staticmethod
    def _to_sliceable(a, info=None):
        return a.to_sliceable(info=info)

    def to_sliceable(self, info=None):
        """
        Get a shallow copy of MPS with SliceableTensor.

        Args:
            info : MPSInfo, optional
                MPSInfo containing the complete basis StateInfo.
                If not specified, the StateInfo will be generated from the MPS,
                which may be incomplete.
        """

        # virtual dims
        bonds = []
        bonds.append(self[0].get_state_info(idx=0))
        for i in range(self.n_sites - 1):
            bonds.append(self[i + 1].get_state_info(idx=0)
                         | self[i].get_state_info(idx=-1))
        bonds.append(self[-1].get_state_info(idx=-1))

        tensors = [None] * self.n_sites
        k = 0
        for i in range(self.n_sites):
            if info is None:
                minfos = tuple(self[i].get_state_info(idx=j)
                               for j in range(1, self[i].ndim - 1))
            else:
                minfos = tuple(info.basis[j]
                               for j in range(k, k + self[i].ndim - 2))
                k += self[i].ndim - 2
            infos = (bonds[i], *minfos, bonds[i + 1])
            tensors[i] = self[i].to_sliceable(infos=infos)

        return MPS(tensors=tensors, const=self.const, opts=self.opts)

    @staticmethod
    def _to_sparse(a):
        return a.to_sparse()

    def to_sparse(self):
        tensors = [ts.to_sparse() for ts in self.tensors]
        return MPS(tensors=tensors, const=self.const, opts=self.opts)

    @staticmethod
    def _amplitude(a, det):
        return a.amplitude(det=det)

    def amplitude(self, det):
        """
        Return overlap <MPS|det>.
        MPS tensors must be sliceable."""
        tensors = [None] * self.n_sites
        k = 0
        for i in range(self.n_sites):
            assert isinstance(self[i], SliceableTensor)
            tensors[i] = self[i][(
                slice(None), *det[k:k + self[i].ndim - 2], slice(None))]
            k += self[i].ndim - 2
        assert k == len(det)
        return reduce(np.dot, tensors).item(0)
