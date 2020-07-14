
from .symmetry import StateInfo, StateFusingInfo
from .tensor import SparseTensor, SubTensor
from collections import Counter
import numpy as np


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

    def set_bond_dimension_fci(self):
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

        for d in range(0, self.n_sites + 1):
            self.left_dims[d].filter(self.right_dims[d])
            self.right_dims[d].filter(self.left_dims[d])

    def set_bond_dimension_occ(self, bond_dim, occ, bias=1):
        """bond dimensions from occupation numbers"""
        return NotImplemented

    def set_bond_dimension(self, bond_dim):
        """Truncated bond dimension based on FCI quantum numbers
            each FCI quantum number has at least one state kept"""

        if self.left_dims is None:
            self.set_bond_dimension_fci()

        for d in range(0, self.n_sites):
            ref = StateInfo.tensor_product(self.left_dims[d], self.basis[d])
            self.left_dims[d + 1].truncate(bond_dim, ref)
        for d in range(self.n_sites - 1, -1, -1):
            ref = StateInfo.tensor_product(
                -self.basis[d], self.right_dims[d + 1])
            self.right_dims[d].truncate(bond_dim, ref)


class MPS:
    """
    Matrix Product State.
    Attributes:
        tensors : list(SparseTensor)
            A list of block-sparse tensors (fused or unfused).
        n_sites : int
            Number of sites.
        center : non-negative int or -1
            If center = -1: uncanonicalized MPS
            Otherwise: canonical center of canonicalized MPS
        fused : bool
            Whether MPS tensors are fused
        opts : dict or None
            options indicating how bond dimension trunctation
            should be done after MPO @ MPS, etc.
            Possible options are: max_bond_dim, cutoff
    """

    def __init__(self, tensors, center=-1, fused=False, opts=None):
        self.tensors = tensors
        self.center = center
        self.fused = fused
        self.opts = opts if opts is not None else {}

    @property
    def n_sites(self):
        """Number of sites"""
        return len(self.tensors)

    @staticmethod
    def from_mps_info(info):
        """Construct unfused MPS from MPSInfo."""
        tensors = [None] * info.n_sites
        for i in range(info.n_sites):
            tensors[i] = SparseTensor.init_mps_tensor(
                info.left_dims[i], info.basis[i], info.left_dims[i + 1])
        return MPS(tensors=tensors)

    def randomize(self, low=0, high=1):
        """Fill random numbers."""
        for i in range(self.n_sites):
            self.tensors[i].randomize(low, high)

    def __mul__(self, other):
        """Scalar multiplication"""
        return MPS(tensors=[self.tensors[0] * other]
                   + self.tensors[1:], center=self.center, fused=self.fused, opts=self.opts)

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        """Times (-1)"""
        return MPS(tensors=[-self.tensors[0]]
                   + self.tensors[1:], center=self.center, fused=self.fused, opts=self.opts)

    def __add__(self, other):
        """Add two MPS"""
        assert isinstance(other, MPS)
        assert self.n_sites == other.n_sites

        bonds = self.get_bond_dims(), other.get_bond_dims()
        sum_bonds = [bondx + bondy for bondx, bondy in zip(*bonds)]

        tensors = []
        for i in range(self.n_sites):
            if self.tensors[i] is None:
                assert other.tensors[i] is None
                tensors.append(None)
                continue
            if i != 0:
                lb = sum_bonds[i - 1]
            if i != self.n_sites - 1:
                rb = sum_bonds[i]
            sub_mp = {}
            # find required new blocks and their shapes
            for block in self.tensors[i].blocks + other.tensors[i].blocks:
                q = block.q_labels
                sh = block.reduced.shape
                if q not in sub_mp:
                    mshape = list(sh)
                    if i != 0:
                        mshape[0] = lb[q[0]]
                    if i != self.n_sites - 1:
                        mshape[-1] = rb[q[-1]]
                    sub_mp[q] = SubTensor(q, np.zeros(tuple(mshape)))
            # copy block self.blocks to smaller index in new block
            for block in self.tensors[i].blocks:
                q = block.q_labels
                sh = block.reduced.shape
                if i == 0:
                    sub_mp[q].reduced[..., : sh[-1]] += block.reduced
                elif i == self.n_sites - 1:
                    sub_mp[q].reduced[: sh[0], ...] += block.reduced
                else:
                    sub_mp[q].reduced[: sh[0], ..., : sh[-1]] += block.reduced
            # copy block other.blocks to greater index in new block
            for block in other.tensors[i].blocks:
                q = block.q_labels
                sh = block.reduced.shape
                if i == 0:
                    sub_mp[q].reduced[..., -sh[-1]:] += block.reduced
                elif i == self.n_sites - 1:
                    sub_mp[q].reduced[-sh[0]:, ...] += block.reduced
                else:
                    sub_mp[q].reduced[-sh[0]:, ..., -sh[-1]:] += block.reduced
            tensors.append(SparseTensor(blocks=list(sub_mp.values())))
        return MPS(tensors=tensors)

    def __sub__(self, other):
        return self + (-other)

    def __or__(self, other):
        """Dot product <MPS|MPS>"""

    def dot(self, other):
        """Dot product <MPS|MPS>"""

    def dot_determinants(self, dets):
        """Return overlap <MPS|det> for each det in dets"""

    def __matmul__(self, other):
        """
        (a) Contraction of two MPS. <MPS|MPS>.
        (b) <MPS|MPO>."""

    def canonicalize(self, center):
        """
        MPS/MPO canonicalization.

        Args:
            center : int
                Site index of canonicalization center.
        """
        for i in range(0, center):
            if self.tensors[i] is None:
                continue
            rs = self.tensors[i].left_canonicalize()
            if i + 1 < self.n_sites and self.tensors[i + 1] is not None:
                self.tensors[i + 1].left_multiply(rs)
            elif i + 2 < self.n_sites:
                self.tensors[i + 2].left_multiply(rs)
        for i in range(self.n_sites - 1, center, -1):
            if self.tensors[i] is None:
                continue
            ls = self.tensors[i].right_canonicalize()
            if i - 1 >= 0 and self.tensors[i - 1] is not None:
                self.tensors[i - 1].right_multiply(ls)
            elif i - 2 >= 0:
                self.tensors[i - 2].right_multiply(ls)
        self.center = center

    def compress(self, bond_dim=-1, cutoff=0.0, left=True):
        """
        MPS/MPO bond dimension compression.
        Args:
            k : int
                Maximal total bond dimension.
                If `k == -1`, no restriction in total bond dimension.
            cutoff : double
                Minimal kept singluar value.
            left : bool
                If left, canonicalize to right boundary and then svd to left.
                Otherwise, canonicalize to left boundary and then svd to right.
        """

    def __getitem__(self, i):
        return self.tensors[i]

    def __setitem__(self, i, ts):
        self.tensors[i] = ts

    def norm(self):
        return np.sqrt(self | self)

    def unfuse(self):
        """Transform fused MPS to unfused MPS."""

    def fuse(self):
        """Transform unfused (already canonicalized) MPS to fused MPS."""
        assert not self.fused
        assert self.center != -1

        for i in range(0, self.n_sites):
            l = self.tensors[i].get_state_info(0)
            m = self.tensors[i].get_state_info(1)
            r = self.tensors[i].get_state_info(2)
            if i <= self.center:
                info = StateFusingInfo.tensor_product(l, m, ref=r)
                self.tensors[i] = self.tensors[i].fuse(0, 1, info)
            else:
                info = StateFusingInfo.tensor_product(-m, r, ref=l)
                self.tensors[i] = self.tensors[i].fuse(1, 2, info, rev=True)

        self.fused = True

class MPO(MPS):
    """
    Matrix Product Operator.
    Attributes:
        tensors : list(SparseTensor)
            A list of MPO tensors.
        const_e : float
            constant energy term.
        n_sites : int
            Number of sites.
    """

    def __init__(self, tensors=None, const_e=0.0):
        self.const_e = const_e
        super().__init__(tensors=tensors)

    def deep_copy(self):
        """Deep copy."""

    def __mul__(self, other):
        """Scalar multiplication."""

    def __neg__(self):
        """Times (-1)."""

    def __add__(self, other):
        """Add two MPO. data in `other` MPO will be put in larger reduced indices."""

    def __matmul__(self, other):
        """
        (a) Contraction of MPO and MPS. MPO |MPS>. (other : MPS)
        (b) Contraction of MPO and MPO. MPO * MPO. (other : MPO)
        """
