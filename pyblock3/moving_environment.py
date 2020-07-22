
import numpy as np
from functools import reduce
from .algebra.linalg import davidson

def implements(np_func):
    global _numpy_func_impls
    return lambda f: (_numpy_func_impls.update({np_func: f}), f)[1]

_me_numpy_func_impls = {}
_numpy_func_impls = _me_numpy_func_impls

class MovingEnvironment:
    """Original and partially contracted tensor network <bra|mpo|ket>."""

    def __init__(self, bra, mpo, ket, opts=None):
        self.bra = bra
        self.mpo = mpo
        self.ket = ket
        assert self.bra.n_sites == self.ket.n_sites
        assert self.mpo.n_sites == self.ket.n_sites
        self.left_envs = {}
        self.right_envs = {}
        if opts is not None:
            self.bra.opts = opts
            self.mpo.opts = opts
            self.ket.opts = opts

    def __array_function__(self, func, types, args, kwargs):
        if func not in _me_numpy_func_impls:
            return NotImplemented
        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented
        return _me_numpy_func_impls[func](*args, **kwargs)

    def _left_canonicalize_site(self, mps, i):
        """Left canonicalize mps at site i."""
        mps[i], r = mps[i].left_canonicalize()
        mps[i + 1] = np.tensordot(r, mps[i + 1], axes=1)

    def _right_canonicalize_site(self, mps, i):
        """Right canonicalize mps at site i."""
        l, mps[i] = mps[i].right_canonicalize()
        mps[i - 1] = np.tensordot(mps[i - 1], l, axes=1)

    def _left_contract_rotate(self, i, prev=None):
        """Left canonicalize bra and ket at site i.
        Contract left-side contracted mpo with mpo at site i."""
        if i == -1:
            return self.mpo[0].__class__.identity(n=1, q=self.ket[0].get_state_info(0).item())
        # contract
        r = np.dot(prev, self.mpo[i])
        # left canonicalize
        self._left_canonicalize_site(self.ket, i)
        if self.ket is not self.bra:
            self._left_canonicalize_site(self.bra, i)
        # rotate
        du, dd = self.bra[i].ndim - 2, self.ket[i].ndim - 2
        r = np.tensordot(r, self.ket[i], axes=([*range(du + 2, du + dd + 3)], [*range(0, dd + 1)]))
        r = np.tensordot(self.bra[i], r, axes=([*range(0, du + 1)], [*range(1, du + 2)]))
        return r.transpose((1, 0, 3, 2))

    def _right_contract_rotate(self, i, prev=None):
        """Right canonicalize bra and ket at site i.
        Contract right-side contracted mpo with mpo at site i."""
        if i == self.n_sites:
            qk, qm = self.ket[-1].get_state_info(-1).item(), self.mpo[-1].get_state_info(-1).item()
            return self.mpo[-1].__class__.identity(n=1, q=(qm, qk, qk, qm))
        # contract
        r = np.dot(self.mpo[i], prev)
        # right canonicalize
        self._right_canonicalize_site(self.ket, i)
        if self.ket is not self.bra:
            self._right_canonicalize_site(self.bra, i)
        # rotate
        du, dd = self.bra[i].ndim - 2, self.ket[i].ndim - 2
        r = np.tensordot(r, self.ket[i], axes=([*range(du + 2, du + dd + 3)], [*range(1, dd + 2)]))
        r = np.tensordot(self.bra[i], r, axes=([*range(1, du + 2)], [*range(1, du + 2)]))
        return r.transpose((1, 0, 3, 2))

    def _initialize(self, l=0, r=2):
        """Canonicalize bra and ket around sites [l, r). Contract mpo around sites [l, r)."""
        self.left_envs[-2] = None
        self.right_envs[self.n_sites + 1] = None
        for i in range(-1, l):
            if i not in self.left_envs or self.left_envs[i] is None:
                self.left_envs[i] = self._left_contract_rotate(i, prev=self.left_envs[i - 1])
        for i in range(self.n_sites, r - 1, -1):
            if i not in self.right_envs or self.right_envs[i] is None:
                self.right_envs[i] = self._right_contract_rotate(i, prev=self.right_envs[i + 1])

    def _effective_mpo(self, l=0, r=2):
        """Get mpo in sub-system with sites [l, r)"""
        tensors = [self.left_envs[l - 1], *self.mpo[l:r], self.right_envs[r]]
        tensors[:2] = [reduce(np.dot, tensors[:2])]
        tensors[-2:] = [reduce(np.dot, tensors[-2:])]
        return self.mpo.__class__(tensors=tensors, const=self.mpo.const, opts=self.mpo.opts)

    def _effective_mps(self, mps, l=0, r=2):
        """Get mps in sub-system with sites [l, r)"""
        tensors = [
            mps[0].__class__.identity(n=1, q=mps[0].get_state_info(0).item(), ndim=1),
            *mps[l:r],
            mps[-1].__class__.identity(n=1, q=mps[-1].get_state_info(-1).item(), ndim=1)
        ]
        tensors[:2] = [reduce(lambda a, b: np.tensordot(a, b, axes=0), tensors[:2])]
        tensors[-2:] = [reduce(lambda a, b: np.tensordot(a, b, axes=0), tensors[-2:])]
        return mps.__class__(tensors=tensors, opts=mps.opts)

    def _effective_ket(self, l=0, r=2):
        """Get ket in sub-system with sites [l, r)"""
        return self._effective_mps(self.ket, l=l, r=r)

    def _effective_bra(self, l=0, r=2):
        """Get bra in sub-system with sites [l, r)"""
        return self._effective_mps(self.bra, l=l, r=r)

    def _embedded_mps(self, mps):
        """Change mps format for embedding into larger system."""
        tensors = mps.tensors
        lt = mps[0].__class__.identity(n=1, q=mps[0].get_state_info(0).item(), ndim=1)
        rt = mps[-1].__class__.identity(n=1, q=mps[-1].get_state_info(-1).item(), ndim=1)
        tensors[0] = np.tensordot(lt, tensors[0], axes=1)
        tensors[-1] = np.tensordot(tensors[-1], rt, axes=1)
        return mps.__class__(tensors=tensors, opts=mps.opts)

    def _embedded_bra(self):
        """Change bra format for embedding into larger system."""
        return self._embedded_mps(self.bra)

    def _embedded_ket(self):
        """Change ket format for embedding into larger system."""
        return self._embedded_mps(self.ket)

    def _effective(self, l=0, r=2):
        """Get sub-system with sites [l, r)"""
        self._initialize(l=l, r=r)
        eff_ket = self._effective_ket(l=l, r=r)
        eff_bra = eff_ket if self.bra is self.ket else self._effective_bra(l=l, r=r)
        eff_mpo = self._effective_mpo(l=l, r=r)
        return MovingEnvironment(bra=eff_bra, mpo=eff_mpo, ket=eff_ket)

    def _embedded(self, me, l=0, r=2):
        """Modify sub-system with sites [l, r)"""
        self.ket[l:r] = me._embedded_ket().tensors
        if me.ket is me.bra:
            self.bra[l:r] = self.ket[l:r]
        else:
            self.bra[l:r] = me._embedded_bra().tensors
        for i in range(l, self.n_sites + 1):
            self.left_envs[i] = None
        for i in range(r - 1, -2, -1):
            self.right_envs[i] = None

    def __getitem__(self, idx):
        """Return sub-system including sites specified by ``idx``"""
        if isinstance(idx, int):
            l, r = idx, idx + 1
        elif isinstance(idx, slice):
            l = 0 if idx.start is None else idx.start
            r = self.n_sites if idx.stop is None else idx.stop
            l = l if l >= 0 else self.n_sites + l
            r = r if r >= 0 else self.n_sites + r
            assert r > l
        else:
            raise TypeError("Unknown index %r" % idx)
        return self._effective(l, r)

    def __setitem__(self, idx, me):
        """Modify sub-system including sites specified by ``idx``"""
        if isinstance(idx, int):
            l, r = idx, idx + 1
        elif isinstance(idx, slice):
            l = 0 if idx.start is None else idx.start
            r = self.n_sites if idx.stop is None else idx.stop
            l = l if l >= 0 else self.n_sites + l
            r = r if r >= 0 else self.n_sites + r
            assert r > l
        else:
            raise TypeError("Unknown index %r" % idx)
        self._embedded(me, l, r)

    @property
    def expectation(self):
        """<bra|mpo|ket> for the whole system."""
        return np.dot(self.bra, self.mpo @ self.ket)

    @staticmethod
    @implements(np.linalg.eigh)
    def _eigh(x):
        """Return ground-state energy and ground-state system."""
        w, v, ndav = davidson(x.mpo, [x.ket], k=1, iprint=False)
        return w[0], MovingEnvironment(bra=v[0], mpo=x.mpo, ket=v[0]), ndav

    def eigh(self):
        """Return ground-state energy and ground-state system."""
        return np.linalg.eigh(self)

    @property
    def n_sites(self):
        return self.ket.n_sites
