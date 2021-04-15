import numpy as np

class U1:
    '''
    U1 symmetry class
    '''
    def __init__(self, n=0):
        self.n = n

    def __add__(self, other):
        return self.__class__(self.n + other.n)

    def __sub__(self, other):
        return self.__class__(self.n - other.n)

    def __neg__(self):
        return self.__class__(-self.n)

    def __eq__(self, other):
        return self.n == other.n

    def __lt__(self, other):
        return self.n < other.n

    def __hash__(self):
        return hash((self.n, ))

    def __repr__(self):
        return "< N=%d >" % (self.n)

    @property
    def parity(self):
        return self.n % 2

    @classmethod
    def qpn_to_flat(cls, n):
        return (n + 8192) * 131072

    @classmethod
    def flat_to_qpn(cls, x):
        return x // 131072 - 8192

    @classmethod
    def from_flat(cls, x):
        return cls(cls.flat_to_qpn(x))

    def to_flat(self):
        return self.qpn_to_flat(self.n)

    @classmethod
    def flat_to_parity(cls, x):
        '''
        quick drive to compute parity directly from flat numbers
        '''
        return (x // 131072) % 2

    @classmethod
    def flip_flat(cls, flat_array):
        '''
        quick drive to compute flat numbers of the negative symmetry label
        '''
        return 2147483648 - flat_array

    @classmethod
    def _compute(cls, pattern, qpns, offset=None, neg=False):
        '''
        quick drive to compute symmetry label calculus
        '''
        is_cls = isinstance(qpns[0], cls)
        if is_cls:
            assert(len(pattern)==len(qpns))
            out = cls()
            if offset is not None:
                ix, dq = offset
                pattern += ix
                qpns = list(qpns) + [dq, ]
            for iq, ix in zip(qpns, pattern):
                if ix=="+":
                    out += iq
                else:
                    out -= iq
            if neg:
                out = -out
        elif isinstance(qpns, np.ndarray):
            nsec = len(pattern)
            ndim = qpns.ndim
            if ndim ==1:
                newdat = qpns.copy().astype(int).reshape(1,nsec)
            else:
                newdat = qpns.copy().astype(int)
            nplus = pattern.count("+")
            nmin = pattern.count("-")
            narr = cls.flat_to_qpn(newdat)
            if offset is not None:
                ix, dq = offset
                if not isinstance(dq, cls):
                    dq = cls.from_flat(dq)
                if ix=="-":
                    dq = -dq
            else:
                dq = cls(0)
            offn = dq.n
            for ix, istr in enumerate(pattern):
                if istr=="-":
                    narr[:,ix] *= -1
            if neg:
                narr = -np.sum(narr, axis=1) - offn
            else:
                narr = np.sum(narr, axis=1) + offn
            out = cls.qpn_to_flat(narr)
            if ndim==1:
                out = out[0]
        else:
            raise TypeError("qpns must be a list/tuple of fermion symmetry class or flat numbers")
        return out

class Z2(U1):
    '''
    Z2 symmetry class
    '''
    _modulus = 2
    def __init__(self, n=0):
        self.n = n % self._modulus

    @classmethod
    def flat_to_parity(cls, x):
        return x % 2

    @classmethod
    def qpn_to_flat(cls, n):
        return n % cls._modulus

    @classmethod
    def flat_to_qpn(cls, x):
        return x % cls._modulus

    @classmethod
    def flip_flat(cls, flat_array):
        return (-flat_array) % cls._modulus

class Z4(Z2):
    '''
    Z4 symmetry class
    '''
    _modulus = 4

class U11:
    '''
    U1 prod U1 symmetry class
    '''
    def __init__(self, n=0, sz=0):
        self.n = n
        self.sz = sz

    def __add__(self, other):
        return self.__class__(self.n + other.n, self.sz + other.sz)

    def __sub__(self, other):
        return self.__class__(self.n - other.n, self.sz - other.sz)

    def __eq__(self, other):
        return self.n == other.n and self.sz == other.sz

    def __neg__(self):
        return self.__class__(-self.n, -self.sz)

    def __lt__(self, other):
        return (self.n, self.sz) < (other.n, other.sz)

    def __hash__(self):
        return hash((self.n, self.sz))

    @property
    def parity(self):
        return self.n % 2

    @classmethod
    def flat_to_parity(cls, x):
        return (x // 131072) % 2

    @classmethod
    def qpn_to_flat(cls, n, sz):
        return ((n + 8192) * 16384 + (sz + 8192)) * 8

    @classmethod
    def flat_to_qpn(cls, x):
        return (x // 131072) % 16384 - 8192, (x // 8) % 16384 - 8192

    @classmethod
    def flip_flat(cls, flat_array):
        return 2147614720 - flat_array

    def to_flat(self):
        return self.qpn_to_flat(self.n, self.sz)

    @classmethod
    def from_flat(cls, x):
        return cls(*cls.flat_to_qpn(x))

    def __repr__(self):
        return "< N=%d SZ=%d >" % (self.n, self.sz)

    @classmethod
    def _compute(cls, pattern, qpns, offset=None, neg=False):
        is_cls = isinstance(qpns[0], cls)
        if is_cls:
            assert(len(pattern)==len(qpns))
            out = cls()
            if offset is not None:
                ix, dq = offset
                pattern += ix
                qpns = list(qpns) + [dq, ]
            for iq, ix in zip(qpns, pattern):
                if ix=="+":
                    out += iq
                else:
                    out -= iq
            if neg:
                out = -out
        elif isinstance(qpns, np.ndarray):
            nsec = len(pattern)
            ndim = qpns.ndim
            if ndim ==1:
                newdat = qpns.copy().astype(int).reshape(1,nsec)
            else:
                newdat = qpns.copy().astype(int)
            nplus = pattern.count("+")
            nmin = pattern.count("-")
            narr, zarr = cls.flat_to_qpn(newdat)
            if offset is not None:
                ix, dq = offset
                if not isinstance(dq, cls):
                    dq = cls.from_flat(dq)
                if ix=="-":
                    dq = -dq
            else:
                dq = cls(0)
            offn = dq.n
            offz = dq.sz
            for ix, istr in enumerate(pattern):
                if istr=="-":
                    narr[:,ix] *= -1
                    zarr[:,ix] *= -1
            if neg:
                narr = -np.sum(narr, axis=1) - offn
                zarr = -np.sum(zarr, axis=1) - offz
            else:
                narr = np.sum(narr, axis=1) + offn
                zarr = np.sum(zarr, axis=1) + offz
            out = cls.qpn_to_flat(narr, zarr)
            if ndim==1:
                out = out[0]
        else:
            raise TypeError("qpns must be a list/tuple of fermion symmetry class or flat numbers")
        return out

class Z22(U11):
    '''
    Z2 prod Z2 symmetry class
    '''
    _mod1 = 2
    _mod2 = 2
    def __init__(self, n=0, sz=0):
        self.n = n % self._mod1
        self.sz = sz % self._mod2

    @classmethod
    def qpn_to_flat(cls, n, sz):
        return n % cls._mod1 * cls._mod2 + sz % cls._mod2

    @classmethod
    def flat_to_qpn(cls, x):
        return x // cls._mod2, x % cls._mod2

    @classmethod
    def flat_to_parity(cls, x):
        return x // cls._mod2 % 2

    @classmethod
    def flip_flat(cls, flat_array):
        return - flat_array % (cls._mod1 * cls._mod2)
