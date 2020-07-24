
from collections import Counter
import numpy as np


class SZ:
    """non-spin-adapted spin label"""

    def __init__(self, n=0, twos=0, pg=0):
        self.n = n
        self.twos = twos
        self.pg = pg

    @property
    def is_fermion(self):
        return self.n % 2 == 1

    def __add__(self, other):
        return SZ(self.n + other.n, self.twos + other.twos, self.pg ^ other.pg)

    def __sub__(self, other):
        return SZ(self.n - other.n, self.twos - other.twos, self.pg ^ other.pg)

    def __neg__(self):
        return SZ(-self.n, -self.twos, self.pg)

    def __eq__(self, other):
        return self.n == other.n and self.twos == other.twos and self.pg == other.pg

    def __lt__(self, other):
        return (self.n, self.twos, self.pg) < (other.n, other.twos, other.pg)

    def __hash__(self):
        return hash((self.n, self.twos, self.pg))

    def __repr__(self):
        if self.twos % 2 == 1:
            return "< N=%d SZ=%d/2 PG=%d >" % (self.n, self.twos, self.pg)
        else:
            return "< N=%d SZ=%d PG=%d >" % (self.n, self.twos // 2, self.pg)


class BondInfo(Counter):
    """
    collection of quantum labels

    Attributes:
        self : Counter
            dict of quantum label and number of states
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def n_bonds(self):
        return sum(self.values())

    def item(self):
        assert len(self) == 1 and self[list(self)[0]] == 1
        return list(self)[0]

    @staticmethod
    def tensor_product(a, b, ref=None):
        quanta = BondInfo()
        for ka, va in a.items():
            for kb, vb in b.items():
                if ref is None or ka + kb in ref:
                    quanta[ka + kb] += va * vb
        return quanta
    
    def __or__(self, other):
        return BondInfo(super().__or__(other))

    def __add__(self, other):
        return BondInfo(super().__add__(other))

    def __neg__(self):
        return BondInfo({-k: v for k, v in self.items()})

    def __mul__(self, other):
        return BondInfo.tensor_product(self, other)

    def filter(self, other):
        return BondInfo({k: min(other[k], v)
                         for k, v in self.items() if k in other})

    def truncate(self, bond_dim, ref=None):
        n_total = self.n_bonds
        if n_total > bond_dim:
            for k, v in self.items():
                self[k] = int(np.ceil(v * bond_dim // n_total + 0.1))
                if ref is not None:
                    self[k] = min(self[k], ref[k])

    def __repr__(self):
        print(self.items())
        return " ".join(["%r = %d" % (k, v) for k, v in sorted(self.items(), key=lambda x: x[0])])


class BondFusingInfo(BondInfo):
    """
    collection of quantum labels
    with quantum label information for fusing/unfusing

    Attributes:
        self : Counter
            dict of quantum label and number of states
        finfo : dict(SZ -> Counter((SZ, SZ) -> int))
    """

    def __init__(self, *args, **kwargs):
        finfo = kwargs.pop("finfo", None)
        self.finfo = finfo if finfo is not None else {}
        super().__init__(*args, **kwargs)

    @staticmethod
    def tensor_product(a, b, ref=None):
        quanta = BondInfo()
        finfo = {}
        for ka, va in sorted(a.items(), key=lambda x: x[0]):
            for kb, vb in sorted(b.items(), key=lambda x: x[0]):
                kc = ka + kb
                if ref is None or kc in ref:
                    if kc not in finfo:
                        finfo[kc] = Counter()
                    finfo[kc][(ka, kb)] = quanta[kc]
                    quanta[kc] += va * vb
        return BondFusingInfo(quanta, finfo=finfo)
