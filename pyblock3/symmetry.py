
from collections import Counter
import numpy as np


class SZ:
    """non-spin-adapted spin label"""

    def __init__(self, n=0, twos=0, pg=0):
        self.n = n
        self.twos = twos
        self.pg = pg

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


class StateInfo:
    """
    collection of quantum labels

    Attributes:
        quanta : Counter
            dict of quantum label and number of states
    """

    def __init__(self, quanta=None):
        self.quanta = quanta if quanta is not None else Counter()

    @property
    def n_states_total(self):
        return sum(self.quanta.values())

    @staticmethod
    def tensor_product(a, b, ref=None):
        quanta = Counter()
        for ka, va in a.quanta.items():
            for kb, vb in b.quanta.items():
                if ref is None or ka + kb in ref.quanta:
                    quanta[ka + kb] += va * vb
        return StateInfo(quanta)

    def __neg__(self):
        return StateInfo(quanta=Counter({-k: v for k, v in self.quanta.items()}))

    def filter(self, other):
        return StateInfo(
            quanta=Counter(
                {k: min(other.quanta[k], v)
                 for k, v in self.quanta.items() if k in other.quanta}))
    
    def truncate(self, bond_dim, ref=None):
        n_total = self.n_states_total
        if n_total > bond_dim:
            for k, v in self.quanta.items():
                self.quanta[k] = int(np.ceil(v * bond_dim // n_total + 0.1))
                if ref is not None:
                    self.quanta[k] = min(self.quanta[k], ref.quanta[k])


class StateFusingInfo(StateInfo):
    """
    collection of quantum labels
    with quantum label information for fusing/unfusing

    Attributes:
        quanta : Counter
            dict of quantum label and number of states
        finfo : dict(SZ -> Counter((SZ, SZ) -> int))
    """

    def __init__(self, quanta=None, finfo=None):
        self.finfo = finfo if finfo is not None else {}
        super().__init__(quanta)

    @staticmethod
    def tensor_product(a, b):
        quanta = Counter()
        finfo = {}
        for ka, va in a.quanta.items():
            for kb, vb in b.quanta.items():
                kc = ka + kb
                if kc not in finfo:
                    finfo[kc] = Counter()
                finfo[kc][(ka, kb)] = quanta[ka + kb]
                quanta[ka + kb] += va * vb
        return StateFusingInfo(quanta, finfo)
