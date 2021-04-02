
#  pyblock3: An Efficient python MPS/DMRG Library
#  Copyright (C) 2020 The pyblock3 developers. All Rights Reserved.
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program. If not, see <https://www.gnu.org/licenses/>.
#
#

"""
Definition of quantum numbers.
"""

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

    def to_flat(self):
        return ((self.n + 8192) * 16384 + (self.twos + 8192)) * 8 + self.pg

    @staticmethod
    def from_flat(x):
        return SZ((x // 131072) % 16384 - 8192, (x // 8) % 16384 - 8192, x % 8)

    @staticmethod
    def is_flat_fermion(x):
        return (x & 8) != 0

    def __add__(self, other):
        return self.__class__(self.n + other.n, self.twos + other.twos, self.pg ^ other.pg)

    def __sub__(self, other):
        return self.__class__(self.n - other.n, self.twos - other.twos, self.pg ^ other.pg)

    def __neg__(self):
        return self.__class__(-self.n, -self.twos, self.pg)

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
            dict of quantum label and number of bonds
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def n_bonds(self):
        """Total number of bonds."""
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

    def __and__(self, other):
        """Intersection."""
        return BondInfo(super().__and__(other))

    def __or__(self, other):
        """Union."""
        return BondInfo(super().__or__(other))

    def __add__(self, other):
        """Sum of number of states."""
        return BondInfo(super().__add__(other))

    def __neg__(self):
        """Negattion of keys."""
        return BondInfo({-k: v for k, v in self.items()})

    def __mul__(self, other):
        """Tensor product."""
        return BondInfo.tensor_product(self, other)

    def __xor__(self, other):
        """Tensor product, including fusing information."""
        return BondFusingInfo.tensor_product(self, other)

    def filter(self, other):
        return BondInfo({k: min(other[k], v)
                         for k, v in self.items() if k in other})

    def truncate(self, bond_dim, ref=None):
        n_total = self.n_bonds
        if n_total > bond_dim:
            for k, v in self.items():
                self[k] = int(np.ceil(v * bond_dim / n_total + 0.1))
                if ref is not None:
                    self[k] = min(self[k], ref[k])

    def keep_maximal(self):
        maxk = max(self)
        return BondInfo({maxk: self[maxk]})

    def __repr__(self):
        return " ".join(["%r = %d" % (k, v) for k, v in sorted(self.items(), key=lambda x: x[0])])


class BondFusingInfo(BondInfo):
    """
    collection of quantum labels
    with quantum label information for fusing/unfusing

    Attributes:
        self : Counter
            dict of quantum label and number of states
        finfo : dict(SZ -> dict(tuple(SZ) -> (int, tuple(int))))
            For each fused q and unfused q,
            the starting index in fused dim and the shape of unfused block
        pattern : str
            a str of '+'/'-' indicating how quantum numbers are combined
    """

    def __init__(self, *args, **kwargs):
        finfo = kwargs.pop("finfo", None)
        self.pattern = kwargs.pop("pattern", None)
        self.finfo = finfo if finfo is not None else {}
        super().__init__(*args, **kwargs)

    @staticmethod
    def tensor_product(*infos, ref=None, pattern=None, trans=None):
        """
        Direct product of a collection of BondInfo.

        Args:
            infos : tuple(BondInfo)
                BondInfo for each unfused leg.
            ref : BondInfo (optional)
                Reference fused BondInfo.
        """
        quanta = BondInfo()
        finfo = {}
        if trans is None:
            qs = [sorted(q.items(), key=lambda x: x[0]) for q in infos]
        else:
            qs = [sorted(q.items(), key=lambda x: trans(x[0])) for q in infos]
        if pattern is None:
            pattern = "+" * len(qs)
        it = np.zeros(tuple(len(q) for q in qs), dtype=int)
        nit = np.nditer(it, flags=['multi_index'])
        for _ in nit:
            x = nit.multi_index
            ps = [iq[ix][0] if ip == '+' else -iq[ix][0]
                  for iq, ix, ip in zip(qs, x, pattern)]
            qls = tuple(iq[ix][0] for iq, ix in zip(qs, x))
            shs = tuple(iq[ix][1] for iq, ix in zip(qs, x))
            q = np.add.reduce(ps)
            if ref is None or q in ref:
                if q not in finfo:
                    finfo[q] = {}
                finfo[q][qls] = quanta[q], shs
                quanta[q] += np.prod(shs, dtype=np.uint32)
        return BondFusingInfo(quanta, finfo=finfo, pattern=pattern)

    @staticmethod
    def kron_sum(items, ref=None, pattern=None):
        """
        Direct sum of combination of quantum numbers.

        Args:
            items : list((tuple(SZ), tuple(int)))
                The items to be summed.
                For every item, the q_labels and matrix shape are given.
                Repeated items are okay (will not be considered).
            ref : BondInfo (optional)
                Reference fused BondInfo.
        """
        quanta = BondInfo()
        finfo = {}
        collected = {}
        if pattern is None:
            pattern = "+" * len(items[0][0])
        for qs, shs in items:
            q = np.add.reduce(
                [iq if ip == "+" else -iq for iq, ip in zip(qs, pattern)])
            if ref is None or q in ref:
                if q not in collected:
                    collected[q] = []
                collected[q].append((qs, shs))
        for q, v in collected.items():
            v.sort(key=lambda x: x[0])
            finfo[q] = {}
            for qs, shs in v:
                if qs not in finfo[q]:
                    finfo[q][qs] = quanta[q], shs
                    quanta[q] += np.prod(shs, dtype=np.uint32)
                else:
                    assert finfo[q][qs][1] == shs
        return BondFusingInfo(quanta, finfo=finfo, pattern=pattern)
