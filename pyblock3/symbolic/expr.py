
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

"""Operator symbols and expressions."""

import numpy as np
from enum import Enum, auto
from collections import Counter
import numbers


class OpNames(Enum):
    """Operator Names."""
    H = auto()
    I = auto()
    N = auto()
    NN = auto()
    NUD = auto()
    C = auto()
    D = auto()
    R = auto()
    RD = auto()
    A = auto()
    AD = auto()
    P = auto()
    PD = auto()
    B = auto()
    Q = auto()
    X = auto()
    XL = auto()
    XR = auto()

    def __repr__(self):
        return self.name

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        else:
            return NotImplemented


op_names = list(OpNames.__members__.items())


class OpExpr:
    pass


class OpElement(OpExpr):
    """
    Single operator symbol.

    Attributes:
        name : :class:`OpNames`
            Type of the operator.
        site_index : () or tuple(int..)
            Site indices of the operator.
        factor : float / complex
            scalar factor.
        q_label : SZ
            Quantum label of the operator.
    """
    __slots__ = ['name', 'site_index', 'factor', 'q_label']

    def __init__(self, name, site_index, factor=1, q_label=None):
        assert isinstance(site_index, tuple)
        if not isinstance(factor, complex):
            factor = float(factor)
        self.name = name
        self.site_index = site_index
        self.factor = factor
        self.q_label = q_label

    def __repr__(self):
        if self.factor != 1:
            if self.factor == -1:
                return '(-1 %r)' % abs(self)
            else:
                return '(%f %r)' % (self.factor, abs(self))
        if len(self.site_index) == 0:
            return repr(self.name)
        elif len(self.site_index) == 1:
            return repr(self.name) + repr(self.site_index[0])
        else:
            ssi = " ".join([str(x) for x in self.site_index])
            return repr(self.name) + "[ " + ssi + " ]"

    def __mul__(self, other):
        if other == 0:
            return 0
        elif isinstance(other, OpSum):
            return OpSum([OpString([self] + st.ops, st.factor) for st in other.strings])
        elif isinstance(other, OpElement):
            return OpString([self, other])
        elif isinstance(other, numbers.Number):
            return OpElement(self.name, self.site_index, self.factor * other, self.q_label)
        else:
            return NotImplemented

    def __rmul__(self, other):
        if other == 0:
            return 0
        elif isinstance(other, numbers.Number):
            return OpElement(self.name, self.site_index, self.factor * other, self.q_label)
        elif isinstance(other, OpSum):
            return OpSum([OpString(st.ops + [self], st.factor) for st in other.strings])
        else:
            return NotImplemented

    def __add__(self, other):
        if isinstance(other, OpElement):
            return OpSum([OpString([self]), OpString([other])])
        elif other == 0:
            return self
        elif isinstance(other, OpSum):
            return OpSum([OpString([self])] + other.strings)
        else:
            return NotImplemented

    def __radd__(self, other):
        if other == 0:
            return self
        if isinstance(other, OpSum):
            return OpSum(other.strings + [OpString([self])])
        else:
            return NotImplemented

    def __abs__(self):
        return OpElement(self.name, self.site_index, 1, self.q_label)

    def __neg__(self):
        return OpElement(self.name, self.site_index, -self.factor, self.q_label)

    def __eq__(self, other):
        if not isinstance(other, OpElement):
            return False
        else:
            return self.name == other.name and self.site_index == other.site_index \
                and self.factor == other.factor

    def __lt__(self, other):
        if other == 0:
            return False
        elif not isinstance(other, OpElement):
            return NotImplemented
        elif self.name != other.name:
            return self.name < other.name
        elif self.site_index != other.site_index:
            return self.site_index < other.site_index
        elif self.factor != other.factor:
            return self.factor < other.factor
        else:
            return False

    def __hash__(self):
        return hash((self.name, self.site_index, self.factor))

    @staticmethod
    def parse_site_index(expr):
        if len(expr) == 0:
            return ()
        elif expr.startswith('('):
            return tuple([int(x.strip()) for x in expr[1:-1].split(',')])
        elif expr.startswith('['):
            return tuple([int(x.strip()) for x in expr[1:-1].strip().split(' ')])
        else:
            return (int(expr), )

    @staticmethod
    def parse(expr):
        """Parse a str to operator symbol."""
        for name, op in sorted(op_names, key=lambda x: -len(x[0])):
            if expr.startswith(name):
                return OpElement(op, OpElement.parse_site_index(expr[len(name):].strip()))


class OpString(OpExpr):
    """
    String of operator symbols representing direct product of single operator symbols.

    Attributes:
        ops : list(:class:`OpElement`)
            A list of single operator symbols.
        sign : int (1 or -1)
            Sign factor. With SU(2) factor considered
    """
    __slots__ = ['ops', 'factor']

    def __init__(self, ops, factor=1):
        self.factor = np.prod([x.factor for x in ops]) * factor
        self.ops = [abs(x) for x in ops]

    @property
    def op(self):
        assert len(self.ops) == 1
        return self.factor * self.ops[0]

    def sort(self, fermion=True):
        if fermion:
            for i, ix in enumerate(self.ops):
                for jx in self.ops[i + 1:]:
                    if ix.site_index > jx.site_index:
                        self.factor = -self.factor
        self.ops.sort(key=lambda x: x.site_index[0])

    def __repr__(self):
        if self.factor == 1:
            return " ".join([repr(x) for x in self.ops])
        elif isinstance(self.factor, float):
            return '(%10.5f %r)' % (self.factor, abs(self))
        else:
            return '(%10.5f + %10.5fj %r)' % (self.factor.real, self.factor.imag, abs(self))

    def __truediv__(self, other):
        if isinstance(other, float) or isinstance(other, complex):
            return OpString(self.ops, self.factor / other)
        else:
            return NotImplemented

    def __abs__(self):
        return OpString(self.ops, 1)

    def __hash__(self):
        return hash((self.factor, *self.ops))

    def __mul__(self, other):
        if isinstance(other, OpElement):
            return OpString(self.ops + [other], self.factor)
        elif other == 0:
            return 0
        elif isinstance(other, float) or isinstance(other, complex) or isinstance(other, int):
            return OpString(self.ops, self.factor * other) if abs(self.factor * other) > 1E-12 else 0
        elif isinstance(other, OpString):
            return OpString(self.ops + other.ops, self.factor * other.factor)
        else:
            return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, OpElement):
            return OpString([other] + self.ops, self.factor)
        elif other == 0:
            return 0
        elif isinstance(other, float) or isinstance(other, complex) or isinstance(other, int):
            return OpString(self.ops, self.factor * other) if abs(self.factor * other) > 1E-12 else 0
        else:
            return NotImplemented

    def __add__(self, other):
        if other == 0:
            return self
        if isinstance(other, OpSum):
            return OpSum([self] + other.strings)
        elif isinstance(other, OpString):
            return OpSum([self, other])
        else:
            return NotImplemented

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return NotImplemented

    def __eq__(self, other):
        if not isinstance(other, OpString):
            return False
        else:
            return (len(other.ops) == len(self.ops) and
                    self.factor == other.factor and
                    all([pa == pb for pa, pb in zip(self.ops, other.ops)]))


class OpSum(OpExpr):
    """
    Sum of direct product of single operator symbols.

    Attributes:
        strings : list(:class:`OpString`)
    """
    __slots__ = ['strings']

    def __init__(self, strings):
        assert isinstance(strings, list)
        self.strings = strings

    def simplify(self):
        mp = Counter()
        for string in self.strings:
            mp[tuple(string.ops)] += string.factor
        self.strings = [OpString(list(k), v)
                        for k, v in mp.items() if abs(v) > 1E-12]

    def sort(self, fermion=True):
        self.strings = [x for x in self.strings if x != 0]
        for string in self.strings:
            string.sort(fermion=fermion)

    def __repr__(self):
        return " + ".join([repr(x) for x in self.strings])

    def __add__(self, other):
        if other == 0:
            return self
        elif isinstance(other, OpString):
            return OpSum(self.strings + [other])
        elif isinstance(other, OpSum):
            return OpSum(self.strings + other.strings)
        else:
            return NotImplemented

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, float) or isinstance(other, complex):
            return OpSum([x / other for x in self.strings])
        else:
            return NotImplemented

    def __mul__(self, other):
        if other == 0:
            return 0
        elif isinstance(other, numbers.Number) or isinstance(other, OpString):
            return OpSum([x * other for x in self.strings])
        elif isinstance(other, OpSum):
            strings = []
            for x in self.strings:
                strings.extend((x * other).strings)
            return OpSum(strings)
        else:
            return NotImplemented

    def __rmul__(self, other):
        if other == 0:
            return 0
        elif isinstance(other, numbers.Number) or isinstance(other, OpString):
            return OpSum([other * x for x in self.strings])
        elif isinstance(other, OpSum):
            strings = []
            for x in self.strings:
                strings.extend((other * x).strings)
            return OpSum(strings)
        else:
            return NotImplemented

    def __eq__(self, other):
        if not isinstance(other, OpSum):
            return False
        else:
            return (len(other.strings) == len(self.strings) and
                    all([sa == sb for sa, sb in zip(self.strings, other.strings)]))
