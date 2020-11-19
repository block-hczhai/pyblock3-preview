
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
Functions at module level.
The actual implementation depends on argument types.
"""

from .core import method_alias
from .linalg import *


@method_alias('amplitude')
def amplitude(a, *args, **kwargs):
    return NotImplemented


@method_alias('fuse')
def fuse(a, *args, **kwargs):
    return NotImplemented


@method_alias('tensordot')
def tensordot(a, *args, **kwargs):
    return NotImplemented


@method_alias('dot')
def dot(a, *args, **kwargs):
    return NotImplemented


@method_alias('hdot')
def hdot(a, *args, **kwargs):
    return NotImplemented


@method_alias('to_dense')
def to_dense(a, *args, **kwargs):
    return NotImplemented


@method_alias('to_sparse')
def to_sparse(a, *args, **kwargs):
    return NotImplemented


@method_alias('to_sliceable')
def to_sliceable(a, *args, **kwargs):
    return NotImplemented
