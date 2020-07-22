
from .core import method_alias

"""
Functions at module level.
The actual implementation depends on argument types.
"""

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


@method_alias('to_dense')
def to_dense(a, *args, **kwargs):
    return NotImplemented


@method_alias('to_sparse')
def to_sparse(a, *args, **kwargs):
    return NotImplemented


@method_alias('to_sliceable')
def to_sliceable(a, *args, **kwargs):
    return NotImplemented
