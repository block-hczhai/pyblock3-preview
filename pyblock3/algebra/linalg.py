
from .core import method_alias


@method_alias('lq')
def lq(a, *args, **kwargs):
    return NotImplemented


@method_alias('qr')
def qr(a, *args, **kwargs):
    return NotImplemented


@method_alias('norm')
def norm(a, *args, **kwargs):
    return NotImplemented


@method_alias('left_svd')
def left_svd(a, *args, **kwargs):
    return NotImplemented


@method_alias('right_svd')
def right_svd(a, *args, **kwargs):
    return NotImplemented


def truncate_svd(l, s, r, *args, **kwargs):
    if l.__class__ != s.__class__:
        return l.__class__.truncate_svd(l, s, r, *args, **kwargs)
    elif r.__class__ != s.__class__:
        return r.__class__.truncate_svd(l, s, r, *args, **kwargs)
    elif hasattr(s.__class__, 'truncate_svd'):
        return s.__class__.truncate_svd(l, s, r, *args, **kwargs)
    else:
        return NotImplemented


@method_alias('canonicalize')
def canonicalize(a, *args, **kwargs):
    return NotImplemented


@method_alias('left_canonicalize')
def left_canonicalize(a, *args, **kwargs):
    return NotImplemented


@method_alias('right_canonicalize')
def right_canonicalize(a, *args, **kwargs):
    return NotImplemented


@method_alias('compress')
def compress(a, *args, **kwargs):
    return NotImplemented
