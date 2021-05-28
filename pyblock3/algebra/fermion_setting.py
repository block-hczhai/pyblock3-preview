import sys
from .fermion_symmetry import U11, U1, Z4, Z2, Z22

this = sys.modules[__name__]
this.SVD_SCREENING = 1e-10
this.DEFAULT_SYMMETRY = U11
this.DEFAULT_FLAT = True
this.DEFAULT_FERMION = True

symmetry_map = {"U11": U11,
                "U1": U1,
                "Z2": Z2,
                "Z4": Z4,
                "Z22": Z22}

def set_symmetry(symmetry):
    if isinstance(symmetry, str):
        symmetry = symmetry.upper()
        if symmetry not in symmetry_map:
            raise KeyError("input symmetry %s not supported"%symmetry_string)
        this.DEFAULT_SYMMETRY = symmetry_map[symmetry]
    else:
        this.DEFAULT_SYMMETRY = symmetry

def set_flat(flat):
    this.DEFAULT_FLAT = flat

def set_fermion(fermion):
    this.DEFAULT_FERMION = fermion

def set_options(**kwargs):
    symmetry = kwargs.pop("symmetry", None)
    fermion = kwargs.pop("fermion", None)
    flat = kwargs.pop("flat", None)
    if symmetry is not None:
        set_symmetry(symmetry)
    if fermion is not None:
        set_fermion(fermion)
    if flat is not None:
        set_flat(flat)

def dispatch_settings(**kwargs):
    keys = list(kwargs.keys())
    for ikey in keys:
        if kwargs.get(ikey) is None:
            kwargs[ikey] = getattr(this, "DEFAULT_"+ikey.upper())
    _settings = [kwargs.pop(ikey) for ikey in keys]
    if len(_settings) == 1:
        _settings = _settings[0]
    return _settings
