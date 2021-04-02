from pyblock3.algebra.fermion_symmetry import U11, U1, Z4, Z2

SVD_SCREENING = 1e-10
DEFAULT_SYMMETRY = U11

symmetry_map = {"U11": U11,
                "U1": U1,
                "Z2": Z2,
                "Z4": Z4}

def set_symmetry(symmetry_string):
    global DEFAULT_SYMMETRY
    symmetry_string = symmetry_string.upper()
    if symmetry_string not in symmetry_map:
        raise KeyError("input symmetry %s not supported"%symmetry_string)
    DEFAULT_SYMMETRY = symmetry_map[symmetry_string]
