
import sys
sys.path[:0] = ['..']
from pyblock3.algebra.symmetry import SZ
from pyblock3.heisenberg import Heisenberg
import pyblock3.algebra.funcs as pbalg
from functools import reduce
import numpy as np
import os

np.random.seed(1000)

n_threads = 16
os.environ['OMP_NUM_THREADS'] = str(n_threads)
os.environ['MKL_NUM_THREADS'] = str(n_threads)

n = 8
topology = [
    (1, 2, 1.00),
    (2, 3, 1.00),
    (3, 4, 1.00),
    (1, 4, 1.00),
    (1, 3, 1.00),
    (2, 4, 1.00),

    (5, 6, 1.00),
    (5, 7, 1.00),
    (6, 8, 1.00),
    (7, 8, 1.00),
    (5, 8, 1.00),
    (6, 7, 1.00),

    (2, 5, 1.00),
    (2, 6, 1.00),
    (2, 7, 1.00),

    (3, 5, 1.00),
    (3, 6, 1.00),
    (3, 7, 1.00),

    (4, 5, 1.00),
    (4, 6, 1.00),
    (4, 7, 1.00),
]

topology = [(i - 1, j - 1, k) for i, j, k in topology]
hamil = Heisenberg(twos=1, n_sites=n, topology=topology, flat=False)
mpo = hamil.build_mpo().to_sparse()
print('MPO = ', mpo.show_bond_dims())

mpo[:] = [reduce(pbalg.hdot, mpo[:])]
info = mpo[-1].kron_product_info(*range(1, n + 1))
for k in list(info.keys()):
    if k != SZ(0, 0, 0):
        del info[k]

mpo[-1] = mpo[-1].fuse(*range(9, n + 9), info=info).fuse(*range(1, n + 1), info=info)
mat = mpo[-1].to_dense()
mat = mat.reshape(mat.shape[1:-1])
print(mat.shape, np.linalg.norm(mat - mat.T))
v = np.linalg.eigh(mat)[0]
ref = -4.18649167
print("%8.3f" * len(v[:20]) % tuple(v[:20]))
print("%8.3f" * len(v[:20]) % tuple((v - ref)[:20]))

#   -0.000   0.917   1.070   1.070   1.070   1.070   1.463   1.463   1.463   1.463
#    1.729   1.729   1.729   1.729   1.936   1.936   1.936   1.936   2.436   2.436
