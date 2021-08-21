
import sys
sys.path[:0] = ['..', "../../block2-old/build"]

from pyblock3.hamiltonian import Hamiltonian
from pyblock3.fcidump import FCIDUMP
from pyblock3.algebra.mpe import MPE
import pyblock3.algebra.funcs as pbalg
import numpy as np
from functools import reduce

flat = True

# hubbard
def build_hubbard(u=2, t=1, n=8, cutoff=1E-9):
    fcidump = FCIDUMP(pg='c1', n_sites=n, n_elec=n, twos=0, ipg=0, orb_sym=[0] * n)
    hamil = Hamiltonian(fcidump, flat=flat)

    def generate_terms(n_sites, c, d):
        for i in range(0, n_sites):
            for s in [0, 1]:
                if i - 1 >= 0:
                    yield t * c[i, s] * d[i - 1, s]
                if i + 1 < n_sites:
                    yield t * c[i, s] * d[i + 1, s]
            yield u * (c[i, 0] * c[i, 1] * d[i, 1] * d[i, 0])

    return hamil, hamil.build_mpo(generate_terms, cutoff=cutoff).to_sparse()

bond_dim = 100
hamil, mpo = build_hubbard(n=8)
mpo, _ = mpo.compress(left=True, cutoff=1E-9, norm_cutoff=1E-9)
mps = hamil.build_mps(bond_dim)


bdims = [250] * 5 + [500] * 5
noises = [1E-5] * 2 + [1E-6] * 4 + [1E-7] * 3 + [0]
davthrds = [5E-3] * 4 + [1E-3] * 4 + [1E-4]

dmrg = MPE(mps, mpo, mps).dmrg(bdims=bdims, noises=noises,
                                     dav_thrds=davthrds, iprint=2, n_sweeps=10)
ener = dmrg.energies[-1]
print("Energy = %20.12f" % ener)
print(len(mps))
print(len(mpo))

print('MPS energy = ', np.dot(mps, mpo @ mps))

mps[-3:] = [reduce(pbalg.hdot, mps[-3:])]
mpo[-3:] = [reduce(pbalg.hdot, mpo[-3:])]
print(len(mps))
print(len(mpo))

print('MPO (hdot) = ', mpo.show_bond_dims())
print('MPS (hdot) = ', mps.show_bond_dims())

print('MPS energy = ', mps @ (mpo @ mps))

info = mps[-1].kron_product_info(1, 2, 3)
mps[-1] = mps[-1].fuse(1, 2, 3, info=info)
mpo[-1] = mpo[-1].fuse(4, 5, 6, info=info).fuse(1, 2, 3, info=info)
print(len(mps))
print(len(mpo))

print('MPS energy = ', mps @ (mpo @ mps))

me = MPE(mps, mpo, mps)
mex = me[0:2]
# print(mex.ket[-2])
# print(mex.ket[-1])
mex.bra[:] = [reduce(pbalg.hdot, mex.bra[:])]
print(len(mex.bra))
mex.ket[:] = [reduce(pbalg.hdot, mex.ket[:])]
print(len(mex.ket))
print(mex.expectation)
