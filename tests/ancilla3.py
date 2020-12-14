
import sys
sys.path[:0] = ['..']

from pyblock3.hamiltonian import Hamiltonian
from pyblock3.fcidump import FCIDUMP
from pyblock3.algebra.mpe import MPE, CachedMPE
import numpy as np
import time

flat = True
cutoff = 1E-12

fd = '../data/H8.STO6G.R1.8.FCIDUMP'
hamil = Hamiltonian(FCIDUMP(pg='d2h', mu=-1.0).read(fd), flat=flat)

mps = hamil.build_ancilla_mps()
mpo = hamil.build_qc_mpo()
mpo = hamil.build_ancilla_mpo(mpo)
mpo.const = 0.0

print('MPS = ', mps.show_bond_dims())
print('MPO = ', mpo.show_bond_dims())
mpo, error = mpo.compress(cutoff=cutoff)
print('MPO = ', mpo.show_bond_dims(), error)

init_e = np.dot(mps, mpo @ mps) / np.dot(mps, mps)
print('Initial Energy = ', init_e)
print('Error          = ', init_e - 0.3124038410492045)

beta = 0.05
mpe = CachedMPE(mps, mpo, mps)
mpe.tddmrg(bdims=[500], dt=-beta / 2, iprint=2, n_sweeps=1, n_sub_sweeps=6)
mpe.tddmrg(bdims=[500], dt=-beta / 2, iprint=2, n_sweeps=9, n_sub_sweeps=2)
