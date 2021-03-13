
import sys
sys.path[:0] = ['..', "../../block2-old/build"]

from functools import reduce
import time
import numpy as np
import pyblock3.algebra.funcs as pbalg
from pyblock3.algebra.mpe import MPE
# from pyblock3.aux.hamil import HamilTools
from pyblock3.hamiltonian import Hamiltonian
from pyblock3.fcidump import FCIDUMP
from pyblock3.symbolic.symbolic_mpo import QCSymbolicMPO
from pyblock3.algebra.mps import MPSInfo, MPS
from pyinstrument import Profiler
profiler = Profiler()

flat = True
profile = False

np.random.seed(1000)

# fd = '../data/HUBBARD-L8.FCIDUMP'
fd = '../data/HUBBARD-L16.FCIDUMP'
# fd = '../data/N2.STO3G.FCIDUMP'
# fd = '../data/H8.STO6G.R1.8.FCIDUMP'
# fd = '../my_test/n2/N2.FCIDUMP'
bdims = 200

# with HamilTools.hubbard(n_sites=4, u=2, t=1) as hamil:
# # with HamilTools.hubbard(n_sites=16, u=2, t=1) as hamil:
# # with HamilTools.from_fcidump(fd) as hamil:
#     mps = hamil.get_init_mps(bond_dim=bdims)
#     # mps = hamil.get_ground_state_mps(bond_dim=bdims, noise=0)
#     # exit(0)
#     mpo = hamil.get_mpo()

tx = time.perf_counter()
fcidump = FCIDUMP(pg='d2h').read(fd)
hamil = Hamiltonian(fcidump)
mpo = QCSymbolicMPO(hamil).to_sparse().to_flat()
print('build mpo time = ', time.perf_counter() - tx)

fhamil = Hamiltonian(fcidump, flat=True)
mps_info = MPSInfo(fhamil.n_sites, fhamil.vacuum, fhamil.target, fhamil.basis)
mps_info.set_bond_dimension(bdims)
mps = MPS.random(mps_info)

print('MPS = ', mps.show_bond_dims())
print('MPO (NC) =         ', mpo.show_bond_dims())
mpo, _ = mpo.compress(left=True, cutoff=1E-9, norm_cutoff=1E-9)
print('MPO (compressed) = ', mpo.show_bond_dims())

print(MPE(mps, mpo, mps).dmrg(bdims=[bdims]))
