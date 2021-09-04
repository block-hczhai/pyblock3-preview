
"""Transformation between pyblock3 and block2 MPO and MPS."""

import sys
sys.path[:0] = ['..', "../../block2-old/build_seq"]

from pyblock3.block2.hamil import HamilTools
from pyblock3.block2.io import MPOTools, MPSTools, SymbolicMPOTools
from pyblock3.algebra.mpe import MPE, CachedMPE
from pyblock3.hamiltonian import Hamiltonian
from pyblock3.symbolic.symbolic_mpo import QCSymbolicMPO
from pyblock3.fcidump import FCIDUMP
import pyblock3.algebra.funcs as pbalg
import numpy as np
import time
from functools import reduce

# FCIDUMP
fd = "../data/N2.STO3G.FCIDUMP"

# MPO build options:

# max_bond_dim >= -1: SVD
# max_bond_dim = -2: NC
# max_bond_dim = -3: CN
# max_bond_dim = -4: bipartite O(K^5)
# max_bond_dim = -5: fast bipartite O(K^4)
# max_bond_dim = -6: SVD (rescale)
# max_bond_dim = -7: SVD (rescale, fast)
# max_bond_dim = -8: SVD (fast)

SPIN, SITE, OP = 1, 2, 16384
def generate_qc_terms(n_sites, h1e, g2e, cutoff=1E-9):
    OP_C, OP_D = 0 * OP, 1 * OP
    h_values = []
    h_terms = []
    for i in range(0, n_sites):
        for j in range(0, n_sites):
            t = h1e[i, j]
            if abs(t) > cutoff:
                for s in [0, 1]:
                    h_values.append(t)
                    h_terms.append([OP_C + i * SITE + s * SPIN,
                                    OP_D + j * SITE + s * SPIN, -1, -1])
    for i in range(0, n_sites):
        for j in range(0, n_sites):
            for k in range(0, n_sites):
                for l in range(0, n_sites):
                    v = g2e[i, j, k, l]
                    if abs(v) > cutoff:
                        for sij in [0, 1]:
                            for skl in [0, 1]:
                                h_values.append(0.5 * v)
                                h_terms.append([OP_C + i * SITE + sij * SPIN,
                                                OP_C + k * SITE + skl * SPIN,
                                                OP_D + l * SITE + skl * SPIN,
                                                OP_D + j * SITE + sij * SPIN])
    if len(h_values) == 0:
        return np.zeros((0, ), dtype=np.float64), np.zeros((0, 4), dtype=np.int32)
    else:
        return np.array(h_values, dtype=np.float64), np.array(h_terms, dtype=np.int32)

def build_qc(hamil, cutoff=1E-9, max_bond_dim=-1):
    terms = generate_qc_terms(
            hamil.fcidump.n_sites, hamil.fcidump.h1e, hamil.fcidump.g2e, 1E-13)
    mm = hamil.build_mpo(terms, cutoff=cutoff, max_bond_dim=max_bond_dim,
        const=hamil.fcidump.const_e)
    return mm

# pyblock3 MPO

hamil = Hamiltonian(FCIDUMP(pg='d2h').read(fd), flat=True)
hamil_non_flat = Hamiltonian(FCIDUMP(pg='d2h').read(fd), flat=False)
print('FCIDUMP = ', fd)

mpo_flat_bip = build_qc(hamil, cutoff=0, max_bond_dim=-5)
print("MPO FLAT BIPARTITE    = ", mpo_flat_bip.show_bond_dims())
mpo_flat_svd = build_qc(hamil, cutoff=0, max_bond_dim=-1)
mpo_flat_svd, error = mpo_flat_svd.compress(left=True, cutoff=1E-9, norm_cutoff=1E-9)
print("MPO FLAT SVD COMPRESS = ", mpo_flat_svd.show_bond_dims())
mpo_flat_qc = hamil.build_qc_mpo()
print("MPO FLAT QC           = ", mpo_flat_qc.show_bond_dims())
mpo_flat_qc_cps, error = mpo_flat_qc.compress(left=True, cutoff=1E-9, norm_cutoff=1E-9)
print("MPO FLAT QC COMPRESS  = ", mpo_flat_qc_cps.show_bond_dims())

mpo_sym_qc = QCSymbolicMPO(hamil_non_flat)
print("MPO SYMBOLIC QC       = ", mpo_sym_qc.show_bond_dims())

# initial MPS

mps = hamil.build_mps(250)

print("INIT MPS              = ", mps.show_bond_dims())
print("INIT ENERGY           = ", np.dot(mps, mpo_flat_bip @ mps) / np.dot(mps, mps))
print("INIT ENERGY           = ", np.dot(mps, mpo_flat_svd @ mps) / np.dot(mps, mps))
print("INIT ENERGY           = ", np.dot(mps, mpo_flat_qc @ mps) / np.dot(mps, mps))
print("INIT ENERGY           = ", np.dot(mps, mpo_flat_qc_cps @ mps) / np.dot(mps, mps))

# DMRG

bdims = [250] * 5 + [500] * 5
noises = [1E-5] * 2 + [1E-6] * 4 + [1E-7] * 3 + [0]
davthrds = [5E-3] * 4 + [1E-3] * 4 + [1E-4]

scratch3 = "./tmp3"
scratch2 = "./tmp2"

dmrg = CachedMPE(mps, mpo_flat_bip, mps, scratch=scratch3).dmrg(bdims=bdims, noises=noises,
    dav_thrds=davthrds, iprint=2, n_sweeps=10)
ener = dmrg.energies[-1]
print("FINAL ENERGY          = ", ener)

assert abs(ener - -107.65412244752497) < 1E-5

# transform MPS from pyblock3 to block2

with HamilTools.from_fcidump(fd, scratch=scratch2, n_threads=8) as hamil:
    bmps = MPSTools.to_block2(mps, basis=hamil_non_flat.basis)
    with hamil.get_simplified_mpo_block2() as bmpo:
        print("ENERGY = ", hamil.get_expectation(bmpo, bmps))

# transform MPO and MPS from block2 to pyblock3

with HamilTools.from_fcidump(fd, scratch=scratch2, n_threads=8) as hamil:
    mps_from_b2 = hamil.get_ground_state_mps(bond_dim=250)
    mpo_from_b2 = hamil.get_mpo()

energy = np.dot(mps_from_b2, mpo_from_b2 @ mps_from_b2) / np.dot(mps_from_b2, mps_from_b2)
print("FINAL ENERGY                = ", energy)

print('=== PYBLOCK3 EXPT (algebra) ===')

print("MPO ORIG BIPARTITE ENERGY   = ", np.dot(mps, mpo_flat_bip @ mps) / np.dot(mps, mps))
print("MPO ORIG SVD ENERGY         = ", np.dot(mps, mpo_flat_svd @ mps) / np.dot(mps, mps))
print("MPO ORIG QC ENERGY          = ", np.dot(mps, mpo_flat_qc @ mps) / np.dot(mps, mps))
print("MPO ORIG QC COMPRESS ENERGY = ", np.dot(mps, mpo_flat_qc_cps @ mps) / np.dot(mps, mps))

print('=== PYBLOCK3 EXPT (mpe) ===')

print("MPO ORIG BIPARTITE ENERGY   = ", MPE(mps, mpo_flat_bip, mps)[0:2].expectation)
print("MPO ORIG SVD ENERGY         = ", MPE(mps, mpo_flat_svd, mps)[0:2].expectation)
print("MPO ORIG QC ENERGY          = ", MPE(mps, mpo_flat_qc, mps)[0:2].expectation)
print("MPO ORIG QC COMPRESS ENERGY = ", MPE(mps, mpo_flat_qc_cps, mps)[0:2].expectation)

print('=== BLOCK2 EXPT ===')

# transform MPO from pyblock3 to block2

with HamilTools.from_fcidump(fd, scratch=scratch2, n_threads=8) as hamil:
    from block2.sz import Rule, SimplifiedMPO, IdentityAddedMPO
    bmps = MPSTools.to_block2(mps, basis=hamil_non_flat.basis)
    bmpo = SymbolicMPOTools.to_block2(mpo_sym_qc)
    print("MPO FROM SYMBOLIC QC ENERGY = ", hamil.get_expectation(bmpo, bmps))
    bmpo = IdentityAddedMPO(SimplifiedMPO(bmpo, Rule()))
    print("MPO FROM SYMBOLIC QC ENERGY = ", hamil.get_expectation(bmpo, bmps))
    bmpo = MPOTools.to_block2(mpo_flat_bip)
    bmpo = IdentityAddedMPO(SimplifiedMPO(bmpo, Rule()))
    print("MPO FROM BIPARTITE ENERGY   = ", hamil.get_expectation(bmpo, bmps))
    bmpo = MPOTools.to_block2(mpo_flat_svd)
    bmpo = IdentityAddedMPO(SimplifiedMPO(bmpo, Rule()))
    print("MPO FROM SVD ENERGY         = ", hamil.get_expectation(bmpo, bmps))
    bmpo = MPOTools.to_block2(mpo_flat_qc)
    bmpo = IdentityAddedMPO(SimplifiedMPO(bmpo, Rule()))
    print("MPO FROM QC ENERGY          = ", hamil.get_expectation(bmpo, bmps))
    bmpo = MPOTools.to_block2(mpo_flat_qc_cps)
    bmpo = IdentityAddedMPO(SimplifiedMPO(bmpo, Rule()))
    print("MPO FROM QC COMPRESS ENERGY = ", hamil.get_expectation(bmpo, bmps))

# fusing

mpo = mpo_flat_bip
mps[-4:] = [reduce(pbalg.hdot, mps[-4:])]
mpo[-4:] = [reduce(pbalg.hdot, mpo[-4:])]

print('MPO HDOT = ', mpo.show_bond_dims())
print('MPS HDOT = ', mps.show_bond_dims())

print('ENERGY UNFUSED = ', np.dot(mps, mpo @ mps))

info = mps[-1].kron_product_info(1, 2, 3, 4)
mps[-1] = mps[-1].fuse(1, 2, 3, 4, info=info)
mpo[-1] = mpo[-1].fuse(5, 6, 7, 8, info=info).fuse(1, 2, 3, 4, info=info)
print('MPO FUSE = ', mpo.show_bond_dims())
print('MPS FUSE = ', mps.show_bond_dims())

print('ENERGY FUSED   = ', np.dot(mps, mpo @ mps))

# transform big site MPO from pyblock3 to block2

with HamilTools.from_fcidump(fd, scratch=scratch2, n_threads=8) as hamil:
    from block2.sz import Rule, SimplifiedMPO, IdentityAddedMPO
    bmps = MPSTools.to_block2(mps)
    bmpo = MPOTools.to_block2(mpo)
    bmpo = IdentityAddedMPO(SimplifiedMPO(bmpo, Rule()))
    print("MPO BIG SITE ENERGY   = ", hamil.get_expectation(bmpo, bmps))
    assert bmps.center == 0
    bmps.dot = 2
    print("MPO BIG SITE DMRG     = ", hamil.dmrg(bmpo, bmps))
