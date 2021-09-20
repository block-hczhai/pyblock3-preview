
from pyscf import gto, scf, ci, mcscf, ao2mo, symm
import numpy as np

mol = gto.M(
    atom=[['N', (0, 0, 0)], ['N', (0, 0, 1.1)]],
    basis='sto3g',
    verbose=3, symmetry='d2h', spin=0)

mf = scf.RHF(mol).run()

n_sites = mol.nao
n_elec = sum(mol.nelec)
tol = 1E-13

fcidump_sym = ["Ag", "B3u", "B2u", "B1g", "B1u", "B2g", "B3g", "Au"]
optimal_reorder = ["Ag", "B1u", "B3u", "B2g", "B2u", "B3g", "B1g", "Au"]
orb_sym_str = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mf.mo_coeff)
orb_sym = np.array([fcidump_sym.index(i) + 1 for i in orb_sym_str])
idx = np.argsort([optimal_reorder.index(i) for i in orb_sym_str])
orb_sym = orb_sym[idx]
mf.mo_coeff = mf.mo_coeff[:, idx]

print(orb_sym)

mc = mcscf.CASCI(mf, n_sites, n_elec)
h1e, e_core = mc.get_h1cas()
g2e = mc.get_h2cas()
g2e = ao2mo.restore(1, g2e, n_sites)

h1e[np.abs(h1e) < tol] = 0
g2e[np.abs(g2e) < tol] = 0

n = n_sites
gh1e = np.zeros((n * 2, n * 2))
gg2e = np.zeros((n * 2, n * 2, n * 2, n * 2))

for i in range(n * 2):
    for j in range(i % 2, n * 2, 2):
        gh1e[i, j] = h1e[i // 2, j // 2]

for i in range(n * 2):
    for j in range(i % 2, n * 2, 2):
        for k in range(n * 2):
            for l in range(k % 2, n * 2, 2):
                gg2e[i, j, k, l] = g2e[i // 2, j // 2, k // 2, l // 2]

from pyblock3.fcidump import FCIDUMP
from pyblock3.hamiltonian import Hamiltonian
from pyblock3.algebra.mpe import CachedMPE

fd = FCIDUMP(pg='d2h', n_sites=n * 2, n_elec=n_elec, twos=n_elec, ipg=0, h1e=gh1e,
    g2e=gg2e, orb_sym=[orb_sym[i // 2] for i in range(n * 2)], const_e=e_core)

hamil = Hamiltonian(fd, flat=True)
mpo = hamil.build_qc_mpo()
mpo, error = mpo.compress(left=True, cutoff=1E-9, norm_cutoff=1E-9)
mps = hamil.build_mps(250)

bdims = [250] * 5 + [500] * 5
noises = [1E-5] * 2 + [1E-6] * 4 + [1E-7] * 3 + [0]
davthrds = [5E-3] * 4 + [1E-3] * 4 + [1E-4]

dmrg = CachedMPE(mps, mpo, mps, scratch='./tmp').dmrg(bdims=bdims, noises=noises,
    dav_thrds=davthrds, iprint=2, n_sweeps=10)
ener = dmrg.energies[-1]
print("FINAL ENERGY          = ", ener)
