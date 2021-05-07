
from pyblock3.algebra.mpe import MPE, CachedMPE
from pyblock3.hamiltonian import Hamiltonian
from pyblock3.fcidump import FCIDUMP, PointGroup
from pyblock3.symbolic.expr import OpElement, OpNames
from pyblock3.algebra.symmetry import SZ
from pyscf import gto, scf, lo, symm, ao2mo
import numpy as np

# H chain
N = 10
BOHR = 0.52917721092  # Angstroms
R = 1.8 * BOHR
mol = gto.M(atom=[['H', (0, 0, i * R)] for i in range(N)],
            basis='sto6g', verbose=0, symmetry='c1')
pg = mol.symmetry.lower()
mf = scf.RHF(mol)
ener = mf.kernel()
print("SCF Energy = %20.15f" % ener)

fcidump_sym = ["A"]

# ao mox
mo_coeff_lowdin = lo.orth.lowdin(mol.intor('cint1e_ovlp_sph'))
# ao mo
mo_coeff_mo = mf.mo_coeff
n_mo = mo_coeff_lowdin.shape[1]

u = np.linalg.inv(mo_coeff_mo) @ mo_coeff_lowdin

print(np.linalg.norm(u.T - np.linalg.inv(u)))
print(np.linalg.norm((u @ u.T) - np.identity(len(u))))

if np.linalg.det(u) < 0:
    u[-1] = -u[-1]
    mo_coeff_mo[:, -1] = -mo_coeff_mo[:, -1]

ev, v = np.linalg.eig(u)

print(np.linalg.norm(u - v @ np.diag(ev) @ np.linalg.inv(v)))

kappa = v @ np.diag(np.log(ev)) @ np.linalg.inv(v)
print(np.linalg.norm(kappa + np.conj(kappa.T)))
print('imag = ', np.linalg.norm(np.imag(kappa)))
print('diag = ', np.linalg.norm(np.diag(kappa)))
assert np.linalg.norm(np.imag(kappa)) < 1E-12
kappa = np.real(kappa)

kv, kvv = np.linalg.eig(kappa)
ku = kvv @ np.diag(np.exp(kv)) @ np.linalg.inv(kvv)
print(np.linalg.norm((ku @ ku.T) - np.identity(len(ku))))

mo_coeff = mo_coeff_lowdin
orb_sym = [0] * n_mo
na = nb = mol.nelectron // 2
h1e = mo_coeff.T @ mf.get_hcore() @ mo_coeff
g2e = ao2mo.restore(1, ao2mo.kernel(mol, mo_coeff), n_mo)
ecore = mol.energy_nuc()
fd = FCIDUMP(pg='c1', n_sites=n_mo, n_elec=na + nb, twos=na - nb, ipg=0, uhf=False,
             h1e=h1e, g2e=g2e, orb_sym=orb_sym, const_e=ecore, mu=0)
hamil = Hamiltonian(fd, flat=True)

bond_dim = 500
bdims = [bond_dim]
mpo = hamil.build_qc_mpo()
mpo, _ = mpo.compress(left=True, cutoff=1E-12, norm_cutoff=1E-12)
mps = hamil.build_mps(bond_dim)
print('MPO (NC) =         ', mpo.show_bond_dims())
print('MPS = ', mps.show_bond_dims())

noises = [1E-4, 1E-5, 1E-6, 0]
davthrds = [1E-3] + [1E-4] * 100

dmrg = MPE(mps, mpo, mps).dmrg(bdims=bdims, noises=noises,
                              dav_thrds=davthrds, iprint=2, n_sweeps=20, tol=1E-12)
ener = dmrg.energies[-1]
print("Energy = %20.12f" % ener)

def generate_terms(n_sites, c, d):
    for i in range(0, n_sites):
        for j in range(0, n_sites):
            for s in [0, 1]:
                t = kappa[i, j]
                yield t * c[i, s] * d[j, s]

mpo_rot = hamil.build_mpo(generate_terms, cutoff=1E-12).to_sparse().to_flat()
print('MPO (NC) =         ', mpo_rot.show_bond_dims())
mpo_rot.const = 0
print('MPO (NC) =         ', mpo_rot.show_bond_dims())

ket = mps.copy()
dt = 0.1
t = 1.0

nstep = int(t / dt)

mo_coeff = mo_coeff_mo
orb_sym = [0] * n_mo
na = nb = mol.nelectron // 2
h1e = mo_coeff.T @ mf.get_hcore() @ mo_coeff
g2e = ao2mo.restore(1, ao2mo.kernel(mol, mo_coeff), n_mo)
ecore = mol.energy_nuc()
fd = FCIDUMP(pg='c1', n_sites=n_mo, n_elec=na + nb, twos=na - nb, ipg=0, uhf=False,
             h1e=h1e, g2e=g2e, orb_sym=orb_sym, const_e=ecore, mu=0)
hamil = Hamiltonian(fd, flat=True)

mpo2 = hamil.build_qc_mpo()
mpo2, _ = mpo2.compress(left=True, cutoff=1E-12, norm_cutoff=1E-12)

mpe = MPE(ket, mpo_rot, ket)
print('total step = ', nstep)
for it in range(0, nstep):
    cur_t = it * dt
    mpe.tddmrg(bdims=[1000], dt=dt, iprint=2, n_sweeps=1, n_sub_sweeps=2, normalize=True)
    print(it, 'ener = ', MPE(ket, mpo2, ket)[0:2].expectation)
