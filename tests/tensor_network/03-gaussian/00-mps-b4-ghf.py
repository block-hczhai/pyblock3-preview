
import numpy as np
from pyscf import M
import torch

L = 16
NE = L
U = 2
LB, LC = 4, 2

mol = M()
mol.nelectron = NE

h1e = np.zeros((L, L))
for i in range(L - 1):
    h1e[i, i + 1] = h1e[i + 1, i] = -1.0

g2e = np.zeros((L, L, L, L))
for i in range(L):
    g2e[i, i, i, i] = U

gh1e = np.zeros((L * 2, L * 2))
gh1e[0::2, 0::2] = h1e
gh1e[1::2, 1::2] = h1e

gg2e = np.zeros((L * 2, L * 2, L * 2, L * 2))
gg2e[0::2, 0::2, 0::2, 0::2] = g2e
gg2e[0::2, 0::2, 1::2, 1::2] = g2e
gg2e[1::2, 1::2, 0::2, 0::2] = g2e
gg2e[1::2, 1::2, 1::2, 1::2] = g2e

mf = mol.GHF()
mf.get_hcore = lambda *_: gh1e
mf.get_ovlp = lambda *_: np.eye(L * 2)
mf._eri = g2e

def ghf_make_rdm2():
    dm1 = mf.make_rdm1()
    dm2 = (np.einsum('ij,kl->ijkl', dm1, dm1)
            - np.einsum('ij,kl->iklj', dm1, dm1))
    return dm2

mf.make_rdm2 = ghf_make_rdm2

dm0a = [1, 0] * (L // 2)
dm0b = [0, 1] * (L // 2)

gdm0 = np.zeros((L * 2, ))
gdm0[0::2] = dm0a
gdm0[1::2] = dm0b

mf.conv_tol = 1E-14
mf.kernel(dm0=np.diag(gdm0))

print('RDM1 trace = ', np.trace(mf.make_rdm1()))

rdm1 = torch.tensor(np.array(mf.make_rdm1()))
rdm2 = torch.tensor(np.array(mf.make_rdm2()))

from pyblock3.gaussian import GaussianMPS, GaussianOptimizer

gmps = GaussianMPS(L * 2, LB * 2, LC * 2).ghf().fit_rdm1(rdm1)
print(gmps)

print('tn n elec = ', np.sum(gmps.get_occupations()))
print('rdm1 diff = ', np.linalg.norm(gmps.make_rdm1() - rdm1))
print('rdm2 diff = ', np.linalg.norm(gmps.make_rdm2() - rdm2))

gh1e = torch.tensor(gh1e)
gg2e = torch.tensor(gg2e)

print('init  ener = ', float(gmps.energy_tot(gh1e, gg2e)))

opt = GaussianOptimizer(gmps, gh1e, gg2e, iprint=0)
ener, x = opt.optimize()
print('final ener = ', ener, 'niter = ', opt.niter)

# converged SCF energy = -11.6759028949188  <S^2> = 0.84295667  2S+1 = 2.0908914
# rdm1 diff =  0.1049089920627553
# rdm2 diff =  0.8125546190332478
# init  ener =  -11.663689962215582
# final ener =  -11.861765207589771 niter =  102
