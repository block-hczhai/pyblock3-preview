
import numpy as np
from pyscf import M
import torch

L = 16
NE = L
U = 2
LB, LC = 6, 3

mol = M()
mol.nelectron = NE

h1e = np.zeros((L, L))
for i in range(L - 1):
    h1e[i, i + 1] = h1e[i + 1, i] = -1.0

g2e = np.zeros((L, L, L, L))
for i in range(L):
    g2e[i, i, i, i] = U

gh1e = np.zeros((L * 2, L * 2))
gh1e[:L, :L] = h1e
gh1e[L:, L:] = h1e

gg2e = np.zeros((L * 2, L * 2, L * 2, L * 2))
gg2e[:L, :L, :L, :L] = g2e
gg2e[:L, :L, L:, L:] = g2e
gg2e[L:, L:, :L, :L] = g2e
gg2e[L:, L:, L:, L:] = g2e

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
gdm0[:L] = dm0a
gdm0[L:] = dm0b

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
ener, x = opt.optimize(x0='random', maxiter=1000)
print('final ener = ', ener, 'niter = ', opt.niter)

mf.get_hcore = lambda *_: gh1e.detach().numpy()
mf.kernel(dm0=gmps.make_rdm1().detach().numpy())

# converged SCF energy = -11.8657704004292  <S^2> = 1.5694297  2S+1 = 2.6977248
# rdm1 diff =  2.139622042549017e-06
# rdm2 diff =  1.657344107594207e-05
# init  ener =  -11.865770400421331
# final ener =  -11.865770400421331 niter =  2
# final ener =  -11.865760279931013 niter =  205
# converged SCF energy = -11.8657704004292  <S^2> = 1.5694297  2S+1 = 2.6977248
