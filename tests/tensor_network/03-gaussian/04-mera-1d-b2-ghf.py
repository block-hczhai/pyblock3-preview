
import numpy as np
from pyscf import M
import torch

L = 16
NE = L
U = 2
LB, LC = 2, 1

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

from pyblock3.gaussian import GaussianMERA1D, GaussianOptimizer

gmera = GaussianMERA1D(L * 2, LB * 2, LC * 2, dis_ent=True, periodic=False).ghf().fit_rdm1(rdm1)
print(gmera)

print('tn n elec = ', np.sum(gmera.get_occupations()))
print('rdm1 diff = ', np.linalg.norm(gmera.make_rdm1() - rdm1))
print('rdm2 diff = ', np.linalg.norm(gmera.make_rdm2() - rdm2))

gh1e = torch.tensor(gh1e)
gg2e = torch.tensor(gg2e)

print('init  ener = ', float(gmera.energy_tot(gh1e, gg2e)))

opt = GaussianOptimizer(gmera, gh1e, gg2e, iprint=0)
ener, x = opt.optimize(maxiter=1000)
print('final ener = ', ener, 'niter = ', opt.niter)
ener, x = opt.optimize(x0='random', maxiter=1000)
print('final ener = ', ener, 'niter = ', opt.niter)

gmera = gmera.no_grad_()
print('rdm1 diff = ', np.linalg.norm(gmera.make_rdm1() - rdm1))
print('rdm2 diff = ', np.linalg.norm(gmera.make_rdm2() - rdm2))

mf.get_hcore = lambda *_: gh1e.detach().numpy()
mf.kernel(dm0=gmera.make_rdm1().detach().numpy())

# converged SCF energy = -11.8657704004292  <S^2> = 1.5694297  2S+1 = 2.6977248
# rdm1 diff =  0.30806786186917157
# rdm2 diff =  2.384656401517744
# init  ener =  -11.745012254590188
# final ener =  -11.864935286083055 niter =  660
# final ener =  -11.648108613041709 niter =  381
# rdm1 diff =  1.8075179849774736
# rdm2 diff =  13.773815244866343
# converged SCF energy = -11.6509740107719  <S^2> = 2.4111855  2S+1 = 3.2626281
