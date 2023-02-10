
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

mf = mol.UHF()
mf.get_hcore = lambda *_: h1e
mf.get_ovlp = lambda *_: np.eye(L)
mf._eri = g2e

dm0a = [1, 0] * (L // 2)
dm0b = [0, 1] * (L // 2)

mf.conv_tol = 1E-14
mf.kernel(dm0=(np.diag(dm0a), np.diag(dm0b)))

print('RDM1 trace = ', np.trace(sum(mf.make_rdm1())))

rdm1 = torch.tensor(np.array(mf.make_rdm1()))
rdm2 = torch.tensor(np.array(mf.make_rdm2()))

from pyblock3.gaussian import GaussianMERA1D, GaussianOptimizer

gmera = GaussianMERA1D(L, LB, LC, dis_ent=True, periodic=False).uhf().fit_rdm1(rdm1)
print(gmera)

print('tn n elec = ', np.sum(gmera.get_occupations()))
print('rdm1 diff = ', np.linalg.norm(gmera.make_rdm1() - rdm1))
print('rdm2 diff = ', np.linalg.norm(gmera.make_rdm2() - rdm2))

h1e = torch.tensor(h1e)
g2e = torch.tensor(g2e)

print('init  ener = ', float(gmera.energy_tot(h1e, g2e)))

opt = GaussianOptimizer(gmera, h1e, g2e, iprint=0)
ener, x = opt.optimize(maxiter=1000)
print('final ener = ', ener, 'niter = ', opt.niter)
ener, x = opt.optimize(x0='random', maxiter=1000)
print('final ener = ', ener, 'niter = ', opt.niter)

gmera = gmera.no_grad_()
print('rdm1 diff = ', np.linalg.norm(gmera.make_rdm1() - rdm1))
print('rdm2 diff = ', np.linalg.norm(gmera.make_rdm2() - rdm2))

# converged SCF energy = -11.8657704004292  <S^2> = 1.5694297  2S+1 = 2.6977248
# rdm1 diff =  1.5116013351281585
# rdm2 diff =  8.930503087195595
# init  ener =  -9.450618336179865
# final ener =  -10.949869922410672 niter =  297
# final ener =  -10.933147731389406 niter =  154
# rdm1 diff =  1.845849118627673
# rdm2 diff =  10.924913806087199
