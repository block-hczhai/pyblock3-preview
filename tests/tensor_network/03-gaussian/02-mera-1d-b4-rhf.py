
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

mf = mol.RHF()
mf.get_hcore = lambda *_: h1e
mf.get_ovlp = lambda *_: np.eye(L)
mf._eri = g2e

mf = mf.run(conv_tol=1E-14)
print('RDM1 trace = ', np.trace(mf.make_rdm1()))

rdm1 = torch.tensor(mf.make_rdm1())
rdm2 = torch.tensor(mf.make_rdm2())

from pyblock3.gaussian import GaussianMERA1D, GaussianOptimizer

gmera = GaussianMERA1D(L, LB, LC, dis_ent=True, periodic=False).rhf().fit_rdm1(rdm1)
print(gmera)

print('tn n elec = ', sum(gmera.get_occupations()))
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

# converged SCF energy = -11.6759028949188
# rdm1 diff =  0.6496097201215104
# rdm2 diff =  5.503641924535073
# init  ener =  -11.412701569993416
# final ener =  -11.675068315349609 niter =  444
# final ener =  -11.675062081153172 niter =  614