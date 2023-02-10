
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

mf = mol.RHF()
mf.get_hcore = lambda *_: h1e
mf.get_ovlp = lambda *_: np.eye(L)
mf._eri = g2e

mf = mf.run(conv_tol=1E-14)
print('RDM1 trace = ', np.trace(mf.make_rdm1()))

rdm1 = torch.tensor(mf.make_rdm1())
rdm2 = torch.tensor(mf.make_rdm2())

from pyblock3.gaussian import GaussianMPS, GaussianOptimizer

gmps = GaussianMPS(L, LB, LC).rhf().fit_rdm1(rdm1)
print(gmps)

print('tn n elec = ', sum(gmps.get_occupations()))
print('rdm1 diff = ', np.linalg.norm(gmps.make_rdm1() - rdm1))
print('rdm2 diff = ', np.linalg.norm(gmps.make_rdm2() - rdm2))

h1e = torch.tensor(h1e)
g2e = torch.tensor(g2e)

print('init  ener = ', float(gmps.energy_tot(h1e, g2e)))

opt = GaussianOptimizer(gmps, h1e, g2e, iprint=0)
ener, x = opt.optimize()
print('final ener = ', ener, 'niter = ', opt.niter)

# converged SCF energy = -11.6759028949188
# rdm1 diff =  0.06861284827245893
# rdm2 diff =  0.5821920708110683
# init  ener =  -11.673136413376161
# final ener =  -11.673616810997284 niter =  86
