
import numpy as np
from pyscf import M
import torch

torch.set_num_threads(28)

LX, LY = 16, 16
NE = LX * LY
L = LX * LY
U = 0

mol = M()
mol.nelectron = NE
mol.incore_anyway = True

h1e = np.zeros((L, L))
for ix in range(LX):
    for iy in range(LY):
        ixl, iyl = ix - 1, iy
        ixr, iyr = ix + 1, iy
        ixd, iyd = ix, iy - 1
        ixu, iyu = ix, iy + 1
        if ix > 0:
            h1e[ix * LY + iy, ixl * LY + iyl] = -1.0
            h1e[ixl * LY + iyl, ix * LY + iy] = -1.0
        if ix < LX - 1:
            h1e[ix * LY + iy, ixr * LY + iyr] = -1.0
            h1e[ixr * LY + iyr, ix * LY + iy] = -1.0
        if iy > 0:
            h1e[ix * LY + iy, ixd * LY + iyd] = -1.0
            h1e[ixd * LY + iyd, ix * LY + iy] = -1.0
        if iy < LY - 1:
            h1e[ix * LY + iy, ixu * LY + iyu] = -1.0
            h1e[ixu * LY + iyu, ix * LY + iy] = -1.0

g2e = np.zeros((L, L, L, L))
for i in range(L):
    g2e[i, i, i, i] = U

mf = mol.UHF(verbose=4)
mf.get_hcore = lambda *_: h1e
mf.get_ovlp = lambda *_: np.eye(L)
mf._eri = g2e

mo_coeff = np.linalg.eigh(h1e)[1]
mf.mo_coeff = np.concatenate([mo_coeff[None], mo_coeff[None]], axis=0)
mf.mo_occ = np.array([[1.0] * (NE // 2) + [0.0] * (L - NE // 2) for _ in range(2)])
print('E[tot] = ', mf.energy_tot())

print('RDM1 trace = ', np.trace(sum(mf.make_rdm1())))

rdm1 = torch.tensor(np.array(mf.make_rdm1()))

from pyblock3.gaussian import GaussianMERA2D, GaussianOptimizer

gmera = GaussianMERA2D(n_sites=(LX, LY), n_tensor_sites=(8, 8),
    n_core=(4, 0), dis_ent_width=(1, 1), core_depth=4)

print(gmera)

gmera = gmera.uhf()
gmera = gmera.fit_rdm1(rdm1)
gmera = gmera.set_occupations([[0, 1] * (L // 4) + [1, 0] * (L // 4), [1, 0] * (L // 4) + [0, 1] * (L // 4)])

layers = gmera.get_layers()
current_idxs = gmera.get_initial_indices()
terminal_idxs = gmera.get_terminal_indices()

print(gmera.repr_layers())

print('tn n elec = ', np.sum(gmera.get_occupations()))
print('rdm1 diff = ', np.linalg.norm(gmera.make_rdm1() - rdm1))

h1e = torch.tensor(h1e)
g2e = None

print('init  ener = ', float(gmera.energy_tot(h1e, g2e)))

opt = GaussianOptimizer(gmera, h1e, g2e, iprint=1)
ener, x = opt.optimize(maxiter=1000)
print('final ener = ', ener, 'niter = ', opt.niter)

import pickle

pickle.dump((ener, x, h1e, gmera), open("02-opt-x.bin", "wb"))

for ig in range(4):
    ener, x = opt.optimize(x0='random', maxiter=1000)
    pickle.dump((ener, x, h1e, gmera), open("02-opt-%d.bin" % ig, "wb"))
    print('final ener = ', ener, 'niter = ', opt.niter)
