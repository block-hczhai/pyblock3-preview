
import numpy as np
from pyscf import M
import torch

LX, LY = 4, 4
NE = LX * LY
L = LX * LY
U = 2

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

mf = mol.UHF()
mf.get_hcore = lambda *_: h1e
mf.get_ovlp = lambda *_: np.eye(L)
mf._eri = g2e

dm0a = np.array(
    [[0, 1, 0, 1],
     [1, 0, 1, 0],
     [0, 1, 0, 1],
     [1, 0, 1, 0]]
).ravel()

dm0b = ~dm0a & 1
mf.conv_tol = 1E-14
mf.max_cycle = 200
mf.kernel(dm0=(np.diag(dm0a), np.diag(dm0b)))

print('RDM1 trace = ', np.trace(sum(mf.make_rdm1())))

rdm1 = torch.tensor(np.array(mf.make_rdm1()))
rdm2 = torch.tensor(np.array(mf.make_rdm2()))

from pyblock3.gaussian import GaussianTensorNetwork, GaussianTensor, GaussianOptimizer

#  3  7 11 15   |  * 22 23  *   |   *  *  *  *  |  44 45 46 47  |   *  *  *  *
#  2  6 10 14   |  * 20 21  *   |  25 27 29 31  |  40 41 42 43  |  52 53 54 55
#  1  5  9 13   |  * 18 19  *   |  24 26 28 30  |  36 37 38 39  |   *  *  *  *
#  0  4  8 12   |  * 16 17  *   |   *  *  *  *  |  32 33 34 35  |  48 49 50 51

gmera = GaussianTensorNetwork([
    GaussianTensor(u_idx=[16, 17], d_idx=[4, 8], n_core=0),
    GaussianTensor(u_idx=[18, 19], d_idx=[5, 9], n_core=0),
    GaussianTensor(u_idx=[20, 21], d_idx=[6, 10], n_core=0),
    GaussianTensor(u_idx=[22, 23], d_idx=[7, 11], n_core=0),

    GaussianTensor(u_idx=[24, 25], d_idx=[1, 2], n_core=0),
    GaussianTensor(u_idx=[26, 27], d_idx=[18, 20], n_core=0),
    GaussianTensor(u_idx=[28, 29], d_idx=[19, 21], n_core=0),
    GaussianTensor(u_idx=[30, 31], d_idx=[13, 14], n_core=0),

    GaussianTensor(u_idx=[36, 37, 32, 33], d_idx=[24, 26, 0, 16], n_core=2),
    GaussianTensor(u_idx=[38, 39, 34, 35], d_idx=[28, 30, 17, 12], n_core=2),
    GaussianTensor(u_idx=[44, 45, 40, 41], d_idx=[3, 22, 25, 27], n_core=2),
    GaussianTensor(u_idx=[46, 47, 42, 43], d_idx=[23, 15, 29, 31], n_core=2),

    GaussianTensor(u_idx=[52, 53, 54, 55, 48, 49, 50, 51], d_idx=[40, 41, 42, 43, 32, 33, 34, 35], n_core=8),
])

gmera = gmera.uhf()
gmera = gmera.fit_rdm1(rdm1)
gmera = gmera.set_occupations([[0, 1] * 4 + [1, 0] * 4, [1, 0] * 4 + [0, 1] * 4])
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

for _ in range(10):
    ener, x = opt.optimize(x0='random', maxiter=1000)
    print('final ener = ', ener, 'niter = ', opt.niter)

# converged SCF energy = -14.8692937269376  <S^2> = 2.5877562  2S+1 = 3.3691282
# rdm1 diff =  4.242110483033439
# rdm2 diff =  21.888905491425604
# init  ener =  9.399250485605434
# final ener =  -14.051330152047203 niter =  216
# final ener =  -14.593093645353406 niter =  158
# final ener =  -14.766880403902825 niter =  133
# final ener =  -14.766880352379008 niter =  106
# final ener =  -14.594505440763097 niter =  131
# final ener =  -14.594505465994674 niter =  128
# final ener =  -14.434659920639 niter =  136
# final ener =  -14.766880274767226 niter =  109
# final ener =  -14.766880391714176 niter =  154
# final ener =  -14.766880409273579 niter =  197
# final ener =  -14.59450542679254 niter =  175
