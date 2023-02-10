
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

# 1-electron integrals from spin orbital space half factor from half the 1pdm
h1e_spin = np.zeros((2*L, 2*L))
for ix in range(LX):
    for iy in range(LY):
        ixl, iyl = ix - 1, iy
        ixr, iyr = ix + 1, iy
        ixd, iyd = ix, iy - 1
        ixu, iyu = ix, iy + 1
        if ix > 0:
            h1e_spin[2*ix * LY + iy, 2*ixl * LY + iyl] = -1.0 / 2.0 
            h1e_spin[2*ixl * LY + iyl, 2*ix * LY + iy] = -1.0 / 2.0
            h1e_spin[(2*ix + 1) * LY + iy, (2*ixl + 1) * LY + iyl] = -1.0 / 2.0
            h1e_spin[(2*ixl + 1) * LY + iyl, (2*ix + 1) * LY + iy] = -1.0 / 2.0
        if ix < LX - 1:
            h1e_spin[2*ix * LY + iy, 2*ixr * LY + iyr] = -1.0 / 2.0
            h1e_spin[2*ixr * LY + iyr, 2*ix * LY + iy] = -1.0 / 2.0
            h1e_spin[(2*ix + 1) * LY + iy, (2*ixr + 1) * LY + iyr] = -1.0 / 2.0
            h1e_spin[(2*ixr + 1) * LY + iyr, (2*ix + 1) * LY + iy] = -1.0 / 2.0
        if iy > 0:
            h1e_spin[2*ix * LY + iy, 2*ixd * LY + iyd] = -1.0 / 2.0
            h1e_spin[2*ixd * LY + iyd, 2*ix * LY + iy] = -1.0 / 2.0
            h1e_spin[(2*ix + 1) * LY + iy, (2*ixd + 1) * LY + iyd] = -1.0 / 2.0
            h1e_spin[(2*ixd + 1) * LY + iyd, (2*ix + 1) * LY + iy] = -1.0 / 2.0
        if iy < LY - 1:
            h1e_spin[2*ix * LY + iy, 2*ixu * LY + iyu] = -1.0 / 2.0
            h1e_spin[2*ixu * LY + iyu, 2*ix * LY + iy] = -1.0 / 2.0
            h1e_spin[(2*ix + 1) * LY + iy, (2*ixu + 1) * LY + iyu] = -1.0 / 2.0
            h1e_spin[(2*ixu + 1) * LY + iyu, (2*ix + 1) * LY + iy] = -1.0 / 2.0

# 2-electron integrals from spin orbital space hard-coded factors
g2e_spin = np.zeros((2*L, 2*L, 2*L, 2*L))
for i in range(L):
    g2e_spin[2*i, 2*i, 2*i, 2*i] = U / 4.0
    g2e_spin[2*i + 1, 2*i + 1, 2*i + 1, 2*i + 1] = U / 4.0
    g2e_spin[2*i+1, 2*i+1, 2*i, 2*i] = U / 6.0 
    g2e_spin[2*i , 2*i , 2*i + 1, 2*i + 1] = U / 6.0
    g2e_spin[2*i+1, 2*i, 2*i, 2*i+1] = U / 12.0 
    g2e_spin[2*i , 2*i + 1, 2*i + 1, 2*i] = U / 12.0


mf = mol.HF()
mf.get_hcore = lambda *_: h1e
mf.get_ovlp = lambda *_: np.eye(L)
mf._eri = g2e

mf = mf.newton().run(conv_tol=1E-9)
print('RDM1 trace = ', np.trace(mf.make_rdm1()))

rdm1 = torch.tensor(mf.make_rdm1())
rdm2 = torch.tensor(mf.make_rdm2())

from pyblock3.gaussian.core import GaussianTensorNetwork, GaussianTensor, GaussianOptimizer

# without spins
#  3  7 11 15   |  * 22 23  *   |   *  *  *  *  |  44 45 46 47  |   *  *  *  *
#  2  6 10 14   |  * 20 21  *   |  25 27 29 31  |  40 41 42 43  |  52 53 54 55
#  1  5  9 13   |  * 18 19  *   |  24 26 28 30  |  36 37 38 39  |   *  *  *  *
#  0  4  8 12   |  * 16 17  *   |   *  *  *  *  |  32 33 34 35  |  48 49 50 51

#with spins
#  67  1415 2223 3031   |  * 4445 4647  *   |    *    *    *    *   |  8889 9091 9293 9495  |     *      *      *     *
#  45  1213 2021 2829   |  * 4041 4243  *   |  5051 5455 5859 6263  |  8081 8283 8485 8687  |  104105 106107 108109 110111
#  23  1011 1819 2627   |  * 3637 3839  *   |  4849 5253 5657 6061  |  7273 7475 7677 7879  |     *      *      *     *
#  01   89  1617 2425   |  * 3233 3435  *   |    *    *    *    *   |  6465 6667 6869 7071  |   9697   9899  100101 102103


gmera = GaussianTensorNetwork([
    GaussianTensor(u_idx=[32, 33, 34, 35], d_idx=[8, 9, 16, 17], n_core=0),
    GaussianTensor(u_idx=[36, 37, 38, 39], d_idx=[10, 11, 18, 19], n_core=0),
    GaussianTensor(u_idx=[40, 41, 42, 43], d_idx=[12, 13, 20, 21], n_core=0),
    GaussianTensor(u_idx=[44, 45, 46, 47], d_idx=[14, 15, 22, 23], n_core=0),

    GaussianTensor(u_idx=[48, 49, 50, 51], d_idx=[2, 3, 4, 5], n_core=0),
    GaussianTensor(u_idx=[52, 53, 54, 55], d_idx=[36, 37, 40, 41], n_core=0),
    GaussianTensor(u_idx=[56, 57, 58, 59], d_idx=[38, 39, 42, 43], n_core=0),
    GaussianTensor(u_idx=[60, 61, 62, 63], d_idx=[26, 27, 28, 29], n_core=0),

    GaussianTensor(u_idx=[72, 73, 74, 75, 64, 65, 66, 67], d_idx=[48, 49, 52, 53, 0, 1, 32, 33], n_core=4),
    GaussianTensor(u_idx=[76, 77, 78, 79, 68, 69, 70, 71], d_idx=[56, 57, 60, 61, 34, 35, 24, 25], n_core=4),
    GaussianTensor(u_idx=[88, 89, 90, 91, 80, 81, 82, 83], d_idx=[6, 7, 44, 45, 50, 51, 54, 55], n_core=4),
    GaussianTensor(u_idx=[92, 93, 94, 95, 84, 85, 86, 87], d_idx=[46, 47, 30, 31, 58, 59, 62, 63], n_core=4),

    GaussianTensor(u_idx=[104, 105, 106, 107, 108, 109, 110, 111, 96, 97, 98, 99, 100, 101, 102, 103], d_idx=[80, 81, 82, 83, 84, 85, 86, 87, 64, 65, 66, 67, 68, 69, 70, 71], n_core=16),
])

norb = rdm1.shape[0]
rdm1_spin = np.zeros((2*norb,2*norb))
rdm1_spin[:norb,:norb] = rdm1
rdm1_spin[norb:,norb:] = rdm1
order = [x+norb*j for x in range(norb) for j in range(2)]
rdm1_spin = torch.tensor(rdm1_spin[order,:][:,order])

gmera = gmera.fit_rdm1(rdm1_spin)
#gmera = gmera.set_occ_half_filling()
print(gmera)

h1e_spin = torch.tensor(h1e_spin)
g2e_spin = torch.tensor(g2e_spin)

print('tn n elec = ', sum(gmera.get_occupations())/2)

print('init  ener = ', float(gmera.energy_tot(h1e_spin, g2e_spin)))

opt = GaussianOptimizer(gmera, h1e_spin, g2e_spin, iprint=0)
ener, x = opt.optimize(maxiter=1000)
print('final ener = ', ener, 'niter = ', opt.niter)

for _ in range(10):
    ener, x = opt.optimize(x0='random', maxiter=1000)
    print('final ener = ', ener, 'niter = ', opt.niter)

# converged energy E = -13.88854 for all ten different initial guesses
# HF energy E = -13.7433833