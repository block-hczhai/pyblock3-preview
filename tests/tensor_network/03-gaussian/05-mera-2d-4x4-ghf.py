
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

dm0a = np.array(
    [[0, 1, 0, 1],
     [1, 0, 1, 0],
     [0, 1, 0, 1],
     [1, 0, 1, 0]]
).flatten()

dm0b = ~dm0a & 1

gdm0 = np.zeros((L * 2, ))
gdm0[0::2] = dm0a
gdm0[1::2] = dm0b

mf.conv_tol = 1E-14
mf.max_cycle = 200
mf.kernel(dm0=np.diag(gdm0))

print('RDM1 trace = ', np.trace(mf.make_rdm1()))

rdm1 = torch.tensor(np.array(mf.make_rdm1()))
rdm2 = torch.tensor(np.array(mf.make_rdm2()))

from pyblock3.gaussian import GaussianTensorNetwork, GaussianTensor, GaussianOptimizer

# without spins
#  3  7 11 15   |  * 22 23  *   |   *  *  *  *  |  44 45 46 47  |   *  *  *  *
#  2  6 10 14   |  * 20 21  *   |  25 27 29 31  |  40 41 42 43  |  52 53 54 55
#  1  5  9 13   |  * 18 19  *   |  24 26 28 30  |  36 37 38 39  |   *  *  *  *
#  0  4  8 12   |  * 16 17  *   |   *  *  *  *  |  32 33 34 35  |  48 49 50 51

# with spins
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

gmera = gmera.ghf()
gmera = gmera.fit_rdm1(rdm1)
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

for _ in range(10):
    ener, x = opt.optimize(x0='random', maxiter=1000)
    print('final ener = ', ener, 'niter = ', opt.niter)

# converged SCF energy = -14.8269007837547  <S^2> = 3.5946321  2S+1 = 3.9215467
# rdm1 diff =  1.4045635202501676
# rdm2 diff =  10.724698622241661
# init  ener =  -11.30464073943204
# final ener =  -14.766878617656943 niter =  338
# final ener =  -14.766880324063017 niter =  165
# final ener =  -14.69582137850006 niter =  265
# final ener =  -14.76688000976375 niter =  195
# final ener =  -14.766879857021873 niter =  182
# final ener =  -14.76688036502653 niter =  179
# final ener =  -14.766880214949722 niter =  201
# final ener =  -14.766880182043035 niter =  221
# final ener =  -14.766878953960633 niter =  352
# final ener =  -14.766880327070155 niter =  266
# final ener =  -14.7668801551673 niter =  225
