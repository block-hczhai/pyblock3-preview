
import sys
sys.path[:0] = ['..']

import time
import numpy as np
from pyblock3.algebra.mpe import MPE
from pyblock3.hamiltonian import Hamiltonian
from pyblock3.algorithms.core import DecompositionTypes
from pyblock3.fcidump import FCIDUMP
from pyblock3.symbolic.expr import OpElement, OpNames
from pyblock3.algebra.symmetry import SZ

np.random.seed(1000)

# fd = '../data/HUBBARD-L8.FCIDUMP'
# fd = '../data/HUBBARD-L16.FCIDUMP'
# fd = '../data/N2.STO3G.FCIDUMP'
# fd = '../data/H8.STO6G.R1.8.FCIDUMP'
fd = '../data/H10.STO6G.R1.8.FCIDUMP'
# fd = '../my_test/n2/N2.FCIDUMP'
# fd = '../data/CR2.SVP.FCIDUMP'
occ = None
ket_bond_dim = 500
bra_bond_dim = 750

# occf = '../data/CR2.SVP.OCC'
# occ = [float(x) for x in open(occf, 'r').readline().split()]

hamil = Hamiltonian(FCIDUMP(pg='d2h').read(fd), flat=True)

tx = time.perf_counter()
mpo = hamil.build_qc_mpo()
print('MPO (NC) =         ', mpo.show_bond_dims())
print('build mpo time = ', time.perf_counter() - tx)

tx = time.perf_counter()
mpo, _ = mpo.compress(left=True, cutoff=1E-9, norm_cutoff=1E-9)
print('MPO (compressed) = ', mpo.show_bond_dims())
print('compress mpo time = ', time.perf_counter() - tx)

mps = hamil.build_mps(ket_bond_dim, occ=occ)
print('MPS = ', mps.show_bond_dims())

bdims = [500]
noises = [1E-4, 1E-5, 1E-6, 0]
davthrds = None

dmrg = MPE(mps, mpo, mps).dmrg(bdims=bdims, noises=noises,
                              dav_thrds=davthrds, iprint=2, n_sweeps=20, tol=1E-12)
ener = dmrg.energies[-1]
print("Energy = %20.12f" % ener)

isite = 4
mpo.const -= ener
omega, eta = -0.17, 0.05

dop = OpElement(OpNames.D, (isite, 0), q_label=SZ(-1, -1, hamil.orb_sym[isite]))
bra = hamil.build_mps(bra_bond_dim, target=SZ.to_flat(
    dop.q_label + SZ.from_flat(hamil.target)))
dmpo = hamil.build_site_mpo(dop)
print('DMPO =         ', dmpo.show_bond_dims())
MPE(bra, dmpo, mps).linear(bdims=[bra_bond_dim], noises=noises,
                                 cg_thrds=davthrds, iprint=2, n_sweeps=20, tol=1E-12)

ket = bra.copy()
dt = 0.1
eta = 0.05
t = 100.0

nstep = int(t / dt)

mpe = MPE(ket, mpo, ket)
rtgf = np.zeros((nstep, ), dtype=complex)
print('total step = ', nstep)
for it in range(0, nstep):
    cur_t = it * dt
    mpe.tddmrg(bdims=[500], dt=-dt * 1j, iprint=2, n_sweeps=1, n_sub_sweeps=2, normalize=False)
    rtgf[it] = np.dot(bra.conj(), ket)
    print("=== T = %10.5f EX = %20.15f + %20.15f i" % (cur_t, rtgf[it].real, rtgf[it].imag))

def gf_fft(eta, dt, rtgf):

    dt = abs(dt)
    npts = len(rtgf)

    frq = np.fft.fftfreq(npts, dt)
    frq = np.fft.fftshift(frq) * 2.0 * np.pi
    fftinp = -1j * rtgf * np.exp(-eta * dt * np.arange(0, npts))

    Y = np.fft.fft(fftinp)
    Y = np.fft.fftshift(Y)

    Y_real = Y.real * dt
    Y_imag = Y.imag * dt

    return frq, Y_real, Y_imag
