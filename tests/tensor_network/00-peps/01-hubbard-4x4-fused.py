
# PEPS 4x4 simple update
# need fermiyang branch of quimb
# https://github.com/jcmgray/quimb/tree/fermiyang

Lx = 4
Ly = 4
Ne = 16
D = 4
u = 8
t = 1

d = 4
chi = 2 * D ** 2
energy_interval = 100000

import numpy as np
import time
import quimb.tensor as qtn
from quimb.tensor.fermion.block_interface import set_options

from pyblock3.algebra import fermion
fermion.ENABLE_FUSED_IMPLS = True

symmetry = 'u1'
set_options(symmetry=symmetry, use_cpp=False)

su_tau_dict = {0.5: 100, 0.1: 200, 0.05: 200, 0.01: 400}

np.random.seed(99)
print('D =', D)

from quimb.tensor.fermion.fermion_2d_tebd import Hubbard2D, SimpleUpdate


def get_u1_init_fermionic_peps(Nx, Ny, Ne):

    from quimb.tensor.fermion.fermion_core import FermionTensor, FermionTensorNetwork
    from pyblock3.algebra.fermion_ops import creation, bonded_vaccum
    from quimb.tensor.fermion.fermion_2d import FPEPS
    import itertools

    peps = qtn.PEPS.rand(Nx, Ny, bond_dim=1, phys_dim=1)
    fpeps = FermionTensorNetwork([])

    ind_to_pattern_map = dict()
    inv_pattern = {"+": "-", "-": "+"}

    # Needed operators
    cre = creation('sum')  # a^{\dagger}_{alpha} + a^{\dagger}_{beta}
    # a^{\dagger}_{alpha}a^{\dagger}_{beta}
    cre_double = np.tensordot(creation('a'), creation('b'), axes=([-1], [0]))

    # Function to get pattern for tensor inds
    def get_pattern(inds):
        """
        make sure patterns match in input tensors, eg,

        --->A--->B--->
         i    j    k
        pattern for A_ij = +-
        pattern for B_jk = +-
        the pattern of j index must be reversed in two operands
        """
        pattern = ''
        for ix in inds[:-1]:
            if ix in ind_to_pattern_map:
                ipattern = inv_pattern[ind_to_pattern_map[ix]]
            else:
                nmin = pattern.count("-")
                ipattern = "-" if nmin * 2 < len(pattern) else "+"
                ind_to_pattern_map[ix] = ipattern
            pattern += ipattern
        pattern += "+"  # assuming last index is the physical index
        return pattern

    # Assign number of particles for each site
    nelec = dict()
    sites = [(x, y) for x in range(Nx) for y in range(Ny)]

    sites = [sites[_] for _ in np.random.permutation(Nx * Ny)]
    if Ne == Nx * Ny:
        # Half filling
        for site in sites:
            nelec[site] = 1
    elif Ne < Nx * Ny:
        # Less than half filling
        for siteind in range(Ne):
            nelec[sites[siteind]] = 1
        for siteind in range(Ne, Nx * Ny):
            nelec[sites[siteind]] = 0
    elif Ne > Nx * Ny:
        # More than half filling
        for siteind in range(Nx * Ny):
            if siteind < Ne - Nx * Ny:
                nelec[sites[siteind]] = 2
            else:
                nelec[sites[siteind]] = 1

    for ix, iy in itertools.product(range(peps.Lx), range(peps.Ly)):

        T = peps[ix, iy]
        pattern = get_pattern(T.inds)

        vac = bonded_vaccum((1,) * (T.ndim - 1), pattern=pattern)
        trans_order = list(range(1, T.ndim)) + [0]

        if nelec[ix, iy] == 1:
            data = np.tensordot(cre, vac, axes=(
                (1,), (-1,))).transpose(trans_order)
        elif nelec[ix, iy] == 2:
            data = np.tensordot(cre_double, vac, axes=(
                (1,), (-1,))).transpose(trans_order)
        elif nelec[ix, iy] == 0:
            data = vac

        new_T = FermionTensor(data, inds=T.inds, tags=T.tags)
        fpeps.add_tensor(new_T, virtual=False)

    fpeps.view_as_(FPEPS, like=peps)
    print(
        f'Number of electrons in fpeps: {np.sum([i.data.dq for i in fpeps.tensors])}')

    return fpeps


peps = get_u1_init_fermionic_peps(Lx, Ly, Ne)
hamil = Hubbard2D(t, u, Lx, Ly)
su = SimpleUpdate(peps, hamil, D=D, chi=chi)

# Run the simple update evolution
taus = sorted(su_tau_dict.keys())[::-1]
for tau in taus:
    niters = su_tau_dict[tau]
    tx = time.perf_counter()
    Ei = None
    with qtn.contraction.contract_backend("numpy"):
        for step in range(niters):
            su.sweep(tau)
            if (step + 1) % energy_interval == 0:
                Ei = su.get_state().compute_local_expectation(hamil.terms,
                                                              max_bond=chi, normalized=True)
        if Ei is None:
            Ei = su.get_state().compute_local_expectation(hamil.terms,
                                                          max_bond=chi, normalized=True)
    tt = time.perf_counter() - tx
    from pyblock3.algebra.fermion import clear_timing, format_timing
    print('SU Sweep %5d / %5d NSTEP = %5d ENERGY = %20.6f Time = %10.3f %s' %
          (taus.index(tau), len(taus), niters, Ei, tt, format_timing()))
    clear_timing()

with qtn.contraction.contract_backend("numpy"):
    Ei = su.get_state().compute_local_expectation(hamil.terms,
                                                  max_bond=chi, normalized=True)
    peps = su.get_state()

print('Final energy = ', Ei)
