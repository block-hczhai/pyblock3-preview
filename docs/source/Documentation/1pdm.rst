
One-Particle Density Matrix
===========================

N2(10o, 7e) (STO3G)
-------------------

Ground-state DMRG (N2 STO3G) with C++ optimized core functions:

.. code:: python3

    import numpy as np
    from pyblock3.algebra.mpe import MPE
    from pyblock3.hamiltonian import Hamiltonian
    from pyblock3.fcidump import FCIDUMP
    from pyblock3.symbolic.expr import OpElement, OpNames
    from pyblock3.algebra.symmetry import SZ

    fd = 'data/N2.STO3G.FCIDUMP'
    bond_dim = 500
    hamil = Hamiltonian(FCIDUMP(pg='d2h').read(fd), flat=True)
    mpo = hamil.build_qc_mpo()
    mpo, _ = mpo.compress(cutoff=1E-9, norm_cutoff=1E-9)
    mps = hamil.build_mps(bond_dim)

    dmrg = MPE(mps, mpo, mps).dmrg(bdims=[bond_dim], noises=[1E-6, 0], dav_thrds=[1E-4], iprint=2, n_sweeps=10)
    ener = dmrg.energies[-1]
    print("Energy = %20.12f" % ener)

    print('energy error = ', np.abs(ener - -107.654122447525))
    assert np.abs(ener - -107.654122447525) < 1E-6

Now we can calculate the 1pdm based on ground-state MPS, and compare it with the FCI result.

.. code:: python3

    # FCI results
    pdm1_std = np.zeros((hamil.n_sites, hamil.n_sites))
    pdm1_std[0, 0] = 1.999989282592
    pdm1_std[0, 1] = -0.000025398134
    pdm1_std[0, 2] = 0.000238560621
    pdm1_std[1, 0] = -0.000025398134
    pdm1_std[1, 1] = 1.991431489457
    pdm1_std[1, 2] = -0.005641787787
    pdm1_std[2, 0] = 0.000238560621
    pdm1_std[2, 1] = -0.005641787787
    pdm1_std[2, 2] = 1.985471515555
    pdm1_std[3, 3] = 1.999992764813
    pdm1_std[3, 4] = -0.000236022833
    pdm1_std[3, 5] = 0.000163863520
    pdm1_std[4, 3] = -0.000236022833
    pdm1_std[4, 4] = 1.986371259953
    pdm1_std[4, 5] = 0.018363506969
    pdm1_std[5, 3] = 0.000163863520
    pdm1_std[5, 4] = 0.018363506969
    pdm1_std[5, 5] = 0.019649294772
    pdm1_std[6, 6] = 1.931412559660
    pdm1_std[7, 7] = 0.077134636900
    pdm1_std[8, 8] = 1.931412559108
    pdm1_std[9, 9] = 0.077134637190

    pdm1 = np.zeros((hamil.n_sites, hamil.n_sites))
    for i in range(hamil.n_sites):
        diop = OpElement(OpNames.D, (i, 0), q_label=SZ(-1, -1, hamil.orb_sym[i]))
        di = hamil.build_site_mpo(diop)
        for j in range(hamil.n_sites):
            djop = OpElement(OpNames.D, (j, 0), q_label=SZ(-1, -1, hamil.orb_sym[j]))
            dj = hamil.build_site_mpo(djop)
            # factor 2 due to alpha + beta spins
            pdm1[i, j] = 2 * np.dot((di @ mps).conj(), dj @ mps)

    # 1pdm error is often approximately np.sqrt(error in energy)
    print('max 1pdm error = ', np.abs(pdm1 - pdm1_std).max())
    assert np.abs(pdm1 - pdm1_std).max() < 1E-6
