
Determinant Tools
=================

N2(10o, 7e) (STO3G)
-------------------

Ground-state DMRG (N2 STO3G) with C++ optimized core functions:

.. code:: python3

    import numpy as np
    from pyblock3.algebra.mpe import MPE
    from pyblock3.hamiltonian import Hamiltonian
    from pyblock3.fcidump import FCIDUMP

    fd = 'data/N2.STO3G.FCIDUMP'
    bond_dim = 250
    hamil = Hamiltonian(FCIDUMP(pg='d2h').read(fd), flat=True)
    mpo = hamil.build_qc_mpo()
    mpo, _ = mpo.compress(cutoff=1E-9, norm_cutoff=1E-9)
    mps = hamil.build_mps(bond_dim)

    dmrg = MPE(mps, mpo, mps).dmrg(bdims=[bond_dim], noises=[1E-6, 0], dav_thrds=[1E-4], iprint=2, n_sweeps=10)
    ener = dmrg.energies[-1]
    print("Energy = %20.12f" % ener)

Check MPO:

.. code:: python3

    print('MPO = ', mpo.show_bond_dims())
    mpo, error = mpo.compress(cutoff=1E-12)
    print('MPO = ', mpo.show_bond_dims())

Check MPS:

.. code:: python3

    print('MPS = ', mps.show_bond_dims())
    print(mps.norm())

Check ground-state energy:

.. code:: python3

    mps @ (mpo @ mps)

MPO-MPS Contraction
-------------------

.. code:: python3

    mps.opts = {}
    hmps = mpo @ mps
    print(hmps.show_bond_dims())
    print(hmps.norm())

The result MPS can be compressed:

.. code:: python3

    hmps, _ = hmps.compress(cutoff=1E-12)
    print(hmps.show_bond_dims())
    print(hmps.norm())

MPO truncation to bond dimension 50:

.. code:: python3

    cmpo, cps_error = mpo.compress(max_bond_dim=50, cutoff=1E-12)
    print('error = ', cps_error)
    print(cmpo.show_bond_dims())

Apply contracted MPO to MPS:

.. code:: python3

    hmps = cmpo @ mps
    print(hmps.show_bond_dims())
    print(hmps.norm())

Determinants
------------

Using SliceableTensor:

.. code:: python3

    smps = mps.to_sliceable()
    print(smps[0])
    print('-'*20)
    print(smps[0][:, 2:, 2])
    print('-'*20)
    print(smps[0][:, :2, 2].infos)
    print('-'*20)
    print(smps[0])
    print('-'*20)
    print(smps.amplitude([3, 3, 0, 3, 0, 3, 3, 3, 3, 0]))

If the determinant belongs to another symmetry sector, the overlap should be zero:

.. code:: python3

    print(smps.amplitude([3, 3, 0, 0, 0, 3, 3, 3, 3, 0]))

Check the overlap for all doubly occupied determinants:

.. code:: python3

    import itertools
    coeffs = []
    for ocp in itertools.combinations(range(10), 7):
        det = [0] * mps.n_sites
        for t in ocp:
            det[t] = 3
        tt = time.perf_counter()
        coeffs.append(smps.amplitude(det))
        print(np.array(det), "%10.5f" % coeffs[-1])

Check the sum of probabilities:

.. code:: python3

    print((np.array(coeffs) ** 2).sum())
