
Finite-Temperature DMRG
=======================

Here is an example for calculating :math:`\exp(-\beta/2\cdot H) |\psi\rangle`.

Imports:

.. code:: python3

    from pyblock3.algebra.integrate import rk4_apply
    from pyblock3.hamiltonian import Hamiltonian
    from pyblock3.fcidump import FCIDUMP
    import numpy as np
    import time
    from functools import reduce

    flat = False
    cutoff = 1E-12

Ancilla Approach
----------------

.. code:: python3

    fd = 'data/H8.STO6G.R1.8.FCIDUMP'
    hamil = Hamiltonian(FCIDUMP(pg='d2h', mu=-1.0).read(fd), flat=flat)

    mps = hamil.build_ancilla_mps()
    mpo = hamil.build_ancilla_mpo(hamil.build_qc_mpo())
    mpo.const = 0.0

    print('MPS = ', mps.show_bond_dims())
    print('MPO = ', mpo.show_bond_dims())
    mpo, error = mpo.compress(cutoff=cutoff)
    print('MPO = ', mpo.show_bond_dims(), error)

    init_e = np.dot(mps, mpo @ mps) / np.dot(mps, mps)
    print('Initial Energy = ', init_e)
    print('Error          = ', init_e - 0.3124038410492045)

    mps.opts = dict(max_bond_dim=200, cutoff=cutoff)
    beta = 0.01
    tt = time.perf_counter()
    fmps = rk4_apply((-beta / 2) * mpo, mps)
    ener = np.dot(fmps, mpo @ fmps) / np.dot(fmps, fmps)
    print('time = ', time.perf_counter() - tt)
    print('Energy = ', ener)
    print('Error  = ', ener - 0.2408363230374028)

Matrix Product Density Operator (MPDO) Approach
-----------------------------------------------

.. code:: python3

    mps = hamil.build_ancilla_mps()
    mps.tensors = [a.hdot(b) for a, b in zip(mps.tensors[0::2], mps.tensors[1::2])]
    mpo = hamil.build_qc_mpo()
    mpo.const = 0.0

    print('MPS = ', mps.show_bond_dims())
    print('MPO = ', mpo.show_bond_dims())
    mpo, _ = mpo.compress(cutoff=cutoff)
    print('MPO = ', mpo.show_bond_dims())

    init_e = np.dot(mps, mpo @ mps) / np.dot(mps, mps)
    print('Initial Energy = ', init_e)
    print('Error          = ', init_e - 0.3124038410492045)

    mps.opts = dict(max_bond_dim=200, cutoff=cutoff)
    beta = 0.01
    tt = time.perf_counter()
    fmps = rk4_apply((-beta / 2) * mpo, mps)
    ener = np.dot(fmps, mpo @ fmps) / np.dot(fmps, fmps)
    print('time = ', time.perf_counter() - tt)
    print('Energy = ', ener)
    print('Error  = ', ener - 0.2408363230374028)

Time-Step Targeting Approach
----------------------------

The more efficient way of imaginary time evolution is using Time-Step Targeting Approach:

.. code:: python3

    from pyblock3.hamiltonian import Hamiltonian
    from pyblock3.fcidump import FCIDUMP
    from pyblock3.algebra.mpe import MPE
    import numpy as np
    import time

    flat = True
    cutoff = 1E-12

    fd = '../data/H8.STO6G.R1.8.FCIDUMP'
    hamil = Hamiltonian(FCIDUMP(pg='d2h', mu=-1.0).read(fd), flat=flat)

    mps = hamil.build_ancilla_mps()
    mpo = hamil.build_qc_mpo()
    mpo = hamil.build_ancilla_mpo(mpo)
    mpo.const = 0.0

    print('MPS = ', mps.show_bond_dims())
    print('MPO = ', mpo.show_bond_dims())
    mpo, error = mpo.compress(cutoff=cutoff)
    print('MPO = ', mpo.show_bond_dims(), error)

    init_e = np.dot(mps, mpo @ mps) / np.dot(mps, mps)
    print('Initial Energy = ', init_e)
    print('Error          = ', init_e - 0.3124038410492045)

    beta = 0.05
    mpe = MPE(mps, mpo, mps)
    mpe.tddmrg(bdims=[500], dt=-beta / 2, iprint=2, n_sweeps=1, n_sub_sweeps=6)
    mpe.tddmrg(bdims=[500], dt=-beta / 2, iprint=2, n_sweeps=9, n_sub_sweeps=2)
