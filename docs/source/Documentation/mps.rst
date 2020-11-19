
MPS Algebra
===========

Construct MPO (set ``flat=False`` if you want to test pure python code):

.. code:: python3

    from pyblock3.hamiltonian import Hamiltonian
    from pyblock3.fcidump import FCIDUMP

    fd = 'data/HUBBARD-L8.FCIDUMP'
    hamil = Hamiltonian(FCIDUMP(pg='c1').read(fd), flat=True)
    mpo = hamil.build_qc_mpo()

Construct (random initial) MPS:

.. code:: python3

    bond_dim = 100
    mps = hamil.build_mps(bond_dim)

Expectation value:

.. code:: python3

    import numpy as np
    np.dot(mps, mpo @ mps)

Block-sparse tensor algebra:

.. code:: python3

    np.tensordot(mps[0], mps[1], axes=1)

MPS canonicalization:

.. code:: python3

    print("MPS = ", mps.show_bond_dims())
    mps = mps.canonicalize(center=0)
    mps /= mps.norm()
    print("MPS = ", mps.show_bond_dims()

Check norm after normalization:

.. code:: python3

    np.dot(mps, mps)

MPO Compression:

.. code:: python3

    print("MPO = ", mpo.show_bond_dims())
    mpo, _ = mpo.compress(left=True, cutoff=1E-12, norm_cutoff=1E-12)
    print("MPO = ", mpo.show_bond_dims())

DMRG:

.. code:: python3

    from pyblock3.algebra.mpe import MPE
    dmrg = MPE(mps, mpo, mps).dmrg(bdims=[bond_dim], noises=[1E-6, 0], dav_thrds=[1E-3], iprint=2, n_sweeps=10)
    ener = dmrg.energies[-1]
    print("Energy = %20.12f" % ener)

Check ground-state energy:

.. code:: python3

    print('MPS energy = ', np.dot(mps, mpo @ mps))

Check that ground-state MPS is normalized:

.. code:: python3

    print('MPS = ', mps.show_bond_dims())
    print('MPS norm = ', mps.norm())

MPS Scaling
-----------

MPS scaling (by scaling the first MPS tensor):

.. code:: python3

    mps.opts = {}
    print('2 MPS = ', (2 * mps).show_bond_dims())
    print((2 * mps).norm())

Check the first MPS tensor:

.. code:: python3

    mps[0]

and

.. code:: python3

    (2 * mps)[0]

MPS Addition
------------

MPS addition will increase the bond dimension:

.. code:: python3

    mps_add = mps + mps
    print('MPS + MPS = ', mps_add.show_bond_dims())
    print(mps_add.norm())

Check the overlap :math:`<2MPS|MPS+MPS>`:

.. code:: python3

    mps_add @ (2 * mps)

MPS Canonicalization
--------------------

Left canonicalization:

.. code:: python3

    lmps = mps_add.canonicalize(center=mps_add.n_sites - 1)
    print('L-MPS = ', lmps.show_bond_dims())

Right canonicalization:

.. code:: python3

    rmps = mps_add.canonicalize(center=0)
    print('R-MPS = ', rmps.show_bond_dims())

Check the overlap :math:`<LMPS|RMPS>`:

.. code:: python3

    lmps @ rmps

MPS Compression
---------------

Compression will first do canonicalization from left to right, then do SVD from right to left.

This can further decrease bond dimension of MPS.

.. code:: python3

    print('MPS + MPS = ', mps_add.show_bond_dims())
    mps_add, _ = mps_add.compress(cutoff=1E-9)
    print('MPS + MPS = ', mps_add.show_bond_dims())
    print(mps_add.norm())

MPS Subtraction
---------------

Subtractoin will also increase bond dimension:

.. code:: python3

    mps_minus = mps - mps
    print('MPS - MPS = ', mps_minus.show_bond_dims())

After compression, this is zero:

.. code:: python3

    mps_minus, _ = mps_minus.compress(cutoff=1E-12)
    print('MPS - MPS = ', mps_minus.show_bond_dims())
    print(mps_minus.norm())

MPS Bond Dimension Truncation
-----------------------------

Apply MPO two times to MPS:

.. code:: python3

    hhmps = mpo @ (mpo @ mps)
    print(hhmps.show_bond_dims())
    print(np.sqrt(hhmps @ mps))

MPS compression can be used to reduce bond dimension (to FCI):

.. code:: python3

    hhmps, cps_error = hhmps.compress(cutoff=1E-12)
    print('error = ', cps_error)
    print(hhmps.show_bond_dims())
    print(np.sqrt(hhmps @ mps))

Truncation to bond dimension 100 will introduce a small error:

.. code:: python3

    hhmps, cps_error = hhmps.compress(max_bond_dim=100, cutoff=1E-12)
    print('error = ', cps_error)
    print(hhmps.show_bond_dims())
    print(np.sqrt(hhmps @ mps))

Truncation to bond dimension 30 will introduce a larger error:

.. code:: python3

    hhmps, cps_error = hhmps.compress(max_bond_dim=30, cutoff=1E-12)
    print('error = ', cps_error)
    print(hhmps.show_bond_dims())
    print(np.sqrt(hhmps @ mps))

MPO-MPO Contraction
-------------------

One can also first contract two MPO:

.. code:: python3

    h2 = mpo @ mpo
    print(h2.show_bond_dims())

Check expectation value:

.. code:: python3

    print(np.sqrt((h2 @ mps) @ mps))

MPO Bond Dimension Truncation
-----------------------------

Compression MPO (keeping accuracy):

.. code:: python3

    h2, cps_error = h2.compress(cutoff=1E-12)
    print('error = ', cps_error)
    print(h2.show_bond_dims())
    print(np.sqrt((h2 @ mps) @ mps))

MPO Truncated to bond dimension 15:

.. code:: python3

    h2, cps_error = h2.compress(max_bond_dim=15, cutoff=1E-12)
    print('error = ', cps_error)
    print(h2.show_bond_dims())
    print(np.sqrt((h2 @ mps) @ mps))

MPO Truncated to bond dimension 12:

.. code:: python3

    h2, cps_error = h2.compress(max_bond_dim=12, cutoff=1E-12)
    print('error = ', cps_error)
    print(h2.show_bond_dims())
    print(np.sqrt((h2 @ mps) @ mps))
