
MPO Construction
================

From FCIDUMP
------------

MPO can be constructed from a FCIDUMP file:

.. code:: python3

    from pyblock3.hamiltonian import Hamiltonian
    from pyblock3.fcidump import FCIDUMP

    fd = 'data/H8.STO6G.R1.8.FCIDUMP'
    hamil = Hamiltonian(FCIDUMP(pg='d2h').read(fd), flat=False)
    mpo = hamil.build_qc_mpo()

This will build a ``MPS`` object (representing MPO) using a list of ``FermionTensor``.

If ``flat`` parameter is set to ``True`` in ``Hamiltonian``, the code will use more efficient C++ code for building
MPO, and the resulting MPO is a ``MPS`` object with a list of ``FlatFermionTensor`` included.

.. code:: python3

    from pyblock3.hamiltonian import Hamiltonian
    from pyblock3.fcidump import FCIDUMP

    fd = 'data/H8.STO6G.R1.8.FCIDUMP'
    hamil = Hamiltonian(FCIDUMP(pg='d2h').read(fd), flat=True)
    mpo = hamil.build_qc_mpo()

One can also use ``mpo.to_flat()`` to transform a ``FermionTensor`` MPO to a ``FlatFermionTensor`` MPO.

From Hamiltonian expression (pure python)
-----------------------------------------

A slower but more general way of build MPO is from Hamiltonian expression.

One can set the explicit expression for Hamiltonian for Hubbard model and quantum chemistry:

.. code:: python3

    def build_hubbard(u=2, t=1, n=8, cutoff=1E-9):
        fcidump = FCIDUMP(pg='c1', n_sites=n, n_elec=n, twos=0, ipg=0, orb_sym=[0] * n)
        hamil = Hamiltonian(fcidump, flat=False)

        def generate_terms(n_sites, c, d):
            for i in range(0, n_sites):
                for s in [0, 1]:
                    if i - 1 >= 0:
                        yield t * c[i, s] * d[i - 1, s]
                    if i + 1 < n_sites:
                        yield t * c[i, s] * d[i + 1, s]
                yield u * (c[i, 0] * c[i, 1] * d[i, 1] * d[i, 0])

        return hamil, hamil.build_mpo(generate_terms, cutoff=cutoff).to_sparse()

.. code:: python3

    def build_qc(filename, pg='d2h', cutoff=1E-9):
        fcidump = FCIDUMP(pg=pg).read(fd)
        hamil = Hamiltonian(fcidump, flat=False)

        def generate_terms(n_sites, c, d):
            for i in range(0, n_sites):
                for j in range(0, n_sites):
                    for s in [0, 1]:
                        t = fcidump.t(s, i, j)
                        if abs(t) > cutoff:
                            yield t * c[i, s] * d[j, s]
            for i in range(0, n_sites):
                for j in range(0, n_sites):
                    for k in range(0, n_sites):
                        for l in range(0, n_sites):
                            for sij in [0, 1]:
                                for skl in [0, 1]:
                                    v = fcidump.v(sij, skl, i, j, k, l)
                                    if abs(v) > cutoff:
                                        yield (0.5 * v) * (c[i, sij] * c[k, skl] * d[l, skl] * d[j, sij])

        return hamil, hamil.build_mpo(generate_terms, cutoff=cutoff).to_sparse()

Then the MPO can be built by:

.. code:: python3

    hamil, mpo = build_hubbard(n=4)

or

.. code:: python3

    fd = 'data/H8.STO6G.R1.8.FCIDUMP'
    hamil, mpo = build_qc(fd, cutoff=1E-12)

From Hamiltonian expression (fast)
----------------------------------

If C++ optimized code and ``numba`` are available, when there are very large number of terms in Hamiltonian,
the MPO building process can be accelerated:

First, we can use ``numba`` optimized functions to set the Hamiltonian terms:

.. code:: python3

    import numpy as np
    import numba as nb

    flat = True

    SPIN, SITE, OP = 1, 2, 16384
    @nb.njit(nb.types.Tuple((nb.float64[:], nb.int32[:, :]))(nb.int32, nb.float64, nb.float64))
    def generate_hubbard_terms(n_sites, u, t):
        OP_C, OP_D = 0 * OP, 1 * OP
        h_values = []
        h_terms = []
        for i in range(0, n_sites):
            for s in [0, 1]:
                if i - 1 >= 0:
                    h_values.append(t)
                    h_terms.append([OP_C + i * SITE + s * SPIN, OP_D + (i - 1) * SITE + s * SPIN, -1, -1])
                if i + 1 < n_sites:
                    h_values.append(t)
                    h_terms.append([OP_C + i * SITE + s * SPIN, OP_D + (i + 1) * SITE + s * SPIN, -1, -1])
                h_values.append(0.5 * u)
                h_terms.append([OP_C + i * SITE + s * SPIN, OP_C + i * SITE + (1 - s) * SPIN,
                                OP_D + i * SITE + (1 - s) * SPIN, OP_D + i * SITE + s * SPIN])
        return np.array(h_values, dtype=np.float64), np.array(h_terms, dtype=np.int32)

    @nb.njit(nb.types.Tuple((nb.float64[:], nb.int32[:, :]))
            (nb.int32, nb.float64[:, :], nb.float64[:, :, :, :], nb.float64))
    def generate_qc_terms(n_sites, h1e, g2e, cutoff=1E-9):
        OP_C, OP_D = 0 * OP, 1 * OP
        h_values = []
        h_terms = []
        for i in range(0, n_sites):
            for j in range(0, n_sites):
                t = h1e[i, j]
                if abs(t) > cutoff:
                    for s in [0, 1]:
                        h_values.append(t)
                        h_terms.append([OP_C + i * SITE + s * SPIN,
                                        OP_D + j * SITE + s * SPIN, -1, -1])
        for i in range(0, n_sites):
            for j in range(0, n_sites):
                for k in range(0, n_sites):
                    for l in range(0, n_sites):
                        v = g2e[i, j, k, l]
                        if abs(v) > cutoff:
                            for sij in [0, 1]:
                                for skl in [0, 1]:
                                    h_values.append(0.5 * v)
                                    h_terms.append([OP_C + i * SITE + sij * SPIN,
                                                    OP_C + k * SITE + skl * SPIN,
                                                    OP_D + l * SITE + skl * SPIN,
                                                    OP_D + j * SITE + sij * SPIN])
        return np.array(h_values, dtype=np.float64), np.array(h_terms, dtype=np.int32)

    def build_hubbard(u=2, t=1, n=8, cutoff=1E-9):
        fcidump = FCIDUMP(pg='c1', n_sites=n, n_elec=n,
                        twos=0, ipg=0, orb_sym=[0] * n)
        hamil = Hamiltonian(fcidump, flat=flat)
        terms = generate_hubbard_terms(n, u, t)
        return hamil, hamil.build_mpo(terms, cutoff=cutoff).to_sparse()

    def build_qc(filename, pg='d2h', cutoff=1E-9, max_bond_dim=-1):
        fcidump = FCIDUMP(pg=pg).read(filename)
        hamil = Hamiltonian(fcidump, flat=flat)
        terms = generate_qc_terms(
            fcidump.n_sites, fcidump.h1e, fcidump.g2e, cutoff)
        return hamil, hamil.build_mpo(terms, cutoff=cutoff, max_bond_dim=max_bond_dim).to_sparse()

Then the MPO can be built by:

.. code:: python3

    hamil, mpo = build_hubbard(n=16, cutoff=cutoff)

or

.. code:: python3

    fd = 'data/H8.STO6G.R1.8.FCIDUMP'
    hamil, mpo = build_qc(fd, cutoff=cutoff, max_bond_dim=-1)

From ``pyscf``
--------------

The ``FCIDUMP`` can also be initialized using integral arrays, such as those obtained from ``pyscf``.
Here is an example for H10 (STO6G).

Note that running pyblock3 and pyscf in the same python script with openMP activated may cause some conflicts
in parallel MKL library, in some cases. One need to check number of threads used by pyblock3 during DMRG,
to make sure that number of openMP threads is correct.

.. code:: python3

   from pyscf import gto, scf, lo, symm, ao2mo
    # H chain
    N = 10
    BOHR = 0.52917721092  # Angstroms
    R = 1.8 * BOHR
    mol = gto.M(atom=[['H', (0, 0, i * R)] for i in range(N)],
                basis='sto6g', verbose=0, symmetry=mpg)
    pg = mol.symmetry.lower()
    mf = scf.RHF(mol)
    ener = mf.kernel()
    print("SCF Energy = %20.15f" % ener)

    if pg == 'd2h':
        fcidump_sym = ["Ag", "B3u", "B2u", "B1g", "B1u", "B2g", "B3g", "Au"]
    elif pg == 'c1':
        fcidump_sym = ["A"]

    mo_coeff = mf.mo_coeff
    n_mo = mo_coeff.shape[1]
    orb_sym_str = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mo_coeff)
    xorb_sym = np.array([fcidump_sym.index(i) + 1 for i in orb_sym_str])
    h1e = mo_coeff.T @ mf.get_hcore() @ mo_coeff
    g2e = ao2mo.restore(1, ao2mo.kernel(mol, mo_coeff), n_mo)
    ecore = mol.energy_nuc()
    na = nb = mol.nelectron // 2

    orb_sym = [PointGroup[mpg][i] for i in xorb_sym]
    fd = FCIDUMP(pg='c1', n_sites=n_mo, n_elec=na + nb, twos=na - nb, ipg=0, uhf=False,
                h1e=h1e, g2e=g2e, orb_sym=orb_sym, const_e=ecore, mu=0)
    hamil = Hamiltonian(fd, flat=True)
    mpo = hamil.build_qc_mpo()
