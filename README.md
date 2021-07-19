
[![Documentation Status](https://readthedocs.org/projects/pyblock3/badge/?version=latest)](https://pyblock3.readthedocs.io/en/latest/?badge=latest)

pyblock3
========

An Efficient python block-sparse tensor and MPS/DMRG Library

Copyright (C) 2020-2021 The pyblock3 developers. All Rights Reserved.

Authors:

* Huanchen Zhai [@hczhai](https://github.com/hczhai): MPS/MPO/DMRG
* Yang Gao [@yangcal](https://github.com/yangcal): General fermionic tensor
* Garnet Kin-Lic Chan [@gkc1000](https://github.com/gkc1000): Library design

Please cite this work as:

Huanchen Zhai, Yang Gao, and Garnet K.-L. Chan. pyblock3: an efficient python block-sparse tensor and MPS/DMRG library. 2021;
[https://github.com/block-hczhai/pyblock3-preview](https://github.com/block-hczhai/pyblock3-preview).

Documentation: https://pyblock3.readthedocs.io/en/latest

Tutorial: https://colab.research.google.com/drive/1grQyYP9oTivjqQRZiwU40tF9SdWyrPfV?usp=sharing

Features
--------

* Block-sparse tensor algebra with quantum number symmetries:
    * U(1) particle number
    * U(1) spin
    * Abelian point group symmetry
* MPO construction
    * SVD approach for general fermionic Hamiltonian
    * Conventional approach for quantum chemistry Hamiltonian
* MPO/MPS algebra
    * MPO compression
* Efficient sweep algorithms for ab initio systems (2-site):
    * Ground-state DMRG with perturbative noise
    * MPS compression
    * Green's function (DDMRG++)
    * Imaginary time evolution (time-step targeting approach)
    * Real time evolution (time-step targeting approach)
    * Finite-temperature DMRG (ancilla approach)

Installation
------------

Dependence: `python3`, `psutil`, `numba`, and `numpy` (version >= 1.17.0). pyblock3 can run in pure python mode,
in which no C++ source code is required to be compiled.

For optimal performance, the C++ source code is used and there are some additional dependences:

* `pybind11` (https://github.com/pybind/pybind11)
* `cmake` (version >= 3.0)
* `MKL` (or `blas + lapack`)
    * When `MKL` is available, add `cmake` option: `-DUSE_MKL=ON`.
    * If `cmake` cannot find `MKL`, one can add environment variable hint `MKLROOT`.
* C++ compiler: `g++` or `clang`
    * `icpc` currently not tested/supported
* High performance Tensor Transposition library: `hptt` (https://github.com/springer13/hptt) (optional)
    * For CPU with AVX512 flag, one can use this AVX512 version (https://github.com/hczhai/hptt)
    * HPTT is important for optimal performance
    * When `HPTT` is available, add `cmake` option: `-DUSE_HPTT=ON`.
    * If `cmake` cannot find `HPTT`, one can add environment variable hint `HPTTHOME`.
* openMP library `gomp` or `iomp5` (optional)
    * This is required for multi-threading parallelization.
    * For openMP disabled: add `cmake` option: `-DOMP_LIB=SEQ`.
    * For gnu openMP (`gomp`): add `cmake` option: `-DOMP_LIB=GNU`.
    * For intel openMP (`iomp5`): add `cmake` option: `-DOMP_LIB=INTEL` (default).
    * If `cmake` cannot find openMP library, one can add the path to `libgomp.so` or `libiomp5.so` to environment variable `PATH`.

To compile the C++ part of the code (for better performance):

    mkdir build
    cd build
    cmake .. -DUSE_MKL=ON -DUSE_HPTT=ON
    make

Examples
--------

Add package root directory to `PYTHONPATH` before running the following examples.

If you used directory names other than `build` for the build directory (which contains the compiled python extension),
you also need to add the build directory to `PYTHONPATH`.

Ground-state DMRG (H8 STO6G) in pure python (52 seconds):

    import numpy as np
    from pyblock3.algebra.mpe import MPE
    from pyblock3.hamiltonian import Hamiltonian
    from pyblock3.fcidump import FCIDUMP

    fd = 'data/H8.STO6G.R1.8.FCIDUMP'
    bond_dim = 250
    hamil = Hamiltonian(FCIDUMP(pg='d2h').read(fd), flat=False)
    mpo = hamil.build_qc_mpo()
    mpo, _ = mpo.compress(cutoff=1E-9, norm_cutoff=1E-9)
    mps = hamil.build_mps(bond_dim)

    dmrg = MPE(mps, mpo, mps).dmrg(bdims=[bond_dim], noises=[1E-6, 0],
        dav_thrds=[1E-3], iprint=2, n_sweeps=10)
    ener = dmrg.energies[-1]
    print("Energy = %20.12f" % ener)

Ground-state DMRG (H8 STO6G) with C++ optimized core functions (0.87 seconds):

    import numpy as np
    from pyblock3.algebra.mpe import MPE
    from pyblock3.hamiltonian import Hamiltonian
    from pyblock3.fcidump import FCIDUMP

    fd = 'data/H8.STO6G.R1.8.FCIDUMP'
    bond_dim = 250
    hamil = Hamiltonian(FCIDUMP(pg='d2h').read(fd), flat=True)
    mpo = hamil.build_qc_mpo()
    mpo, _ = mpo.compress(cutoff=1E-9, norm_cutoff=1E-9)
    mps = hamil.build_mps(bond_dim)

    dmrg = MPE(mps, mpo, mps).dmrg(bdims=[bond_dim], noises=[1E-6, 0],
        dav_thrds=[1E-3], iprint=2, n_sweeps=10)
    ener = dmrg.energies[-1]
    print("Energy = %20.12f" % ener)

The printed ground-state energy for this system should be `-4.345079402665`.

Adding Extra Symmetry Class
---------------------------

1. Write the C++ definition of the class (named `QPN`, for example) in `src/qpn.hpp`, which should be similar to `src/sz.hpp`.
2. Add the following in `src/symmetry_tmpl.hpp` after `add other symmetries here` line:

        #include "qpn.hpp"
        #define TMPL_Q QPN
        #include NAME_IMPL(TMPL_NAME,_tmpl.hpp)
        #undef TMPL_Q

   Note that if multiple symmetry class are defined in the same file `src/qpn.hpp`, you may only write `#include "qpn.hpp"` once. The other three lines have to be repeated for each symmetry class.
   If you do not need the default symmetry class `SZ` and you want to save compiling time, the four lines for `SZ` can be removed/commented.
3. Add the following in `src/main.hpp` after `bind extra symmetry here` line:

        py::module m_qpn = m.def_submodule("qpn", "General other symmetry.");
        bind_sparse_tensor<QPN>(m_qpn, m, "QPN");
    
    If you do not need the default symmetry class `SZ` and you want to save compiling time, the two lines `bind_ ...` for `SZ` can be removed/commented.

4. In python script, use the following to indicate which symmetry class is being used:

        if DEFAULT_SYMMETRY == SZ:
            import block3.sz as block3
        elif DEFAULT_SYMMETRY == QPN:
            import block3.qpn as block3
