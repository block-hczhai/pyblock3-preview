import unittest
import tempfile
import numpy as np
from pyblock3.algebra.mpe import MPE
from pyblock3.hamiltonian import Hamiltonian
from pyblock3.fcidump import FCIDUMP

class TestDMRG(unittest.TestCase):

    def test_hubbard(self):

        with tempfile.TemporaryDirectory() as scratch:
            fcidump = """&FCI NORB= 8,NELEC= 8,MS2= 0,
                ORBSYM=1,1,1,1,1,1,1,1,
                ISYM=1,
                /
                2.00 1 1 1 1
                2.00 2 2 2 2
                2.00 3 3 3 3
                2.00 4 4 4 4
                2.00 5 5 5 5
                2.00 6 6 6 6
                2.00 7 7 7 7
                2.00 8 8 8 8
                1.00 1 2 0 0
                1.00 2 3 0 0
                1.00 3 4 0 0
                1.00 4 5 0 0
                1.00 5 6 0 0
                1.00 6 7 0 0
                1.00 7 8 0 0
                0.00 0 0 0 0
                """
            open(scratch + "/FCIDUMP", 'w').write(fcidump)

            fd = scratch + '/FCIDUMP'
            hamil = Hamiltonian(FCIDUMP(pg='d2h').read(fd), flat=True)
            mpo = hamil.build_qc_mpo()
            mpo, err = mpo.compress(cutoff=1E-9, norm_cutoff=1E-9)
            assert mpo.show_bond_dims() == "1|6|6|6|6|6|6|6|1"
            assert err < 1E-12

            bond_dim = 200
            mps = hamil.build_mps(200)

            mps = mps.canonicalize(center=0)
            mps /= mps.norm()
            assert np.abs(np.dot(mps.conj(), mps) - 1.0) < 1E-10

            dmrg = MPE(mps, mpo, mps).dmrg(bdims=[bond_dim], noises=[1E-6, 0],
                dav_thrds=[1E-3, 1E-6], iprint=0, n_sweeps=10, tol=1E-14)
            ener = dmrg.energies[-1]
            ener_std = -6.225634144681
            print(ener)
            self.assertAlmostEqual(ener, ener_std, 7)

            ener = np.dot(mps, mpo @ mps)
            self.assertAlmostEqual(ener, ener_std, 7)

            self.assertAlmostEqual(mps.norm(), 1.0, 7)
    
    def test_n2(self):

        fd = 'data/N2.STO3G.FCIDUMP'
        hamil = Hamiltonian(FCIDUMP(pg='d2h').read(fd), flat=True)
        mpo = hamil.build_qc_mpo()
        mpo, err = mpo.compress(cutoff=1E-9, norm_cutoff=1E-9)
        assert err < 1E-12

        bond_dim = 200
        mps = hamil.build_mps(200)

        mps = mps.canonicalize(center=0)
        mps /= mps.norm()
        assert np.abs(np.dot(mps.conj(), mps) - 1.0) < 1E-10

        dmrg = MPE(mps, mpo, mps).dmrg(bdims=[bond_dim], noises=[1E-6, 0],
            dav_thrds=[1E-3, 1E-6], iprint=0, n_sweeps=10, tol=1E-14)
        ener = dmrg.energies[-1]
        ener_std = -107.654122447525
        print(ener)
        self.assertAlmostEqual(ener, ener_std, 7)

        ener = np.dot(mps, mpo @ mps)
        self.assertAlmostEqual(ener, ener_std, 7)

        self.assertAlmostEqual(mps.norm(), 1.0, 7)

if __name__ == "__main__":
    unittest.main()
