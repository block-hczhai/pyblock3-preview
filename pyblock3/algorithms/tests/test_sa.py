import unittest
import tempfile
import numpy as np
from pyblock3.algebra.mpe import CachedMPE
from pyblock3.hamiltonian import Hamiltonian
from pyblock3.fcidump import FCIDUMP

class TestDMRG(unittest.TestCase):
    
    def test_n2_sa(self):

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

        with tempfile.TemporaryDirectory() as scratch:

            nroots = 5
            extra_mpes = [None] * (nroots - 1)
            for ix in range(nroots - 1):
                xmps = hamil.build_mps(250)
                extra_mpes[ix] = CachedMPE(xmps, mpo, xmps, tag='CP%d' % ix, scratch=scratch)

            dmrg = CachedMPE(mps, mpo, mps, scratch=scratch).dmrg(bdims=[400], noises=[1E-6, 0],
                dav_thrds=[1E-3, 1E-6], iprint=2, n_sweeps=10, extra_mpes=extra_mpes, tol=1E-14)
            eners = dmrg.energies[-1]
            eners_std = (-107.6541224475, -107.0314494716, -106.9596261547)

            for i in range(3):
                self.assertAlmostEqual(eners[i], eners_std[i], 7)

if __name__ == "__main__":
    unittest.main()
