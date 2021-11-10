import unittest
import tempfile
import numpy as np
from pyblock3.algebra.mpe import CachedMPE
from pyblock3.hamiltonian import Hamiltonian
from pyblock3.fcidump import FCIDUMP

class TestDMRG(unittest.TestCase):
    
    def test_n2_ghf(self):

        fd = 'data/N2.STO3G.FCIDUMP'
        rfd = FCIDUMP(pg='d2h').read(fd)

        n = rfd.n_sites
        gh1e = np.zeros((n * 2, n * 2))
        gg2e = np.zeros((n * 2, n * 2, n * 2, n * 2))

        for i in range(n * 2):
            for j in range(i % 2, n * 2, 2):
                gh1e[i, j] = rfd.h1e[i // 2, j // 2]

        for i in range(n * 2):
            for j in range(i % 2, n * 2, 2):
                for k in range(n * 2):
                    for l in range(k % 2, n * 2, 2):
                        gg2e[i, j, k, l] = rfd.g2e[i // 2, j // 2, k // 2, l // 2]
        

        gfd = FCIDUMP(pg='d2h', n_sites=n * 2, n_elec=rfd.n_elec, twos=rfd.n_elec,
            ipg=0, h1e=gh1e, g2e=gg2e,
            orb_sym=[rfd.orb_sym[i // 2] for i in range(n * 2)],
            const_e=rfd.const_e)

        hamil = Hamiltonian(gfd, flat=True)
        mpo = hamil.build_qc_mpo()
        mpo, err = mpo.compress(cutoff=1E-9, norm_cutoff=1E-9)
        assert err < 1E-12

        bond_dim = 200
        mps = hamil.build_mps(200)

        mps = mps.canonicalize(center=0)
        mps /= mps.norm()
        assert np.abs(np.dot(mps.conj(), mps) - 1.0) < 1E-10

        bdims = [250] * 5 + [500] * 5
        noises = [1E-5] * 2 + [1E-6] * 4 + [1E-7] * 3 + [0]
        davthrds = [5E-3] * 4 + [1E-3] * 4 + [1E-4]

        with tempfile.TemporaryDirectory() as scratch:
            dmrg = CachedMPE(mps, mpo, mps, scratch=scratch).dmrg(bdims=bdims,
                noises=noises, dav_thrds=davthrds, iprint=2, n_sweeps=5, tol=1E-14)

            ener = dmrg.energies[-1]
            ener_std = -107.654122447525
            print(ener)
            self.assertAlmostEqual(ener, ener_std, 5)
            self.assertAlmostEqual(mps.norm(), 1.0, 7)

if __name__ == "__main__":
    unittest.main()
