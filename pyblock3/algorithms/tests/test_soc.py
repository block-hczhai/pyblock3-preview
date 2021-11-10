import unittest
import tempfile
import numpy as np
from pyblock3.algebra.mpe import CachedMPE
from pyblock3.hamiltonian import Hamiltonian
from pyblock3.fcidump import FCIDUMP

class TestDMRG(unittest.TestCase):
    
    def test_water_soc(self):

        fd = 'data/H2O.STO3G.FCIDUMP'
        rfd = FCIDUMP(pg='c1').read(fd)
        fd = 'data/H2O.STO3G.MOCOEFF'
        mo_coeff = FCIDUMP(pg='c1').read(fd).h1e

        n = rfd.n_sites
        gh1e = np.zeros((n * 2, n * 2), dtype=complex)
        gg2e = np.zeros((n * 2, n * 2, n * 2, n * 2), dtype=complex)

        for i in range(n * 2):
            for j in range(i % 2, n * 2, 2):
                gh1e[i, j] = rfd.h1e[i // 2, j // 2]

        for i in range(n * 2):
            for j in range(i % 2, n * 2, 2):
                for k in range(n * 2):
                    for l in range(k % 2, n * 2, 2):
                        gg2e[i, j, k, l] = rfd.g2e[i // 2, j // 2, k // 2, l // 2]

        # atomic mean-field spin-orbit integral (AMFI)
        hsoao = np.zeros((3, 7, 7), dtype=complex)
        v = 2.1281747964476273E-004j * 2
        hsoao[:] = 0
        hsoao[0, 4, 3] = hsoao[1, 2, 4] = hsoao[2, 3, 2] = v
        hsoao[0, 3, 4] = hsoao[1, 4, 2] = hsoao[2, 2, 3] = -v

        hso = np.einsum('rij,ip,jq->rpq', hsoao, mo_coeff, mo_coeff)

        for i in range(n * 2):
            for j in range(n * 2):
                if i % 2 == 0 and j % 2 == 0: # aa
                    gh1e[i, j] += hso[2, i // 2, j // 2] * 0.5
                elif i % 2 == 1 and j % 2 == 1: # bb
                    gh1e[i, j] -= hso[2, i // 2, j // 2] * 0.5
                elif i % 2 == 0 and j % 2 == 1: # ab
                    gh1e[i, j] += (hso[0, i // 2, j // 2] - hso[1, i // 2, j // 2] * 1j) * 0.5
                elif i % 2 == 1 and j % 2 == 0: # ba
                    gh1e[i, j] += (hso[0, i // 2, j // 2] + hso[1, i // 2, j // 2] * 1j) * 0.5
                else:
                    assert False

        gfd = FCIDUMP(pg='d2h', n_sites=n * 2, n_elec=rfd.n_elec, twos=rfd.n_elec,
            ipg=0, h1e=gh1e, g2e=gg2e,
            orb_sym=[rfd.orb_sym[i // 2] for i in range(n * 2)],
            const_e=rfd.const_e)

        hamil = Hamiltonian(gfd, flat=True)
        mpo = hamil.build_complex_qc_mpo(max_bond_dim=-5)
        mpo, err = mpo.compress(cutoff=1E-9, norm_cutoff=1E-9)
        assert err < 1E-12

        bond_dim = 250
        mps = hamil.build_mps(250)
        mps = mps.canonicalize(center=0)
        mps /= mps.norm()

        assert np.abs(np.dot(mps.conj(), mps) - 1.0) < 1E-10

        with tempfile.TemporaryDirectory() as scratch:

            nroots = 4
            extra_mpes = [None] * (nroots - 1)
            for ix in range(nroots - 1):
                xmps = hamil.build_mps(250)
                extra_mpes[ix] = CachedMPE(xmps, mpo, xmps, tag='CP%d' % ix, scratch=scratch)

            dmrg = CachedMPE(mps, mpo, mps, scratch=scratch).dmrg(bdims=[400], noises=[1E-6, 0],
                dav_thrds=[1E-3, 1E-6], iprint=2, n_sweeps=10, extra_mpes=extra_mpes,
                tol=1E-14)
            eners = dmrg.energies[-1]
            eners_std = (-74.931919519513997, -74.513038645127153, -74.513038556765679)

            for i in range(3):
                self.assertAlmostEqual(eners[i], eners_std[i], 5)

if __name__ == "__main__":
    unittest.main()
