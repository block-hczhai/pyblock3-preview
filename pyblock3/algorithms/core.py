
import numpy as np
import pyblock3.algebra.funcs as pbalg
from enum import Enum, auto


def fmt_size(i, suffix='B'):
    if i < 1000:
        return "%d %s" % (i, suffix)
    else:
        a = 1024
        for pf in "KMGTPEZY":
            p = 2
            for k in [10, 100, 1000]:
                if i < k * a:
                    return "%%.%df %%s%%s" % p % (i / a, pf, suffix)
                p -= 1
            a *= 1024
    return "??? " + suffix

class DecompositionTypes(Enum):
    DensityMatrix = auto()
    SVD = auto()


class NoiseTypes(Enum):
    Perturbative = auto()
    Random = auto()


class SweepAlgorithm:
    def __init__(self, cutoff=1E-14,
                 decomp_type=DecompositionTypes.DensityMatrix,
                 noise_type=NoiseTypes.Perturbative):
        self.cutoff = cutoff
        self.decomp_type = decomp_type
        self.noise_type = noise_type

    def add_dm_noise(self, dm, mpo, wfn, noise, forward):
        if noise == 0:
            return dm
        if self.noise_type == NoiseTypes.Random:
            pdm = dm.__class__.random_like(dm)
            return dm + (noise / np.linalg.norm(pdm)) * pdm
        elif self.noise_type == NoiseTypes.Perturbative:
            if forward:
                pket = np.tensordot(
                    wfn[0], mpo[0], axes=((1, 2), (3, 4)))
                pket.normalize_along_axis(axis=7)
                pdm = np.tensordot(pket, pket, axes=((1, 2, 3, 4, 7), ) * 2)
            else:
                pket = np.tensordot(
                    mpo[1], wfn[0], axes=((3, 4), (3, 4)))
                pket.normalize_along_axis(axis=0)
                pdm = np.tensordot(pket, pket, axes=((0, 3, 4, 5, 6), ) * 2)
            norm = np.linalg.norm(pdm)
            return dm + (noise / norm) * pdm if abs(norm) > 1E-12 else dm
        else:
            assert False

    def add_wfn_noise(self, wfn, noise, forward):
        if noise == 0:
            return wfn
        if self.noise_type == NoiseTypes.Random:
            pwfn = wfn.__class__.random_like(wfn)
            return wfn + (np.sqrt(noise) / np.linalg.norm(pwfn)) * pwfn
        else:
            assert False

    def decomp_two_site(self, mpo, wfns, forward, noise, bond_dim, weights=None):
        wfn = wfns[0] if isinstance(wfns, list) else wfns
        if weights is None:
            weights = [1.0 / len(wfns)] * len(wfns)
        if self.decomp_type == DecompositionTypes.DensityMatrix:
            if forward:
                dm = np.tensordot(
                    wfn[0], wfn[0], axes=((-3, -2, -1), ) * 2)
                if isinstance(wfns, list):
                    dm = dm * weights[0]
                    for ex_wfn, w in zip(wfns[1:], weights[1:]):
                        dm = dm + w * np.tensordot(
                            ex_wfn[0], ex_wfn[0], axes=((-3, -2, -1), ) * 2)
                dm = self.add_dm_noise(dm, mpo, wfn, noise, forward)
                lsr = dm.tensor_svd(idx=3, pattern='++++++')
                l, _, _, error = pbalg.truncate_svd(
                    *lsr, cutoff=self.cutoff, max_bond_dim=bond_dim, eigen_values=True)
                wfn[:] = [l, np.tensordot(
                    l, wfn[0], axes=((0, 1, 2), ) * 2)]
            else:
                dm = np.tensordot(
                    wfn[0], wfn[0], axes=((0, 1, 2), ) * 2)
                if isinstance(wfns, list):
                    dm = dm * weights[0]
                    for ex_wfn, w in zip(wfns[1:], weights[1:]):
                        dm = dm + w * np.tensordot(
                            ex_wfn[0], ex_wfn[0], axes=((0, 1, 2), ) * 2)
                dm = self.add_dm_noise(dm, mpo, wfn, noise, forward)
                lsr = dm.tensor_svd(idx=3, pattern='-+--+-')
                _, _, r, error = pbalg.truncate_svd(
                    *lsr, cutoff=self.cutoff, max_bond_dim=bond_dim, eigen_values=True)
                wfn[:] = [np.tensordot(
                    wfn[0], r, axes=((-3, -2, -1), ) * 2), r]
        elif self.decomp_type == DecompositionTypes.SVD:
            assert not isinstance(wfns, list)
            wfn = self.add_wfn_noise(
                wfn[0], noise, forward)
            lsr = wfn.tensor_svd(idx=3, pattern='+++-+-')
            l, s, r, error = pbalg.truncate_svd(
                *lsr, cutoff=self.cutoff, max_bond_dim=bond_dim)
            if noise == 0:
                if forward:
                    wfn[:] = [l, np.tensordot(
                        s.diag(), r, axes=1)]
                else:
                    wfn[:] = [np.tensordot(
                        l, s.diag(), axes=1), r]
            else:
                if forward:
                    wfn[:] = [l, np.tensordot(
                        l, wfn[0], axes=((0, 1, 2), ) * 2)]
                else:
                    wfn[:] = [np.tensordot(
                        wfn[0], r, axes=((-3, -2, -1), ) * 2), r]
        else:
            assert False
        return error
