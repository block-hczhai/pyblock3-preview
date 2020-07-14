
from .symmetry import StateFusingInfo


class MovingEnvironment:

    def __init__(self, bra, mpo, ket, dot, bra_info, ket_info):
        self.bra = bra
        self.mpo = mpo
        self.ket = ket
        self.bra_info = bra_info
        self.ket_info = ket_info
        assert self.bra.center == self.ket.center
        assert self.bra.n_sites == self.ket.n_sites
        assert self.mpo.n_sites == self.ket.n_sites
        self.left_envs = [None] * self.n_sites
        self.right_envs = [None] * self.n_sites
        self.dot = dot

    def initialize(self):
        for i in range(1, self.center + 1):
            self.left_envs[i] = self._left_contract_rotate(
                i - 1, prev=self.left_envs[i - 1])
        for i in range(self.n_sites - self.dot - 1, self.center - 1, -1):
            self.right_envs[i] = self._right_contract_rotate(
                i + self.dot, prev=self.right_envs[i + 1])

    def _left_contract_rotate(self, i, prev=None):

        left = prev @ self.mpo[i] if prev is not None else self.mpo[i]

        # fusing info
        if prev is None:
            kl = self.ket_info.vacuum
        else:
            kl = self.ket[i - 1].get_state_info(1)
        km = self.ket_info.basis[i]
        kr = self.ket[i].get_state_info(1)
        kinfo = StateFusingInfo.tensor_product(kl, km, ref=kr)

        if prev is None:
            bl = self.bra_info.vacuum
        else:
            bl = self.bra[i - 1].get_state_info(1)
        bm = self.bra_info.basis[i]
        br = self.bra[i].get_state_info(1)
        binfo = StateFusingInfo.tensor_product(bl, bm, ref=br)

        # fuse and rotate
        for k, v in left.ops.items():
            fv = v.fuse(0, 2, binfo).fuse(1, 2, kinfo)
            # contracted index will be determined by left fusing type in MPS tensor
            # so no need to transpose
            left.ops[k] = self.bra[i] @ fv @ self.ket[i]

        return left

    def _right_contract_rotate(self, i, prev=None):

        right = self.mpo[i] @ prev if prev is not None else self.mpo[i]

        # fusing info
        if prev is None:
            kr = self.ket_info.target
        else:
            kr = self.ket[i + 1].get_state_info(0)
        km = self.ket_info.basis[i]
        kl = self.ket[i].get_state_info(0)
        kinfo = StateFusingInfo.tensor_product(-km, kr, ref=kl)

        if prev is None:
            br = self.bra_info.target
        else:
            br = self.bra[i + 1].get_state_info(0)
        bm = self.bra_info.basis[i]
        bl = self.bra[i].get_state_info(0)
        binfo = StateFusingInfo.tensor_product(-bm, br, ref=bl)

        # fuse and rotate
        for k, v in right.ops.items():
            fv = v.fuse(0, 2, binfo, rev=True).fuse(1, 2, kinfo, rev=True)
            # contracted index will be determined by right fusing type in MPS tensor
            # so no need to transpose
            right.ops[k] = self.bra[i] @ fv @ self.ket[i]

        return right

    @property
    def n_sites(self):
        return self.ket.n_sites

    @property
    def center(self):
        return self.ket.center
