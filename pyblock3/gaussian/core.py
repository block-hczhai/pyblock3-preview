
import numpy as np
import torch


def pack(x):
    size = sum(z.nelement() for z in x)
    r = np.zeros(size)
    iz = 0
    for z in x:
        r[iz:iz + z.nelement()] = z.detach().numpy().flatten()
        iz += z.nelement()
    assert iz == r.size
    return r


def unpack(r, x):
    iz = 0
    for ix, z in enumerate(x):
        x[ix] = torch.tensor(
            r[iz:iz + z.nelement()].reshape(z.shape)).requires_grad_(True)
        iz += z.nelement()
    assert iz == r.size
    return x


class GaussianOptimizer:

    def __init__(self, tn, h1e, g2e, jit=True, iprint=1):
        self.tn = tn
        self.niter = 0

        def get_ener(x):
            tn.set_params(x)
            tn.unitize_()
            return tn.energy_tot(h1e, g2e)

        if jit:
            self.fn = torch.jit.trace(
                get_ener, example_inputs=[tn.get_params()])
        else:
            self.fn = get_ener

        def energy_and_grad(x):
            params = unpack(x, tn.get_params())
            energy = get_ener(params)
            grad = torch.autograd.grad(energy, params)
            self.niter += 1
            if iprint >= 1:
                print("%5d = %20.12f" % (self.niter, float(energy.detach())))
            return float(energy.detach()), pack(grad)

        self.grad = energy_and_grad

    def optimize(self, x0=None, maxiter=100):

        self.niter = 0

        if x0 is None:
            x0 = pack(self.tn.get_params())
        elif isinstance(x0, str) and x0 == "random":
            x0 = pack(self.tn.get_params())
            x0[:] = np.random.random(x0.shape)

        from scipy.optimize import minimize

        res = minimize(fun=self.grad, jac=True, hessp=None, x0=x0,
                       tol=None, method='L-BFGS-B', options={'maxiter': maxiter})

        return self.grad(res.x)[0], res.x


class GaussianTensor:
    """
    Attrs:
        u_idx : list(any), up (output) indices
        d_idx : list(any), down (output) indices
        data : square matrix, anti-symm K matrix for orbital rotation
        n_core : int, first n_core up indices have fixed occ (can be core or virtual)
        core_occ : list(float), the fixed occ (0 ~ 1)
    """

    def __init__(self, u_idx, d_idx, n_core=0, data=None):
        self.u_idx = u_idx
        self.d_idx = d_idx
        self.data = data
        self.n_core = n_core
        self.core_occ = None

    def __repr__(self):
        return "[u = %r d = %r nc = %r]" % (self.u_idx, self.d_idx, self.n_core)

    @property
    def width(self):
        return max(len(self.u_idx), len(self.d_idx))

    def get_du_map(self):
        return {i: j for i, j in zip(self.d_idx, self.u_idx)}

    def get_ud_map(self):
        return {i: j for i, j in zip(self.u_idx, self.d_idx)}


class GaussianTensorNetwork:

    def __init__(self, tensors):
        self.tensors = tensors

    def get_params(self):
        return [ts.data for ts in self.tensors]

    def set_params(self, params):
        for p, ts in zip(params, self.tensors):
            ts.data = p

    def unitize_(self):
        for ts in self.tensors:
            ts.data = 0.5 * (ts.data - ts.data.T)

    def get_initial_indices(self):
        """
        Initial indices are down (d_idx) indices that are not connected to any up indices.
        """

        all_d_idx = set()

        for ts in self.tensors:
            for di in ts.d_idx:
                all_d_idx.add(di)

        for ts in self.tensors:
            for ui in ts.u_idx:
                if ui in all_d_idx:
                    all_d_idx.remove(ui)

        return sorted(list(all_d_idx))

    def get_terminal_indices(self):
        """
        Terminal indices are up (d_idx) indices that are not connected to any down indices.
        """

        all_u_idx = set()

        for ts in self.tensors:
            for ui in ts.u_idx:
                all_u_idx.add(ui)

        for ts in self.tensors:
            for di in ts.d_idx:
                if di in all_u_idx:
                    all_u_idx.remove(di)

        return sorted(list(all_u_idx))

    def get_layers(self):
        """
        Return a list of layers, each layer is a list of indices of tensors
        that can be computed simultaneously, as long as all previous layers
        are computed.
        """

        current_idxs = set(self.get_initial_indices())
        layers = []
        current_tensor_idxs = range(0, len(self.tensors))

        while current_tensor_idxs != []:

            current_layer = []
            next_tensor_idxs = []
            next_idxs = current_idxs.copy()

            for its in current_tensor_idxs:
                ts = self.tensors[its]
                if all(ix in current_idxs for ix in ts.d_idx):
                    current_layer.append(its)
                    for ix in ts.d_idx:
                        next_idxs.remove(ix)
                    for ix in ts.u_idx:
                        next_idxs.add(ix)
                else:
                    next_tensor_idxs.append(its)

            current_idxs = next_idxs
            current_tensor_idxs = next_tensor_idxs

            layers.append(current_layer)

        return layers

    def __repr__(self):

        layers = self.get_layers()
        current_idxs = self.get_initial_indices()
        terminal_idxs = set(self.get_terminal_indices())

        pp = ['┆' * len(current_idxs)]

        for l in layers:
            pl = [' '] * len(current_idxs)
            layer_min_ip = min(min(current_idxs.index(xd) for xd in self.tensors[its].d_idx)
                               for its in l)
            layer_max_ip = max(max(current_idxs.index(xd) for xd in self.tensors[its].d_idx)
                               for its in l)
            for its in l:
                nx = self.tensors[its].width
                min_ip = min(current_idxs.index(xd)
                             for xd in self.tensors[its].d_idx)
                max_ip = max(current_idxs.index(xd)
                             for xd in self.tensors[its].d_idx)
                if len(l) == 1 or not (max_ip == layer_max_ip and min_ip == layer_min_ip):
                    pl[min_ip:max_ip] = '━' * (max_ip - min_ip)
                for ix, (xu, xd) in enumerate(zip(self.tensors[its].u_idx, self.tensors[its].d_idx)):
                    ip = current_idxs.index(xd)
                    current_idxs[ip] = xu
                    pl[ip] = '┃' if nx == 1 else (
                        '┣' if ix == 0 else ('┫' if ix == nx - 1 else '┿'))

            pp.append(pl)
            if l != layers[-1] or self.tensors[its].core_occ is not None:
                pl = [' '] * len(current_idxs)
                uset = set()
                for its in l:
                    uset |= set(self.tensors[its].u_idx)
                for ic, c in enumerate(current_idxs):
                    if c not in terminal_idxs:
                        if c in uset:
                            pl[ic] = '┆' or '|'
                        else:
                            pl[ic] = pp[-1][ic] = '┆'
                    elif self.tensors[its].core_occ is not None:
                        for its in l:
                            if c in self.tensors[its].u_idx:
                                iuc = self.tensors[its].u_idx.index(c)
                                occ = self.tensors[its].core_occ[iuc]
                                pl[ic] = str(int(np.round(occ)))
                pp.append(pl)

        return '\n'.join([''.join(x) for x in pp[::-1]])

    def fit_rdm1(self, dm, dm_idxs=None):

        if dm_idxs is None:
            dm_idxs = self.get_initial_indices()

        dm_idxs_map = {x: ix for ix, x in enumerate(dm_idxs)}
        dm = torch.clone(dm) / 2

        layers = self.get_layers()

        def my_logm(mrot):
            rs = mrot + mrot.T
            _, rv = torch.linalg.eigh(rs)
            rd = rv.T @ mrot @ rv
            ld = torch.zeros_like(rd)
            for i in range(0, len(rd) // 2 * 2, 2):
                xcos = (rd[i, i] + rd[i + 1, i + 1]) / 2
                xsin = (rd[i, i + 1] - rd[i + 1, i]) / 2
                theta = torch.arctan2(xsin, xcos)
                ld[i, i + 1] = theta
                ld[i + 1, i] = -theta
            return rv @ ld @ rv.T

        for l in layers:
            for its in l:
                # find the small 1pdm for indices at this tensor
                eff_dm_idxs = np.array([dm_idxs_map[x]
                                        for x in self.tensors[its].d_idx])
                eff_dm = dm[eff_dm_idxs, :][:, eff_dm_idxs]
                xb, ub = torch.linalg.eigh(eff_dm)
                # sort eigvals to put 0 and 1 near the beginning
                xg = torch.tensor([list(xb), list(1 - xb)])
                xg[xg <= 1E-50] = 1E-50
                p = torch.argsort(-(xg[0] * torch.log(xg[0]) +
                                    xg[1] * torch.log(xg[1])))
                xb = xb[p]
                ub = ub[:, p]
                if torch.linalg.det(ub) < 0:
                    ub[:, 0] *= -1
                self.tensors[its].data = my_logm(ub)
                self.tensors[its].core_occ = np.round(
                    xb[:self.tensors[its].n_core])
                dm[eff_dm_idxs, :] = ub.T @ dm[eff_dm_idxs, :]
                dm[:, eff_dm_idxs] = dm[:, eff_dm_idxs] @ ub
                # update index tags after apply this gate
                du_map = self.tensors[its].get_du_map()
                dm_idxs_map = {du_map.get(
                    x, x): ix for x, ix in dm_idxs_map.items()}

        return self

    def make_rdm1(self):

        dm_idxs = self.get_terminal_indices()
        dm_idxs_map = {x: ix for ix, x in enumerate(dm_idxs)}

        layers = self.get_layers()

        dm = torch.zeros((len(dm_idxs), len(dm_idxs)), dtype=torch.float64)

        for l in layers[::-1]:
            for its in l[::-1]:
                # find the small 1pdm for indices at this tensor
                eff_dm_idxs = np.array([dm_idxs_map[x]
                                        for x in self.tensors[its].u_idx])
                dmc_idxs = np.array([dm_idxs_map[u] for u
                                     in self.tensors[its].u_idx[:self.tensors[its].n_core]])
                dm[dmc_idxs, dmc_idxs] = self.tensors[its].core_occ
                ub = torch.matrix_exp(self.tensors[its].data)
                dm[eff_dm_idxs, :] = ub @ dm[eff_dm_idxs, :]
                dm[:, eff_dm_idxs] = dm[:, eff_dm_idxs] @ ub.T
                # update index tags after apply this gate
                ud_map = self.tensors[its].get_ud_map()
                dm_idxs_map = {ud_map.get(
                    x, x): ix for x, ix in dm_idxs_map.items()}

        init_idxs = self.get_initial_indices()
        reidx = np.array([dm_idxs_map[ix] for ix in init_idxs])
        return dm[reidx, :][:, reidx] * 2

    def make_rdm2(self, dm1=None):
        if dm1 is None:
            dm1 = self.make_rdm1()
        dm2 = (torch.einsum('ij,kl->ijkl', dm1, dm1)
               - torch.einsum('ij,kl->iklj', dm1, dm1) / 2)
        return dm2

    def energy_tot(self, h1e, g2e, dm1=None, dm2=None):
        if dm1 is None:
            dm1 = self.make_rdm1()
        if dm2 is None:
            dm2 = self.make_rdm2(dm1=dm1)
        e_tot = torch.einsum('ij,ij->', dm1, h1e)
        e_tot += torch.einsum('ijkl,ijkl->', dm2, g2e) * 0.5
        return e_tot

    def get_occupations(self):

        term_occ = []

        layers = self.get_layers()

        for l in layers:
            for its in l:
                for occ in self.tensors[its].core_occ:
                    term_occ.append(occ)

        return np.array(term_occ) * 2

    def set_occ_smearing(self, sigma=0.0):

        layers = self.get_layers()

        for l in layers:
            for its in l:
                occs = self.tensors[its].core_occ
                if len(occs) > 0:
                    occs = torch.round(occs)
                    occs[occs > 0.5] -= sigma
                    occs[occs < 0.5] += sigma
                    self.tensors[its].core_occ = occs

    def set_occ_half_filling(self):

        layers = self.get_layers()

        ix = 0
        for l in layers:
            for its in l:
                occs = self.tensors[its].core_occ
                for i in range(len(occs)):
                    occs[i] = 1.0 if ix % 2 else 0.0
                    ix += 1
        
        return self


class GaussianMERA1D(GaussianTensorNetwork):

    def __init__(self, n_sites, n_tensor_sites=5, n_core=2, dis_ent=True, periodic=True):
        """
        Args:
            n_sites : number of sites in the MERA
            n_tensor_sites : number of sites in each MERA tensor
            n_core : number of core indices in the upper bond of MERA tensor
            dis_ent : if true, add disentangler layers
            periodic : if true, add extra non-local disentangler tensor for boundary sites
        """
        assert n_core >= 1
        tensors = []
        cur_idx = list(range(0, n_sites))
        idx = n_sites
        n_act = n_tensor_sites - n_core
        while True:
            if dis_ent:
                if not periodic:
                    next_idx = cur_idx[:n_tensor_sites // 2]
                    i = n_tensor_sites // 2
                    while True:
                        d = min(n_tensor_sites, len(cur_idx) - i)
                        d_idx = cur_idx[i:i + d]
                        u_idx = list(range(idx, idx + d))
                        if d == n_tensor_sites:
                            idx += d
                            tensors.append(GaussianTensor(
                                u_idx, d_idx, n_core=0))
                            next_idx += u_idx
                        else:
                            next_idx += d_idx
                        if i + d == len(cur_idx):
                            break
                        i += d
                else:
                    next_idx = []
                    i = n_tensor_sites // 2
                    while True:
                        d = min(n_tensor_sites, len(cur_idx) -
                                i + n_tensor_sites // 2)
                        d_idx = cur_idx[i:i + d]
                        u_idx = list(range(idx, idx + d))
                        if i + d > len(cur_idx):
                            if d == n_tensor_sites and d != len(cur_idx):
                                idx += d
                                tensors.append(GaussianTensor(
                                    u_idx[:len(d_idx)] + u_idx[len(d_idx):],
                                    d_idx + cur_idx[:i + d - len(cur_idx)],
                                    n_core=0))
                                next_idx = u_idx[len(
                                    d_idx):] + next_idx + u_idx[:len(d_idx)]
                            else:
                                next_idx = cur_idx[:i + d -
                                                   len(cur_idx)] + next_idx + d_idx
                        else:
                            if d == n_tensor_sites and d != len(cur_idx):
                                idx += d
                                tensors.append(GaussianTensor(
                                    u_idx, d_idx, n_core=0))
                                next_idx += u_idx
                            else:
                                next_idx += d_idx
                        if i + d == len(cur_idx) + n_tensor_sites // 2:
                            break
                        i += d
                cur_idx = next_idx
            next_idx = []
            i = 0
            while True:
                d = min(n_tensor_sites, len(cur_idx) - i)
                d_idx = cur_idx[i:i + d]
                u_idx = list(range(idx, idx + d))
                idx += d
                tensors.append(GaussianTensor(u_idx, d_idx, n_core=n_core))
                tensors[-1].n_core = max(0, tensors[-1].width - n_act)
                next_idx += u_idx[tensors[-1].n_core:]
                if i + d == len(cur_idx):
                    break
                i += d
            if tensors[-1].width == len(cur_idx):
                tensors[-1].n_core = tensors[-1].width
                break
            cur_idx = next_idx

        super().__init__(tensors)


class GaussianMPS(GaussianTensorNetwork):

    def __init__(self, n_sites, n_tensor_sites=5, n_core=2):
        """
        Args:
            n_sites : number of sites in the MPS
            n_tensor_sites : number of sites in each MPS tensor
            n_core : number of core indices in the upper bond of MPS tensor
        """
        assert n_core >= 1
        tensors = []
        cur_idx = list(range(0, n_sites))
        idx = n_sites
        i = 0
        while True:
            d = min(n_tensor_sites, n_sites - i)
            d_idx = cur_idx[i:i + d]
            u_idx = list(range(idx, idx + d))
            idx += d
            cur_idx[i:i + d] = u_idx
            tensors.append(GaussianTensor(u_idx, d_idx, n_core=n_core))
            if i + d == n_sites:
                tensors[-1].n_core = tensors[-1].width
                break
            i += n_core

        super().__init__(tensors)
