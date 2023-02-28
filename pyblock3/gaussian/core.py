
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

    @property
    def params_length(self):
        return len(pack(self.tn.get_params()))

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

    def __init__(self, tensors, grad_idxs=None):
        self.tensors = tensors
        self.grad_idxs = grad_idxs

    @property
    def grad_tensors(self):
        return [ts for its, ts in enumerate(self.tensors)
            if self.grad_idxs is None or its in self.grad_idxs]

    def get_params(self):
        return [ts.data for ts in self.grad_tensors]

    def set_params(self, params):
        for p, ts in zip(params, self.grad_tensors):
            ts.data = p
        return self

    def unitize_(self):
        for ts in self.tensors:
            if ts.data.ndim == 2:
                ts.data = 0.5 * (ts.data - ts.data.T)
            else:
                ts.data = 0.5 * (ts.data - ts.data.permute(0, 2, 1))
        return self

    def no_grad_(self):
        for ts in self.tensors:
            ts.data = ts.data.detach()
        return self

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

    def get_core_terminal_indices(self):

        all_u_idx = set()

        for ts in self.tensors:
            for ui in ts.u_idx[:ts.n_core]:
                all_u_idx.add(ui)

        return sorted(list(all_u_idx))

    def get_non_core_terminal_indices(self):

        idx = set(self.get_terminal_indices()) - set(self.get_core_terminal_indices())
        return sorted(list(idx))

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
                                if not isinstance(self.tensors[its].core_occ, tuple):
                                    occ = self.tensors[its].core_occ[iuc]
                                    pl[ic] = str(int(np.round(occ)))
                                else:
                                    occa = self.tensors[its].core_occ[0][iuc]
                                    occb = self.tensors[its].core_occ[1][iuc]
                                    pl[ic] = "0ba2"[
                                        int(np.round(occa)) * 2 + int(np.round(occb))]

                pp.append(pl)

        return '\n'.join([''.join(x) for x in pp[::-1]])

    def get_occupations(self):

        term_occ = []

        layers = self.get_layers()

        for l in layers:
            for its in l:
                for occ in self.tensors[its].core_occ:
                    term_occ.append(occ)

        return np.array(term_occ)

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

    def repr_layers(self):

        minx = min([g[1] for t in self.tensors for g in t.u_idx])
        maxx = max([g[1] for t in self.tensors for g in t.u_idx])
        miny = min([g[2] for t in self.tensors for g in t.u_idx])
        maxy = max([g[2] for t in self.tensors for g in t.u_idx])

        layers = self.get_layers()
        r = "NLayers = %d\n" % len(layers)

        for il, l in enumerate(layers):
            r += "Layer %d NTensors = %d\n" % (il, len(l))
            mp = [[""] * (maxy - miny + 1) for _ in range(minx, maxx + 1)]
            for its in l:
                for ix, xu in enumerate(self.tensors[its].u_idx):
                    mp[xu[1] - minx][xu[2] - miny] = str(its)
                    if ix < self.tensors[its].n_core:
                        if self.tensors[its].core_occ is None:
                            mp[xu[1] - minx][xu[2] - miny] = "?" + mp[xu[1] - minx][xu[2] - miny]
                        else:
                            occa = self.tensors[its].core_occ[0][ix]
                            occb = self.tensors[its].core_occ[1][ix]
                            mp[xu[1] - minx][xu[2] - miny] = "0ba2"[
                                int(np.round(occa)) * 2 + int(np.round(occb))] + mp[xu[1] - minx][xu[2] - miny]
            maxl_mp = max([max(len(x) for x in mx) for mx in mp]) + 1
            mp = "\n".join(["".join(["%%%ds" % maxl_mp % x for x in mx]) for mx in mp])
            r += mp + "\n"

        return r

    def truncate_layers(self, n_layer):

        layers = self.get_layers()[:n_layer]
        layers = [xx for x in layers for xx in x]

        return GaussianTensorNetwork(tensors=[ts for its, ts in enumerate(self.tensors)
            if its in layers]).view_as(self.__class__)

    def view_as(self, cls):
        return cls(self.tensors)

    def rhf(self):
        return self.view_as(RHFTensorNetwork)

    def uhf(self):
        return self.view_as(UHFTensorNetwork)

    def ghf(self):
        return self.view_as(GHFTensorNetwork)


class RHFTensorNetwork(GaussianTensorNetwork):
    _high_occ = 2.0

    def __init__(self, tensors):
        super().__init__(tensors)

    def fit_rdm1(self, dm, dm_idxs=None, fit_ent=True):

        if dm_idxs is None:
            dm_idxs = self.get_initial_indices()

        dm_idxs_map = {x: ix for ix, x in enumerate(dm_idxs)}
        dm = torch.clone(dm)

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

        if fit_ent == True:
            for l in layers:
                for its in l:
                    # find the small 1pdm for indices at this tensor
                    eff_dm_idxs = np.array([dm_idxs_map[x]
                                            for x in self.tensors[its].d_idx])
                    eff_dm = dm[eff_dm_idxs, :][:, eff_dm_idxs]
                    xb, ub = torch.linalg.eigh(eff_dm)
                    # sort eigvals to put 0 and 2 near the beginning
                    xg = torch.tensor([list(xb), list(self._high_occ - xb)])
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
        else:
            for l in layers:
                for its in l:
                    ncore = self.tensors[its].n_core
                    if ncore > 0:
                        # find the small 1pdm for indices at this tensor
                        eff_dm_idxs = np.array([dm_idxs_map[x]
                                                for x in self.tensors[its].d_idx])
                        eff_dm = dm[eff_dm_idxs, :][:, eff_dm_idxs]
                        xb, ub = torch.linalg.eigh(eff_dm)
                        # sort eigvals to put 0 and 2 near the beginning
                        xg = torch.tensor([list(xb), list(self._high_occ - xb)])
                        xg[xg <= 1E-50] = 1E-50
                        p = torch.argsort(-(xg[0] * torch.log(xg[0]) +
                                            xg[1] * torch.log(xg[1])))
                        xb = xb[p]
                        ub = ub[:, p]
                        if torch.linalg.det(ub) < 0:
                            ub[:, 0] *= -1
                        self.tensors[its].data = my_logm(ub)
                        self.tensors[its].core_occ = np.round(
                            xb[:ncore])
                        dm[eff_dm_idxs, :] = ub.T @ dm[eff_dm_idxs, :]
                        dm[:, eff_dm_idxs] = dm[:, eff_dm_idxs] @ ub
                    else:
                        dim = len(self.tensors[its].u_idx)
                        self.tensors[its].data = torch.zeros((dim, dim), dtype=float)
                        self.tensors[its].core_occ = torch.tensor([], dtype=float)
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
        return dm[reidx, :][:, reidx]

    def make_rdm2(self, dm1=None):
        if dm1 is None:
            dm1 = self.make_rdm1()
        dm2 = (torch.einsum('ij,kl->ijkl', dm1, dm1)
               - torch.einsum('ij,kl->iklj', dm1, dm1) / 2)
        return dm2

    def energy_tot(self, h1e, g2e=None, dm1=None, dm2=None):
        if dm1 is None:
            dm1 = self.make_rdm1()
        e_tot = torch.einsum('ij,ij->', dm1, h1e)
        if g2e is not None:
            if dm2 is not None:
                e_tot += torch.einsum('ijkl,ijkl->', dm2, g2e) * 0.5
            else:
                e_tot += torch.einsum('ij,kl,ijkl->', dm1, dm1, g2e) * 0.5
                e_tot -= torch.einsum('ij,kl,iklj->', dm1, dm1, g2e) * 0.25
        return e_tot

    def set_occ_half_filling(self):

        layers = self.get_layers()

        ix = 0
        for l in layers:
            for its in l:
                occs = self.tensors[its].core_occ
                for i in range(len(occs)):
                    occs[i] = self._high_occ if ix % 2 else 0.0
                    ix += 1

        return self


class UHFTensorNetwork(GaussianTensorNetwork):

    def __init__(self, tensors):
        super().__init__(tensors)

    def fit_rdm1(self, dm, dm_idxs=None):

        if dm_idxs is None:
            dm_idxs = self.get_initial_indices()

        dm_idxs_map = {x: ix for ix, x in enumerate(dm_idxs)}

        if not hasattr(dm, "shape"):
            dm = torch.tensor(dm)
        if len(dm.shape) == 2:
            dm = torch.tensor(
                np.array([dm.detach().numpy(), dm.detach().numpy()])) / 2
        dm = torch.clone(dm)

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
                eff_dm = dm[:, eff_dm_idxs, :][:, :, eff_dm_idxs]
                xa, ua = torch.linalg.eigh(eff_dm[0])
                xb, ub = torch.linalg.eigh(eff_dm[1])
                # sort eigvals to put 0 and 1 near the beginning
                xg = torch.tensor(
                    [list(xa), list(1 - xa), list(xb), list(1 - xb)])
                xg[xg <= 1E-50] = 1E-50
                pa = torch.argsort(-(xg[0] * torch.log(xg[0]) +
                                     xg[1] * torch.log(xg[1])))
                pb = torch.argsort(-(xg[2] * torch.log(xg[2]) +
                                     xg[3] * torch.log(xg[3])))
                xa, xb = xa[pa], xb[pb]
                ua, ub = ua[:, pa], ub[:, pb]
                if torch.linalg.det(ua) < 0:
                    ua[:, 0] *= -1
                if torch.linalg.det(ub) < 0:
                    ub[:, 0] *= -1
                self.tensors[its].data = torch.tensor(
                    np.array([my_logm(ua).detach().numpy(), my_logm(ub).detach().numpy()]))
                self.tensors[its].core_occ = (
                    np.round(xa[:self.tensors[its].n_core]),
                    np.round(xb[:self.tensors[its].n_core]),
                )
                dm[0, eff_dm_idxs, :] = ua.T @ dm[0, eff_dm_idxs, :]
                dm[0, :, eff_dm_idxs] = dm[0, :, eff_dm_idxs] @ ua
                dm[1, eff_dm_idxs, :] = ub.T @ dm[1, eff_dm_idxs, :]
                dm[1, :, eff_dm_idxs] = dm[1, :, eff_dm_idxs] @ ub
                # update index tags after apply this gate
                du_map = self.tensors[its].get_du_map()
                dm_idxs_map = {du_map.get(
                    x, x): ix for x, ix in dm_idxs_map.items()}

        return self

    def make_rdm1(self):

        dm_idxs = self.get_terminal_indices()
        dm_idxs_map = {x: ix for ix, x in enumerate(dm_idxs)}

        layers = self.get_layers()

        dm = torch.zeros((2, len(dm_idxs), len(dm_idxs)), dtype=torch.float64)

        for l in layers[::-1]:
            for its in l[::-1]:
                # find the small 1pdm for indices at this tensor
                eff_dm_idxs = np.array([dm_idxs_map[x]
                                        for x in self.tensors[its].u_idx])
                dmc_idxs = np.array([dm_idxs_map[u] for u
                                     in self.tensors[its].u_idx[:self.tensors[its].n_core]])
                dm[0, dmc_idxs, dmc_idxs] = self.tensors[its].core_occ[0]
                dm[1, dmc_idxs, dmc_idxs] = self.tensors[its].core_occ[1]
                ua = torch.matrix_exp(self.tensors[its].data[0])
                ub = torch.matrix_exp(self.tensors[its].data[1])
                dm[0, eff_dm_idxs, :] = ua @ dm[0, eff_dm_idxs, :]
                dm[0, :, eff_dm_idxs] = dm[0, :, eff_dm_idxs] @ ua.T
                dm[1, eff_dm_idxs, :] = ub @ dm[1, eff_dm_idxs, :]
                dm[1, :, eff_dm_idxs] = dm[1, :, eff_dm_idxs] @ ub.T
                # update index tags after apply this gate
                ud_map = self.tensors[its].get_ud_map()
                dm_idxs_map = {ud_map.get(
                    x, x): ix for x, ix in dm_idxs_map.items()}

        init_idxs = self.get_initial_indices()
        reidx = np.array([dm_idxs_map[ix] for ix in init_idxs])
        return dm[:, reidx, :][:, :, reidx]

    def make_rdm2(self, dm1=None):
        if dm1 is None:
            dm1 = self.make_rdm1()
        dm1a, dm1b = dm1[0], dm1[1]
        dm2aa = (torch.einsum('ij,kl->ijkl', dm1a, dm1a)
                 - torch.einsum('ij,kl->iklj', dm1a, dm1a))
        dm2bb = (torch.einsum('ij,kl->ijkl', dm1b, dm1b)
                 - torch.einsum('ij,kl->iklj', dm1b, dm1b))
        dm2ab = torch.einsum('ij,kl->ijkl', dm1a, dm1b)
        return torch.cat([dm2aa[None], dm2ab[None], dm2bb[None]])

    def energy_tot(self, h1e, g2e=None, dm1=None, dm2=None):
        if dm1 is None:
            dm1 = self.make_rdm1()
        if len(h1e.shape) == 2:
            h1e = torch.cat([h1e[None], h1e[None]])
        e_tot = torch.einsum('ij,ij->', dm1[0], h1e[0])
        e_tot += torch.einsum('ij,ij->', dm1[1], h1e[1])
        if g2e is not None:
            if len(g2e.shape) == 4:
                g2e = torch.cat([g2e[None], g2e[None], g2e[None]])
            if dm2 is not None:
                e_tot += torch.einsum('ijkl,ijkl->', dm2[0], g2e[0]) * 0.5
                e_tot += torch.einsum('ijkl,ijkl->', dm2[1], g2e[1]) * 1.0
                e_tot += torch.einsum('ijkl,ijkl->', dm2[2], g2e[2]) * 0.5
            else:
                e_tot += torch.einsum('ij,kl,ijkl->', dm1[0], dm1[0], g2e[0]) * 0.5
                e_tot -= torch.einsum('ij,kl,iklj->', dm1[0], dm1[0], g2e[0]) * 0.5
                e_tot += torch.einsum('ij,kl,ijkl->', dm1[0], dm1[1], g2e[1]) * 1.0
                e_tot += torch.einsum('ij,kl,ijkl->', dm1[1], dm1[1], g2e[2]) * 0.5
                e_tot -= torch.einsum('ij,kl,iklj->', dm1[1], dm1[1], g2e[2]) * 0.5
        return e_tot

    def get_occupations(self):

        term_occ = [], []

        layers = self.get_layers()

        for l in layers:
            for its in l:
                for occ in self.tensors[its].core_occ[0]:
                    term_occ[0].append(float(occ))
                for occ in self.tensors[its].core_occ[1]:
                    term_occ[1].append(float(occ))

        return np.array(term_occ)

    def set_occ_half_filling(self):

        layers = self.get_layers()

        ix = 0
        for l in layers:
            for its in l:
                occa, occb = self.tensors[its].core_occ
                for i in range(len(occa)):
                    occa[i] = 1.0 if ix % 2 == 0 else 0.0
                    occb[i] = 1.0 if ix % 2 == 1 else 0.0
                    ix += 1

        return self

    def set_occupations(self, occs):

        layers = self.get_layers()

        ix = 0
        for l in layers:
            for its in l:
                if self.tensors[its].core_occ is None:
                    self.tensors[its].core_occ = (
                        torch.tensor([0.0] * self.tensors[its].n_core, dtype=torch.float64),
                        torch.tensor([0.0] * self.tensors[its].n_core, dtype=torch.float64)
                    )
                occa, occb = self.tensors[its].core_occ
                for i in range(len(occa)):
                    occa[i] = occs[0][ix]
                    occb[i] = occs[1][ix]
                    ix += 1
        
        assert ix == len(occs[0]) and ix == len(occs[1])
        return self


class GHFTensorNetwork(RHFTensorNetwork):
    _high_occ = 1.0

    def __init__(self, tensors):
        super().__init__(tensors)

    def make_rdm2(self, dm1=None):
        if dm1 is None:
            dm1 = self.make_rdm1()
        dm2 = (torch.einsum('ij,kl->ijkl', dm1, dm1)
               - torch.einsum('ij,kl->iklj', dm1, dm1))
        return dm2


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


class GaussianMERA2D(GaussianTensorNetwork):

    def __init__(self, n_sites, n_tensor_sites, n_core, dis_ent_width=None,
        starting_idxs=None, core_depth=None, add_cap=True,
        n_dis_ent_tensor_sites=None, large_dis_ent=False):
        """
            n_sites : (int, int).
                Number of sites along x, y in the MERA
            n_tensor_sites : (int, int).
                Number of sites along x, y in each MERA tensor
            n_dis_ent_tensor_sites : (int, int) or None.
                Number of sites along x, y in each MERA tensor for dis_ent
            dis_ent_width : (int, int).
                Number of sites in smaller dim in each dis ent tensor
            n_core : (int, int).
                Number of core indices along x, y in the upper bond of MERA tensor
            starting_idxs : (int, int, int).
                Starting layer, x, and y indices
            core_depth : int or None.
                If not None, use mod to determine core position. None is equiv to core_depth == 1
            add_cap : bool.
                If True, make all upper indices core indices in the last tensor
            large_dis_ent : bool.
                If True, with suitable n_dis_ent_tensor_sites, dis_ent will cover all sites along x/y.
        """
        from functools import reduce
        assert any(x >= 1 for x in n_core)
        tensors = []
        stl, stx, sty = starting_idxs if starting_idxs is not None else (0, 0, 0)
        cur_idx = [[(stl, stx + i, sty + j) for j in range(0, n_sites[1])] for i in range(0, n_sites[1])]
        cur_l = stl + 1
        nax, nay = tuple(tsz - c for tsz, c in zip(n_tensor_sites, n_core))
        if dis_ent_width is None:
            dis_ent_width = tuple(x // 2 for x in n_tensor_sites)
        if n_dis_ent_tensor_sites is None:
            n_dis_ent_tensor_sites = n_tensor_sites
        ncx, ncy = n_sites
        nwx, nwy = dis_ent_width
        flatten = lambda l: reduce(lambda x, y: x + y, l, [])
        upper = lambda k, l: [(l, x[1], x[2]) for x in k]
        while True:
            ntx, nty = n_dis_ent_tensor_sites
            nax, nay = nay, nax
            # dis ent along x
            i = ntx // 2 if not large_dis_ent else 0
            while True:
                dx = min(ntx, ncx - i)
                for j in range(0, ncy, nwy):
                    dy = min(nwy, ncy - j)
                    d_idx = flatten([k[j:j + dy] for k in cur_idx[i:i + dx]])
                    u_idx = upper(d_idx, cur_l)
                    if dx == ntx:
                        tensors.append(GaussianTensor(u_idx, d_idx, n_core=0))
                        for k in cur_idx[i:i + dx]:
                            k[j:j + dy] = upper(k[j:j + dy], cur_l)
                if i + dx == ncx:
                    break
                i += dx
            cur_l += 1
            # dis ent along y
            j = nty // 2 if not large_dis_ent else 0
            while True:
                dy = min(nty, ncy - j)
                for i in range(0, ncx, nwx):
                    dx = min(nwx, ncx - i)
                    d_idx = flatten([k[j:j + dy] for k in cur_idx[i:i + dx]])
                    u_idx = upper(d_idx, cur_l)
                    if dy == nty:
                        tensors.append(GaussianTensor(u_idx, d_idx, n_core=0))
                        for k in cur_idx[i:i + dx]:
                            k[j:j + dy] = upper(k[j:j + dy], cur_l)
                if j + dy == ncy:
                    break
                j += dy
            cur_l += 1
            # isometry
            ntx, nty = n_tensor_sites
            new_cur_idx = [[None] * ncy for _ in range(ncx)]
            ni = 0
            nts = 0
            for i in range(0, ncx, ntx):
                nj = 0
                dx = min(ncx - i, ntx)
                for j in range(0, ncy, nty):
                    dy = min(ncy - j, nty)
                    xc, yc = i + dx - nax, j + dy - nay
                    if core_depth is None:
                        d_idx = flatten([[(gi, gj) for gj in range(j, j + dy)] for gi in range(i, i + dx)])
                        d_idx = sorted(d_idx, key=lambda k: ((k[0] >= xc) + (k[1] >= yc), ) + k)
                    else:
                        d_idx = flatten([[(gi, gj) for gj in range(dy)] for gi in range(dx)])
                        gxc, gyc = (dx - nax) // core_depth, (dy - nay) // core_depth
                        d_idx = sorted(d_idx, key=lambda k:
                            ((k[0] % (dx // core_depth) >= gxc) + (k[1] % (dy // core_depth) >= gyc), ) + k)
                        d_idx = [(i + gi, j + gj) for (gi, gj) in d_idx]

                    d_idx = [cur_idx[gi][gj] for (gi, gj) in d_idx]
                    u_idx = upper(d_idx, cur_l)
                    n_core_this = max(0, dx * dy - nax * nay)
                    tensors.append(GaussianTensor(u_idx, d_idx, n_core=n_core_this))
                    nts += 1
                    if core_depth is None:
                        for k, kd in zip(new_cur_idx[ni:ni + nax], cur_idx[xc:i + dx]):
                            k[nj:nj + nay] = upper(kd[yc:j + dy], cur_l)
                    else:
                        gxc, gyc = (dx - nax) // core_depth, (dy - nay) // core_depth
                        igi, igj = 0, 0
                        for gi in range(dx):
                            igj = 0
                            if gi % (dx // core_depth) >= gxc:
                                for gj in range(dy):
                                    if gj % (dy // core_depth) >= gyc:
                                        new_cur_idx[ni + igi][nj + igj] = (cur_l, ) + cur_idx[i + gi][j + gj][1:]
                                        igj += 1
                                igi += 1
                        assert igi * igj == tensors[-1].width - n_core_this
                    nj += nay
                ni += nax
            cur_l += 1
            cur_idx = [[k for k in j if k is not None] for j in new_cur_idx
                if len([k for k in j if k is not None]) != 0]
            ncx, ncy = len(cur_idx), len(cur_idx[0])
            if nts == 1:
                if add_cap:
                    tensors[-1].n_core = tensors[-1].width
                break

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
