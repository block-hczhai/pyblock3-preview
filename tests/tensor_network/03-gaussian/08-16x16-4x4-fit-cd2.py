
import numpy as np
from pyscf import M
import torch

torch.set_num_threads(28)

LX, LY = 16, 16
NE = LX * LY
L = LX * LY
U = 0

import pickle

ener, x, h1e, gmera = pickle.load(open("02-opt-x.bin", "rb"))
gmera = gmera.no_grad_()

g2e = None
print('init  ener = ', float(gmera.energy_tot(h1e, g2e)))

from pyblock3.gaussian import GaussianMERA2D, GaussianOptimizer, GaussianTensorNetwork

layers = gmera.get_layers()

max_l = lambda gtn: max([max(u[0], d[0]) for ts in gtn.tensors for u, d in zip(ts.u_idx, ts.d_idx)])

il = max_l(gmera) + 1
orig_tn = gmera

print(orig_tn.repr_layers())

for its in layers[2]:

    ref_tn = GaussianTensorNetwork(tensors=[orig_tn.tensors[its]]).uhf()
    ref_init_idxs = sorted(ref_tn.get_initial_indices(), key=lambda p: p[1:] + p[:1])
    ref_cterm_idxs = sorted(ref_tn.get_core_terminal_indices(), key=lambda p: p[1:] + p[:1])
    ref_nterm_idxs = sorted(ref_tn.get_non_core_terminal_indices(), key=lambda p: p[1:] + p[:1])

    fit_tn = GaussianMERA2D(n_sites=(LX // 2, LY // 2), n_tensor_sites=(4, 4), n_core=(2, 0),
        dis_ent_width=(1, 1), starting_idxs=(il, 0, 0), core_depth=2)
    # fit_tn = GaussianMERA2D(n_sites=(LX // 2, LY // 2), n_tensor_sites=(4, 4), n_core=(2, 0),
    #     dis_ent_width=(1, 1), starting_idxs=(il, 0, 0), core_depth=2, n_dis_ent_tensor_sites=(8, 8), large_dis_ent=True)
    # fit_tn = GaussianMERA2D(n_sites=(LX // 2, LY // 2), n_tensor_sites=(8, 8), n_core=(4, 0),
    #     dis_ent_width=(1, 1), starting_idxs=(il, 0, 0), core_depth=4, add_cap=False)
    fit_tn = fit_tn.uhf()
    fit_tn = fit_tn.truncate_layers(3)
    fit_tn.fit_rdm1(torch.tensor(np.identity((LX // 2) * (LY // 2))))
    fit_init_idxs = sorted(fit_tn.get_initial_indices(), key=lambda p: p[1:] + p[:1])
    fit_cterm_idxs = sorted(fit_tn.get_core_terminal_indices(), key=lambda p: p[1:] + p[:1])
    fit_nterm_idxs = sorted(fit_tn.get_non_core_terminal_indices(), key=lambda p: p[1:] + p[:1])

    fit_tn.set_occupations(ref_tn.get_occupations())

    print(ref_tn.repr_layers())
    print(fit_tn.repr_layers())

    il = max(il, max_l(fit_tn)) + 1

    idx_mp = {}
    for ri, fi in zip(ref_init_idxs, fit_init_idxs):
        assert fi[1:] == ri[1:]
        idx_mp[fi] = ri
    for ri, fi in zip(ref_cterm_idxs, fit_cterm_idxs):
        assert fi[1:] == ri[1:]
        idx_mp[fi] = ri
    for ri, fi in zip(ref_nterm_idxs, fit_nterm_idxs):
        assert fi[1:] == ri[1:]
        idx_mp[fi] = ri

    print(len(idx_mp), idx_mp)

    for ts in fit_tn.tensors:
        ts.u_idx = [idx_mp.get(x, x) for x in ts.u_idx]
        ts.d_idx = [idx_mp.get(x, x) for x in ts.d_idx]

    print(fit_tn.repr_layers())

    new_tn = GaussianTensorNetwork(tensors=
        [ts for ts in gmera.tensors if ts != orig_tn.tensors[its]] + fit_tn.tensors).uhf()
    new_tn.grad_idxs = list(range(len(new_tn.tensors) - len(fit_tn.tensors), len(new_tn.tensors)))

    print(new_tn.repr_layers())

    opt = GaussianOptimizer(new_tn, h1e, g2e, iprint=1)
    print('param length = ', opt.params_length)
    ener, x = opt.optimize(x0='random', maxiter=1000)
    print('final ener = ', ener, 'niter = ', opt.niter)

    quit()
