
import numpy as np
from pyscf import M
import torch
import sys

# parent mera is 07-16x16
# for example: LTS, ... = 4, 4, 2, 1, 0

LTS = int(sys.argv[1]) # isometry size
LAY = int(sys.argv[2]) # n layers
SHF = int(sys.argv[3]) # shift
LOC = bool(int(sys.argv[4])) # is local update
REC = bool(int(sys.argv[5])) # is square

print('PARAMS = LTS = %s, LAY = %s, SHF = %s, LOC = %s, REC = %s' % (LTS, LAY, SHF, LOC, REC))

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

from pyblock3.gaussian import GaussianOptimizer, GaussianTensorNetwork
from pyblock3.gaussian import GaussianLinearGates2D, GaussianRectGates2D

layers = gmera.get_layers()

max_l = lambda gtn: max([max(u[0], d[0]) for ts in gtn.tensors for u, d in zip(ts.u_idx, ts.d_idx)])

il = max_l(gmera) + 1
orig_tn = gmera

print(orig_tn.repr_layers())

for iitt, its in enumerate(layers[2]):

    print("---- TENSOR ---- %d / %d" % (iitt, len(layers[2])))

    ref_tn = GaussianTensorNetwork(tensors=[orig_tn.tensors[its]]).uhf()
    if REC:
        fit_tn = GaussianRectGates2D(n_sites=(LX // 2, LY // 2), n_tensor_sites=(LTS, LTS), n_layers=LAY,
            n_shift_sites=(SHF, SHF), starting_idxs=(il, 0, 0)).uhf()
    else:
        fit_tn = GaussianLinearGates2D(n_sites=(LX // 2, LY // 2), n_tensor_sites=(LTS, LTS), n_layers=LAY,
            n_shift_sites=(SHF, SHF), starting_idxs=(il, 0, 0)).uhf()

    fit_tn.fit_to_reference_2d(ref_tn)
    fit_tn.fit_rdm1(ref_tn.make_rdm1())
    fit_tn.set_occupations(ref_tn.get_occupations())

    print(ref_tn.repr_layers())
    print(fit_tn.repr_layers())

    il = max(il, max_l(fit_tn)) + 1

    new_tn = GaussianTensorNetwork(tensors=
        [ts for ts in gmera.tensors if ts != orig_tn.tensors[its]] + fit_tn.tensors).uhf()
    if LOC:
        new_tn.restrict_grad_idxs(sub_tn=fit_tn)
    else:
        new_tn.restrict_grad_idxs(sub_tn=None)

    print([np.trace(x) for x in orig_tn.make_rdm1()])
    print([np.trace(x) for x in new_tn.make_rdm1()])

    opt = GaussianOptimizer(new_tn, h1e, g2e, iprint=1)
    print('param length = ', opt.params_length)
    ener, x = opt.optimize(x0='random', maxiter=2000)
    print('final ener = ', ener, 'niter = ', opt.niter)

    # gmera = new_tn
    break
