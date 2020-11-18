
import sys
sys.path[:0] = ['..', "../../block2-old/build"]

from functools import reduce
import numpy as np
import shutil
import time
from pyinstrument import Profiler
profiler = Profiler()

from block2 import SZ, Global
from block2 import init_memory, release_memory, set_mkl_num_threads, DataFrame
from block2 import VectorUInt8, VectorUInt16, VectorDouble, PointGroup
from block2 import Random, FCIDUMP, QCTypes, SeqTypes, NoiseTypes
from block2.sz import HamiltonianQC, MPS, MPSInfo, VectorStateInfo, StateInfo
from block2.sz import PDM1MPOQC, SimplifiedMPO, Rule, RuleQC, MPOQC, IdentityMPO
from block2.sz import Expect, MovingEnvironment, NoTransposeRule
from pyblock3.algebra.mpe import MPE as PYME
from pyblock3.aux.hamil import MPSTools, MPOTools, HamilTools
from pyblock3.aux.io import SymbolicMPOTools
import pyblock3.algebra.funcs as pbalg
import os
import block3

scratch = './my_tmp'
empty_scratch = True
rand_seed = 1234
memory = int(4E10)
n_threads = 1
pg = 'd2h'
bond_dim = 25600
dot = 1
mode = ["PY"]
operator = "H"
system = "Hubbard"
# system = "N2"
flat = True
prop = False
do_canon = False
profile = False

Random.rand_seed(rand_seed)
set_mkl_num_threads(n_threads)

if system == "N2":
    fd = '../data/N2.STO3G.FCIDUMP'
    hamil = HamilTools.from_fcidump(fd, memory=memory).__enter__().hamil
    n_sites = hamil.n_sites
    target = SZ(hamil.fcidump.n_elec, hamil.fcidump.twos, 0)
    cutoff = 4E-10
else:
    n_sites = 16
    hamil = HamilTools.hubbard(n_sites, memory=memory).__enter__().hamil
    target = SZ(n_sites, 0, 0)
    cutoff = 4E-10
vacuum = SZ(0, 0, 0)

mps_info = MPSInfo(n_sites, vacuum, target, hamil.basis)
mps_info.set_bond_dimension(bond_dim)
mps = MPS(n_sites, 0, 1)
mps.initialize(mps_info)
mps.random_canonicalize()
mps.dot = dot

print('left bond dims = ', [x.n_states_total for x in mps_info.left_dims])
print('right bond dims = ', [x.n_states_total for x in mps_info.right_dims])

print('BL MPS MEM = ', [x.total_memory for x in mps.tensors])
mps.save_mutable()
mps.deallocate()
mps_info.save_mutable()
mps_info.deallocate_mutable()

if operator == "I":
    orig_mpo = IdentityMPO(hamil)
    mpo = SimplifiedMPO(orig_mpo, NoTransposeRule(RuleQC()), True)
else:
    orig_mpo = MPOQC(hamil, QCTypes.NC)
    mpo = SimplifiedMPO(orig_mpo, RuleQC(), True)

if "PY" in mode:
    mps.dot = 1
    pymps = MPSTools.from_block2(mps)
    print('PY MPS MEM = ', [sum([y.size for y in x.blocks]) for x in pymps.tensors])
    pympo = MPOTools.from_block2(orig_mpo)
    pympo, _ = pympo.compress(left=True, cutoff=cutoff, norm_cutoff=cutoff)
    print(pympo.show_bond_dims())
    if flat:
        fpymps = pymps.to_flat()
        me = PYME(fpymps, pympo.to_flat(),fpymps, do_canon=do_canon)
    else:
        me = PYME(pymps, pympo, pymps, do_canon=do_canon)

    if profile:
        profiler.start()
    t0 = time.perf_counter()
    for i in range(0, n_sites - 1 if prop else 1):
        t = time.perf_counter()
        eff = me[i:i + dot]
        # print("PY time ctr/rot = ", me.t_ctr, me.t_rot)
        # print("PY Init elapsed = ", time.perf_counter() - t0)
        # if dot == 2:
        #     eff.ket[:] = [reduce(pbalg.hdot, eff.ket[:])]
        #     eff.bra[:] = [reduce(pbalg.hdot, eff.bra[:])]
        # # ener, eff, ndav = eff.eigs(iprint=True)
        # if dot == 2:
        #     l, s, r = eff.ket[0].tensor_svd(
        #         idx=3, pattern='++++--', full_matrices=False)
        #     eff.ket[:] = [np.tensordot(l, s.diag(), axes=1), r]
        # me[i:i + dot] = eff
        # print(ener)
        print(eff.expectation, "T = %5.3f" % (time.perf_counter() - t))
    print("PY Time elapsed = ", time.perf_counter() - t0)
    print(block3.time_cost())
    if profile:
        profiler.stop()

if "SYMPY" in mode:
    mps.dot = 1
    pymps = MPSTools.from_block2(mps)
    print('PY MPS MEM = ', [sum([y.size for y in x.blocks]) for x in pymps.tensors])
    pympo = SymbolicMPOTools.from_block2(orig_mpo)
    pympo = pympo.simplify()
    print(pympo.show_bond_dims())
    me = PYME(pymps, pympo, pymps, do_canon=False)
    t0 = time.perf_counter()
    for i in range(0, n_sites - 1 if prop else 1):
        t = time.perf_counter()
        eff = me[i:i + dot]
        if dot == 2:
            eff.ket[:] = [reduce(pbalg.hdot, eff.ket[:])]
        print(eff.expectation, "T = %5.3f" % (time.perf_counter() - t))
    print("SP Time elapsed = ", time.perf_counter() - t0)

if "BLOCK2" in mode:
    t0 = time.perf_counter()
    mps.dot = dot
    me = MovingEnvironment(mpo, mps, mps, "EX")
    me.init_environments(False)
    # print("BL time ctr/rot = ", me.tctr, me.trot)
    # print("BL Init elapsed = ", time.perf_counter() - t0)
    ex = Expect(me, bond_dim * 2, bond_dim * 2)
    ex.cutoff = 1E-10
    ener = ex.solve(prop)
    if mpo.const_e != 0.0:
        mps.load_tensor(0)
        ener = ener + mps.tensors[0].norm() ** 2 * mpo.const_e
        mps.unload_tensor(0)
    print(ener)
    print("BL Time elapsed = ", time.perf_counter() - t0)
    # print("BL time read/write = ", Global.frame.tread, Global.frame.twrite)

# ---------- clean up ------------

mpo.deallocate()
mps_info.deallocate()
hamil.deallocate()

release_memory()
if empty_scratch:
    shutil.rmtree(scratch)

if profile:
    print(profiler.output_text(unicode=True, color=True))
