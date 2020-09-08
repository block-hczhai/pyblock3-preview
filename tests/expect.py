
import sys
sys.path[:0] = ['..', "../../block2-old/build"]

from functools import reduce
import numpy as np
import shutil
import time

from block2 import SZ, Global
from block2 import init_memory, release_memory, set_mkl_num_threads, DataFrame
from block2 import VectorUInt8, VectorUInt16, VectorDouble, PointGroup
from block2 import Random, FCIDUMP, QCTypes, SeqTypes, NoiseTypes
from block2.sz import HamiltonianQC, MPS, MPSInfo, VectorStateInfo, StateInfo
from block2.sz import PDM1MPOQC, SimplifiedMPO, Rule, RuleQC, MPOQC, IdentityMPO
from block2.sz import Expect, MovingEnvironment, NoTransposeRule
from pyblock3.moving_environment import MovingEnvironment as PYME
from pyblock3.aux.hamil import MPSTools, MPOTools, HamilTools
from pyblock3.aux.io import SymbolicMPOTools
import pyblock3.algebra.funcs as pbalg
import os

scratch = './my_tmp'
empty_scratch = True
rand_seed = 1234
memory = int(4E10)
n_threads = 1
pg = 'd2h'
bond_dim = 1000
dot = 1
mode = ["BLOCK2", "PY"]
operator = "H"
flat = True
prop = False

Random.rand_seed(rand_seed)
set_mkl_num_threads(n_threads)

swap_pg = getattr(PointGroup, "swap_" + pg)
vacuum = SZ(0, 0, 0)
n_sites = 16
orb_sym = VectorUInt8([0] * n_sites)
target = SZ(n_sites, 0, 0)
hamil = HamilTools.hubbard(n_sites, memory=memory).__enter__().hamil

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
    pympo, _ = pympo.compress(left=True, cutoff=4E-10, norm_cutoff=4E-10)
    print(pympo.show_bond_dims())
    if flat:
        me = PYME(pymps.to_flat(), pympo.to_flat(), pymps.to_flat(), do_canon=False)
    else:
        me = PYME(pymps, pympo, pymps, do_canon=False)
    t0 = time.perf_counter()
    for i in range(0, n_sites - 1 if prop else 1):
        t = time.perf_counter()
        eff = me[i:i + dot]
        # print("PY time ctr/rot = ", me.t_ctr, me.t_rot)
        # print("PY Init elapsed = ", time.perf_counter() - t0)
        eff.ket[:] = [reduce(pbalg.hdot, eff.ket[:])]
        eff.bra[:] = [reduce(pbalg.hdot, eff.bra[:])]
        # ener, eff, ndav = eff.eigh(iprint=True)
        # print(ener)
        print(eff.expectation, "T = %5.3f" % (time.perf_counter() - t))
    print("PY Time elapsed = ", time.perf_counter() - t0)

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
        print("SP Init elapsed = ", time.perf_counter() - t0)
        # eff.ket[:] = [reduce(pbalg.hdot, eff.ket[:])]
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
    norm = ex.solve(prop)
    print(norm)
    print("BL Time elapsed = ", time.perf_counter() - t0)
    # print("BL time read/write = ", Global.frame.tread, Global.frame.twrite)

# ---------- clean up ------------

mpo.deallocate()
mps_info.deallocate()
hamil.deallocate()

release_memory()
if empty_scratch:
    shutil.rmtree(scratch)
