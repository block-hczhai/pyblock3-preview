
from pyscf import gto, scf, ci, mcscf, ao2mo, symm
from pyscf.data import nist
import copy
import numpy as np

import ctypes
mkl_rt = ctypes.CDLL('libmkl_rt.so')
mkl_set_num_threads = mkl_rt.MKL_Set_Num_Threads
mkl_get_max_threads = mkl_rt.MKL_Get_Max_Threads

np.random.seed(1000)

scratch = './tmp'
n_threads = 16

import os
if not os.path.isdir(scratch):
    os.mkdir(scratch)
os.environ['TMPDIR'] = scratch
os.environ['OMP_NUM_THREADS'] = str(n_threads)

mkl_set_num_threads(n_threads)
print(mkl_get_max_threads())

def get_jk(mol, dm0):
    hso2e = mol.intor('int2e_p1vxp1', 3).reshape(
        3, mol.nao, mol.nao, mol.nao, mol.nao)
    vj = np.einsum('yijkl,lk->yij', hso2e, dm0)
    vk = np.einsum('yijkl,jk->yil', hso2e, dm0)
    vk += np.einsum('yijkl,li->ykj', hso2e, dm0)
    return vj, vk


def get_jk_amfi(mol, dm0):
    '''Atomic-mean-field approximation'''
    ao_loc = mol.ao_loc_nr()
    nao = ao_loc[-1]
    vj = np.zeros((3, nao, nao))
    vk = np.zeros((3, nao, nao))
    atom = copy.copy(mol)
    aoslice = mol.aoslice_by_atom(ao_loc)
    for ia in range(mol.natm):
        b0, b1, p0, p1 = aoslice[ia]
        atom._bas = mol._bas[b0:b1]
        vj1, vk1 = get_jk(atom, dm0[p0:p1, p0:p1])
        vj[:, p0:p1, p0:p1] = vj1
        vk[:, p0:p1, p0:p1] = vk1
    return vj, vk


def compute_hso_ao(mol, dm0, qed_fac=1, amfi=False):
    """hso (complex, pure imag)"""
    alpha2 = nist.ALPHA ** 2
    hso1e = mol.intor_asymmetric('int1e_pnucxp', 3)
    vj, vk = get_jk_amfi(mol, dm0) if amfi else get_jk(mol, dm0)
    hso2e = vj - vk * 1.5
    hso = qed_fac * (alpha2 / 4) * (hso1e + hso2e)
    return hso * 1j

ano = gto.basis.parse("""
#BASIS SET: (21s,15p,10d,6f,4g) -> [6s,5p,3d,2f]
Cu    S
9148883.                     0.00011722            -0.00003675             0.00001383            -0.00000271             0.00000346            -0.00001401
1369956.                     0.00033155            -0.00010403             0.00003914            -0.00000766             0.00000979            -0.00003959
 311782.6                    0.00088243            -0.00027733             0.00010439            -0.00002047             0.00002626            -0.00010612
  88318.80                   0.00217229            -0.00068417             0.00025745            -0.00005030             0.00006392            -0.00025900
  28815.53                   0.00529547            -0.00167686             0.00063203            -0.00012442             0.00016087            -0.00064865
  10403.46                   0.01296630            -0.00413460             0.00155718            -0.00030269             0.00038023            -0.00154770
   4057.791                  0.03188050            -0.01035127             0.00391659            -0.00077628             0.00101716            -0.00408342
   1682.974                  0.07576569            -0.02534791             0.00959476            -0.00185240             0.00229152            -0.00940206
    733.7543                 0.16244637            -0.05836947             0.02238837            -0.00448270             0.00597497            -0.02389117
    333.2677                 0.28435959            -0.11673284             0.04528606            -0.00866024             0.01044695            -0.04359832
    156.4338                 0.34173643            -0.18466860             0.07498378            -0.01544726             0.02151583            -0.08588857
     74.69721                0.20955087            -0.14700899             0.06188170            -0.01089864             0.01030128            -0.04734865
     33.32262                0.03544751             0.18483257            -0.09047327             0.01467867            -0.00984324             0.06382592
     16.62237               -0.00243996             0.56773609            -0.39288472             0.09154122            -0.14292630             0.59363541
      8.208260               0.00146990             0.37956092            -0.35664956             0.06330857            -0.05612707             0.41506062
      3.609400              -0.00064379             0.04703755             0.34554793            -0.06093879             0.09135650            -1.70027942
      1.683449               0.00024738            -0.00072783             0.70639197            -0.27148545             0.45831402            -0.29799956
      0.733757              -0.00008118             0.00102976             0.25335911            -0.10138944            -0.19726740             1.70719425
      0.110207               0.00001615            -0.00007020             0.00658749             0.71212180            -1.30310157            -0.65600053
      0.038786              -0.00001233             0.00007090            -0.00044247             0.34613175             0.87975568            -0.31924524
      0.015514               0.00000452            -0.00002454             0.00072338             0.07198709             0.57310889             0.61772456
Cu    P
   9713.253                  0.00039987            -0.00014879             0.00005053            -0.00015878             0.00012679
   2300.889                  0.00210157            -0.00078262             0.00026719            -0.00085359             0.00068397
    746.7706                 0.01001097            -0.00376229             0.00127718            -0.00399919             0.00320291
    284.6806                 0.03836039            -0.01457968             0.00499760            -0.01608092             0.01301351
    119.9999                 0.11626756            -0.04571093             0.01561376            -0.04924864             0.04021371
     54.07386                0.25899831            -0.10593013             0.03689701            -0.12165679             0.10150109
     25.37321                0.38428226            -0.16836228             0.05796469            -0.17713367             0.13700954
     12.20962                0.29911210            -0.10373213             0.03791966            -0.14448197             0.10628972
      5.757421               0.08000672             0.21083155            -0.11706499             0.64743311            -0.83390265
      2.673402               0.00319440             0.48916443            -0.20912257             0.54797692             0.14412002
      1.186835               0.00147252             0.38468638            -0.05800532            -0.78180321             0.98301239
      0.481593              -0.00023391             0.08529338             0.16430736            -0.52607778            -0.52956418
      0.192637               0.00020761            -0.00161045             0.52885466             0.22437177            -0.75303984
      0.077055              -0.00008963             0.00206530             0.37373016             0.33392453             0.44288700
      0.030822               0.00002783            -0.00064175             0.08399866             0.14446374             0.57511589
Cu    D
    249.3497                 0.00134561            -0.00158359             0.00191116
     74.63837                0.01080983            -0.01281116             0.01772495
     28.37641                0.04733793            -0.05642679             0.07056381
     11.94893                0.13772582            -0.16641393             0.22906101
      5.317646               0.26263833            -0.35106014             0.43938511
      2.364417               0.33470401            -0.23717890            -0.26739908
      1.012386               0.31000597             0.21994641            -0.62590624
      0.406773               0.21316819             0.48841363             0.12211340
      0.147331               0.07865365             0.29537131             0.53683629
      0.058932               0.00603096             0.04295539             0.19750055
Cu    F
     15.4333                 0.03682063            -0.06200336
      6.1172                 0.24591418            -0.52876669
      2.4246                 0.50112688            -0.29078386
      0.9610                 0.35850648             0.52124157
      0.3809                 0.14728062             0.37532205
      0.1510                 0.02564748             0.10659798
""")

mol = gto.M(atom='Cu 0 0 0',
            symmetry=False,
            basis=ano, spin=1, charge=0, verbose=4, max_memory=150000)
mf = scf.newton(scf.RHF(mol).sfx2c1e())
mf.kernel()

print("\n CASSCF(12, 11):\n")

from pyscf.mcscf import avas
ao_labels = ['Cu 3d', 'Cu 4s']
norb, ne_act, orbs = avas.avas(mf, ao_labels, threshold=0.001, canonicalize=True)
print(norb, ne_act)

mc = mcscf.CASSCF(mf, norb, ne_act)
mc.state_average_((0.5, 0.1, 0.1, 0.1, 0.1, 0.1))
mc.kernel(orbs)

dmao = mc.make_rdm1()

n_sites = norb
n_elec = int(ne_act)
tol = 1E-13

h1e, e_core = mc.get_h1cas()
g2e = mc.get_h2cas()
g2e = ao2mo.restore(1, g2e, n_sites)

h1e[np.abs(h1e) < tol] = 0
g2e[np.abs(g2e) < tol] = 0

n = n_sites
gh1e = np.zeros((n * 2, n * 2), dtype=complex)
gg2e = np.zeros((n * 2, n * 2, n * 2, n * 2), dtype=complex)

for i in range(n * 2):
    for j in range(i % 2, n * 2, 2):
        gh1e[i, j] = h1e[i // 2, j // 2]

for i in range(n * 2):
    for j in range(i % 2, n * 2, 2):
        for k in range(n * 2):
            for l in range(k % 2, n * 2, 2):
                gg2e[i, j, k, l] = g2e[i // 2, j // 2, k // 2, l // 2]

# atomic mean-field spin-orbit integral (AMFI)
hsoao = compute_hso_ao(mol, dmao, amfi=True) * 2
hso = np.einsum('rij,ip,jq->rpq', hsoao,
                mc.mo_coeff[:, mc.ncore:mc.ncore + n],
                mc.mo_coeff[:, mc.ncore:mc.ncore + n])

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

from pyblock3.fcidump import FCIDUMP
from pyblock3.hamiltonian import Hamiltonian
from pyblock3.algebra.mpe import CachedMPE, MPE

orb_sym = [0] * (n * 2)

fd = FCIDUMP(pg='d2h', n_sites=n * 2, n_elec=n_elec, twos=n_elec, ipg=0, h1e=gh1e,
    g2e=gg2e, orb_sym=orb_sym, const_e=e_core)

SPIN, SITE, OP = 1, 2, 16384
def generate_qc_terms(n_sites, h1e, g2e, cutoff=1E-9):
    OP_C, OP_D = 0 * OP, 1 * OP
    h_values = []
    h_terms = []
    for i in range(0, n_sites):
        for j in range(0, n_sites):
            t = h1e[i, j]
            if abs(t) > cutoff:
                for s in [0, 1]:
                    h_values.append(t)
                    h_terms.append([OP_C + i * SITE + s * SPIN,
                                    OP_D + j * SITE + s * SPIN, -1, -1])
    for i in range(0, n_sites):
        for j in range(0, n_sites):
            for k in range(0, n_sites):
                for l in range(0, n_sites):
                    v = g2e[i, j, k, l]
                    if abs(v) > cutoff:
                        for sij in [0, 1]:
                            for skl in [0, 1]:
                                h_values.append(0.5 * v)
                                h_terms.append([OP_C + i * SITE + sij * SPIN,
                                                OP_C + k * SITE + skl * SPIN,
                                                OP_D + l * SITE + skl * SPIN,
                                                OP_D + j * SITE + sij * SPIN])
    if len(h_values) == 0:
        return np.zeros((0, ), dtype=np.complex128), np.zeros((0, 4), dtype=np.int32)
    else:
        return np.array(h_values, dtype=np.complex128), np.array(h_terms, dtype=np.int32)

def build_qc(hamil, cutoff=1E-9, max_bond_dim=-1):
    terms = generate_qc_terms(
            hamil.fcidump.n_sites, hamil.fcidump.h1e, hamil.fcidump.g2e, 1E-13)
    mm = hamil.build_mpo(terms, cutoff=cutoff, max_bond_dim=max_bond_dim,
        const=hamil.fcidump.const_e)
    return mm

hamil = Hamiltonian(fd, flat=True)
mpo = build_qc(hamil, max_bond_dim=-5)
mpo, error = mpo.compress(left=True, cutoff=1E-9, norm_cutoff=1E-9)
mps = hamil.build_mps(250)

bdims = [500] * 5 + [750] * 5 + [1000] * 5
noises = [1E-5] * 5 + [1E-6] * 5 + [1E-7] * 5 + [0]
davthrds = [5E-3] * 5 + [1E-4] * 5 + [1E-5]

nroots = 12
extra_mpes = [None] * (nroots - 1)
weights = (0.25, ) * 2 + (0.05, ) * 10
for ix in range(nroots - 1):
    xmps = hamil.build_mps(250)
    extra_mpes[ix] = CachedMPE(xmps, mpo, xmps, tag='CP%d' % ix)

dmrg = CachedMPE(mps, mpo, mps).dmrg(bdims=bdims, noises=noises, cutoff=1E-16, max_iter=2000,
    dav_thrds=davthrds, iprint=3, n_sweeps=18, extra_mpes=extra_mpes, weights=weights, init_site=3)
ener = dmrg.energies[-1]
print("FINAL ENERGY          = ", ener)

e0 = np.average(ener[0:2])
e1 = np.average(ener[2:8])
e2 = np.average(ener[8:12])

au2ev = 27.21139
print("")
print("E 2D(5/2)         = %10.4f eV" % ((e1 - e0) * au2ev))
print("E 2D(3/2)         = %10.4f eV" % ((e2 - e0) * au2ev))
print("2D(5/2) - 2D(3/2) = %10.4f eV" % ((e2 - e1) * au2ev))
