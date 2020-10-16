
import numpy as np
from itertools import accumulate, groupby
from collections import Counter
from pyblock3.algebra.symmetry import SZ, BondInfo
from pyblock3.algebra.core import SparseTensor
from pyblock3.algebra.flat import FlatSparseTensor


def flat_sparse_tensordot(aqs, ashs, adata, aidxs, bqs, bshs, bdata, bidxs, idxa, idxb):
    if len(aqs) == 0:
        return aqs, ashs, adata, aidxs
    elif len(bqs) == 0:
        return bqs, bshs, bdata, bidxs
    la, ndima = aqs.shape
    lb, ndimb = bqs.shape
    out_idx_a = np.delete(np.arange(0, ndima), idxa)
    out_idx_b = np.delete(np.arange(0, ndimb), idxb)

    ctrqas, outqas = aqs[:, idxa], aqs[:, out_idx_a]
    ctrqbs, outqbs = bqs[:, idxb], bqs[:, out_idx_b]

    map_idx_b = {}
    for ib in range(lb):
        ctrq = ctrqbs[ib].tobytes()
        if ctrq not in map_idx_b:
            map_idx_b[ctrq] = []
        map_idx_b[ctrq].append(ib)

    blocks_map = {}
    idxs = [0]
    qs = []
    shapes = []
    mats = []
    def compute_phase(nlist, idx, direction='left'):
        counted = []
        phase = 1
        for x in idx:
            if direction == 'left':
                counter = sum([nlist[i] for i in range(x) if i not in counted]) * nlist[x]
            elif direction == 'right':
                counter = sum([nlist[i] for i in range(x+1, len(nlist)) if i not in counted]) * nlist[x]
            phase *= (-1) ** counter
            counted.append(x)
        return phase

    for ia in range(la):
        ptca = [SZ.from_flat(i).n for i in aqs[ia]]
        phase_a = compute_phase(ptca, idxa, 'right')
        ctrq = ctrqas[ia].tobytes()
        mata = adata[aidxs[ia]: aidxs[ia + 1]].reshape(ashs[ia])
        if ctrq in map_idx_b:
            for ib in map_idx_b[ctrq]:
                ptcb = [SZ.from_flat(i).n for i in bqs[ib]]
                phase_b = compute_phase(ptcb, idxb, 'left')
                alpha = phase_a * phase_b
                outq = np.concatenate((outqas[ia], outqbs[ib]))
                matb = bdata[bidxs[ib]: bidxs[ib + 1]].reshape(bshs[ib])
                mat = np.tensordot(mata, matb, axes=(idxa, idxb)) * alpha
                outqk = outq.tobytes()
                if outqk not in blocks_map:
                    blocks_map[outqk] = len(qs)
                    idxs.append(idxs[-1] + mat.size)
                    qs.append(outq)
                    shapes.append(mat.shape)
                    mats.append(mat.flatten())
                else:
                    mats[blocks_map[outqk]] += mat.flatten()

    return (np.array(qs, dtype=np.uint32),
            np.array(shapes, dtype=np.uint32),
            np.concatenate(mats) if len(mats) != 0 else np.array(
                [], dtype=np.float64),
            np.array(idxs, dtype=np.uint32))


np.random.seed(3)

x = SZ(0,0,0)
y = SZ(1,0,0)
infox = BondInfo({x:3, y: 2})
infoy = BondInfo({x:2, y: 3})

arra = SparseTensor.random((infox,infoy))
a = FlatSparseTensor.from_sparse(arra)
arrb = SparseTensor.random((infoy,infox))
b = FlatSparseTensor.from_sparse(arrb)

x = flat_sparse_tensordot(a.q_labels, a.shapes, a.data, a.idxs, b.q_labels, b.shapes, b.data, b.idxs, (1,), (0,))

from block3.fermion_sparse_tensor import tensordot

y = tensordot(a.q_labels, a.shapes, a.data, a.idxs, b.q_labels, b.shapes, b.data, b.idxs, (1,), (0,))
