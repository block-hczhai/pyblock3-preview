
#  pyblock3: An Efficient python MPS/DMRG Library
#  Copyright (C) 2020 The pyblock3 developers. All Rights Reserved.
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program. If not, see <https://www.gnu.org/licenses/>.
#
#

"""Vector/Matrix of operator symbols."""

import numpy as np
from .expr import OpSum, OpString, OpElement, OpNames
from collections import Counter
from ..algebra.core import FermionTensor, SparseTensor
from ..algebra.symmetry import BondFusingInfo


class Symbolic:
    def __init__(self, n_rows, n_cols, data=None):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.data = data if data is not None else np.zeros(
            (n_rows, n_cols), dtype=object)

    def __getitem__(self, idx):
        return self.data[idx[0], idx[1]]

    def __setitem__(self, idx, v):
        self.data[idx[0], idx[1]] = v

    def get_symbols(self):
        return set([abs(op) for op in self if op != 0])

    def __iter__(self):
        yield from self.data.flatten()

    def __len__(self):
        return self.data.size

    def deflate(self):
        pass


class SymbolicIdentity(Symbolic):
    def __init__(self, n_rows, op):
        data = np.zeros((n_rows, n_rows), dtype=object)
        for i in range(n_rows):
            data[i, i] = op
        super().__init__(n_rows, n_rows, data)

    def __matmul__(self, other):
        assert isinstance(other, SymbolicRowVector)
        r = SymbolicRowVector(other.n_cols)
        for idx, d in enumerate(other.data):
            r.data[idx] = self.data[0, 0] * d
        return r

    def __rmatmul__(self, other):
        assert isinstance(other, SymbolicColumnVector)
        r = SymbolicColumnVector(other.n_rows)
        for idx, d in enumerate(other.data):
            r.data[idx] = d * self.data[0, 0]
        return r

    def copy(self):
        return SymbolicIdentity(self.n_rows, self.data[0, 0])


class SymbolicRowVector(Symbolic):
    def __init__(self, n_cols, data=None):
        if data is None:
            data = np.zeros((n_cols, ), dtype=object)
        super().__init__(1, n_cols, data)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.data[idx]
        else:
            assert idx[0] == 0
            return self.data[idx[1]]

    def __matmul__(self, other):
        if isinstance(other, SymbolicColumnVector):
            r = SymbolicRowVector(1)
            for idx, d in enumerate(self.data):
                r.data[0] += d * other.data[idx]
            return r
        else:
            return NotImplemented

    def __setitem__(self, idx, v):
        if isinstance(idx, int):
            self.data[idx] = v
        else:
            assert idx[0] == 0
            self.data[idx[1]] = v

    def copy(self):
        return SymbolicRowVector(self.n_cols, self.data.copy())


class SymbolicColumnVector(Symbolic):
    def __init__(self, n_rows, data=None):
        if data is None:
            data = np.zeros((n_rows, ), dtype=object)
        super().__init__(n_rows, 1, data)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.data[idx]
        else:
            assert idx[1] == 0
            return self.data[idx[0]]

    def __setitem__(self, idx, v):
        if isinstance(idx, int):
            self.data[idx] = v
        else:
            assert idx[1] == 0
            self.data[idx[0]] = v

    def copy(self):
        return SymbolicColumnVector(self.n_rows, self.data.copy())


class SymbolicMatrix(Symbolic):
    def __init__(self, n_rows, n_cols, indices=None, data=None):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.indices = indices if indices is not None else []
        self.data = data if data is not None else []

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.data[idx]
        else:
            raise RuntimeError("Cannot read SymbolicMatrix elems by indices!")

    def __setitem__(self, idx, v):
        # add assert here will greatly slow down MPO init
        self.indices.append(idx)
        self.data.append(v)

    def __iter__(self):
        yield from self.data

    def __len__(self):
        return len(self.data)

    def __matmul__(self, other):
        assert isinstance(other, SymbolicColumnVector)
        r = SymbolicColumnVector(self.n_rows)
        for idx, d in zip(self.indices, self.data):
            r.data[idx[0]] += d * other.data[idx[1]]
        return r

    def __rmatmul__(self, other):
        assert isinstance(other, SymbolicRowVector)
        r = SymbolicRowVector(self.n_cols)
        for idx, d in zip(self.indices, self.data):
            r.data[idx[1]] += other.data[idx[0]] * d
        return r

    def deflate(self):
        indices = []
        data = []
        for idx, d in zip(self.indices, self.data):
            if d != 0:
                indices.append(idx)
                data.append(d)
        self.indices = indices
        self.data = data

    def copy(self):
        return SymbolicMatrix(self.n_rows, self.n_cols, self.indices.copy(), self.data.copy())


class SymbolicBondFusingInfo:
    """Information for transform between sparse symbolic bond and block-sparse bond."""

    def __init__(self, names, q_labels, n_states):
        self.names = names
        self.q_labels = q_labels
        self.n_states = n_states

    @property
    def n_bonds(self):
        return len(self.q_labels)

    def __or__(self, other):
        return self

    @staticmethod
    def from_op_names(opv):
        assert isinstance(opv, SymbolicRowVector) or isinstance(
            opv, SymbolicColumnVector)
        mp = {}
        ns = Counter()
        qs = [0 if op == 0 else op.q_label for op in opv.data]
        names = [op for op in opv.data]
        for ip, q in enumerate(qs):
            if q not in mp:
                mp[q] = []
            mp[q].append(ip)
            ns[q] += 1
        for q, v in mp.items():
            for iiv, iv in enumerate(v):
                qs[iv] = (q, iiv)
        return SymbolicBondFusingInfo(names, qs, ns)


def implements(np_func):
    global _numpy_func_impls
    return lambda f: (_numpy_func_impls.update({np_func: f})
                      if np_func not in _numpy_func_impls else None,
                      _numpy_func_impls[np_func])[1]


_sym_sparse_numpy_func_impls = {}
_numpy_func_impls = _sym_sparse_numpy_func_impls


class SymbolicSparseTensor:
    """MPO tensor represented as matrix of symbols"""

    def __init__(self, mat, ops, lop, rop, idx_perm=None):
        self.mat = mat
        self.ops = ops
        self.lop = lop
        self.rop = rop
        # outer idx -> inner idx
        self.idx_perm = idx_perm
        # delta quantum of the operator
        self.target = None

    @property
    def ndim(self):
        """Number of dimensions."""
        for v in self.ops.values():
            if not isinstance(v, int):
                return v.ndim + 2
        return 0

    def __array_function__(self, func, types, args, kwargs):
        if func not in _sym_sparse_numpy_func_impls:
            return NotImplemented
        if not all(issubclass(t, self.__class__) or issubclass(t, SparseTensor) for t in types):
            return NotImplemented
        return _sym_sparse_numpy_func_impls[func](*args, **kwargs)

    def deflate(self):
        for i, mop in enumerate(self.mat):
            op = abs(mop)
            if op not in self.ops:
                self.mat.data[i] = 0
            elif (isinstance(self.ops[op], int) and self.ops[op] == 0) or self.ops[op].n_blocks == 0:
                self.mat.data[i] = 0
                del self.ops[op]
        self.mat.deflate()
        return self

    @staticmethod
    def ones(bond_infos, pattern=None, dtype=float):
        """Create operator tensor with ones."""
        assert bond_infos[0].n_bonds == 1
        assert bond_infos[-1].n_bonds == 1
        assert bond_infos[0].q_labels[0] == bond_infos[-1].q_labels[0]
        spt = FermionTensor.ones(
            bond_infos[1:-1], pattern=pattern[1:-1], dtype=dtype)
        iop = OpElement(OpNames.I, (), q_label=bond_infos[0].q_labels[0])
        mat = SymbolicIdentity(1, iop)
        ops = {iop: spt}
        return SymbolicSparseTensor(mat, ops, mat, mat)

    def copy(self):
        return SymbolicSparseTensor(self.mat.copy(), self.ops.copy(), self.lop.copy(), self.rop.copy(), self.idx_perm)

    def simplify(self, idx_mp, left=True):
        """Reduce virtual bond dimensions for symbolic sparse tensors."""
        assert self.idx_perm is None
        ops = self.lop if left else self.rop
        new_ops = ops.__class__(len(idx_mp))
        new_ops.data[:] = ops.data[idx_mp]
        if left:
            lop, rop = new_ops, self.rop
        else:
            lop, rop = self.lop, new_ops
        if isinstance(self.mat, SymbolicRowVector) or isinstance(self.mat, SymbolicColumnVector):
            new_mat = self.mat.__class__(len(idx_mp))
            new_mat.data[:] = self.mat.data[idx_mp]
        else:
            mp = {xr: ir for ir, xr in enumerate(idx_mp)}
            new_idx = []
            new_data = []
            for i, (j, k) in enumerate(self.mat.indices):
                if left and j in mp:
                    new_idx.append((mp[j], k))
                    new_data.append(self.mat.data[i])
                elif not left and k in mp:
                    new_idx.append((j, mp[k]))
                    new_data.append(self.mat.data[i])
            if left:
                new_mat = SymbolicMatrix(
                    len(idx_mp), self.mat.n_cols, new_idx, new_data)
            else:
                new_mat = SymbolicMatrix(
                    self.mat.n_rows, len(idx_mp), new_idx, new_data)
        return SymbolicSparseTensor(new_mat, self.ops, lop, rop)

    def get_remap(self, left=True):
        assert self.idx_perm is None
        r = []
        if isinstance(self.mat, SymbolicRowVector) or isinstance(self.mat, SymbolicColumnVector):
            for i, mop in enumerate(self.mat):
                if mop != 0:
                    r.append(i)
        else:
            for (j, k), expr in zip(self.mat.indices, self.mat.data):
                if expr != 0:
                    r.append(j if left else k)
            r = list(set(r))
        r.sort()
        return np.array(r, dtype=int)

    @property
    def physical_infos(self):
        minfos = ()
        for spmat in self.ops.values():
            if not isinstance(spmat, int):
                if minfos == ():
                    minfos = spmat.infos
                else:
                    assert len(minfos) == len(spmat.infos)
                    minfos = [m | s for m, s in zip(minfos, spmat.infos)]
        return minfos

    @property
    def virtual_infos(self):
        linfo = SymbolicBondFusingInfo.from_op_names(self.lop)
        rinfo = SymbolicBondFusingInfo.from_op_names(self.rop)
        return linfo, rinfo

    @property
    def infos(self):
        linfo, rinfo = self.virtual_infos
        minfos = self.physical_infos
        r = (linfo, *minfos, rinfo)
        if self.idx_perm is None:
            return r
        else:
            return tuple(r[i] for i in self.idx_perm)

    # Right svd needs to collect all right indices for each specific left index.
    # works when self.ops values are exprs
    # used to build mpo from hamiltonian expression
    def right_svd(self, idx, cutoff=1E-12):
        i_op = self.rop[0]
        ident = (i_op, )
        q_map = {}
        lr_map = []
        mats = []
        npm = len(self.mat.data)
        dtype = float
        for ip, op in enumerate(self.mat.data):
            qll = self.lop[ip].q_label
            if op == 0:
                continue
            expr = self.ops[abs(op)]
            if isinstance(expr, OpSum):
                terms = (op.factor * expr).strings
            elif isinstance(expr, OpString):
                terms = [op.factor * expr]
            elif isinstance(expr, OpElement):
                terms = [op.factor * OpString([expr])]
            else:
                assert False
            for term in terms:
                if term == 0:
                    continue
                k = 0
                for op in term.ops:
                    if op == i_op or op.site_index[0] <= idx:
                        k += 1
                    else:
                        break
                l = ident if k == 0 else tuple(term.ops[:k])
                r = ident if k == len(term.ops) else tuple(term.ops[k:])
                ql = -qll + np.add.reduce([x.q_label for x in l])
                qr = np.add.reduce([x.q_label for x in r])
                assert ql + qr == i_op.q_label
                if (ql, qr) not in q_map:
                    iq = len(q_map)
                    q_map[(ql, qr)] = iq
                    lr_map.append(({}, {}))
                    mats.append([])
                else:
                    iq = q_map[(ql, qr)]
                mpl, mpr = lr_map[iq]
                if (ip, l) not in mpl:
                    il = len(mpl)
                    mpl[(ip, l)] = il
                else:
                    il = mpl[(ip, l)]
                if r not in mpr:
                    ir = len(mpr)
                    mpr[r] = ir
                else:
                    ir = mpr[r]
                if isinstance(term.factor, complex):
                    dtype = complex
                mats[iq].append((il, ir, term.factor))
        mqlr = [None] * len(mats)
        m, mq = 0, len(q_map)
        for (ql, qr), iq in q_map.items():
            matvs = mats[iq]
            mpl, mpr = lr_map[iq]
            mat = np.zeros((len(mpl), len(mpr)), dtype=dtype)
            for il, ir, v in matvs:
                mat[il, ir] += v
            l, s, r = np.linalg.svd(mat, full_matrices=False)
            mask = s > cutoff
            ll, rr = s[None, mask] * l[:, mask], r[mask, :]
            mqlr[iq] = ll, rr, ql, qr
            m += ll.shape[1]
        lrop = SymbolicRowVector(m)
        rlop = SymbolicColumnVector(m)
        lmat = np.zeros((npm, m), dtype=object)
        rmat = np.zeros((m, ), dtype=object)
        im = 0
        for iq in range(mq):
            mpl, mpr = lr_map[iq]
            ll, rr, ql, qr = mqlr[iq]
            nm = ll.shape[1]
            for imm in range(im, im + nm):
                lrop[imm] = OpElement(OpNames.XR, (imm, ), q_label=ql)
                rlop[imm] = OpElement(OpNames.XL, (imm, ), q_label=qr)
            lxmat = np.zeros((npm, len(mpl)), dtype=object)
            rxmat = np.zeros((len(mpr), ), dtype=object)
            for (ip, l), il in mpl.items():
                lxmat[ip, il] = OpString(l)
            for r, ir in mpr.items():
                rxmat[ir] = OpString(r)
            lmat[:, im:im + nm] = lxmat @ ll
            rmat[im:im + nm] = rr @ rxmat
            im += nm
        splmat = SymbolicMatrix(npm, m)
        sprmat = SymbolicColumnVector(m)
        lops, rops = {}, {}
        ilop, irop = 0, 0
        im = 0
        for iq in range(mq):
            ll, _, ql, qr = mqlr[iq]
            nm = ll.shape[1]
            for ip in range(splmat.n_rows):
                q = ql + self.lop[ip].q_label
                for imm in range(im, im + nm):
                    if lmat[ip, imm] != 0:
                        if isinstance(lmat[ip, imm], OpString) and len(lmat[ip, imm].ops) == 1:
                            xop = lmat[ip, imm].op
                            op = abs(xop)
                            lops[op] = op
                            splmat[ip, imm] = xop
                        else:
                            op = OpElement(OpNames.X, (ilop, ), q_label=q)
                            lops[op] = lmat[ip, imm]
                            splmat[ip, imm] = op
                            ilop += 1
            for imm in range(im, im + nm):
                if rmat[imm] != 0:
                    if isinstance(rmat[imm], OpString) and len(rmat[imm].ops) == 1:
                        xop = rmat[imm].op
                        op = abs(xop)
                        rops[op] = op
                        sprmat[imm] = xop
                    else:
                        op = OpElement(OpNames.X, (irop, ), q_label=qr)
                        rops[op] = rmat[imm]
                        sprmat[imm] = op
                        irop += 1
            im += nm
        tl = SymbolicSparseTensor(splmat, lops, self.lop, lrop)
        tr = SymbolicSparseTensor(sprmat, rops, rlop, self.rop)
        return tl, tr

    def to_sparse(self, infos=None, dq=None):

        mat = self.mat
        ops = self.ops
        map_blocks = {}, {}
        linfo, rinfo = infos if infos is not None else self.virtual_infos

        def add_block(spmat, dq, factor, ql, il, nl, qr, ir, nr):
            if dq is None:
                dq = ql.__class__()
            for ioe, oe in enumerate([spmat.odd, spmat.even]):
                for block in oe.to_sparse():
                    qx = (dq - ql, *block.q_labels, qr)
                    sh = (nl, *block.shape, nr)
                    dtype = block.dtype
                    if isinstance(factor, complex):
                        dtype = complex
                    if qx not in map_blocks[ioe]:
                        map_blocks[ioe][qx] = block.__class__.zeros(
                            shape=sh, q_labels=qx, dtype=dtype)
                    map_blocks[ioe][qx][il, ..., ir] += factor * \
                        np.asarray(block)

        if isinstance(mat, SymbolicRowVector):
            for k, expr in enumerate(mat.data):
                if expr == 0:
                    continue
                assert isinstance(expr, OpElement)
                spmat = ops[abs(expr)]
                if spmat.n_blocks == 0:
                    continue
                qr, ir = rinfo.q_labels[k]
                nr = rinfo.n_states[qr]
                ql, il, nl = qr.__class__(), 0, 1
                add_block(spmat, dq, expr.factor, ql, il, nl, qr, ir, nr)
        elif isinstance(mat, SymbolicColumnVector):
            for k, expr in enumerate(mat.data):
                if expr == 0:
                    continue
                assert isinstance(expr, OpElement)
                spmat = ops[abs(expr)]
                if spmat.n_blocks == 0:
                    continue
                ql, il = linfo.q_labels[k]
                nl = linfo.n_states[ql]
                qr, ir, nr = ql.__class__(), 0, 1
                add_block(spmat, dq, expr.factor, ql, il, nl, qr, ir, nr)
        else:
            for (j, k), expr in zip(mat.indices, mat.data):
                if expr == 0:
                    continue
                assert isinstance(expr, OpElement)
                spmat = ops[abs(expr)]
                if spmat.n_blocks == 0:
                    continue
                ql, il = linfo.q_labels[j]
                qr, ir = rinfo.q_labels[k]
                nl, nr = linfo.n_states[ql], rinfo.n_states[qr]
                add_block(spmat, dq, expr.factor, ql, il, nl, qr, ir, nr)
        r = FermionTensor(
            odd=list(map_blocks[0].values()), even=list(map_blocks[1].values()))
        if self.idx_perm is None:
            return r
        else:
            return r.transpose(tuple(self.idx_perm))

    @staticmethod
    def _pdot(a, b, out=None):
        """Vertical contraction (all middle/physical dims)."""

        if isinstance(a, list):
            p, ps = 1, []
            for i in range(len(a)):
                ps.append(p)
                p += a[i].ndim // 2 - 1
            r = b
            d2 = r.ndim - 2
            # [MPO] x MPS
            if isinstance(b, SparseTensor):
                for i in range(len(a))[::-1]:
                    d = a[i].ndim // 2 - 1
                    aidx = list(range(d, d + d))
                    bidx = list(range(ps[i], ps[i] + d))
                    tr = (*range(d, ps[i] + d), *range(0, d), *range(ps[i] + d, d2 + 2))
                    if isinstance(a[i].mat, SymbolicColumnVector):
                        ar = [0] * len(a[i].mat)
                        for ia in range(len(a[i].mat)):
                            if a[i].mat[ia] == 0:
                                continue
                            mop = a[i].mat[ia]
                            op = abs(mop)
                            ar[ia] = np.tensordot(a[i].ops[op] * mop.factor, r, axes=(aidx, bidx)).transpose(axes=tr)
                    elif isinstance(a[i].mat, SymbolicRowVector):
                        ar = 0
                        assert len(a[i].mat) == len(r)
                        for ia in range(len(a[i].mat)):
                            if (isinstance(r[ia], int) and r[ia] == 0) or a[i].mat[ia] == 0:
                                continue
                            mop = a[i].mat[ia]
                            op = abs(mop)
                            ar = np.tensordot(a[i].ops[op] * mop.factor, r[ia], axes=(aidx, bidx)).transpose(axes=tr) + ar
                    elif isinstance(a[i].mat, SymbolicMatrix):
                        ar = [0] * a[i].mat.n_rows
                        for (j, k), mop in zip(a[i].mat.indices, a[i].mat.data):
                            if mop == 0 or (isinstance(r[k], int) and r[k] == 0):
                                continue
                            assert isinstance(mop, OpElement)
                            op = abs(mop)
                            ar[j] = np.tensordot(a[i].ops[op] * mop.factor, r[k], axes=(aidx, bidx)).transpose(axes=tr) + ar[j]
                    r = ar
                if isinstance(r, list):
                    assert len(r) == 1
                    r = r[0]
                ir = r.infos
                rl = r.ones(bond_infos=(ir[0], ir[0], ir[0]), pattern="--+")
                rr = r.ones(bond_infos=(ir[-1], ir[0], ir[-1]), pattern="+--")
                r = np.tensordot(rl, r, axes=1)
                r = np.tensordot(r, rr, axes=1)
            else:
                raise RuntimeError(
                    "Cannot matmul tensors with types %r x %r" % (a.__class__, b.__class__))
        elif isinstance(b, list):
            raise NotImplementedError("not implemented.")
        else:
            assert isinstance(a, SymbolicSparseTensor) or isinstance(b, SymbolicSparseTensor)
            raise NotImplementedError("not implemented.")

        if out is not None:
            assert isinstance(r, SparseTensor)
            out.blocks = r.blocks

        return r

    def pdot(self, b, out=None):
        return SymbolicSparseTensor._pdot(self, b, out=out)

    @staticmethod
    @implements(np.tensordot)
    def _tensordot(a, b, axes):
        """
        Contract with a SparseTensor to form a new SymbolicSparseTensor.
        Only physical dims are allowed to be contracted.

        Args:
            a : SymbolicSparseTensor/SparseTensor
                SymbolicSparseTensor/SparseTensor a, as left operand.
            b : SymbolicSparseTensor/SparseTensor
                SymbolicSparseTensor/SparseTensor b, as right operand.
            axes : (2,) array_like
                A list of axes to be summed over, first sequence applying to a, second to b.

        Returns:
            c : SymbolicSparseTensor
                The contracted SymbolicSparseTensor.
        """
        idxa, idxb = axes
        idxa = [x if x >= 0 else a.ndim + x for x in idxa]
        idxb = [x if x >= 0 else b.ndim + x for x in idxb]
        assert len(idxa) == len(idxb)

        if isinstance(a, SymbolicSparseTensor) and isinstance(b, SparseTensor):
            out_idx_b = list(set(range(0, b.ndim)) - set(idxb))
            if a.idx_perm is not None:
                pidxa = [a.idx_perm[x] for x in idxa]
                # expt output -> outer
                out_idx_a = list(set(range(0, a.ndim)) - set(idxa))
                pout_idx_a = list(set(range(0, a.ndim)) -
                                  set(pidxa))  # real output -> real
                # real output -> outer
                ppout_idx_a = [a.idx_perm.index(ip) for ip in pout_idx_a]
                # expt output -> real output
                pppout_idx_a = [ppout_idx_a.index(ip) for ip in out_idx_a]
                # last vir adj
                p4out_idx_a = [ip if ip != len(
                    ppout_idx_a) - 1 else ip + len(out_idx_b) for ip in pppout_idx_a]
                new_perm = p4out_idx_a + \
                    [x + len(ppout_idx_a) - 1 for x in range(len(out_idx_b))]
                idxa = pidxa
            else:
                out_idx_a = list(set(range(0, a.ndim)) - set(idxa))
                new_perm = list(range(0, len(out_idx_a) - 1)) + [len(out_idx_a) + len(
                    out_idx_b) - 1] + [x + len(out_idx_a) - 1 for x in range(len(out_idx_b))]
            assert all([x != 0 and x != a.ndim - 1 for x in idxa])
            idxa = [x - 1 for x in idxa]
            new_ops = {op: np.tensordot(expr, b, axes=(idxa, idxb))
                       for op, expr in a.ops.items()}
            if new_perm == list(range(0, len(new_perm))):
                new_perm = None
            return SymbolicSparseTensor(a.mat, new_ops, a.lop, a.rop, idx_perm=new_perm)
        elif isinstance(b, SymbolicSparseTensor) and isinstance(a, SparseTensor):
            out_idx_a = list(set(range(0, a.ndim)) - set(idxa))
            if b.idx_perm is not None:
                pidxb = [b.idx_perm[x] for x in idxb]
                # expt output -> outer
                out_idx_b = list(set(range(0, b.ndim)) - set(idxb))
                pout_idx_b = list(set(range(0, b.ndim)) -
                                  set(pidxb))  # real output -> real
                # real output -> outer
                ppout_idx_b = [b.idx_perm.index(ip) for ip in pout_idx_b]
                # expt output -> real output
                pppout_idx_b = [ppout_idx_b.index(ip) for ip in out_idx_b]
                # first vir adj
                p4out_idx_b = [ip if ip == 0 else ip +
                               len(out_idx_a) for ip in pppout_idx_b]
                new_perm = list(range(1, len(out_idx_a) + 1)) + p4out_idx_b
                idxb = pidxb
            else:
                out_idx_b = list(set(range(0, b.ndim)) - set(idxb))
                new_perm = list(range(1, len(out_idx_a) + 1)) + [0] + list(
                    range(len(out_idx_a) + 1, len(out_idx_a) + len(out_idx_b)))
            assert all([x != 0 and x != b.ndim - 1 for x in idxb])
            idxb = [x - 1 for x in idxb]
            new_ops = {op: np.tensordot(a, expr, axes=(idxa, idxb))
                       for op, expr in b.ops.items()}
            if new_perm == list(range(0, len(new_perm))):
                new_perm = None
            return SymbolicSparseTensor(b.mat, new_ops, b.lop, b.rop, idx_perm=new_perm)
        else:
            raise TypeError('Unsupported tensordot for %r and %r' %
                            (a.__class__, b.__class__))

    def tensordot(self, b, axes):
        return np.tensordot(self, b, axes)

    @staticmethod
    @implements(np.transpose)
    def _transpose(a, axes=None):
        """
        Reverse or permute the axes of an array; returns the modified array.

        Args:
            a : array_like
                Input array.
            axes : tuple or list of ints, optional
                If specified, it must be a tuple or list which contains a permutation of [0,1,..,N-1]
                where N is the number of axes of a.
                The i’th axis of the returned array will correspond to the axis numbered axes[i] of the input.
                If not specified, defaults to ``range(a.ndim)[::-1]``, which reverses the order of the axes.

        Returns
            p : FermionTensor
                a with its axes permuted. A view is returned whenever possible.
        """
        if axes is None:
            axes = range(a.ndim)[::-1]  # new outer -> old outer
        if a.idx_perm is not None:
            new_perm = [a.idx_perm[x] for x in axes]
        else:
            new_perm = axes
        if new_perm[0] == 0 and new_perm[-1] == a.ndim - 1:
            paxes = [x - 1 for x in new_perm[1:-1]]
            new_ops = {op: expr.transpose(axes=paxes)
                       for op, expr in a.ops.items()}
            return SymbolicSparseTensor(a.mat, new_ops, a.lop, a.rop, idx_perm=None)
        else:
            return SymbolicSparseTensor(a.mat, a.ops, a.lop, a.rop, idx_perm=new_perm)

    def transpose(self, axes=None):
        return np.transpose(self, axes=axes)

    def hdot(self, other):
        assert self.idx_perm is None and other.idx_perm is None
        exprs = self.mat @ other.mat
        if isinstance(self.mat, SymbolicIdentity):
            mat = other.rop.copy()
        elif isinstance(other.mat, SymbolicIdentity):
            mat = self.lop.copy()
        elif isinstance(other.mat, SymbolicColumnVector):
            mat = self.lop.copy()
        elif isinstance(self.mat, SymbolicRowVector):
            mat = other.rop.copy()
        else:
            raise RuntimeError("Cannot perform symbolic tensor multiplication %r @ %r" %
                               (self.mat.__class__, other.mat.__class__))

        assert len(mat) == len(exprs)

        ops = {}
        lops, rops = self.ops, other.ops
        for mop, expr in zip(mat, exprs):
            op = abs(mop)
            factor = 1 / mop.factor
            if isinstance(expr, OpString):
                if expr.ops[0] in lops and expr.ops[1] in rops:
                    ops[op] = (lops[expr.ops[0]] ^ rops[expr.ops[1]]) * \
                        (factor * expr.factor)
                else:
                    ops[op] = 0
            elif isinstance(expr, OpSum):
                ops[op] = 0
                for pd in expr.strings:
                    if pd.ops[0] in lops and pd.ops[1] in rops:
                        ops[op] = (lops[pd.ops[0]] ^ rops[pd.ops[1]]) * \
                            (factor * pd.factor) + ops[op]
            elif isinstance(expr, int) and expr == 0:
                ops[op] = 0
            else:
                raise RuntimeError(
                    "Unknown expression type: %r" % expr.__class__)

        if isinstance(self.mat, SymbolicIdentity):
            return SymbolicSparseTensor(mat, ops, other.lop, other.rop).deflate()
        elif isinstance(other.mat, SymbolicIdentity):
            return SymbolicSparseTensor(mat, ops, self.lop, self.rop).deflate()
        else:
            return SymbolicSparseTensor(mat, ops, self.lop, other.rop).deflate()

    @staticmethod
    def _unfuse(a, i, info):
        return a.unfuse(i, info)

    def unfuse(self, i, info):
        """Unfuse one leg to several legs.
        Only works for physical legs.

        Args:
            i : int
                index of the leg to be unfused. The new unfused indices will be i, i + 1, ...
            info : BondFusingInfo
                Indicating how quantum numbers are collected.
        """
        assert self.idx_perm is None
        i = i if i >= 0 else self.ndim + i
        assert i != 0 and i != self.ndim - 1
        ops = {}
        for k, v in self.ops.items():
            ops[k] = v.unfuse(
                i - 1, info=info) if not isinstance(v, int) else 0
        return SymbolicSparseTensor(self.mat, ops, self.lop, self.rop)

    @staticmethod
    def _fuse(a, *idxs, info=None, pattern=None):
        return a.fuse(*idxs, info=info, pattern=pattern)

    def fuse(self, *idxs, info=None, pattern=None):
        """Fuse several legs to one leg.
        Only works for physical legs.

        Args:
            idxs : tuple(int)
                Leg indices to be fused. The new fused index will be idxs[0].
            info : BondFusingInfo (optional)
                Indicating how quantum numbers are collected.
                If not specified, the direct sum of quantum numbers will be used.
                This will generate minimal and (often) incomplete fused shape.
            pattern : str (optional)
                A str of '+'/'-'. Only required when info is not specified.
                Indicating how quantum numbers are linearly combined.
        """
        assert self.idx_perm is None
        idxs = [i if i >= 0 else self.ndim + i for i in idxs]
        for idx in idxs:
            assert idx != 0 and idx != self.ndim - 1
        idxs = [i - 1 for i in idxs]
        if info is None:
            items = []
            for sp in self.ops.values():
                if not isinstance(sp, int):
                    for block in sp.odd.blocks + sp.even.blocks:
                        qs = tuple(block.q_labels[i] for i in idxs)
                        shs = tuple(block.shape[i] for i in idxs)
                        items.append((qs, shs))
            # using minimal fused dimension
            info = BondFusingInfo.kron_sum(items, pattern=pattern)
        ops = {}
        for k, v in self.ops.items():
            ops[k] = v.fuse(*idxs, info=info,
                            pattern=pattern) if not isinstance(v, int) else 0
        return SymbolicSparseTensor(self.mat, ops, self.lop, self.rop)
