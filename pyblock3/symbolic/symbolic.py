
import numpy as np
from .expr import OpSum, OpString, OpElement
from collections import Counter
from ..algebra.core import FermionTensor
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


class SymbolicSparseTensor:
    """MPO tensor represented as matrix of symbols"""

    def __init__(self, mat, ops, lop, rop):
        self.mat = mat
        self.ops = ops
        self.lop = lop
        self.rop = rop

    @property
    def ndim(self):
        """Number of dimensions."""
        for v in self.ops.values():
            if not isinstance(v, int):
                return v.ndim + 2
        return 0

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
    
    def copy(self):
        return SymbolicSparseTensor(self.mat.copy(), self.ops.copy(), self.lop.copy(), self.rop.copy())

    def simplify(self, idx_mp, left=True):
        """Reduce virtual bond dimensions for symbolic sparse tensors."""
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
                if minfos is ():
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
        return (linfo, *minfos, rinfo)

    def to_sparse(self, infos=None):

        mat = self.mat
        ops = self.ops
        map_blocks = {}, {}
        linfo, rinfo = infos if infos is not None else self.virtual_infos

        def add_block(spmat, factor, ql, il, nl, qr, ir, nr):
            for ioe, oe in enumerate([spmat.odd, spmat.even]):
                for block in oe:
                    qx = (-ql, *block.q_labels, qr)
                    sh = (nl, *block.shape, nr)
                    if qx not in map_blocks[ioe]:
                        map_blocks[ioe][qx] = block.__class__.zeros(
                            shape=sh, q_labels=qx)
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
                add_block(spmat, expr.factor, ql, il, nl, qr, ir, nr)
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
                add_block(spmat, expr.factor, ql, il, nl, qr, ir, nr)
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
                add_block(spmat, expr.factor, ql, il, nl, qr, ir, nr)
        return FermionTensor(odd=list(map_blocks[0].values()), even=list(map_blocks[1].values()))

    def __matmul__(self, other):
        return self.__class__.hdot(self, other)

    def hdot(self, other):
        exprs = self.mat @ other.mat
        if isinstance(other.mat, SymbolicColumnVector):
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
