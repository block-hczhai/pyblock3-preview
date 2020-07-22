
import numpy as np
from .expr import OpSum, OpString


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
        r = SymbolicRowVector(self.n_rows)
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


class SparseSymbolicTensor:
    """MPO tensor represented as matrix of symbols"""

    def __init__(self, mat, ops, lop, rop):
        self.mat = mat
        self.ops = ops
        self.lop = lop
        self.rop = rop

    def deflate(self):
        for i, mop in enumerate(self.mat):
            op = abs(mop)
            if op not in self.ops:
                self.mat.data[i] = 0
            elif self.ops[op] == 0:
                self.mat.data[i] = 0
                del self.ops[op]
        self.mat.deflate()
        return self

    def __matmul__(self, other):
        exprs = self.mat @ other.mat
        if isinstance(other.mat, SymbolicColumnVector):
            mat = self.mat.lop
        elif isinstance(self.mat, SymbolicRowVector):
            mat = other.mat.rop
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
                if pd.ops[0] in lops and pd.ops[1] in rops:
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
            else:
                raise RuntimeError("Unknown expression type: %r" % expr.__class__)
        return SparseSymbolicTensor(mat, ops, self.mat.lop, other.mat.rop).deflate()
