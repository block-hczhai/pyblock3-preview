import numpy as np
from .core import SparseTensor, SubTensor, implements
from .flat import FlatSparseTensor, flat_sparse_skeleton
from .symmetry import SZ
import numbers

_sub_tensor_numpy_func_impls = {}
_numpy_func_impls = _sub_tensor_numpy_func_impls

class FermionSparseTensor(SparseTensor):

    def __init__(self, *args, **kwargs):
        SparseTensor.__init__(self, *args, **kwargs)
        self.check_sanity()

    @property
    def pairty(self):
        self.check_sanity()
        parity_uniq = np.unique(self.parity_list)
        return parity_uniq[0]

    @property
    def parity_per_block(self):
        parity_list = []
        for block in self.blocks:
            pval = sum([q_label.n for q_label in block.q_labels])
            parity_list.append(int(pval)%2)
        return parity_list

    def check_sanity(self):
        parity_uniq = np.unique(self.parity_per_block)
        if len(parity_uniq) >1:
            raise ValueError("both odd/even parity detected in blocks, \
                              this is prohibited")

    def _local_flip(self, axes):
        if isinstance(axes, int):
            ax = [axes]
        else:
            ax = list(axes)
        for i in range(self.n_blocks):
            block_parity = sum([self.blocks[i].q_labels[j].n for j in ax]) %2
            if block_parity == 1:
                self.blocks[i] *= -1

    def _global_flip(self):
        for i in range(self.n_blocks):
            self.blocks[i] *= -1

    @staticmethod
    def zeros(bond_infos, pattern=None, dq=None, dtype=float):
        """Create tensor from tuple of BondInfo with zero elements."""
        blocks = []
        for sh, qs in SparseTensor._skeleton(bond_infos, pattern=pattern, dq=dq):
            blocks.append(SubTensor.zeros(shape=sh, q_labels=qs, dtype=dtype))
        return FermionSparseTensor(blocks=blocks)

    @staticmethod
    def ones(bond_infos, pattern=None, dq=None, dtype=float):
        """Create tensor from tuple of BondInfo with ones."""
        blocks = []
        for sh, qs in SparseTensor._skeleton(bond_infos, pattern=pattern, dq=dq):
            blocks.append(SubTensor.ones(shape=sh, q_labels=qs, dtype=dtype))
        return FermionSparseTensor(blocks=blocks)

    @staticmethod
    def random(bond_infos, pattern=None, dq=None, dtype=float):
        """Create tensor from tuple of BondInfo with random elements."""
        blocks = []
        for sh, qs in SparseTensor._skeleton(bond_infos, pattern=pattern, dq=dq):
            blocks.append(SubTensor.random(shape=sh, q_labels=qs, dtype=dtype))
        return FermionSparseTensor(blocks=blocks)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if ufunc in _sparse_tensor_numpy_func_impls:
            types = tuple(
                x.__class__ for x in inputs if not isinstance(x, numbers.Number))
            return self.__array_function__(ufunc, types, inputs, kwargs)
        out = kwargs.pop("out", None)
        if out is not None and not all(isinstance(x, self.__class__) for x in out):
            return NotImplemented

        if method == "__call__":
            if ufunc.__name__ in ["matmul"]:
                a, b = inputs
                if isinstance(a, numbers.Number):
                    blocks = [a * block for block in b.blocks]
                elif isinstance(b, numbers.Number):
                    blocks = [block * b for block in a.blocks]
                else:
                    blocks = self._tensordot(a, b, axes=([-1], [0])).blocks
            elif ufunc.__name__ in ["multiply", "divide", "true_divide"]:
                a, b = inputs
                if isinstance(a, numbers.Number):
                    blocks = [getattr(ufunc, method)(a, block)
                              for block in b.blocks]
                elif isinstance(b, numbers.Number):
                    blocks = [getattr(ufunc, method)(block, b)
                              for block in a.blocks]
                else:
                    return NotImplemented
            elif len(inputs) == 1:
                blocks = [getattr(ufunc, method)(block)
                          for block in inputs[0].blocks]
            else:
                return NotImplemented
        else:
            return NotImplemented
        if out is not None:
            out.blocks = blocks
        return self.__class__(blocks=blocks)

    @staticmethod
    def _hdot(a, b, out=None):
        """Horizontal contraction (contracting connected virtual dimensions)."""
        if isinstance(a, numbers.Number) or isinstance(b, numbers.Number):
            return np.multiply(a, b, out=out)

        assert isinstance(a, FermionSparseTensor) and isinstance(b, FermionSparseTensor)
        r = np.tensordot(a, b, axes=1)

        if out is not None:
            out.blocks = r.blocks

        return r

    @staticmethod
    def _pdot(a, b, out=None):
        """Vertical contraction (all middle/physical dims)."""
        if isinstance(a, numbers.Number) or isinstance(b, numbers.Number):
            return np.multiply(a, b, out=out)

        assert isinstance(a, FermionSparseTensor) and isinstance(b, FermionSparseTensor)

        assert a.ndim == b.ndim
        d = a.ndim - 2
        aidx = list(range(1, d + 1))
        bidx = list(range(1, d + 1))
        tr = (0, 2, 1, 3)

        r = np.tensordot(a, b, axes=(aidx, bidx)).transpose(axes=tr)

        if out is not None:
            out.blocks = r.blocks

        return r


class FlatFermionTensor(FlatSparseTensor):

    def __init__(self, *args, **kwargs):
        FlatSparseTensor.__init__(self, *args, **kwargs)
        self.check_sanity()

    @property
    def pairty(self):
        self.check_sanity()
        parity_uniq = np.unique(self.parity_list)
        return parity_uniq[0]

    @property
    def parity_per_block(self):
        parity_list = []
        for q_label in self.q_labels:
            pval = sum([SZ.from_flat(qlab).n for qlab in q_label])
            parity_list.append(int(pval)%2)
        return parity_list

    def check_sanity(self):
        parity_uniq = np.unique(self.parity_per_block)
        if len(parity_uniq) >1:
            raise ValueError("both odd/even parity detected in blocks, \
                              this is prohibited")

    def _local_flip(self, axes):
        if isinstance(axes, int):
            ax = [axes]
        else:
            ax = list(axes)
        idx = self.idxs
        for i in range(len(self.q_labels)):
            block_parity = sum([SZ.from_flat(self.q_labels[i][j]).n for j in ax]) % 2
            if block_parity == 1:
                self.data[idx[i]:idx[i+1]] *=-1

    def _global_flip(self):
        self.data *= -1

    def to_sparse(self):
        blocks = [None] * self.n_blocks
        for i in range(self.n_blocks):
            qs = tuple(map(SZ.from_flat, self.q_labels[i]))
            blocks[i] = SubTensor(
                self.data[self.idxs[i]:self.idxs[i + 1]].reshape(self.shapes[i]), q_labels=qs)
        return FermionSparseTensor(blocks=blocks)

    @staticmethod
    def get_zero():
        zu = np.zeros((0, 0), dtype=np.uint32)
        zd = np.zeros((0, ), dtype=float)
        iu = np.zeros((1, ), dtype=np.uint32)
        return FlatFermionTensor(zu, zu, zd, iu)

    @staticmethod
    def from_sparse(spt):
        ndim = spt.ndim
        n_blocks = spt.n_blocks
        shapes = np.zeros((n_blocks, ndim), dtype=np.uint32)
        q_labels = np.zeros((n_blocks, ndim), dtype=np.uint32)
        for i in range(n_blocks):
            shapes[i] = spt.blocks[i].shape
            q_labels[i] = list(map(SZ.to_flat, spt.blocks[i].q_labels))
        idxs = np.zeros((n_blocks + 1, ), dtype=np.uint32)
        idxs[1:] = np.cumsum(shapes.prod(axis=1))
        data = np.zeros((idxs[-1], ), dtype=np.float64)
        for i in range(n_blocks):
            data[idxs[i]:idxs[i + 1]] = spt.blocks[i].flatten()
        return FlatFermionTensor(q_labels, shapes, data, idxs)


    @staticmethod
    def zeros(bond_infos, pattern=None, dq=None, dtype=float):
        """Create tensor from tuple of BondInfo with zero elements."""
        qs, shs, idxs = flat_sparse_skeleton(bond_infos, pattern, dq)
        data = np.zeros((idxs[-1], ), dtype=dtype)
        return FlatFermionTensor(qs, shs, data, idxs)

    @staticmethod
    def ones(bond_infos, pattern=None, dq=None, dtype=float):
        """Create tensor from tuple of BondInfo with ones."""
        qs, shs, idxs = flat_sparse_skeleton(bond_infos, pattern, dq)
        data = np.ones((idxs[-1], ), dtype=dtype)
        return FlatFermionTensor(qs, shs, data, idxs)

    @staticmethod
    def random(bond_infos, pattern=None, dq=None, dtype=float):
        """Create tensor from tuple of BondInfo with random elements."""
        qs, shs, idxs = flat_sparse_skeleton(bond_infos, pattern, dq)
        if dtype == float:
            data = np.random.random((idxs[-1], ))
        elif dtype == complex:
            data = np.random.random(
                (idxs[-1], )) + np.random.random((idxs[-1], )) * 1j
        else:
            return NotImplementedError('dtype %r not supported!' % dtype)
        return FlatFermionTensor(qs, shs, data, idxs)

    def cast_assign(self, other):
        """assign other to self, removing blocks not in self."""
        aqs, adata, aidxs = other.q_labels, other.data, other.idxs
        bqs, bdata, bidxs = self.q_labels, self.data, self.idxs
        blocks_map = {q.tobytes(): iq for iq, q in enumerate(aqs)}
        for ib in range(self.n_blocks):
            q = bqs[ib].tobytes()
            if q in blocks_map:
                ia = blocks_map[q]
                bdata[bidxs[ib]:bidxs[ib + 1]] = adata[aidxs[ia]:aidxs[ia + 1]]
        return self

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if ufunc in _flat_sparse_tensor_numpy_func_impls:
            types = tuple(
                x.__class__ for x in inputs if not isinstance(x, numbers.Number))
            return self.__array_function__(ufunc, types, inputs, kwargs)
        out = kwargs.pop("out", None)
        if out is not None and not all(isinstance(x, self.__class__) for x in out):
            return NotImplemented

        if method == "__call__":
            if ufunc.__name__ in ["matmul"]:
                a, b = inputs
                if isinstance(a, numbers.Number):
                    shs, qs, data, idxs = b.shapes, b.q_labels, a * b.data, b.idxs
                elif isinstance(b, numbers.Number):
                    shs, qs, data, idxs = a.shapes, a.q_labels, a.data * b, a.idxs
                else:
                    c = self._tensordot(a, b, axes=([-1], [0]))
                    shs, qs, data, idxs = c.shapes, c.q_labels, c.data, c.idxs
            elif ufunc.__name__ in ["multiply", "divide", "true_divide"]:
                a, b = inputs
                if isinstance(a, numbers.Number):
                    shs, qs, data, idxs = b.shapes, b.q_labels, getattr(
                        ufunc, method)(a, b.data), b.idxs
                elif isinstance(b, numbers.Number):
                    shs, qs, data, idxs = a.shapes, a.q_labels, getattr(
                        ufunc, method)(a.data, b), a.idxs
                else:
                    return NotImplemented
            elif len(inputs) == 1:
                a = inputs[0]
                shs, qs, data, idxs = a.shapes, a.q_labels, getattr(
                    ufunc, method)(a.data), a.idxs
            else:
                return NotImplemented
        else:
            return NotImplemented
        if out is not None:
            out.shapes[...] = shs
            out.q_labels[...] = qs
            out.data[...] = data
            out.idxs[...] = idxs
        return FlatFermionTensor(q_labels=qs, shapes=shs, data=data, idxs=idxs)


    @staticmethod
    def _hdot(a, b, out=None):
        """Horizontal contraction (contracting connected virtual dimensions)."""
        if isinstance(a, numbers.Number) or isinstance(b, numbers.Number):
            return np.multiply(a, b, out=out)

        assert isinstance(a, FlatFermionTensor) and isinstance(
            b, FlatFermionTensor)
        r = np.tensordot(a, b, axes=1)

        if out is not None:
            out.q_labels[...] = r.q_labels
            out.shapes[...] = r.shapes
            out.data[...] = r.data
            out.idxs[...] = r.idxs

        return r



    @staticmethod
    def _pdot(a, b, out=None):
        """Vertical contraction (all middle/physical dims)."""
        if isinstance(a, numbers.Number) or isinstance(b, numbers.Number):
            return np.multiply(a, b, out=out)

        assert isinstance(a, FlatFermionTensor) and isinstance(
            b, FlatFermionTensor)

        assert a.ndim == b.ndim
        d = a.ndim - 2
        aidx = np.arange(1, d + 1, dtype=np.uint32)
        bidx = np.arange(1, d + 1, dtype=np.uint32)
        tr = (0, 2, 1, 3)

        r = np.tensordot(a, b, axes=(aidx, bidx)).transpose(axes=tr)

        if out is not None:
            out.q_labels[...] = r.q_labels
            out.shapes[...] = r.shapes
            out.data[...] = r.data
            out.idxs[...] = r.idxs

        return r
