import numpy as np
from .core import SparseTensor, SubTensor, _sparse_tensor_numpy_func_impls
from .flat import FlatSparseTensor, flat_sparse_skeleton, _flat_sparse_tensor_numpy_func_impls
from .symmetry import SZ
import numbers
from block3 import flat_fermion_tensor, flat_sparse_tensor
def method_alias(name):
    def ff(f):
        def fff(obj, *args, **kwargs):
            if hasattr(obj, name):
                if isinstance(getattr(obj, name), staticmethod):
                    return getattr(obj, name)(obj, *args, **kwargs)
                else:
                    return getattr(obj, name)(*args, **kwargs)
            else:
                return f(obj, *args, **kwargs)
        return fff
    return ff

def implements(np_func):
    global _numpy_func_impls
    return lambda f: (_numpy_func_impls.update({np_func: f})
                      if np_func not in _numpy_func_impls else None,
                      _numpy_func_impls[np_func])[1]

def compute_phase(q_labels, axes):
    nlist = [qlab.n for qlab in q_labels]
    counted = []
    phase = 1
    for x in axes:
        parity = sum([nlist[i] for i in range(x) if i not in counted]) * nlist[x]
        phase *= (-1) ** parity
        counted.append(x)
    return phase

NEW_METHODS = [np.transpose, np.tensordot]

_sparse_fermion_tensor_numpy_func_impls = _sparse_tensor_numpy_func_impls.copy()
[_sparse_fermion_tensor_numpy_func_impls.pop(key) for key in NEW_METHODS]
_numpy_func_impls = _sparse_fermion_tensor_numpy_func_impls

class SparseFermionTensor(SparseTensor):

    def __init__(self, *args, **kwargs):
        SparseTensor.__init__(self, *args, **kwargs)
        self.check_sanity()

    @property
    def pairty(self):
        self.check_sanity()
        parity_uniq = np.unique(self.parity_per_block)
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
        return SparseFermionTensor(blocks=blocks)

    @staticmethod
    def ones(bond_infos, pattern=None, dq=None, dtype=float):
        """Create tensor from tuple of BondInfo with ones."""
        blocks = []
        for sh, qs in SparseTensor._skeleton(bond_infos, pattern=pattern, dq=dq):
            blocks.append(SubTensor.ones(shape=sh, q_labels=qs, dtype=dtype))
        return SparseFermionTensor(blocks=blocks)

    @staticmethod
    def random(bond_infos, pattern=None, dq=None, dtype=float):
        """Create tensor from tuple of BondInfo with random elements."""
        blocks = []
        for sh, qs in SparseTensor._skeleton(bond_infos, pattern=pattern, dq=dq):
            blocks.append(SubTensor.random(shape=sh, q_labels=qs, dtype=dtype))
        return SparseFermionTensor(blocks=blocks)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if ufunc in _sparse_fermion_tensor_numpy_func_impls:
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

        assert isinstance(a, SparseFermionTensor) and isinstance(b, SparseFermionTensor)
        r = np.tensordot(a, b, axes=1)

        if out is not None:
            out.blocks = r.blocks

        return r

    @staticmethod
    def _pdot(a, b, out=None):
        """Vertical contraction (all middle/physical dims)."""
        if isinstance(a, numbers.Number) or isinstance(b, numbers.Number):
            return np.multiply(a, b, out=out)

        assert isinstance(a, SparseFermionTensor) and isinstance(b, SparseFermionTensor)

        assert a.ndim == b.ndim
        d = a.ndim - 2
        aidx = list(range(1, d + 1))
        bidx = list(range(1, d + 1))
        tr = (0, 2, 1, 3)

        r = np.tensordot(a, b, axes=(aidx, bidx)).transpose(axes=tr)

        if out is not None:
            out.blocks = r.blocks

        return r

    @staticmethod
    @implements(np.tensordot)
    def _tensordot(a, b, axes=2):
        if isinstance(axes, int):
            idxa, idxb = list(range(-axes, 0)), list(range(0, axes))
        else:
            idxa, idxb = axes
        idxa = [x if x >= 0 else a.ndim + x for x in idxa]
        idxb = [x if x >= 0 else b.ndim + x for x in idxb]
        out_idx_a = list(set(range(0, a.ndim)) - set(idxa))
        out_idx_b = list(set(range(0, b.ndim)) - set(idxb))
        assert len(idxa) == len(idxb)

        map_idx_b = {}
        for block in b.blocks:
            ctrq = tuple(block.q_labels[id] for id in idxb)
            outq = tuple(block.q_labels[id] for id in out_idx_b)
            if ctrq not in map_idx_b:
                map_idx_b[ctrq] = []
            map_idx_b[ctrq].append((block, outq))

        def _compute_phase(nlist, idx, direction='left'):
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

        blocks_map = {}
        for block_a in a.blocks:
            ptca = [i.n for i in block_a.q_labels]
            phase_a = _compute_phase(ptca, idxa, 'right')
            ctrq = tuple(block_a.q_labels[id] for id in idxa)
            if ctrq in map_idx_b:
                outqa = tuple(block_a.q_labels[id] for id in out_idx_a)
                for block_b, outqb in map_idx_b[ctrq]:
                    ptcb = [i.n for i in block_b.q_labels]
                    phase_b = _compute_phase(ptcb, idxb, 'left')
                    phase = phase_a * phase_b
                    outq = outqa + outqb
                    mat = np.tensordot(np.asarray(block_a), np.asarray(
                        block_b), axes=(idxa, idxb)).view(block_a.__class__)
                    if outq not in blocks_map:
                        mat.q_labels = outq
                        blocks_map[outq] = mat * phase
                    else:
                        blocks_map[outq] += mat * phase

        return a.__class__(blocks=list(blocks_map.values()))

    @staticmethod
    @implements(np.transpose)
    def _transpose(a, axes=None):
        phase = [compute_phase(block.q_labels, axes) for block in a.blocks]
        blocks = [np.transpose(block, axes=axes)*phase[ibk] for ibk, block in enumerate(a.blocks)]
        return a.__class__(blocks=blocks)

    def __array_function__(self, func, types, args, kwargs):
        if func not in _sparse_fermion_tensor_numpy_func_impls:
            return NotImplemented
        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented
        return _sparse_fermion_tensor_numpy_func_impls[func](*args, **kwargs)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if ufunc in _sparse_fermion_tensor_numpy_func_impls:
            types = tuple(
                x.__class__ for x in inputs if not isinstance(x, numbers.Number))
            return self.__array_function__(ufunc, types, inputs, kwargs)
        out = kwargs.pop("out", None)
        if out is not None and not all(isinstance(x, SparseFermionTensor) for x in out):
            return NotImplemented
        if any(isinstance(x, SparseFermionTensor) for x in inputs):
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

_flat_fermion_tensor_numpy_func_impls = _flat_sparse_tensor_numpy_func_impls.copy()
[_flat_fermion_tensor_numpy_func_impls.pop(key) for key in NEW_METHODS]
_numpy_func_impls = _flat_fermion_tensor_numpy_func_impls

class FlatFermionTensor(FlatSparseTensor):

    def __init__(self, *args, **kwargs):
        FlatSparseTensor.__init__(self, *args, **kwargs)
        self.check_sanity()

    @property
    def parity(self):
        self.check_sanity()
        parity_uniq = np.unique(self.parity_per_block)
        return parity_uniq[0]

    @property
    def parity_per_block(self):
        parity_list = []
        for q_label in self.q_labels:
            pval = sum([SZ.from_flat(qlab).n for qlab in q_label])
            parity_list.append(int(pval)%2)
        return parity_list

    @property
    def shape(self):
        return np.amax(self.shapes, axis=1)

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
        return SparseFermionTensor(blocks=blocks)

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

    def __array_function__(self, func, types, args, kwargs):
        if func not in _flat_fermion_tensor_numpy_func_impls:
            return NotImplemented
        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented
        return _flat_fermion_tensor_numpy_func_impls[func](*args, **kwargs)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if ufunc in _flat_fermion_tensor_numpy_func_impls:
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

    @staticmethod
    @implements(np.transpose)
    def _transpose(a, axes=None):
        if axes is None:
            axes = np.arange(a.ndim)[::-1]
        if a.n_blocks == 0:
            return a
        else:
            data = np.zeros_like(a.data)
            axes = np.array(axes, dtype=np.int32)
            flat_fermion_tensor.transpose(a.q_labels, a.shapes, a.data, a.idxs, axes, data)
            return a.__class__(a.q_labels[:,axes], a.shapes[:,axes], \
                               data, a.idxs)

    def permute(self, axes=None):
        if axes is None:
            axes = np.arange(a.ndim)[::-1]
        if self.n_blocks == 0:
            return self
        else:
            axes = np.array(axes, dtype=np.int32)
            data = np.zeros_like(self.data)
            flat_sparse_tensor.transpose(self.shapes, self.data, self.idxs, axes, data)
            return self.__class__(self.q_labels[:, axes], self.shapes[:, axes], data, self.idxs)


    @staticmethod
    @implements(np.tensordot)
    def _tensordot(a, b, axes=2):
        if isinstance(axes, int):
            idxa = np.arange(-axes, 0, dtype=np.int32)
            idxb = np.arange(0, axes, dtype=np.int32)
        else:
            idxa = np.array(axes[0], dtype=np.int32)
            idxb = np.array(axes[1], dtype=np.int32)
        idxa[idxa < 0] += a.ndim
        idxb[idxb < 0] += b.ndim
        return a.__class__(*flat_fermion_tensor.tensordot(
            a.q_labels, a.shapes, a.data, a.idxs,
            b.q_labels, b.shapes, b.data, b.idxs,
            idxa, idxb))

    @staticmethod
    @implements(np.linalg.norm)
    def _norm(a):
        return np.linalg.norm(a.data)

    @staticmethod
    @implements(np.reshape)
    def _reshape(a, newshape, *args, **kwargs):
        return np.reshape(a.data, newshape, *args, **kwargs)

    def reshape(self, newshape, *args, **kwargs):
        return np.reshape(self, newshape, *args, **kwargs)

    def astype(self, dtype, *args, **kwargs):
        return self.__class__(self.q_labels, self.shapes, \
                              self.data.astype(dtype, *args, **kwargs), \
                              self.idxs)
