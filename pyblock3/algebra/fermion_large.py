
import numpy as np

def get_xp_backend(use_cupy):
    if use_cupy:
        import cupy as xp
        to_np = xp.asnumpy
    else:
        import numpy as xp
        to_np = xp.asarray
    return xp, to_np

from . import fermion_setting as setting
from .fermion import FlatFermionTensor, _flip_pattern, _flat_fermion_tensor_numpy_func_impls
from .fermion import Q_LABELS_DTYPE, SHAPES_DTYPE
from .fermion import SparseFermionTensor, SubTensor

def implements(np_func):
    global _numpy_func_impls
    return lambda f: (_numpy_func_impls.update({np_func: f})
                      if np_func not in _numpy_func_impls else None,
                      _numpy_func_impls[np_func])[1]

NEW_METHODS = [np.transpose, np.tensordot, np.add, np.subtract, np.copy]

_large_fermion_tensor_numpy_func_impls = _flat_fermion_tensor_numpy_func_impls.copy()
[_large_fermion_tensor_numpy_func_impls.pop(key) for key in NEW_METHODS]
_numpy_func_impls = _large_fermion_tensor_numpy_func_impls

class LargeFermionTensor(FlatFermionTensor):

    def __init__(self, q_labels, shapes, data,
                 pattern=None, idxs=None, symmetry=None, shape=None, use_cupy=None):
        assert idxs is not None
        super().__init__(q_labels, shapes, data,
            pattern=pattern, idxs=idxs, symmetry=symmetry, shape=shape)
        self.use_cupy = setting.dispatch_settings(cupy=use_cupy)

    @property
    def dagger(self):
        axes = list(range(self.ndim))[::-1]
        axes = np.array(axes, dtype=np.int32)
        xp, _ = get_xp_backend(self.use_cupy)
        data = xp.transpose(self.data.conj(), axes)
        # FIXME missing phase
        return self.__class__(self.q_labels[:, axes], self.shapes[:, axes], data,
            pattern=_flip_pattern(self.pattern[::-1]),
            idxs=self.idxs[:, axes], symmetry=self.symmetry, use_cupy=self.use_cupy,
            shape=self.shape[::-1])

    @staticmethod
    @implements(np.copy)
    def _copy(x):
        return x.__class__(q_labels=x.q_labels.copy(order="K"), shapes=x.shapes.copy(order="K"),
            data=x.data.copy(), pattern=x.pattern, idxs=x.idxs.copy(),
            symmetry=x.symmetry, shape=x.shape, use_cupy=x.use_cupy)

    def new_like(self, data, **kwargs):
        q_labels = kwargs.pop("q_labels", self.q_labels)
        shapes = kwargs.pop("shapes", self.shapes)
        pattern = kwargs.pop("pattern", self.pattern)
        idxs = kwargs.pop("idxs", self.idxs)
        symmetry = kwargs.pop("symmetry", self.symmetry)
        use_cupy = kwargs.pop("use_cupy", self.use_cupy)
        shape = kwargs.pop("shape", self.shape)
        return self.__class__(q_labels, shapes, data, pattern=pattern, idxs=idxs,
            symmetry=symmetry, shape=shape, use_cupy=use_cupy)

    def conj(self):
        return self.new_like(self.data.conj())

    def _local_flip(self, axes):
        if not setting.DEFAULT_FERMION: return
        if isinstance(axes, int):
            axes = [axes]
        idx = self.idxs
        q_labels = np.stack([self.q_labels[:,ix] for ix in axes], axis=1)
        pattern = "".join([self.pattern[ix] for ix in axes])
        net_q = self.symmetry._compute(pattern, q_labels)
        parities = self.symmetry.flat_to_parity(net_q)
        inds = np.where(parities==1)[0]
        for i in inds:
            self.data[idx[i]:idx[i+1]] *=-1

    def _global_flip(self):
        if not setting.DEFAULT_FERMION: return
        self.data *= -1

    def to_sparse(self):
        blocks = [None] * self.n_blocks
        _, to_np = get_xp_backend(self.use_cupy)
        for i in range(self.n_blocks):
            qs = tuple(map(self.symmetry.from_flat, self.q_labels[i]))
            idata = tuple(slice(xi, xi + xe) for xi, xe in zip(self.idxs[i], self.shapes[i]))
            subdata = to_np(self.data[idata])
            blocks[i] = SubTensor(subdata, q_labels=qs)
        return SparseFermionTensor(blocks=blocks, pattern=self.pattern, shape=self.shape)

    @staticmethod
    def from_flat(spt, use_cupy=None):
        use_cupy = setting.dispatch_settings(cupy=use_cupy)
        xp, _ = get_xp_backend(use_cupy)
        data = xp.zeros(spt.shape, dtype=spt.dtype)
        ndim = spt.ndim
        n_blocks = spt.n_blocks
        idxs = np.zeros((n_blocks, ndim), dtype=SHAPES_DTYPE)
        infos = spt.infos
        for i in range(spt.ndim):
            x = 0
            for xx in sorted(list(infos[i].keys())):
                x += infos[i][xx]
                infos[i][xx] = x - infos[i][xx]
        for i in range(spt.n_blocks):
            idxs[i] = [ixx[qxx] for ixx, qxx in zip(infos, spt.q_labels[i])]
            idata = tuple(slice(xi, xi + xe) for xi, xe in zip(idxs[i], spt.shapes[i]))
            data[idata] = xp.asarray(spt.data[spt.idxs[i]:spt.idxs[i + 1]].reshape(spt.shapes[i]))
        return LargeFermionTensor(spt.q_labels, spt.shapes, data, spt.pattern, idxs, symmetry=spt.symmetry,
            shape=spt.shape, use_cupy=use_cupy)

    @staticmethod
    def from_sparse(spt, use_cupy=None):
        use_cupy = setting.dispatch_settings(cupy=use_cupy)
        xp, _ = get_xp_backend(use_cupy)
        data = xp.zeros(spt.shape, dtype=spt.dtype)
        ndim = spt.ndim
        n_blocks = spt.n_blocks
        idxs = np.zeros((n_blocks, ndim), dtype=SHAPES_DTYPE)
        shapes = np.zeros((n_blocks, ndim), dtype=SHAPES_DTYPE)
        q_labels = np.zeros((n_blocks, ndim), dtype=Q_LABELS_DTYPE)
        cls = spt.blocks[0].q_labels[0].__class__
        infos = spt.infos
        for i in range(spt.ndim):
            x = 0
            for xx in sorted(list(map(cls.to_flat, infos[i].keys()))):
                x += infos[i][xx]
                infos[i][xx] = x - infos[i][xx]
        for i in range(n_blocks):
            shapes[i] = spt.blocks[i].shape
            q_labels[i] = list(map(cls.to_flat, spt.blocks[i].q_labels))
            idxs[i] = [ixx[qxx] for ixx, qxx in zip(infos, q_labels[i])]
            idata = tuple(slice(xi, xi + xe) for xi, xe in zip(idxs[i], shapes[i]))
            data[idata] = xp.asarray(spt.blocks[i])
        return LargeFermionTensor(q_labels, shapes, data, spt.pattern, idxs, symmetry=cls,
            shape=spt.shape, use_cupy=use_cupy)
