
from . import fermion_setting as setting
from .fermion import SparseFermionTensor, SubTensor
from .fermion import Q_LABELS_DTYPE, SHAPES_DTYPE, INDEX_DTYPE, SVD_SCREENING, get_backend
from .fermion import FlatFermionTensor, _flip_pattern, _flat_fermion_tensor_numpy_func_impls
from .fermion import _adjust_q_labels, _contract_patterns, _trim_and_renorm_SVD
from .fermion import _maybe_transpose_tensor, _gen_null_qr_info, _svd_preprocess
from .fermion import _absorb_svd, timing
import numpy as np


def get_xp_backend(use_cupy):
    if use_cupy:
        import cupy as xp
        to_np = xp.asnumpy
    else:
        import numpy as xp
        to_np = xp.asarray
    return xp, to_np


def implements(np_func):
    global _numpy_func_impls
    return lambda f: (_numpy_func_impls.update({np_func: f})
                      if np_func not in _numpy_func_impls else None,
                      _numpy_func_impls[np_func])[1]


NEW_METHODS = [np.transpose, np.tensordot, np.add, np.subtract, np.copy]

_large_fermion_tensor_numpy_func_impls = _flat_fermion_tensor_numpy_func_impls.copy()
[_large_fermion_tensor_numpy_func_impls.pop(key) for key in NEW_METHODS]
_numpy_func_impls = _large_fermion_tensor_numpy_func_impls

def large_svd(T, left_idx, right_idx=None, qpn_partition=None, **opts):
    """Perform tensor SVD on LargeFermionTensor

    Parameters
    ----------
    T : LargeFermionTensor
    left_idx : tuple/list of int
        left indices for SVD
    right_idx : optional, tuple/list of int
        right indices for SVD
    qpn_partition: optional, tuple/list of symmetry object
        partition of symmetry on the left and right,
        must add up to total symmetry of the input tensor

    Returns
    -------
    u: LargeFermionTensor
    s: LargeFermionTensor or None
    v: LargeFermionTensor
    """
    absorb = opts.pop("absorb", 0)
    if right_idx is None:
        right_idx = [idim for idim in range(T.ndim)
                        if idim not in left_idx]
    new_T, qpn_partition, symmetry = _svd_preprocess(
                T, left_idx, right_idx,
                qpn_partition, absorb)
    if max(len(left_idx), len(right_idx)) == T.ndim:
        raise NotImplementedError

    split_ax = len(left_idx)
    left_q = symmetry._compute(new_T.pattern[:split_ax], new_T.q_labels[:,:split_ax],
        offset=("-", qpn_partition[0]))
    right_q = symmetry._compute(new_T.pattern[split_ax:], new_T.q_labels[:,split_ax:],
        offset=("-", qpn_partition[1]), neg=True)

    aux_q = list(set(np.unique(left_q)) & set(np.unique(right_q)))

    full_left_qs = np.hstack([new_T.q_labels[:,:split_ax],
                              left_q.reshape(-1,1)])
    full_right_qs = np.hstack([left_q.reshape(-1,1),
                               new_T.q_labels[:,split_ax:]])
    full_qs = [(tuple(il), tuple(ir)) for il, ir \
            in zip(full_left_qs, full_right_qs)]

    row_shapes = np.prod(new_T.shapes[:,:split_ax],
                         axis=1, dtype=int)
    col_shapes = np.prod(new_T.shapes[:,split_ax:],
                         axis=1, dtype=int)
    s_data =[]
    uv_data = []
    all_maps = []
    strides = np.ones((new_T.ndim, ), dtype=INDEX_DTYPE)
    np.cumprod(new_T.shape[:0:-1], dtype=INDEX_DTYPE, out=strides[-2::-1])
    xp, _ = get_xp_backend(new_T.use_cupy)
    
    # finding data mapping to correpsonding symmetry sector
    for sblk_q_label in sorted(aux_q):
        blocks = np.where(left_q == sblk_q_label)[0]
        row_map, col_map = {}, {}
        row_len, col_len = 0, 0
        alldatas = {}
        for iblk in blocks:
            lq, rq = full_qs[iblk]
            if lq not in row_map:
                new_row_len = row_shapes[iblk] + row_len
                row_map[lq] = (row_len, new_row_len, new_T.shapes[iblk,:split_ax],
                    np.int64(new_T.idxs[iblk] // strides[split_ax - 1]))
                ist, ied = row_len, new_row_len
                row_len = new_row_len
            else:
                ist, ied = row_map[lq][:2]
            if rq not in col_map:
                new_col_len = col_shapes[iblk] + col_len
                col_map[rq] = (col_len, new_col_len, new_T.shapes[iblk,split_ax:],
                    np.int64(new_T.idxs[iblk] % strides[split_ax - 1]))
                jst, jed = col_len, new_col_len
                col_len = new_col_len
            else:
                jst, jed = col_map[rq][:2]
            pidx = [np.uint64(new_T.idxs[iblk] // st % sh) for st, sh in zip(strides, new_T.shape)]
            idata = tuple(slice(xi, xi + xe) for xi, xe in zip(pidx, new_T.shapes[iblk]))
            alldatas[(ist, ied, jst, jed)] = new_T.data[idata].reshape(ied - ist, jed - jst)

        data = xp.zeros([row_len, col_len], dtype=new_T.dtype)
        if data.size == 0:
            continue
        for (ist, ied, jst, jed), val in alldatas.items():
            data[ist:ied, jst:jed] = val

        u, s, v = xp.linalg.svd(data, full_matrices=False)
        ind = s > SVD_SCREENING
        s = s[ind]
        if s.size == 0:
            continue
        u = u[:, ind]
        v = v[ind, :]
        s_data.append(s)
        uv_data.append([u, v])
        all_maps.append([sblk_q_label, row_map, col_map])

    # truncate the singular values for each block
    s_data, uv_data = _trim_and_renorm_SVD(s_data, uv_data, **opts)

    if absorb is not None:
        for iblk in range(len(uv_data)):
            s = s_data[iblk]
            if s.size == 0:
                continue
            u, v = uv_data[iblk]
            u, s, v = _absorb_svd(u, s, v, absorb)
            uv_data[iblk] = (u, v)
            s_data[iblk] = s

    udata, vdata = [], []
    qu, qv, qs = [], [], []
    shu, shv, shs = [], [], []
    ixu, ixv, ixs = [], [], []


    imid = 0
    mid_shape = np.sum([u.shape[-1] for u, _ in uv_data])
    sd = xp.zeros((mid_shape, mid_shape), dtype=new_T.dtype)
    for s, (u, v), (sblk_q_label, row_map, col_map) in zip(s_data, uv_data, all_maps):
        if u.size == 0:
            continue
        if absorb is None:
            sd[imid:imid + s.shape[0], imid:imid + s.shape[0]] = xp.diag(s)
            shs.append(s.shape * 2)
            ixs.append(imid * (mid_shape + 1))
            qs.append([sblk_q_label, sblk_q_label])

        for lq, (lst, led, lsh, lix) in row_map.items():
            udata.append(u[lst:led])
            qu.append(lq)
            ixu.append(lix * mid_shape + imid)
            shu.append(tuple(lsh) + (u.shape[-1], ))

        for rq, (rst, red, rsh, rix) in col_map.items():
            vdata.append(v[:, rst:red])
            qv.append(rq)
            ixv.append(rix + imid * strides[split_ax - 1])
            shv.append((v.shape[0], ) + tuple(rsh))
        
        imid += u.shape[-1]
    assert imid == mid_shape

    if absorb is None:
        qs = np.asarray(qs, dtype=Q_LABELS_DTYPE)
        shs = np.asarray(shs, dtype=SHAPES_DTYPE)
        s = T.__class__(qs, shs, sd, pattern="+-", idxs=ixs, symmetry=T.symmetry,
            shape=sd.shape, use_cupy=new_T.use_cupy)
    else:
        s = None

    qu = np.asarray(qu, dtype=Q_LABELS_DTYPE)
    shu = np.asarray(shu, dtype=SHAPES_DTYPE)
    ixu = np.asarray(ixu, dtype=INDEX_DTYPE)
    qv = np.asarray(qv, dtype=Q_LABELS_DTYPE)
    shv = np.asarray(shv, dtype=SHAPES_DTYPE)
    ixv = np.asarray(ixv, dtype=INDEX_DTYPE)

    ushape = new_T.shape[:split_ax] + (mid_shape, )
    vshape = (mid_shape, ) + new_T.shape[split_ax:]
    ud = xp.zeros(ushape, dtype=new_T.dtype)
    vd = xp.zeros(vshape, dtype=new_T.dtype)

    ustrides = np.ones((len(ushape), ), dtype=INDEX_DTYPE)
    np.cumprod(ushape[:0:-1], dtype=INDEX_DTYPE, out=ustrides[-2::-1])
    vstrides = np.ones((len(vshape), ), dtype=INDEX_DTYPE)
    np.cumprod(vshape[:0:-1], dtype=INDEX_DTYPE, out=vstrides[-2::-1])

    for d, di, dh in zip(udata, ixu, shu):
        pidx = [np.uint64(di // st % sh) for st, sh in zip(ustrides, ushape)]
        idata = tuple(slice(xi, xi + xe) for xi, xe in zip(pidx, dh))
        ud[idata] = d.reshape(dh)

    for d, di, dh in zip(vdata, ixv, shv):
        pidx = [np.uint64(di // st % sh) for st, sh in zip(vstrides, vshape)]
        idata = tuple(slice(xi, xi + xe) for xi, xe in zip(pidx, dh))
        vd[idata] = d.reshape(dh)

    u_pattern = new_T.pattern[:split_ax] + "-"
    v_pattern = "+" + new_T.pattern[split_ax:]

    u = T.__class__(qu, shu, ud, pattern=u_pattern, idxs=ixu,
        symmetry=T.symmetry, use_cupy=new_T.use_cupy, shape=ushape)
    v = T.__class__(qv, shv, vd, pattern=v_pattern, idxs=ixv,
        symmetry=T.symmetry, use_cupy=new_T.use_cupy, shape=vshape)
    return u, s, v

def large_qr(T, left_idx, right_idx=None, mod="qr"):
    """Perform tensor QR on LargeFermionTensor, this will partition
    the quantum number fully on Q and net zero symmetry on R

    Parameters
    ----------
    T : LargeFermionTensor
    left_idx : tuple/list of int
        left indices for QR
    right_idx : optional, tuple/list of int
        right indices for QR
    mod: optional, {"qr", "lq"}

    """
    assert mod in ["qr", "lq"]
    if right_idx is None:
        right_idx = [idim for idim in range(T.ndim) if idim not in left_idx]
    new_T = _maybe_transpose_tensor(T, left_idx, right_idx)
    if len(left_idx) == T.ndim or len(right_idx) == T.ndim:
        flat_q = T.dq.to_flat()
        flat_qs = np.asarray([[flat_q]], dtype=Q_LABELS_DTYPE)
        ishapes = np.asarray([[1,]], dtype=SHAPES_DTYPE)
        iidxs = np.asarray([0, 1], dtype=INDEX_DTYPE)
        data = np.asarray([1,])
        Q = T.__class__(flat_qs, ishapes, data, idxs=iidxs,
                        pattern="+", symmetry=T.symmetry)
        new_pattern, inds, return_order = _gen_null_qr_info(T, mod)
        new_shapes = np.insert(T.shapes, inds, 1, axis=1)
        new_q_labels = np.insert(T.q_labels, inds, flat_q, axis=1)
        if return_order == slice(None):
            shape = (1, ) + T.shape
        else:
            shape = T.shape + (1, )
        R = T.__class__(new_q_labels, new_shapes,
                        T.data.copy(), pattern=new_pattern,
                        idxs=T.idxs.copy(), symmetry=T.symmetry, shape=shape)
        return (Q, R)[return_order]
    symmetry = T.dq.__class__
    dq = {"lq": symmetry(0), "qr": T.dq}[mod]
    split_ax = len(left_idx)
    left_q = symmetry._compute(new_T.pattern[:split_ax],
                               new_T.q_labels[:,:split_ax],
                               offset=("-", dq))
    aux_q = list(set(np.unique(left_q)))
    full_left_qs = np.hstack([new_T.q_labels[:, :split_ax],
                              left_q.reshape(-1,1)])
    full_right_qs = np.hstack([left_q.reshape(-1,1),
                               new_T.q_labels[:, split_ax:]])
    full_qs = [(tuple(il), tuple(ir)) for il, ir in
                    zip(full_left_qs, full_right_qs)]
    qdata, rdata = [], []
    qq, qr = [], []
    shq, shr = [], []
    ixq, ixr = [], []
    row_shapes = np.prod(new_T.shapes[:, :split_ax], axis=1, dtype=int)
    col_shapes = np.prod(new_T.shapes[:, split_ax:], axis=1, dtype=int)
    strides = np.ones((new_T.ndim, ), dtype=INDEX_DTYPE)
    np.cumprod(new_T.shape[:0:-1], dtype=INDEX_DTYPE, out=strides[-2::-1])
    xp, _ = get_xp_backend(new_T.use_cupy)
    mid_shape = 0
    mid_qmap = {}
    mid_max = np.prod(new_T.shape[:split_ax], dtype=INDEX_DTYPE)
    for sblk_q_label in sorted(aux_q):
        blocks = np.where(left_q == sblk_q_label)[0]
        row_map, col_map = {}, {}
        row_len, col_len = 0, 0
        alldatas = {}
        for iblk in blocks:
            lq, rq = full_qs[iblk]
            if lq not in row_map:
                new_row_len = row_shapes[iblk] + row_len
                row_map[lq] = (row_len, new_row_len, new_T.shapes[iblk, :split_ax],
                    np.int64(new_T.idxs[iblk] // strides[split_ax - 1]))
                ist, ied = row_len, new_row_len
                row_len = new_row_len
            else:
                ist, ied = row_map[lq][:2]
            if rq not in col_map:
                new_col_len = col_shapes[iblk] + col_len
                col_map[rq] = (col_len, new_col_len, new_T.shapes[iblk, split_ax:],
                    np.int64(new_T.idxs[iblk] % strides[split_ax - 1]))
                jst, jed = col_len, new_col_len
                col_len = new_col_len
            else:
                jst, jed = col_map[rq][:2]
            pidx = [np.uint64(new_T.idxs[iblk] // st % sh) for st, sh in zip(strides, new_T.shape)]
            idata = tuple(slice(xi, xi + xe) for xi, xe in zip(pidx, new_T.shapes[iblk]))
            alldatas[(ist, ied, jst, jed)] = new_T.data[idata].reshape(ied - ist, jed - jst)

        data = xp.zeros([row_len, col_len], dtype=new_T.dtype)
        if data.size == 0:
            continue
        for (ist, ied, jst, jed), val in alldatas.items():
            data[ist:ied, jst:jed] = val

        if mod == "qr":
            q, r = xp.linalg.qr(data)
        else:
            r, q = xp.linalg.qr(data.T)
            q, r = q.T, r.T

        mid_qmap[sblk_q_label] = mid_shape

        for lq, (lst, led, lsh, lix) in row_map.items():
            qdata.append(q[lst:led])
            qq.append(lq)
            ixq.append(lix * mid_max + mid_shape)
            shq.append(tuple(lsh) + (q.shape[-1],))

        for rq, (rst, red, rsh, rix) in col_map.items():
            rdata.append(r[:, rst:red])
            qr.append(rq)
            ixr.append(rix + mid_shape * strides[split_ax - 1])
            shr.append((r.shape[0], ) + tuple(rsh))

        mid_shape += q.shape[-1]

    qq = np.asarray(qq, dtype=Q_LABELS_DTYPE)
    shq = np.asarray(shq, dtype=SHAPES_DTYPE)
    ixq = np.asarray(ixq, dtype=INDEX_DTYPE)
    qr = np.asarray(qr, dtype=Q_LABELS_DTYPE)
    shr = np.asarray(shr, dtype=SHAPES_DTYPE)
    ixr = np.asarray(ixr, dtype=INDEX_DTYPE)

    qshape = new_T.shape[:split_ax] + (mid_shape, )
    rshape = (mid_shape, ) + new_T.shape[split_ax:]
    qd = xp.zeros(qshape, dtype=new_T.dtype)
    rd = xp.zeros(rshape, dtype=new_T.dtype)
    ixq = ixq // mid_max * mid_shape + ixq % mid_max

    qstrides = np.ones((len(qshape), ), dtype=INDEX_DTYPE)
    np.cumprod(qshape[:0:-1], dtype=INDEX_DTYPE, out=qstrides[-2::-1])
    rstrides = np.ones((len(rshape), ), dtype=INDEX_DTYPE)
    np.cumprod(rshape[:0:-1], dtype=INDEX_DTYPE, out=rstrides[-2::-1])

    for d, di, dh in zip(qdata, ixq, shq):
        pidx = [np.uint64(di // st % sh) for st, sh in zip(qstrides, qshape)]
        idata = tuple(slice(xi, xi + xe) for xi, xe in zip(pidx, dh))
        qd[idata] = d.reshape(dh)

    for d, di, dh in zip(rdata, ixr, shr):
        pidx = [np.uint64(di // st % sh) for st, sh in zip(rstrides, rshape)]
        idata = tuple(slice(xi, xi + xe) for xi, xe in zip(pidx, dh))
        rd[idata] = d.reshape(dh)

    q_pattern = new_T.pattern[:split_ax] + "-"
    r_pattern = "+" + new_T.pattern[split_ax:]
    q = T.__class__(qq, shq, qd, pattern=q_pattern, idxs=ixq,
        symmetry=T.symmetry, use_cupy=new_T.use_cupy, shape=qshape)
    r = T.__class__(qr, shr, rd, pattern=r_pattern, idxs=ixr,
        symmetry=T.symmetry, use_cupy=new_T.use_cupy, shape=rshape)
    return q, r

class LargeFermionTensor(FlatFermionTensor):

    def __init__(self, q_labels, shapes, data,
                 pattern=None, idxs=None, symmetry=None, shape=None, use_cupy=None):
        assert idxs is not None
        super().__init__(q_labels, shapes, data,
                         pattern=pattern, idxs=idxs, symmetry=symmetry, shape=shape)
        self.use_cupy = setting.dispatch_settings(cupy=use_cupy)

    @property
    def dagger(self):
        r = self._transpose(self.conj(), do_fermi=False)
        r.pattern = _flip_pattern(self.pattern[::-1])
        return r

    def to_constructor(self, axes):
        return NotImplemented

    @staticmethod
    @implements(np.copy)
    @timing("copy")
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
        if not setting.DEFAULT_FERMION:
            return
        if isinstance(axes, int):
            axes = [axes]
        idx = self.idxs
        q_labels = np.stack([self.q_labels[:, ix] for ix in axes], axis=1)
        pattern = "".join([self.pattern[ix] for ix in axes])
        net_q = self.symmetry._compute(pattern, q_labels)
        parities = self.symmetry.flat_to_parity(net_q)
        inds = np.where(parities == 1)[0]
        strides = np.ones((self.ndim, ), dtype=INDEX_DTYPE)
        np.cumprod(self.shape[:0:-1], dtype=INDEX_DTYPE, out=strides[-2::-1])
        for i in inds:
            pidx = [np.uint64(self.idxs[i] // st % sh) for st, sh in zip(strides, self.shape)]
            idata = tuple(slice(xi, xi + xe) for xi, xe in zip(pidx, self.shapes[i]))
            self.data[idata] *= -1

    def _global_flip(self):
        if not setting.DEFAULT_FERMION:
            return
        self.data *= -1

    def to_sparse(self):
        blocks = [None] * self.n_blocks
        strides = np.ones((self.ndim, ), dtype=INDEX_DTYPE)
        np.cumprod(self.shape[:0:-1], dtype=INDEX_DTYPE, out=strides[-2::-1])
        _, to_np = get_xp_backend(self.use_cupy)
        for i in range(self.n_blocks):
            qs = tuple(map(self.symmetry.from_flat, self.q_labels[i]))
            pidx = [np.uint64(self.idxs[i] // st % sh) for st, sh in zip(strides, self.shape)]
            idata = tuple(slice(xi, xi + xe)
                          for xi, xe in zip(pidx, self.shapes[i]))
            subdata = to_np(self.data[idata])
            blocks[i] = SubTensor(subdata, q_labels=qs)
        return SparseFermionTensor(blocks=blocks, pattern=self.pattern, shape=self.shape)
    
    def to_flat(self):
        idxs = np.zeros((self.n_blocks + 1, ), dtype=np.uint64)
        idxs[1:] = np.cumsum(self.shapes.prod(axis=1), dtype=np.uint64)
        data = np.zeros((idxs[-1], ), dtype=self.data.dtype)
        strides = np.ones((self.ndim, ), dtype=INDEX_DTYPE)
        np.cumprod(self.shape[:0:-1], dtype=INDEX_DTYPE, out=strides[-2::-1])
        _, to_np = get_xp_backend(self.use_cupy)
        for i in range(self.n_blocks):
            pidx = [np.uint64(self.idxs[i] // st % sh) for st, sh in zip(strides, self.shape)]
            idata = tuple(slice(xi, xi + xe)
                          for xi, xe in zip(pidx, self.shapes[i]))
            data[idxs[i]:idxs[i + 1]] = np.ravel(to_np(self.data[idata]))
        return FlatFermionTensor(self.q_labels, self.shapes, data, self.pattern, idxs,
                                 symmetry=self.symmetry, shape=self.shape)

    @staticmethod
    def from_flat(spt, use_cupy=None, infos=None):
        use_cupy = setting.dispatch_settings(cupy=use_cupy)
        xp, _ = get_xp_backend(use_cupy)
        data = xp.zeros(spt.shape, dtype=spt.dtype)
        strides = np.ones((spt.ndim, ), dtype=INDEX_DTYPE)
        np.cumprod(spt.shape[:0:-1], dtype=INDEX_DTYPE, out=strides[-2::-1])
        idxs = np.zeros_like(spt.idxs)
        if infos is None:
            infos = spt.infos
        else:
            infos = infos.__class__(infos)
        for i in range(spt.ndim):
            x = 0
            for xx in sorted(list(infos[i].keys())):
                x += infos[i][xx]
                infos[i][xx] = x - infos[i][xx]
        for i in range(spt.n_blocks):
            pidx = [ixx[qxx] for ixx, qxx in zip(infos, spt.q_labels[i])]
            idxs[i] = np.sum(pidx * strides)
            idata = tuple(slice(xi, xi + xe)
                          for xi, xe in zip(pidx, spt.shapes[i]))
            data[idata] = xp.asarray(
                spt.data[spt.idxs[i]:spt.idxs[i + 1]].reshape(spt.shapes[i]))
        idxs[-1] = data.size
        return LargeFermionTensor(spt.q_labels, spt.shapes, data, spt.pattern, idxs, symmetry=spt.symmetry,
                                  shape=spt.shape, use_cupy=use_cupy)

    @staticmethod
    def from_sparse(spt, use_cupy=None):
        use_cupy = setting.dispatch_settings(cupy=use_cupy)
        xp, _ = get_xp_backend(use_cupy)
        data = xp.zeros(spt.shape, dtype=spt.dtype)
        strides = np.ones((spt.ndim, ), dtype=INDEX_DTYPE)
        np.cumprod(spt.shape[:0:-1], dtype=INDEX_DTYPE, out=strides[-2::-1])
        idxs = np.zeros_like(spt.idxs)
        shapes = np.zeros((spt.n_blocks, spt.ndim), dtype=SHAPES_DTYPE)
        q_labels = np.zeros((spt.n_blocks, spt.ndim), dtype=Q_LABELS_DTYPE)
        cls = spt.blocks[0].q_labels[0].__class__
        infos = spt.infos
        for i in range(spt.ndim):
            x = 0
            for xx in sorted(list(map(cls.to_flat, infos[i].keys()))):
                x += infos[i][xx]
                infos[i][xx] = x - infos[i][xx]
        for i in range(spt.n_blocks):
            shapes[i] = spt.blocks[i].shape
            q_labels[i] = list(map(cls.to_flat, spt.blocks[i].q_labels))
            pidx = np.array([ixx[qxx] for ixx, qxx in zip(infos, q_labels[i])])
            idxs[i] = np.sum(pidx * strides)
            idata = tuple(slice(xi, xi + xe) for xi, xe in zip(pidx, shapes[i]))
            data[idata] = xp.asarray(spt.blocks[i])
        idxs[-1] = data.size
        return LargeFermionTensor(q_labels, shapes, data, spt.pattern, idxs, symmetry=cls,
                                  shape=spt.shape, use_cupy=use_cupy)

    @staticmethod
    @implements(np.add)
    def _add(a, b):
        import numbers
        if isinstance(a, numbers.Number):
            data = a + b.data
            return b.new_like(data)
        elif isinstance(b, numbers.Number):
            data = a.data + b
            return a.new_like(data)
        elif b.n_blocks == 0:
            return a
        elif a.n_blocks == 0:
            return b
        else:
            return NotImplemented
            # flip_axes = [ix for ix in range(b.ndim) if a.pattern[ix] != b.pattern[ix]]
            # q_labels_b = _adjust_q_labels(b.symmetry, b.q_labels, flip_axes)
            # q_labels, shapes, data, idxs = _block3.flat_sparse_tensor.add(a.q_labels, a.shapes, a.data,
            #                                     a.idxs, q_labels_b, b.shapes, b.data, b.idxs)
            # return a.__class__(q_labels, shapes, data, a.pattern, idxs, a.symmetry)

    def add(self, b):
        return self._add(self, b)

    @staticmethod
    @implements(np.subtract)
    def _subtract(a, b):
        import numbers
        if isinstance(a, numbers.Number):
            data = a - b.data
            return b.__class__(b.q_labels, b.shapes, data, b.pattern, b.idxs, b.symmetry)
        elif isinstance(b, numbers.Number):
            data = a.data - b
            return a.__class__(a.q_labels, a.shapes, data, a.pattern, a.idxs, a.symmetry)
        elif b.n_blocks == 0:
            return a
        elif a.n_blocks == 0:
            return b
        else:
            return NotImplemented
            # flip_axes = [ix for ix in range(b.ndim) if a.pattern[ix]!=b.pattern[ix]]
            # q_labels_b = _adjust_q_labels(b.symmetry, b.q_labels, flip_axes)
            # q_labels, shapes, data, idxs = _block3.flat_sparse_tensor.add(a.q_labels, a.shapes, a.data,
            #                                     a.idxs, q_labels_b, b.shapes, -b.data, b.idxs)
            # return a.__class__(q_labels, shapes, data, a.pattern, idxs, a.symmetry)

    def subtract(self, b):
        return self._subtract(self, b)

    def trace(self, ax1, ax2):
        return NotImplemented

    def __array_function__(self, func, types, args, kwargs):
        if func not in _large_fermion_tensor_numpy_func_impls:
            return NotImplemented
        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented
        return _large_fermion_tensor_numpy_func_impls[func](*args, **kwargs)

    @timing('uf')
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        import numbers
        if ufunc in _large_fermion_tensor_numpy_func_impls:
            types = tuple(
                x.__class__ for x in inputs if not isinstance(x, numbers.Number))
            return self.__array_function__(ufunc, types, inputs, kwargs)
        out = kwargs.pop("out", None)
        if out is not None and not all(isinstance(x, self.__class__) for x in out):
            return NotImplemented
        out_shape = None
        out_use_cupy = None
        xp, _ = get_xp_backend(self.use_cupy)
        if method == "__call__":
            if ufunc.__name__ in ["matmul"]:
                a, b = inputs
                if isinstance(a, xp.ndarray) and a.ndim == 0:
                    a = float(a)
                if isinstance(b, xp.ndarray) and b.ndim == 0:
                    b = float(b)
                if isinstance(a, numbers.Number):
                    shs, qs, data, idxs = b.shapes, b.q_labels, a * b.data, b.idxs
                    out_pattern = b.pattern
                    symmetry = b.symmetry
                    out_shape = b.shape
                    out_use_cupy = a.use_cupy
                elif isinstance(b, numbers.Number):
                    shs, qs, data, idxs = a.shapes, a.q_labels, a.data * b, a.idxs
                    out_pattern = a.pattern
                    symmetry = a.symmetry
                    out_shape = a.shape
                    out_use_cupy = a.use_cupy
                else:
                    c = self._tensordot(a, b, axes=([-1], [0]))
                    shs, qs, data, idxs = c.shapes, c.q_labels, c.data, c.idxs
                    out_pattern, _ = _contract_patterns(a.pattern, b.pattern, [a.ndim-1], [0])
                    symmetry = a.symmetry
                    out_shape = c.shape
                    out_use_cupy = a.use_cupy
            elif ufunc.__name__ in ["multiply", "divide", "true_divide"]:
                a, b = inputs
                if isinstance(a, xp.ndarray) and a.ndim == 0:
                    a = float(a)
                if isinstance(b, xp.ndarray) and b.ndim == 0:
                    b = float(b)
                if isinstance(a, numbers.Number):
                    shs, qs, data, idxs = b.shapes, b.q_labels, getattr(
                        ufunc, method)(a, b.data), b.idxs
                    out_pattern = b.pattern
                    symmetry = b.symmetry
                    out_shape = b.shape
                    out_use_cupy = a.use_cupy
                elif isinstance(b, numbers.Number):
                    shs, qs, data, idxs = a.shapes, a.q_labels, getattr(
                        ufunc, method)(a.data, b), a.idxs
                    out_pattern = a.pattern
                    symmetry = a.symmetry
                    out_shape = a.shape
                    out_use_cupy = a.use_cupy
                else:
                    return NotImplemented
            elif len(inputs) == 1:
                a = inputs[0]
                shs, qs, data, idxs = a.shapes, a.q_labels, getattr(
                    ufunc, method)(a.data), a.idxs
                out_pattern = a.pattern
                symmetry = a.symmetry
                out_shape = a.shape
                out_use_cupy = a.use_cupy
            else:
                return NotImplemented
        else:
            return NotImplemented
        if out is not None:
            out[0].shapes[...] = shs
            out[0].q_labels[...] = qs
            out[0].data[...] = data
            out[0].idxs[...] = idxs
            out[0].pattern = out_pattern
            out[0].symmetry = symmetry
            out[0].shape = out_shape
            out[0].use_cupy = out_use_cupy
        return LargeFermionTensor(q_labels=qs, shapes=shs, data=data, pattern=out_pattern,
            idxs=idxs, symmetry=symmetry, shape=out_shape, use_cupy=out_use_cupy)

    @staticmethod
    @implements(np.tensordot)
    @timing('td')
    def _tensordot(a, b, axes=2):
        if isinstance(axes, int):
            idxa = np.arange(-axes, 0, dtype=np.int32)
            idxb = np.arange(0, axes, dtype=np.int32)
        else:
            idxa = np.array(axes[0], dtype=np.int32)
            idxb = np.array(axes[1], dtype=np.int32)
        idxa[idxa < 0] += a.ndim
        idxb[idxb < 0] += b.ndim
        out_shape = [a.shape[ix] for ix in range(a.ndim) if ix not in idxa] + \
                    [b.shape[ix] for ix in range(b.ndim) if ix not in idxb]
        ainfos = a.infos
        binfos = b.infos
        is_matched = True
        for ixa, ixb in zip(idxa, idxb):
            if len(ainfos[ixa]) != len(binfos[ixb]) or sorted(list(ainfos[ixa].keys())) != sorted(list(binfos[ixb].keys())):
                is_matched = False
                cminfo = ainfos[ixa] | binfos[ixb]
                ainfos[ixa] = cminfo
                binfos[ixb] = cminfo
        if not is_matched:
            a = a.to_flat().to_large(use_cupy=a.use_cupy, infos=ainfos)
            b = b.to_flat().to_large(use_cupy=b.use_cupy, infos=binfos)
        xp, _ = get_xp_backend(a.use_cupy)
        data = xp.empty(out_shape, dtype=a.data.dtype)

        out_pattern, b_flip_axes = _contract_patterns(a.pattern, b.pattern, idxa, idxb)
        q_labels_b = _adjust_q_labels(b.symmetry, b.q_labels, b_flip_axes)
        backend = get_backend(a.symmetry)
        if xp == np:
            return NotImplemented
        assert q_labels_b.strides == b.shapes.strides
        if setting.DEFAULT_FERMION:
            q_labels, shapes, idxs = backend.gpu.flat_fermion_tensor.tensordot(
                a.q_labels, a.shapes, a.data.data.ptr, np.array(a.data.shape, dtype=np.int64), a.idxs,
                q_labels_b, b.shapes, b.data.data.ptr, np.array(b.data.shape, dtype=np.int64), b.idxs,
                idxa, idxb, data.data.ptr, a.data.dtype.name, do_fermi=True)
        else:
            q_labels, shapes, idxs = backend.gpu.flat_fermion_tensor.tensordot(
                a.q_labels, a.shapes, a.data.data.ptr, np.array(a.data.shape, dtype=np.int64), a.idxs,
                q_labels_b, b.shapes, b.data.data.ptr, np.array(b.data.shape, dtype=np.int64), b.idxs,
                idxa, idxb, data.data.ptr, a.data.dtype.name, do_fermi=False)

        if len(idxa) == a.ndim and len(idxb) == b.ndim:
            return data.item()
        return a.__class__(q_labels, shapes, data, out_pattern, idxs, a.symmetry,
            shape=tuple(out_shape), use_cupy=a.use_cupy)

    @staticmethod
    @implements(np.transpose)
    @timing('tp')
    def _transpose(a, axes=None, do_fermi=True):
        if axes is None:
            axes = np.arange(a.ndim)[::-1]
        if a.n_blocks == 0:
            return a
        else:
            xp, _ = get_xp_backend(a.use_cupy)
            axes = np.array(axes, dtype=np.int32)
            data = xp.empty(np.array(a.data.shape)[axes], dtype=a.data.dtype)
            backend = get_backend(a.symmetry)
            if xp == np:
                print('warning: large without gpu!')
                return a.from_flat(a.to_flat().transpose(axes), use_cupy=a.use_cupy)
            if setting.DEFAULT_FERMION and do_fermi:
                cqs, cshs, cidxs = backend.gpu.flat_fermion_tensor.transpose(
                    a.q_labels, a.shapes, a.data.data.ptr, np.array(a.data.shape, dtype=np.int64),
                    a.idxs, axes, data.data.ptr, a.data.dtype.name, do_fermi=True)
            else:
                cqs, cshs, cidxs = backend.gpu.flat_fermion_tensor.transpose(
                    a.q_labels, a.shapes, a.data.data.ptr, np.array(a.data.shape, dtype=np.int64),
                    a.idxs, axes, data.data.ptr, a.data.dtype.name, do_fermi=False)
            pattern = "".join([a.pattern[ix] for ix in axes])
            return a.__class__(cqs, cshs, data, pattern, cidxs, a.symmetry,
                               shape=data.shape, use_cupy=a.use_cupy)

    @timing('svd')
    def tensor_svd(self, left_idx, right_idx=None, qpn_partition=None, **opts):
        return large_svd(self, left_idx, right_idx=right_idx, qpn_partition=qpn_partition, **opts)
    
    @timing('qr')
    def tensor_qr(self, left_idx, right_idx=None, mod="qr"):
        return large_qr(self, left_idx, right_idx=right_idx, mod=mod)

    @timing('ex')
    def to_exponential(self, x):
        from pyblock3.algebra.fermion_ops import get_flat_exponential
        return self.__class__.from_flat(get_flat_exponential(self.to_flat(), x), self.use_cupy)
