import numpy as np
from itertools import product
from functools import reduce
from .core import SubTensor
from .fermion import SparseFermionTensor
from . import fermion_encoding
from . import fermion_setting as setting

SVD_SCREENING = setting.SVD_SCREENING
cre_map = fermion_encoding.cre_map
ann_map = fermion_encoding.ann_map
sz_dict = fermion_encoding.sz_dict
pn_dict = fermion_encoding.pn_dict

hop_map = {0:(1,2), 1:(0,3), 2:(0,3), 3:(1,2)}

def _compute_swap_phase(*states, fermion=None):
    fermion = setting.dispatch_settings(fermion=fermion)
    if not fermion: return 1
    nops = len(states) //2
    cre_string = [cre_map[ikey] for ikey in states[:nops]]
    ann_string = [ann_map[ikey] for ikey in states[nops:]]
    full_string = cre_string + ann_string
    tmp = full_string.copy()
    phase = 1
    for ix, ia in enumerate(cre_string):
        ib = full_string[ix+nops]
        shared = set(ia) & set(ib)
        const_off = sum(len(istring) for istring in full_string[ix+1:ix+nops])
        offset = 0
        for char_a in list(shared):
            inda = ia.index(char_a)
            indb = ib.index(char_a)
            ib = ib.replace(char_a,"")
            ia = ia.replace(char_a,"")
            offset += indb + len(ia)-inda + const_off
        phase *= (-1) ** offset
        full_string[ix] = ia
        full_string[nops+ix] = ib
    return phase

def measure_SZ(symmetry=None, flat=None):
    symmetry, flat = setting.dispatch_settings(symmetry=symmetry, flat=flat)
    state_map = fermion_encoding.get_state_map(symmetry)
    block_dict = dict()
    for key in sz_dict.keys():
        qlab, ind, dim = state_map[key]
        if key not in block_dict:
            dat = np.zeros([dim, dim])
            block_dict[key] = (dat, [qlab, qlab])
        dat = block_dict[key][0]
        dat[ind, ind] += sz_dict[key]
    blocks = [SubTensor(reduced=dat, q_labels=q_lab) for dat, q_lab in block_dict.values()]
    T = SparseFermionTensor(blocks=blocks, pattern="+-")
    if flat:
        return T.to_flat()
    else:
        return T

def ParticleNumber(symmetry=None, flat=None):
    symmetry, flat = setting.dispatch_settings(symmetry=symmetry, flat=flat)
    state_map = fermion_encoding.get_state_map(symmetry)
    block_dict = dict()
    for key in pn_dict.keys():
        qlab, ind, dim = state_map[key]
        if key not in block_dict:
            dat = np.zeros([dim, dim])
            block_dict[key] = (dat, [qlab, qlab])
        dat = block_dict[key][0]
        dat[ind, ind] +=  pn_dict[key]
    blocks = [SubTensor(reduced=dat, q_labels=q_lab) for dat, q_lab in block_dict.values()]
    T = SparseFermionTensor(blocks=blocks, pattern="+-")
    if flat:
        return T.to_flat()
    else:
        return T

def onsite_U(u=1, symmetry=None, flat=None):
    symmetry, flat = setting.dispatch_settings(symmetry=symmetry, flat=flat)
    state_map = fermion_encoding.get_state_map(symmetry)
    block_dict = dict()
    for key, pn in pn_dict.items():
        qlab, ind, dim = state_map[key]
        if key not in block_dict:
            dat = np.zeros([dim, dim])
            block_dict[key] = (dat, [qlab, qlab])
        dat = block_dict[key][0]
        dat[ind, ind] += (pn==2) * u
    blocks = [SubTensor(reduced=dat, q_labels=q_lab) for dat, q_lab in block_dict.values()]
    T = SparseFermionTensor(blocks=blocks, pattern="+-")
    if flat:
        return T.to_flat()
    else:
        return T

def H1(h=1, symmetry=None, flat=None):
    symmetry, flat = setting.dispatch_settings(symmetry=symmetry, flat=flat)
    state_map = fermion_encoding.get_state_map(symmetry)
    block_dict = dict()

    for s1, s2 in product(cre_map.keys(), repeat=2):
        q1, ix1, d1 = state_map[s1]
        q2, ix2, d2 = state_map[s2]
        for s3 in hop_map[s1]:
            q3, ix3, d3 = state_map[s3]
            input_string = sorted(cre_map[s1]+cre_map[s2])
            for s4 in hop_map[s2]:
                q4, ix4, d4 = state_map[s4]
                output_string = sorted(cre_map[s3]+cre_map[s4])
                if input_string != output_string:
                    continue
                if (q1, q2, q3, q4) not in block_dict:
                    block_dict[(q1, q2, q3, q4)] = np.zeros([d1, d2, d3, d4])
                dat = block_dict[(q1, q2, q3, q4)]
                phase = _compute_swap_phase(s1, s2, s3, s4)
                dat[ix1, ix2, ix3, ix4] += phase * h
    blocks = [SubTensor(reduced=dat, q_labels=qlab) for qlab, dat in block_dict.items()]
    T = SparseFermionTensor(blocks=blocks, pattern="++--")
    if flat:
        return T.to_flat()
    else:
        return T

def Hubbard(t=1, u=1, mu=0., fac=None, symmetry=None, flat=None):
    symmetry, flat = setting.dispatch_settings(symmetry=symmetry, flat=flat)
    if fac is None:
        fac = (1, 1)
    faca, facb = fac
    state_map = fermion_encoding.get_state_map(symmetry)
    block_dict = dict()

    for s1, s2 in product(cre_map.keys(), repeat=2):
        q1, ix1, d1 = state_map[s1]
        q2, ix2, d2 = state_map[s2]
        val = (pn_dict[s1]==2) * faca * u + pn_dict[s1] * faca * mu +\
              (pn_dict[s2]==2) * facb * u + pn_dict[s2] * facb * mu
        if (q1, q2, q1, q2) not in block_dict:
            block_dict[(q1, q2, q1, q2)] = np.zeros([d1, d2, d1, d2])
        dat = block_dict[(q1, q2, q1, q2)]
        phase = _compute_swap_phase(s1, s2, s1, s2)
        dat[ix1, ix2, ix1, ix2] += phase * val
        for s3 in hop_map[s1]:
            q3, ix3, d3 = state_map[s3]
            input_string = sorted(cre_map[s1]+cre_map[s2])
            for s4 in hop_map[s2]:
                q4, ix4, d4 = state_map[s4]
                output_string = sorted(cre_map[s3]+cre_map[s4])
                if input_string != output_string:
                    continue
                if (q1, q2, q3, q4) not in block_dict:
                    block_dict[(q1, q2, q3, q4)] = np.zeros([d1, d2, d3, d4])
                dat = block_dict[(q1, q2, q3, q4)]
                phase = _compute_swap_phase(s1, s2, s3, s4)
                dat[ix1, ix2, ix3, ix4] += phase * -t

    blocks = [SubTensor(reduced=dat, q_labels=qlab) for qlab, dat in block_dict.items()]
    T = SparseFermionTensor(blocks=blocks, pattern="++--")
    if flat:
        return T.to_flat()
    else:
        return T

def _fuse_data_map(left_q, right_q, from_parent, to_parent, data_map):
    if left_q not in to_parent:
        if right_q not in to_parent:
            data_map[left_q] = []
            from_parent[left_q] = [left_q]
            if right_q != left_q: from_parent[left_q].append(right_q)
            to_parent[left_q] = to_parent[right_q] = parent = left_q
        else:
            parent = to_parent[right_q]
            from_parent[parent].append(left_q)
            to_parent[left_q] = parent
    else:
        if right_q not in to_parent:
            to_parent[right_q] = parent = to_parent[left_q]
            from_parent[parent].append(right_q)
        else:
            parent = to_parent[left_q]
            parent_b = to_parent[right_q]
            if parent != parent_b:
                for sub in from_parent[parent_b]:
                    to_parent[sub] = parent
                    from_parent[parent].append(sub)
                data_map[parent] += data_map[parent_b]
                del data_map[parent_b]
    return parent

def make_phase_dict(state_map, ndim):
    phase_dict = {}
    for states in product(state_map.keys(), repeat=ndim):
        all_info = [state_map[istate] for istate in states]
        qlabs = tuple([info[0] for info in all_info])
        input_string = sorted("".join([cre_map[istate] for istate in states[:ndim//2]]))
        output_string = sorted("".join([ann_map[istate] for istate in states[ndim//2:]]))
        if input_string != output_string:
            continue
        inds = tuple([info[1] for info in all_info])
        shape = tuple([info[2] for info in all_info])
        if qlabs not in phase_dict:
            phase_dict[qlabs] = np.zeros(shape)
        phase_dict[qlabs][inds] = _compute_swap_phase(*states)
    return phase_dict

def get_exponential(T, x):
    if setting.DEFAULT_FLAT:
        return get_flat_exponential(T, x)
    else:
        return get_sparse_exponential(T, x)

def get_flat_exponential(T, x):
    symmetry = T.dq.__class__
    if setting.DEFAULT_FERMION:
        state_map = fermion_encoding.get_state_map(T.symmetry)
        phase_dict = make_phase_dict(state_map, T.ndim)
    def get_phase(qlabels):
        if setting.DEFAULT_FERMION:
            return phase_dict[qlabels]
        else:
            return 1
    split_ax = T.ndim//2
    left_pattern = T.pattern[:split_ax]
    right_pattern = T.pattern[split_ax:]
    data_map = {}
    from_parent = {}
    to_parent = {}
    for iblk in range(T.n_blocks):
        left_q = tuple(T.q_labels[iblk,:split_ax])
        right_q = tuple(T.q_labels[iblk,split_ax:])
        parent = _fuse_data_map(left_q, right_q, from_parent, to_parent, data_map)
        data_map[parent].append(iblk)
    datas = []
    shapes = []
    qlablst = []
    for slab, datasets in data_map.items():
        row_len = col_len = 0
        row_map = {}
        for iblk in datasets:
            lq = tuple(T.q_labels[iblk,:split_ax])
            rq = tuple(T.q_labels[iblk,split_ax:])
            if lq not in row_map:
                new_row_len = row_len + np.prod(T.shapes[iblk,:split_ax], dtype=int)
                row_map[lq] = (row_len, new_row_len, T.shapes[iblk,:split_ax])
                row_len = new_row_len
            if rq not in row_map:
                new_row_len = row_len + np.prod(T.shapes[iblk,split_ax:], dtype=int)
                row_map[rq] = (row_len, new_row_len, T.shapes[iblk,split_ax:])
                row_len = new_row_len
        data = np.zeros([row_len, row_len], dtype=T.dtype)
        for iblk in datasets:
            lq = tuple(T.q_labels[iblk,:split_ax])
            rq = tuple(T.q_labels[iblk,split_ax:])
            ist, ied = row_map[lq][:2]
            jst, jed = row_map[rq][:2]
            qlabs = tuple([symmetry.from_flat(iq) for iq in T.q_labels[iblk]])
            phase = get_phase(qlabs)
            if isinstance(phase, np.ndarray):
                phase = phase.reshape(ied-ist, jed-jst)
            data[ist:ied,jst:jed] = T.data[T.idxs[iblk]:T.idxs[iblk+1]].reshape(ied-ist, jed-jst) * phase
        if data.size ==0:
            continue
        el, ev = np.linalg.eigh(data)
        s = np.diag(np.exp(el*x))
        tmp = reduce(np.dot, (ev, s, ev.conj().T))
        for lq, (ist, ied, ish) in row_map.items():
            for rq, (jst, jed, jsh) in row_map.items():
                q_labels = lq + rq
                qlabs = tuple([symmetry.from_flat(iq) for iq in q_labels])
                phase = get_phase(qlabs)
                chunk = tmp[ist:ied, jst:jed].reshape(tuple(ish)+tuple(jsh)) * phase
                if abs(chunk).max()<SVD_SCREENING:
                    continue
                datas.append(chunk.ravel())
                shapes.append(tuple(ish)+tuple(jsh))
                qlablst.append(q_labels)
    q_labels = np.asarray(qlablst, dtype=np.uint32)
    shapes = np.asarray(shapes, dtype=np.uint32)
    datas = np.concatenate(datas)
    Texp = T.__class__(q_labels, shapes, datas, pattern=T.pattern, symmetry=T.symmetry)
    return Texp

def get_sparse_exponential(T, x):
    symmetry = T.dq.__class__
    if setting.DEFAULT_FERMION:
        state_map = fermion_encoding.get_state_map(symmetry)
        phase_dict = make_phase_dict(state_map, T.ndim)
    def get_phase(qlabels):
        if setting.DEFAULT_FERMION:
            return phase_dict[qlabels]
        else:
            return 1
    split_ax = T.ndim//2
    left_pattern = T.pattern[:split_ax]
    right_pattern = T.pattern[split_ax:]
    data_map = {}
    from_parent = {}
    to_parent = {}
    blocks = []
    for iblk in T.blocks:
        left_q = iblk.q_labels[:split_ax]
        right_q = iblk.q_labels[split_ax:]
        parent = _fuse_data_map(left_q, right_q, from_parent, to_parent, data_map)
        data_map[parent].append(iblk)

    for slab, datasets in data_map.items():
        row_len = col_len = 0
        row_map = {}
        for iblk in datasets:
            lq = iblk.q_labels[:split_ax]
            rq = iblk.q_labels[split_ax:]
            if lq not in row_map:
                new_row_len = row_len + np.prod(iblk.shape[:split_ax], dtype=int)
                row_map[lq] = (row_len, new_row_len, iblk.shape[:split_ax])
                row_len = new_row_len
            if rq not in row_map:
                new_row_len = row_len + np.prod(iblk.shape[split_ax:], dtype=int)
                row_map[rq] = (row_len, new_row_len, iblk.shape[split_ax:])
                row_len = new_row_len
        data = np.zeros([row_len, row_len], dtype=T.dtype)
        for iblk in datasets:
            lq = iblk.q_labels[:split_ax]
            rq = iblk.q_labels[split_ax:]
            ist, ied = row_map[lq][:2]
            jst, jed = row_map[rq][:2]
            phase = get_phase(iblk.q_labels)
            if isinstance(phase, np.ndarray):
                phase = phase.reshape(ied-ist, jed-jst)
            data[ist:ied,jst:jed] = np.asarray(iblk).reshape(ied-ist, jed-jst) * phase
        if data.size ==0:
            continue
        el, ev = np.linalg.eigh(data)
        s = np.diag(np.exp(el*x))
        tmp = reduce(np.dot, (ev, s, ev.conj().T))
        for lq, (ist, ied, ish) in row_map.items():
            for rq, (jst, jed, jsh) in row_map.items():
                q_labels = lq + rq
                phase = get_phase(q_labels)
                chunk = tmp[ist:ied, jst:jed].reshape(tuple(ish)+tuple(jsh)) * phase
                if abs(chunk).max()<SVD_SCREENING:
                    continue
                blocks.append(SubTensor(reduced=chunk, q_labels=q_labels))
    Texp = T.__class__(blocks=blocks, pattern=T.pattern)
    return Texp
