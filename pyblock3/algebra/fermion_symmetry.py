import numpy as np
from pyblock3.algebra.core import SubTensor
from pyblock3.algebra.symmetry import SZ, BondInfo
from itertools import product
USE_FLAT = True

def _compute_swap_phase(cre_map, ann_map, *qpns):
    nops = len(qpns) //2
    cre_string = [cre_map[ikey] for ikey in qpns[:nops]]
    ann_string = [ann_map[ikey] for ikey in qpns[nops:]]
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

def _make_z2_phase_array(cre_map, ann_map, *q_labels):
    ndim = len(q_labels)
    phase_arr = np.ones([2,]*ndim)
    for inds in product(range(2), repeat=ndim):
        qns = ((q_labels[i], inds[i]) for i in range(ndim))
        phase = _compute_swap_phase(cre_map, ann_map, *qns)
        phase_arr[inds] = phase
    return phase_arr

def _blocks_to_tensor(blocks, pattern, FLAT=USE_FLAT):
    from pyblock3.algebra.fermion import SparseFermionTensor
    T = SparseFermionTensor(blocks=blocks, pattern=pattern)
    if FLAT:
        T = T.to_flat()
    return T

class U1(SZ):
    def __init__(self, n=0, sz=0):
        self.n = n
        self.twos = sz
        self.pg = n % 2

    def __add__(self, other):
        return self.__class__(self.n + other.n, self.twos + other.twos)

    def __sub__(self, other):
        return self.__class__(self.n - other.n, self.twos - other.twos)

    def __neg__(self):
        return self.__class__(-self.n, -self.twos)

    @property
    def sz(self):
        return self.twos

    def __hash__(self):
        return hash((self.n, self.twos))

    @property
    def parity(self):
        return self.pg

    @staticmethod
    def from_flat(x):
        return U1((x // 131072) % 16384 - 8192, (x // 8) % 16384 - 8192)

    @classmethod
    def flat_to_parity(cls, x):
        return (x // 131072) % 16384 % 2

    @classmethod
    def compute(cls, pattern, qpns, offset=None, neg=False):
        is_U1 = isinstance(qpns[0], U1)
        if is_U1:
            assert(len(pattern)==len(qpns))
            out = U1()
            if offset is not None:
                ix, dq = offset
                pattern += ix
                qpns = list(qpns) + [dq, ]
            for iq, ix in zip(qpns, pattern):
                if ix=="+":
                    out += iq
                else:
                    out -= iq
            if neg:
                out = -out
        elif isinstance(qpns, np.ndarray):
            nsec = len(pattern)
            nplus = pattern.count("+")
            nmin = pattern.count("-")
            newdat = qpns.copy().astype(int)
            narr = (newdat // 131072) % 16384 - 8192
            zarr = (newdat // 8) % 16384 - 8192
            if offset is not None:
                ix, dq = offset
                if not isinstance(dq, U1):
                    dq = U1.from_flat(dq)
                if ix=="-":
                    dq = -dq
                offn = dq.n
                offz = dq.sz
            else:
                offn = offz = 0
            for ix, istr in enumerate(pattern):
                if istr=="-":
                    narr[:,ix] *= -1
                    zarr[:,ix] *= -1
            if neg:
                narr = -np.sum(narr, axis=1) - offn
                zarr = -np.sum(zarr, axis=1) - offz
            else:
                narr = np.sum(narr, axis=1) + offn
                zarr = np.sum(zarr, axis=1) + offz
            out = ((narr + 8192) * 16384 + (zarr + 8192)) * 8 + narr % 2
        else:
            newdat = np.asarray(qpns)
            narr = (newdat // 131072) % 16384 - 8192
            zarr = (newdat // 8) % 16384 - 8192
            if offset is not None:
                ix, dq = offset
                if not isinstance(dq, U1):
                    dq = U1.from_flat(dq)
                if ix=="-":
                    dq = -dq
                offn = dq.n
                offz = dq.sz
            else:
                offn = offz = 0
            for ix, istr in enumerate(pattern):
                if istr=="-":
                    narr[ix] *= -1
                    zarr[ix] *= -1
            if neg:
                narr = -np.sum(narr) - offn
                zarr = -np.sum(zarr) - offz
            else:
                narr = np.sum(narr) + offn
                zarr = np.sum(zarr) + offz
            out = ((narr + 8192) * 16384 + (zarr + 8192)) * 8 + narr % 2
        return out

    def __repr__(self):
        return "< N=%d SZ=%d >" % (self.n, self.twos)

    @classmethod
    def onsite_U(cls, U=1, FLAT=USE_FLAT):
        blocks = [SubTensor(reduced=np.zeros([1,1]), q_labels=(U1(0), U1(0))),
                  SubTensor(reduced=np.zeros([1,1]), q_labels=(U1(1,1), U1(1,1))),
                  SubTensor(reduced=np.zeros([1,1]), q_labels=(U1(1,-1), U1(1,-1))),
                  SubTensor(reduced=np.eye(1)*U, q_labels=(U1(2), U1(2)))]
        return _blocks_to_tensor(blocks, "+-", FLAT)


    @classmethod
    def Measure_SZ(cls, FLAT=USE_FLAT):
        blocks = [SubTensor(reduced=np.zeros([1,1]), q_labels=(U1(0), U1(0))),
                  SubTensor(reduced=np.eye(1)*.5, q_labels=(U1(1,1), U1(1,1))),
                  SubTensor(reduced=np.eye(1)*-.5, q_labels=(U1(1,-1), U1(1,-1))),
                  SubTensor(reduced=np.zeros([1,1]), q_labels=(U1(2), U1(2)))]
        return _blocks_to_tensor(blocks, "+-", FLAT)

    @classmethod
    def ParticleNumber(cls, FLAT=USE_FLAT):
        blocks = [SubTensor(reduced=np.zeros([1,1]), q_labels=(U1(0), U1(0))),
                  SubTensor(reduced=np.eye(1), q_labels=(U1(1,1), U1(1,1))),
                  SubTensor(reduced=np.eye(1), q_labels=(U1(1,-1), U1(1,-1))),
                  SubTensor(reduced=np.eye(1)*2, q_labels=(U1(2), U1(2)))]
        return _blocks_to_tensor(blocks, "+-", FLAT)

    @classmethod
    def operator_cre_map(cls):
        return {U1(0):"", U1(1,1):"a", U1(1,-1):"b", U1(2,0):"ab"}

    @classmethod
    def operator_ann_map(cls):
        return {U1(0):"", U1(1,1):"a", U1(1,-1):"b", U1(2,0):"ba"}

    @classmethod
    def operator_qpnlst(cls):
        return [U1(0),U1(1,1),U1(1,-1),U1(2,0)]

    @classmethod
    def operator_dqlst(cls):
        return [U1(1,1),U1(1,-1)]

    @classmethod
    def H1(cls, h=1., FLAT=USE_FLAT):
        blocks = []
        qpn_lst = cls.operator_qpnlst()
        cre_map = cls.operator_cre_map()
        ann_map = cls.operator_ann_map()
        dqlst = cls.operator_dqlst()
        for q1, q2 in product(qpn_lst, repeat=2):
            for dq in dqlst:
                if q1 + dq in qpn_lst and q2-dq in qpn_lst:
                    phase = _compute_swap_phase(cre_map, ann_map, q1, q2, q1+dq, q2-dq)
                    blocks.append(SubTensor(reduced=phase*np.ones([1,1,1,1])*h, q_labels=(q1, q2, q1+dq, q2-dq)))
                if q1 - dq in qpn_lst and q2+dq in qpn_lst:
                    phase = _compute_swap_phase(cre_map, ann_map, q1, q2, q1-dq, q2+dq)
                    blocks.append(SubTensor(reduced=phase*np.ones([1,1,1,1])*h, q_labels=(q1, q2, q1-dq, q2+dq)))
            blocks.append(SubTensor(reduced=np.zeros([1,1,1,1]), q_labels=(q1,q2,q1,q2)))
        return _blocks_to_tensor(blocks, "++--", FLAT)

    @classmethod
    def hubbard(cls, t, u, mu=0, fac=None, FLAT=USE_FLAT):
        if fac is None: fac = (1,1)
        faca, facb = fac
        blocks = []
        qpn_lst = cls.operator_qpnlst()
        cre_map = cls.operator_cre_map()
        ann_map = cls.operator_ann_map()
        dqlst = cls.operator_dqlst()
        for q1, q2 in product(qpn_lst, repeat=2):
            for dq in dqlst:
                if q1 + dq in qpn_lst and q2-dq in qpn_lst:
                    phase = _compute_swap_phase(cre_map, ann_map, q1, q2, q1+dq, q2-dq)
                    blocks.append(SubTensor(reduced=phase*np.ones([1,1,1,1])*-t, q_labels=(q1, q2, q1+dq, q2-dq)))
                if q1 - dq in qpn_lst and q2+dq in qpn_lst:
                    phase = _compute_swap_phase(cre_map, ann_map, q1, q2, q1-dq, q2+dq)
                    blocks.append(SubTensor(reduced=phase*np.ones([1,1,1,1])*-t, q_labels=(q1, q2, q1-dq, q2+dq)))
            q_labels = (q1, q2, q1, q2)
            val = (q1==U1(2)) * faca * u + q1.n * faca * mu +\
                  (q2==U1(2)) * facb * u + q2.n * facb * mu
            phase = _compute_swap_phase(cre_map, ann_map, q1, q2, q1, q2)
            blocks.append(SubTensor(reduced=phase*np.ones([1,1,1,1])*val, q_labels=q_labels))
        return _blocks_to_tensor(blocks, "++--", FLAT)

class Z2(SZ):
    _modulus = 2
    def __init__(self, z=0):
        self.n = z % self._modulus
        self.twos = 0
        self.pg = z % 2

    def __add__(self, other):
        return self.__class__(self.n + other.n)

    def __sub__(self, other):
        return self.__class__(self.n - other.n)

    def __neg__(self):
        return self.__class__(-self.n)

    def __eq__(self, other):
        return self.n == other.n

    def __lt__(self, other):
        return self.n < other.n

    def __hash__(self):
        return hash((self.n,))

    @property
    def parity(self):
        return self.pg

    @staticmethod
    def from_flat(x):
        return Z2((x // 131072) % 16384 - 8192)

    @classmethod
    def flat_to_parity(cls, x):
        return (x // 131072) % 2

    @classmethod
    def compute(cls, pattern, qpns, offset=None, neg=False):
        is_cls = isinstance(qpns[0], cls)
        if is_cls:
            assert(len(pattern)==len(qpns))
            out = cls()
            if offset is not None:
                ix, dq = offset
                pattern += ix
                qpns = list(qpns) + [dq, ]
            for iq, ix in zip(qpns, pattern):
                if ix=="+":
                    out += iq
                else:
                    out -= iq
            if neg:
                out = -out
        elif isinstance(qpns, np.ndarray):
            nsec = len(pattern)
            nplus = pattern.count("+")
            nmin = pattern.count("-")
            newdat = qpns.copy().astype(int)
            narr = (newdat // 131072) % 16384 - 8192
            if offset is not None:
                ix, dq = offset
                if not isinstance(dq, cls):
                    dq = cls.from_flat(dq)
                if ix=="-":
                    dq = -dq
                offn = dq.n
            else:
                offn = 0
            for ix, istr in enumerate(pattern):
                if istr=="-":
                    narr[:,ix] *= -1
            if neg:
                narr = -np.sum(narr, axis=1) - offn
            else:
                narr = np.sum(narr, axis=1) + offn
            narr = narr % cls._modulus
            out = (narr + 8192) * 131072 + narr % 2 + 65536
        else:
            newdat = np.asarray(qpns)
            narr = (newdat // 131072) % 16384 - 8192
            if offset is not None:
                ix, dq = offset
                if not isinstance(dq, cls):
                    dq = cls.from_flat(dq)
                if ix=="-":
                    dq = -dq
                offn = dq.n
            else:
                offn = 0
            for ix, istr in enumerate(pattern):
                if istr=="-":
                    narr[ix] *= -1
            if neg:
                narr = -np.sum(narr) - offn
            else:
                narr = np.sum(narr) + offn
            narr = narr % _modulus
            if _modulus == 2:
                out = (narr + 8192) * 131072 + narr + 65536
            else:
                out = (narr + 8192) * 131072 + narr % 2 + 65536
        return out

    def __repr__(self):
        return "< Z=%d >" % self.n

    @classmethod
    def onsite_U(cls, U=1, FLAT=USE_FLAT):
        data = np.zeros([2,2])
        data[1,1] = U
        blocks = [SubTensor(reduced=data, q_labels=(Z2(0),Z2(0))),
                  SubTensor(reduced=np.zeros([2,2]), q_labels=(Z2(1),Z2(1)))]
        return _blocks_to_tensor(blocks, "+-", FLAT)

    @classmethod
    def Measure_SZ(cls, FLAT=USE_FLAT):
        data = np.zeros([2,2])
        data[0,0] = .5
        data[1,1] =-.5
        blocks = [SubTensor(reduced=np.zeros([2,2]), q_labels=(Z2(0),Z2(0))),
                  SubTensor(reduced=data, q_labels=(Z2(1),Z2(1)))]
        return _blocks_to_tensor(blocks, "+-", FLAT)

    @classmethod
    def ParticleNumber(cls, FLAT=USE_FLAT):
        data = np.zeros([2,2])
        data[1,1] = 2
        blocks = [SubTensor(reduced=data, q_labels=(Z2(0),Z2(0))),
                  SubTensor(reduced=np.eye(2), q_labels=(Z2(1),Z2(1)))]
        return _blocks_to_tensor(blocks, "+-", FLAT)

    @classmethod
    def operator_cre_map(cls):
        return {(Z2(0),0):"",(Z2(0),1):"ab",(Z2(1),0):"a",(Z2(1),1):"b"}

    @classmethod
    def operator_ann_map(cls):
        return {(Z2(0),0):"",(Z2(0),1):"ba",(Z2(1),0):"a",(Z2(1),1):"b"}

    @classmethod
    def operator_qpnlst(cls):
        return [(Z2(0),0),(Z2(0),1),(Z2(1),0),(Z2(1),1)]

    @classmethod
    def H1(cls, h=1., FLAT=USE_FLAT):
        block_map = {}
        qpn_lst = cls.operator_qpnlst()
        cre_map = cls.operator_cre_map()
        ann_map = cls.operator_ann_map()
        for q1, q2 in product(qpn_lst, repeat=2):
            for dq in [(Z2(1),0),(Z2(1),1)]:
                q3 = (q1[0]+dq[0], (q1[1]+dq[1])%2)
                q4 = (q2[0]-dq[0], (q2[1]-dq[1])%2)
                if q3 in qpn_lst and q4 in qpn_lst:
                    input = "".join(sorted(cre_map[q1]+cre_map[q2]))
                    output = "".join(sorted(cre_map[q3] + cre_map[q4]))
                    if input != output: continue
                    qlab = (q1[0], q2[0], q3[0], q4[0])
                    i, j, k, l = q1[1], q2[1], q3[1], q4[1]
                    phase = _compute_swap_phase(cre_map, ann_map, q1, q2, q3, q4)
                    if qlab not in block_map:
                        block_map[qlab] = np.zeros([2,2,2,2])
                    data = block_map[qlab]
                    data[i,j,k,l] += phase * h

        blocks = [SubTensor(reduced=val, q_labels=key) for key, val in block_map.items()]
        return _blocks_to_tensor(blocks, "++--", FLAT)

    @classmethod
    def hubbard(cls, t, u, mu=0, fac=None, FLAT=USE_FLAT):
        if fac is None: fac = (1,1)
        faca, facb = fac
        block_map = {}
        qpn_lst = [(Z2(0),0),(Z2(0),1),(Z2(1),0),(Z2(1),1)]
        cre_map = {(Z2(0),0):"",(Z2(0),1):"ab",(Z2(1),0):"a",(Z2(1),1):"b"}
        ann_map = {(Z2(0),0):"",(Z2(0),1):"ba",(Z2(1),0):"a",(Z2(1),1):"b"}
        onsite_map = {(Z2(0),0):0,(Z2(0),1):1,(Z2(1),0):0,(Z2(1),1):0}
        n_map = {(Z2(0),0):0,(Z2(0),1):2,(Z2(1),0):1,(Z2(1),1):1}

        for q1, q2 in product(qpn_lst, repeat=2):
            for dq in [(Z2(1),0),(Z2(1),1)]:
                q3 = (q1[0]+dq[0], (q1[1]+dq[1])%2)
                q4 = (q2[0]-dq[0], (q2[1]-dq[1])%2)
                if q3 in qpn_lst and q4 in qpn_lst:
                    input = "".join(sorted(cre_map[q1]+cre_map[q2]))
                    output = "".join(sorted(cre_map[q3] + cre_map[q4]))
                    if input != output: continue
                    phase = _compute_swap_phase(cre_map, ann_map, q1, q2, q3, q4)
                    qlab = (q1[0], q2[0], q3[0], q4[0])
                    i, j, k, l = q1[1], q2[1], q3[1], q4[1]
                    if qlab not in block_map:
                        block_map[qlab] = np.zeros([2,2,2,2])
                    data = block_map[qlab]
                    data[i,j,k,l] += phase * -t

            val = onsite_map[q1] * faca * u + n_map[q1] * faca * mu +\
                  onsite_map[q2] * facb * u + n_map[q2] * facb * mu
            phase = _compute_swap_phase(cre_map, ann_map, q1, q2, q1, q2)
            qlab = (q1[0], q2[0], q1[0], q2[0])
            i, j, k, l = q1[1], q2[1], q1[1], q2[1]
            if qlab not in block_map:
                block_map[qlab] = np.zeros([2,2,2,2])
            data = block_map[qlab]
            data[i,j,k,l] += phase * val

        blocks = [SubTensor(reduced=val, q_labels=key) for key, val in block_map.items()]
        return _blocks_to_tensor(blocks, "++--", FLAT)

class Z4(Z2):
    _modulus = 4
    @staticmethod
    def from_flat(x):
        return Z4((x // 131072) % 16384 - 8192)

    @classmethod
    def onsite_U(cls, U=1, FLAT=USE_FLAT):
        blocks = [SubTensor(reduced=np.zeros([1,1]), q_labels=(Z4(0), Z4(0))),
                  SubTensor(reduced=np.zeros([1,1]), q_labels=(Z4(1), Z4(1))),
                  SubTensor(reduced=np.eye(1)*U, q_labels=(Z4(2), Z4(2))),
                  SubTensor(reduced=np.zeros([1,1]), q_labels=(Z4(3), Z4(3))),]
        return _blocks_to_tensor(blocks, "+-", FLAT)

    @classmethod
    def Measure_SZ(cls, FLAT=USE_FLAT):
        blocks = [SubTensor(reduced=np.zeros([1,1]), q_labels=(Z4(0), Z4(0))),
                  SubTensor(reduced=np.eye(1)*.5, q_labels=(Z4(1), Z4(1))),
                  SubTensor(reduced=np.zeros([1,1]), q_labels=(Z4(2), Z4(2))),
                  SubTensor(reduced=np.eye(1)*-.5, q_labels=(Z4(3), Z4(3)))]
        return _blocks_to_tensor(blocks, "+-", FLAT)

    @classmethod
    def ParticleNumber(cls, FLAT=USE_FLAT):
        blocks = [SubTensor(reduced=np.zeros([1,1]), q_labels=(Z4(0), Z4(0))),
                  SubTensor(reduced=np.eye(1), q_labels=(Z4(1), Z4(1))),
                  SubTensor(reduced=np.eye(1)*2, q_labels=(Z4(2), Z4(2))),
                  SubTensor(reduced=np.eye(1), q_labels=(Z4(3), Z4(3)))]
        return _blocks_to_tensor(blocks, "+-", FLAT)

    @classmethod
    def operator_cre_map(cls):
        return {Z4(0):"", Z4(1):"a", Z4(2):"ab", Z4(3):"b"}

    @classmethod
    def operator_ann_map(cls):
        return {Z4(0):"", Z4(1):"a", Z4(2):"ba", Z4(3):"b"}

    @classmethod
    def operator_qpnlst(cls):
        return [Z4(0),Z4(1),Z4(2),Z4(3)]

    @classmethod
    def operator_dqlst(cls):
        return [Z4(1),Z4(3)]

    @classmethod
    def H1(cls, h=1., FLAT=USE_FLAT):
        blocks = []
        qpn_lst = cls.operator_qpnlst()
        cre_map = cls.operator_cre_map()
        ann_map = cls.operator_ann_map()
        dqlst = cls.operator_dqlst()
        for q1, q2 in product(qpn_lst, repeat=2):
            for dq1, dq2 in product(dqlst, repeat=2):
                if q1 + dq1 in qpn_lst and q2+dq2 in qpn_lst:
                    input = "".join(sorted(cre_map[q1]+cre_map[q2]))
                    output = "".join(sorted(cre_map[q1+dq1] + cre_map[q2+dq2]))
                    if input != output: continue
                    phase = _compute_swap_phase(cre_map, ann_map, q1, q2, q1+dq1, q2+dq2)
                    blocks.append(SubTensor(reduced=phase*np.ones([1,1,1,1])*h, q_labels=(q1, q2, q1+dq1, q2+dq2)))
            blocks.append(SubTensor(reduced=np.zeros([1,1,1,1]), q_labels=(q1,q2,q1,q2)))
        return _blocks_to_tensor(blocks, "++--", FLAT)

    @classmethod
    def hubbard(cls, t, u, mu=0, fac=None, FLAT=USE_FLAT):
        if fac is None: fac = (1,1)
        faca, facb = fac
        blocks = []
        qpn_lst = cls.operator_qpnlst()
        cre_map = cls.operator_cre_map()
        ann_map = cls.operator_ann_map()
        dqlst = cls.operator_dqlst()
        n_map = {Z4(0):0, Z4(1):1, Z4(2):2, Z4(3):1}
        for q1, q2 in product(qpn_lst, repeat=2):
            for dq1, dq2 in product(dqlst, repeat=2):
                if q1 + dq1 in qpn_lst and q2+dq2 in qpn_lst:
                    input = "".join(sorted(cre_map[q1]+cre_map[q2]))
                    output = "".join(sorted(cre_map[q1+dq1] + cre_map[q2+dq2]))
                    if input != output:
                        continue
                    phase = _compute_swap_phase(cre_map, ann_map, q1, q2, q1+dq1, q2+dq2)
                    blocks.append(SubTensor(reduced=phase*np.ones([1,1,1,1])*-t, q_labels=(q1, q2, q1+dq1, q2+dq2)))
            q_labels = (q1, q2, q1, q2)
            val = (q1==Z4(2)) * faca * u + n_map[q1] * faca * mu +\
                  (q2==Z4(2)) * facb * u + n_map[q2] * facb * mu
            phase = _compute_swap_phase(cre_map, ann_map, q1, q2, q1, q2)
            blocks.append(SubTensor(reduced=phase*np.ones([1,1,1,1])*val, q_labels=q_labels))
        return _blocks_to_tensor(blocks, "++--", FLAT)
