
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

"""
Tools for translating block2 objects.
MPS/MPO format transform between block2 and pyblock3.
"""

import numpy as np
import numbers
from numpy.core.numeric import tensordot
from ..algebra.core import SparseTensor as ASparseTensor, SubTensor, FermionTensor
from ..algebra.mps import MPS
from ..algebra.symmetry import SZ as ASZ
from ..symbolic.expr import OpElement as AOpElement
from ..symbolic.symbolic import SymbolicSparseTensor
from ..symbolic.symbolic import SymbolicMatrix as ASymbolicMatrix
from ..symbolic.symbolic import SymbolicRowVector as ASymbolicRowVector
from ..symbolic.symbolic import SymbolicColumnVector as ASymbolicColumnVector

from block2 import OpTypes, QCTypes, SZ, Tensor, Global, init_memory, release_memory
from block2 import VectorInt, VectorDouble, VectorPIntInt, DoubleVectorAllocator, IntVectorAllocator
from block2 import OpNames, SiteIndex
from block2.sz import StateInfo, MPOQC, UnfusedMPS, CG, MPSInfo, VectorStateInfo
from block2.sz import VectorVectorPSSTensor, VectorPSSTensor, SparseTensor, VectorSpTensor
from block2.sz import OpElement, MPO, OpExpr, VectorOpExpr, OperatorTensor
from block2.sz import SymbolicMatrix, SymbolicRowVector, SymbolicColumnVector
from block2.sz import SparseMatrixInfo, SparseMatrix, TensorFunctions, OperatorFunctions, CG
from block2.sz import VectorVectorPLMatInfo, VectorPLMatInfo, VectorSymbolic, VectorOpTensor

class MPSTools:
    @staticmethod
    def from_block2(bmps):
        """Translate block2 MPS to pyblock3 MPS."""
        tensors = [None] * bmps.n_sites
        cf = [c for c in bmps.canonical_form]
        if cf.count('C') + cf.count('K') + cf.count('S') != 1:
            assert not isinstance(bmps, UnfusedMPS)
            assert cf.count('C') == 2
            ix = cf.index('C')
            assert cf[ix + 1] == 'C'
            bmps.center = ix
            bmps.move_right(CG(), None)
            bmps.save_data()
        if not isinstance(bmps, UnfusedMPS):
            bmps.load_data()
            bmps = UnfusedMPS(bmps)
        for i in range(0, bmps.n_sites):
            spd = bmps.tensors[i].data
            blocks = []
            for im in range(len(spd)):
                qm = bmps.info.basis[i].quanta[im]
                for ((ql, qr), ts) in spd[im]:
                    qx = tuple(ASZ(x.n, x.twos, x.pg) for x in (ql, qm, qr))
                    blocks.append(SubTensor(q_labels=qx, reduced=np.array(ts, copy=True)))
            tensors[i] = ASparseTensor(blocks=blocks)
        return MPS(tensors=tensors)
    
    @staticmethod
    def to_block2(mps, center=None, basis=None, tag='KET', save_dir=None):
        """
        Translate pyblock3 MPS to block2 MPS.
        
        Args:
            mps : pyblock3 MPS
                More than one physical index is not supported.
                But fused index can be supported.
            center : int or None
                If not None, the pyblock3 MPS is already canonicalized
                at the given index of the center.
                If None (default), the pyblock3 MPS is transformed after
                canonicalization at site 0.
            basis : List(BondInfo) or None
                If None (default), the site basis will be constructed based on
                blocks in the MPS tensor, which may be incomplete if the
                bond dimension of the MPS is small.
                If not None, the given basis will be used.
            tag : str
                Tag of the block2 MPS. Default is "KET".
            save_dir : str or None
                If not None, the block2 MPS will be saved to the given dir.
                If None and the block2 global scratch is not set before
                entering this function, the block2 MPS will be saved to './nodex'.
                If None and the block2 global scratch is set, the block2 MPS will
                be saved to the current block2 global scratch folder.
        
        Returns:
            bmps : block2 MPS
                To inspect this MPS, please make sure that the block2 global
                scratch folder and stack memory are properly initialized.
        """
        frame_back = Global.frame
        inited = False
        if Global.frame is None or save_dir is not None:
            if save_dir is None:
                save_dir = './nodex'
            init_memory(isize=1 << 20, dsize=1 << 30, save_dir=save_dir)
            inited = True
        if save_dir is None:
            save_dir = Global.frame.save_dir
        if center is None:
            mps = mps.canonicalize(0)
            center = 0
        mps = mps.to_non_flat()
        mps_infos = [x.infos for x in mps.tensors]
        vacuum = list(mps_infos[0][0].keys())[0]
        target = list(mps_infos[-1][-1].keys())[0]
        vacuum = SZ(vacuum.n, vacuum.twos, vacuum.pg)
        target = SZ(target.n, target.twos, target.pg)
        if basis is None:
            basis = [info[1] for info in mps_infos]
        else:
            basis = basis.copy()
        assert len(basis) == len(mps)
        for ib, b in enumerate(basis):
            p = StateInfo()
            p.allocate(len(b))
            for ix, (k, v) in enumerate(b.items()):
                p.quanta[ix] = SZ(k.n, k.twos, k.pg)
                p.n_states[ix] = v
            basis[ib] = p
            p.sort_states()
        minfo = MPSInfo(mps.n_sites, vacuum, target, VectorStateInfo(basis))
        minfo.left_dims[0] = StateInfo(vacuum)
        for i, info in enumerate(mps_infos):
            p = minfo.left_dims[i + 1]
            p.allocate(len(info[-1]))
            for ix, (k, v) in enumerate(info[-1].items()):
                p.quanta[ix] = SZ(k.n, k.twos, k.pg)
                p.n_states[ix] = v
            p.sort_states()
        minfo.right_dims[mps.n_sites] = StateInfo(vacuum)
        for i, info in enumerate(mps_infos):
            p = minfo.right_dims[i]
            p.allocate(len(info[0]))
            for ix, (k, v) in enumerate(info[0].items()):
                p.quanta[ix] = target - SZ(k.n, k.twos, k.pg)
                p.n_states[ix] = v
            p.sort_states()
        minfo.tag = tag
        minfo.save_mutable()
        minfo.save_data("%s/%s-mps_info.bin" % (save_dir, tag))
        tensors = [None] * len(mps)
        for i, b in enumerate(basis):
            tensors[i] = SparseTensor()
            tensors[i].data = VectorVectorPSSTensor([VectorPSSTensor() for _ in range(b.n)])
            for block in mps[i].blocks:
                ql, qm, qr = [SZ(x.n, x.twos, x.pg) for x in block.q_labels]
                im = b.find_state(qm)
                assert im != -1
                tensors[i].data[im].append(((ql, qr), Tensor(VectorInt(block.shape))))
                tensors[i].data[im][-1][1].data = VectorDouble(np.asarray(block).flatten())
        umps = UnfusedMPS()
        umps.info = minfo
        umps.n_sites = len(mps)
        umps.canonical_form = "L" * center + ("S" if center == len(mps) - 1 else "K") + \
            "R" * (len(mps) - center - 1)
        umps.center = center
        umps.dot = 1
        umps.tensors = VectorSpTensor(tensors)
        umps = umps.finalize()
        if inited:
            release_memory()
            Global.frame = frame_back
        return umps


class SymbolicMPOTools:
    @staticmethod
    def tr_expr_from_block2(expr):
        if expr.get_type() == OpTypes.Zero:
            return 0
        elif expr.get_type() == OpTypes.Elem:
            op = AOpElement.parse(repr(expr.abs())) * expr.factor
            q = expr.q_label
            op.q_label = ASZ(q.n, q.twos, q.pg)
            return op
        else:
            assert False

    @staticmethod
    def tr_expr_to_block2(expr):
        if expr == 0:
            return OpExpr()
        elif isinstance(expr, AOpElement):
            nsi = len(expr.site_index)
            if repr(expr.name) in ["X", "XL", "XR"]:
                si = SiteIndex(expr.site_index, ())
            else:
                si = SiteIndex(expr.site_index[:nsi // 2], expr.site_index[nsi // 2:])
            name = getattr(OpNames, repr(expr.name))
            q = expr.q_label
            ql = SZ(q.n, q.twos, q.pg)
            return OpElement(name, si, ql, expr.factor)
        else:
            assert False

    @staticmethod
    def from_block2(bmpo):
        """Translate block2 (un-simplified) MPO to pyblock2 symbolic MPO."""
        assert bmpo.schemer is None
        if isinstance(bmpo, MPOQC):
            assert bmpo.mode == QCTypes.NC or bmpo.mode == QCTypes.CN
        tensors = [None] * bmpo.n_sites
        for i in range(0, bmpo.n_sites):
            assert bmpo.tensors[i].lmat == bmpo.tensors[i].rmat
            mat = bmpo.tensors[i].lmat
            ops = bmpo.tensors[i].ops
            lop = bmpo.right_operator_names[i]
            rop = bmpo.left_operator_names[i]
            matd = np.array([SymbolicMPOTools.tr_expr_from_block2(expr)
                             for expr in mat.data], dtype=object)
            if i == 0:
                xmat = ASymbolicRowVector(n_cols=len(matd), data=matd)
            elif i == bmpo.n_sites - 1:
                xmat = ASymbolicColumnVector(n_rows=len(matd), data=matd)
            else:
                idxd = [(j, k) for j, k in mat.indices]
                xmat = ASymbolicMatrix(
                    n_rows=mat.m, n_cols=mat.n, indices=idxd, data=list(matd))
            xops = {}
            for expr, spmat in ops.items():
                xexpr = SymbolicMPOTools.tr_expr_from_block2(expr)
                if spmat.factor == 0 or spmat.info.n == 0:
                    xops[xexpr] = 0
                    continue
                map_blocks_odd = {}
                map_blocks_even = {}
                for p in range(spmat.info.n):
                    qu = spmat.info.quanta[p].get_bra(spmat.info.delta_quantum)
                    qd = spmat.info.quanta[p].get_ket()
                    map_blocks = map_blocks_odd if (
                        qu - qd).is_fermion else map_blocks_even
                    qx = (ASZ(qu.n, qu.twos, qu.pg), ASZ(qd.n, qd.twos, qd.pg))
                    map_blocks[qx] = SubTensor(
                        q_labels=qx, reduced=spmat.factor * np.array(spmat[p]))
                xops[xexpr] = FermionTensor(
                    odd=list(map_blocks_odd.values()), even=list(map_blocks_even.values()))
            lopd = np.array([SymbolicMPOTools.tr_expr_from_block2(expr)
                             for expr in lop.data], dtype=object)
            ropd = np.array([SymbolicMPOTools.tr_expr_from_block2(expr)
                             for expr in rop.data], dtype=object)
            xlop = ASymbolicColumnVector(n_rows=len(lopd), data=lopd)
            xrop = ASymbolicRowVector(n_cols=len(ropd), data=ropd)
            tensors[i] = SymbolicSparseTensor(xmat, xops, xlop, xrop)
        return MPS(tensors=tensors, const=bmpo.const_e)

    @staticmethod
    def to_block2(mpo):
        """Translate pyblock2 symbolic MPO to block2 (un-simplified) MPO."""
        bmpo = MPO(mpo.n_sites)
        tensors = [None] * mpo.n_sites
        lops = [None] * mpo.n_sites
        rops = [None] * mpo.n_sites
        site_op_infos = [None] * mpo.n_sites
        site_basis = [None] * mpo.n_sites
        for i in range(0, mpo.n_sites):
            ts = mpo.tensors[i]
            assert isinstance(ts, SymbolicSparseTensor)
            pinfo = ts.physical_infos
            pinfo = pinfo[0] | pinfo[1]
            site_basis[i] = StateInfo()
            site_basis[i].allocate(len(pinfo))
            for ix, (k, v) in enumerate(pinfo.items()):
                site_basis[i].quanta[ix] = SZ(k.n, k.twos, k.pg)
                site_basis[i].n_states[ix] = v
            site_basis[i].sort_states()
            tensors[i] = OperatorTensor()
            site_op_infos[i] = {}
            dalloc = DoubleVectorAllocator()
            ialloc = IntVectorAllocator()
            matd = [SymbolicMPOTools.tr_expr_to_block2(expr) for expr in ts.mat.data]
            if isinstance(ts.mat, ASymbolicRowVector):
                tensors[i].lmat = SymbolicRowVector(ts.mat.n_cols)
                tensors[i].lmat.data = VectorOpExpr(matd)
            elif isinstance(ts.mat, ASymbolicColumnVector):
                tensors[i].lmat = SymbolicColumnVector(ts.mat.n_rows)
                tensors[i].lmat.data = VectorOpExpr(matd)
            elif isinstance(ts.mat, ASymbolicMatrix):
                tensors[i].lmat = SymbolicMatrix(ts.mat.n_rows, ts.mat.n_cols)
                tensors[i].lmat.data = VectorOpExpr(matd)
                tensors[i].lmat.indices = VectorPIntInt(ts.mat.indices)
            tensors[i].rmat = tensors[i].lmat
            zero = SparseMatrix(dalloc)
            zero.factor = 0
            idq = SZ(0, 0, 0)
            iop = OpElement(OpNames.I, SiteIndex(), idq, 1.0)
            for expr, spmat in ts.ops.items():
                xexpr = SymbolicMPOTools.tr_expr_to_block2(expr)
                if isinstance(spmat, numbers.Number) or spmat.n_blocks == 0:
                    tensors[i].ops[xexpr] = zero
                    continue
                assert isinstance(spmat, FermionTensor)
                assert spmat.odd.n_blocks == 0 or spmat.even.n_blocks == 0
                spx = spmat.even if spmat.odd.n_blocks == 0 else spmat.odd
                dq = spx.blocks[0].q_labels[0] - spx.blocks[0].q_labels[1]
                dq = SZ(dq.n, dq.twos, dq.pg)
                if dq not in site_op_infos[i]:
                    site_op_infos[i][dq] = SparseMatrixInfo(ialloc)
                    site_op_infos[i][dq].initialize(site_basis[i], site_basis[i],
                        dq, dq.is_fermion)
                minfo = site_op_infos[i][dq]
                xmat = SparseMatrix(dalloc)
                xmat.allocate(minfo)
                for block in spx.blocks:
                    ql, qr = block.q_labels
                    ql, qr = SZ(ql.n, ql.twos, ql.pg), SZ(qr.n, qr.twos, qr.pg)
                    iq = xmat.info.find_state(dq.combine(ql, qr))
                    xmat[iq] = np.asarray(block).flatten()
                tensors[i].ops[xexpr] = xmat
            if iop not in tensors[i].ops:
                if idq not in site_op_infos[i]:
                    site_op_infos[i][idq] = SparseMatrixInfo(ialloc)
                    site_op_infos[i][idq].initialize(site_basis[i], site_basis[i],
                        idq, idq.is_fermion)
                minfo = site_op_infos[i][idq]
                xmat = SparseMatrix(dalloc)
                xmat.allocate(minfo)
                for ix in range(0, minfo.n):
                    xmat[ix] = np.identity(minfo.n_states_ket[ix]).flatten()
                tensors[i].ops[iop] = xmat
            lopd = [SymbolicMPOTools.tr_expr_to_block2(expr) for expr in ts.lop.data]
            ropd = [SymbolicMPOTools.tr_expr_to_block2(expr) for expr in ts.rop.data]
            lops[i] = SymbolicColumnVector(len(lopd))
            rops[i] = SymbolicRowVector(len(ropd))
            lops[i].data = VectorOpExpr(lopd)
            rops[i].data = VectorOpExpr(ropd)
            site_op_infos[i] = VectorPLMatInfo(
                sorted([(k, v) for k, v in site_op_infos[i].items()], key=lambda x: x[0]))
        bmpo.const_e = mpo.const
        bmpo.tf = TensorFunctions(OperatorFunctions(CG()))
        bmpo.site_op_infos = VectorVectorPLMatInfo(site_op_infos)
        bmpo.basis = VectorStateInfo(site_basis)
        bmpo.sparse_form = "N" * mpo.n_sites
        bmpo.op = rops[-1][0]
        bmpo.right_operator_names = VectorSymbolic(lops)
        bmpo.left_operator_names = VectorSymbolic(rops)
        bmpo.tensors = VectorOpTensor(tensors)
        return bmpo


class MPOTools:
    @staticmethod
    def from_block2(bmpo):
        """Translate block2 (un-simplified) MPO to pyblock2 MPO."""
        assert bmpo.schemer is None
        if isinstance(bmpo, MPOQC):
            assert bmpo.mode == QCTypes.NC or bmpo.mode == QCTypes.CN
        tensors = [None] * bmpo.n_sites
        # translate operator name symbols to quantum labels
        idx_mps, idx_qss, idx_imps = [], [], []
        for i in range(0, bmpo.n_sites - 1):
            lidx_mp = {}
            lidx_qs = [op.q_label for op in bmpo.left_operator_names[i].data]
            for ip, q in enumerate(lidx_qs):
                if q not in lidx_mp:
                    lidx_mp[q] = []
                lidx_mp[q].append(ip)
            limp = {iv: iiv for _, v in lidx_mp.items()
                    for iiv, iv in enumerate(v)}
            idx_mps.append(lidx_mp)
            idx_qss.append(lidx_qs)
            idx_imps.append(limp)
        for i in range(0, bmpo.n_sites):
            assert bmpo.tensors[i].lmat == bmpo.tensors[i].rmat
            mat = bmpo.tensors[i].lmat
            ops = bmpo.tensors[i].ops
            map_blocks_odd = {}
            map_blocks_even = {}
            if i == 0:
                for k, expr in enumerate(mat.data):
                    if expr.get_type() == OpTypes.Zero:
                        continue
                    elif expr.get_type() == OpTypes.Elem:
                        spmat = ops[expr.abs()]
                        if spmat.factor == 0 or spmat.info.n == 0:
                            continue
                        qr = idx_qss[i][k]
                        nr = len(idx_mps[i][qr])
                        ir = idx_imps[i][k]
                        for p in range(spmat.info.n):
                            qu = spmat.info.quanta[p].get_bra(
                                spmat.info.delta_quantum)
                            qd = spmat.info.quanta[p].get_ket()
                            nu = spmat.info.n_states_bra[p]
                            nd = spmat.info.n_states_ket[p]
                            qx = (SZ(0, 0, 0), qu, qd, qr)
                            map_blocks = map_blocks_odd if (
                                qu - qd).is_fermion else map_blocks_even
                            if qx not in map_blocks:
                                map_blocks[qx] = SubTensor(
                                    q_labels=qx, reduced=np.zeros((1, nu, nd, nr)))
                            map_blocks[qx][0, :, :, ir] += expr.factor * \
                                spmat.factor * np.array(spmat[p])
                    else:
                        assert False
            elif i == bmpo.n_sites - 1:
                for k, expr in enumerate(mat.data):
                    if expr.get_type() == OpTypes.Zero:
                        continue
                    elif expr.get_type() == OpTypes.Elem:
                        spmat = ops[expr.abs()]
                        if spmat.factor == 0 or spmat.info.n == 0:
                            continue
                        ql = idx_qss[i - 1][k]
                        nl = len(idx_mps[i - 1][ql])
                        il = idx_imps[i - 1][k]
                        for p in range(spmat.info.n):
                            qu = spmat.info.quanta[p].get_bra(
                                spmat.info.delta_quantum)
                            qd = spmat.info.quanta[p].get_ket()
                            nu = spmat.info.n_states_bra[p]
                            nd = spmat.info.n_states_ket[p]
                            qx = (ql, qu, qd, SZ(0, 0, 0))
                            map_blocks = map_blocks_odd if (
                                qu - qd).is_fermion else map_blocks_even
                            if qx not in map_blocks:
                                map_blocks[qx] = SubTensor(
                                    q_labels=qx, reduced=np.zeros((nl, nu, nd, 1)))
                            map_blocks[qx][il, :, :, 0] += expr.factor * \
                                spmat.factor * np.array(spmat[p])
                    else:
                        assert False
            else:
                for (j, k), expr in zip(mat.indices, mat.data):
                    if expr.get_type() == OpTypes.Zero:
                        continue
                    elif expr.get_type() == OpTypes.Elem:
                        spmat = ops[expr.abs()]
                        if spmat.factor == 0 or spmat.info.n == 0:
                            continue
                        ql, qr = idx_qss[i - 1][j], idx_qss[i][k]
                        nl, nr = len(idx_mps[i - 1][ql]
                                     ), len(idx_mps[i][qr])
                        il, ir = idx_imps[i - 1][j], idx_imps[i][k]
                        for p in range(spmat.info.n):
                            qu = spmat.info.quanta[p].get_bra(
                                spmat.info.delta_quantum)
                            qd = spmat.info.quanta[p].get_ket()
                            nu = spmat.info.n_states_bra[p]
                            nd = spmat.info.n_states_ket[p]
                            qx = (ql, qu, qd, qr)
                            map_blocks = map_blocks_odd if (
                                qu - qd).is_fermion else map_blocks_even
                            if np.linalg.norm(np.array(spmat[p])) == 0:
                                continue
                            if qx not in map_blocks:
                                map_blocks[qx] = SubTensor(
                                    q_labels=qx, reduced=np.zeros((nl, nu, nd, nr)))
                            map_blocks[qx][il, :, :, ir] += expr.factor * \
                                spmat.factor * np.array(spmat[p])
                    else:
                        assert False
            tensors[i] = FermionTensor(
                odd=list(map_blocks_odd.values()), even=list(map_blocks_even.values()))
        for i in range(0, len(tensors)):
            for block in tensors[i].odd.blocks + tensors[i].even.blocks:
                block.q_labels = tuple(ASZ(x.n, x.twos, x.pg)
                                       for x in block.q_labels)
        return MPS(tensors=tensors, const=bmpo.const_e)

    @staticmethod
    def to_block2(mpo):
        """Translate pyblock2 MPO to block2 (un-simplified) MPO."""
        return SymbolicMPOTools.to_block2(mpo.to_symbolic())
