
/*
 * pyblock3: An Efficient python MPS/DMRG Library
 * Copyright (C) 2020 The pyblock3 developers. All Rights Reserved.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 *
 */

#include "qc_hamiltonian.hpp"
#include "flat_sparse.hpp"
#include "qc_mpo.hpp"
#include "sz.hpp"
#include <vector>

#define TINY (1E-20)

namespace py = pybind11;

using namespace std;
using namespace block2;

typedef tuple<py::array_t<uint32_t>, py::array_t<uint32_t>,
              py::array_t<uint64_t>>
    op_skeleton;

void op_matmul(const op_skeleton &ska, const op_skeleton &skb,
               const op_skeleton &skc, const double *pa, const double *pb,
               double *pc, double scale = 1.0, double cfactor = 1.0) {
    int na = get<0>(ska).shape()[0], nb = get<0>(skb).shape()[0],
        nc = get<0>(skc).shape()[0];
    const uint32_t *pqa = get<0>(ska).data(), *pqb = get<0>(skb).data(),
                   *pqc = get<0>(skc).data();
    const uint32_t *psha = get<1>(ska).data(), *pshb = get<1>(skb).data(),
                   *pshc = get<1>(skc).data();
    const uint64_t *pia = get<2>(ska).data(), *pib = get<2>(skb).data(),
                   *pic = get<2>(skc).data();
    for (int ic = 0; ic < nc; ic++)
        for (int ia = 0; ia < na; ia++) {
            if (pqa[ia * 2 + 0] != pqc[ic * 2 + 0])
                continue;
            for (int ib = 0; ib < nb; ib++) {
                if (pqb[ib * 2 + 1] != pqc[ic * 2 + 1] ||
                    pqb[ib * 2 + 0] != pqa[ia * 2 + 1])
                    continue;
                int m = psha[ia * 2 + 0], n = pshb[ib * 2 + 1],
                    k = pshb[ib * 2 + 0];
                dgemm("N", "N", &n, &m, &k, &scale, pb + pib[ib], &n,
                      pa + pia[ia], &k, &cfactor, pc + pic[ic], &n);
            }
        }
}

struct RHFFCIDUMP : FCIDUMP {
    py::array_t<double> t_data;
    py::array_t<double> v_data;
    RHFFCIDUMP(const py::array_t<double> &t_data,
               const py::array_t<double> &v_data)
        : t_data(t_data), v_data(v_data) {}
    double v(uint8_t sl, uint8_t sr, uint16_t i, uint16_t j, uint16_t k,
             uint16_t l) const override {
        return *v_data.data(i, j, k, l);
    }
    double t(uint8_t s, uint16_t i, uint16_t j) const override {
        return *t_data.data(i, j);
    }
};

inline int idx(op_skeleton &skt, SZ sz) {
    int n = get<0>(skt).shape()[0];
    for (int i = 0; i < n; i++)
        if (SZ::from_q(sz) == *get<0>(skt).data(i, 1))
            return i;
    assert(false);
    return -1;
}

vector<tuple<py::array_t<uint32_t>, py::array_t<uint32_t>, py::array_t<double>,
             py::array_t<uint64_t>>>
build_qc_mpo(py::array_t<int32_t> orb_sym, py::array_t<double> t,
             py::array_t<double> v) {
    int n_sites = (int)orb_sym.size();
    vector<uint8_t> v_orb_sym(n_sites);
    for (int i = 0; i < n_sites; i++)
        v_orb_sym[i] = (uint8_t)*orb_sym.data(i);

    shared_ptr<RHFFCIDUMP> fd = make_shared<RHFFCIDUMP>(t, v);
    shared_ptr<MPOQC<SZ>> mpo_qc = make_shared<MPOQC<SZ>>(v_orb_sym, fd);
    vector<tuple<py::array_t<uint32_t>, py::array_t<uint32_t>,
                 py::array_t<double>, py::array_t<uint64_t>>>
        rr(n_sites * 2);

    SZ vacuum;
    vector<unordered_map<uint32_t, uint32_t>> infos(n_sites + 1);
    vector<vector<pair<uint32_t, uint32_t>>> qmps(n_sites + 1);
    infos[0][SZ::from_q(vacuum)] = 1;
    qmps[0].push_back(make_pair(SZ::from_q(vacuum), 0));
    infos[n_sites][SZ::from_q(vacuum)] = 1;
    qmps[n_sites].push_back(make_pair(SZ::from_q(vacuum), 0));
    for (int ii = 0; ii < n_sites - 1; ii++) {
        for (int j = 0; j < mpo_qc->left_operator_names[ii]->data.size(); j++) {
            auto mop = mpo_qc->left_operator_names[ii]->data[j];
            shared_ptr<OpElement<SZ>> op =
                dynamic_pointer_cast<OpElement<SZ>>(mop);
            uint32_t qz = SZ::from_q(op->q_label);
            qmps[ii + 1].push_back(make_pair(qz, infos[ii + 1][qz]));
            infos[ii + 1][qz]++;
        }
    }
    py::array_t<int32_t> perm(vector<ssize_t>{4});
    perm.mutable_data()[0] = 0, perm.mutable_data()[1] = 2,
    perm.mutable_data()[2] = 3, perm.mutable_data()[3] = 1;
    const int32_t *pperm = perm.data();

    for (int ii = 0; ii < n_sites; ii++) {
        cout << "QC MPO site" << setw(4) << ii << " / " << n_sites << endl;
        unordered_map<uint32_t, uint32_t> basis;
        basis[SZ::from_q(SZ(0, 0, 0))] = 1;
        basis[SZ::from_q(SZ(1, 1, v_orb_sym[ii]))] = 1;
        basis[SZ::from_q(SZ(1, -1, v_orb_sym[ii]))] = 1;
        basis[SZ::from_q(SZ(2, 0, 0))] = 1;
        vector<map_uint_uint<SZ>> xinfos = {
            (map_uint_uint<SZ> &)infos.at(ii),
            (map_uint_uint<SZ> &)infos.at(ii + 1), (map_uint_uint<SZ> &)basis,
            (map_uint_uint<SZ> &)basis};
        auto skl = flat_sparse_tensor_skeleton<SZ>(xinfos, "+-+-",
                                                   SZ::from_q(SZ(0, 0, 0)));
        // separate odd and even
        int n_odd = 0, n_total = get<0>(skl).shape()[0];
        ssize_t size_odd = 0;
        vector<bool> skf(n_total, false);
        const uint32_t *psklqs = get<0>(skl).data();
        const uint32_t *psklshs = get<1>(skl).data();
        const uint64_t *psklis = get<2>(skl).data();
        const ssize_t sklqi = get<0>(skl).strides()[0] / sizeof(uint32_t),
                      sklqj = get<0>(skl).strides()[1] / sizeof(uint32_t);
        for (int i = 0; i < n_total; i++)
            if (SZ::to_q(psklqs[i * sklqi + 2 * sklqj]).is_fermion() !=
                SZ::to_q(psklqs[i * sklqi + 3 * sklqj]).is_fermion())
                n_odd++, skf[i] = true, size_odd += psklis[i + 1] - psklis[i];
        int n_even = n_total - n_odd;
        ssize_t size_even = psklis[n_total] - size_odd;
        auto &rodd = rr[ii * 2], &reven = rr[ii * 2 + 1];
        auto &oqs = get<0>(rodd), &oshs = get<1>(rodd);
        auto &oi = get<3>(rodd);
        auto &eqs = get<0>(reven), &eshs = get<1>(reven);
        auto &ei = get<3>(reven);
        oqs = py::array_t<uint32_t>(vector<ssize_t>{n_odd, 4});
        oshs = py::array_t<uint32_t>(vector<ssize_t>{n_odd, 4});
        oi = py::array_t<uint64_t>(vector<ssize_t>{n_odd + 1});
        eqs = py::array_t<uint32_t>(vector<ssize_t>{n_even, 4});
        eshs = py::array_t<uint32_t>(vector<ssize_t>{n_even, 4});
        auto odata = py::array_t<double>(vector<ssize_t>{size_odd});
        auto edata = py::array_t<double>(vector<ssize_t>{size_even});
        ei = py::array_t<uint64_t>(vector<ssize_t>{n_even + 1});
        uint32_t *poqs = oqs.mutable_data(), *poshs = oshs.mutable_data();
        uint64_t *poi = oi.mutable_data();
        uint32_t *peqs = eqs.mutable_data(), *peshs = eshs.mutable_data();
        uint64_t *pei = ei.mutable_data();
        double *po = odata.mutable_data(), *pe = edata.mutable_data();
        memset(po, 0, sizeof(double) * size_odd);
        memset(pe, 0, sizeof(double) * size_even);
        poi[0] = pei[0] = 0;
        // map<uint64_t (uint32_t << 32 + uint32_t), data index>
        unordered_map<uint64_t, size_t> rdt_map;
        for (int i = 0, iodd = 0, ieven = 0; i < n_total; i++)
            if (skf[i]) {
                for (int j = 0; j < 4; j++) {
                    poqs[iodd * 4 + j] = psklqs[i * sklqi + j * sklqj];
                    poshs[iodd * 4 + j] = psklshs[i * sklqi + j * sklqj];
                }
                poi[iodd + 1] = poi[iodd] + psklis[i + 1] - psklis[i];
                uint64_t pk =
                    ((uint64_t)poqs[iodd * 4 + 0] << 32) | poqs[iodd * 4 + 1];
                if (rdt_map.count(pk) == 0)
                    rdt_map[pk] = poi[iodd];
                iodd++;
            } else {
                for (int j = 0; j < 4; j++) {
                    peqs[ieven * 4 + j] = psklqs[i * sklqi + j * sklqj];
                    peshs[ieven * 4 + j] = psklshs[i * sklqi + j * sklqj];
                }
                pei[ieven + 1] = pei[ieven] + psklis[i + 1] - psklis[i];
                uint64_t pk =
                    ((uint64_t)peqs[ieven * 4 + 0] << 32) | peqs[ieven * 4 + 1];
                if (rdt_map.count(pk) == 0)
                    rdt_map[pk] = pei[ieven];
                ieven++;
            }
        // site operators
        unordered_map<uint32_t, op_skeleton> sk_map;
        vector<map_uint_uint<SZ>> op_infos = {(map_uint_uint<SZ> &)basis,
                                              (map_uint_uint<SZ> &)basis};
        vector<uint32_t> sk_qs;
        int ipg = v_orb_sym[ii];
        sk_qs.push_back(SZ::from_q(vacuum));
        for (int iipg = 0; iipg < 8; iipg++)
            for (int n = -1; n <= 1; n += 2)
                for (int s = -1; s <= 1; s += 2)
                    sk_qs.push_back(SZ::from_q(SZ(n, s, iipg)));
        for (int iipg = 0; iipg < 8; iipg++)
            for (int n = -2; n <= 2; n += 2)
                for (int s = -2; s <= 2; s += 2)
                    sk_qs.push_back(SZ::from_q(SZ(n, s, iipg)));
        for (auto &k : sk_qs)
            if (!sk_map.count(k))
                sk_map[k] = flat_sparse_tensor_skeleton<SZ>(op_infos, "+-", k);
        // prims
        vector<map<OpNames, vector<double>>> op_prims(4);
        op_skeleton skt = sk_map[SZ::from_q(vacuum)], skc[2], skd[2];
        op_prims[0][OpNames::I].resize(get<0>(skt).shape()[0]);
        op_prims[0][OpNames::I][idx(skt, SZ(0, 0, 0))] = 1.0;
        op_prims[0][OpNames::I][idx(skt, SZ(1, -1, ipg))] = 1.0;
        op_prims[0][OpNames::I][idx(skt, SZ(1, 1, ipg))] = 1.0;
        op_prims[0][OpNames::I][idx(skt, SZ(2, 0, 0))] = 1.0;
        const int sz[2] = {1, -1};
        for (uint8_t s = 0; s < 2; s++) {
            skc[s] = sk_map[SZ::from_q(SZ(1, sz[s], ipg))];
            op_prims[s][OpNames::C].resize(get<0>(skc[s]).shape()[0]);
            op_prims[s][OpNames::C][idx(skc[s], SZ(0, 0, 0))] = 1.0;
            op_prims[s][OpNames::C][idx(skc[s], SZ(1, -sz[s], ipg))] =
                s ? -1.0 : 1.0;
            skd[s] = sk_map[SZ::from_q(SZ(-1, -sz[s], ipg))];
            op_prims[s][OpNames::D].resize(get<0>(skd[s]).shape()[0]);
            op_prims[s][OpNames::D][idx(skd[s], SZ(1, sz[s], ipg))] = 1.0;
            op_prims[s][OpNames::D][idx(skd[s], SZ(2, 0, 0))] = s ? -1.0 : 1.0;
        }
        // low (&1): left index, high (>>1): right index
        const int sz_plus[4] = {2, 0, 0, -2}, sz_minus[4] = {0, -2, 2, 0};
        for (uint8_t s = 0; s < 4; s++) {
            skt = sk_map[SZ::from_q(SZ(2, sz_plus[s], 0))];
            op_prims[s][OpNames::A].resize(get<0>(skt).shape()[0]);
            op_matmul(skc[s & 1], skc[s >> 1], skt,
                      op_prims[s & 1][OpNames::C].data(),
                      op_prims[s >> 1][OpNames::C].data(),
                      op_prims[s][OpNames::A].data());
            skt = sk_map[SZ::from_q(SZ(-2, -sz_plus[s], 0))];
            op_prims[s][OpNames::AD].resize(get<0>(skt).shape()[0]);
            op_matmul(skd[s >> 1], skd[s & 1], skt,
                      op_prims[s >> 1][OpNames::D].data(),
                      op_prims[s & 1][OpNames::D].data(),
                      op_prims[s][OpNames::AD].data());
            skt = sk_map[SZ::from_q(SZ(0, sz_minus[s], 0))];
            op_prims[s][OpNames::B].resize(get<0>(skt).shape()[0]);
            op_matmul(skc[s & 1], skd[s >> 1], skt,
                      op_prims[s & 1][OpNames::C].data(),
                      op_prims[s >> 1][OpNames::D].data(),
                      op_prims[s][OpNames::B].data());
            skt = sk_map[SZ::from_q(SZ(0, -sz_minus[s], 0))];
            op_prims[s][OpNames::BD].resize(get<0>(skt).shape()[0]);
            op_matmul(skd[s & 1], skc[s >> 1], skt,
                      op_prims[s & 1][OpNames::D].data(),
                      op_prims[s >> 1][OpNames::C].data(),
                      op_prims[s][OpNames::BD].data());
        }
        // low (&1): R index, high (>>1): B index
        for (uint8_t s = 0; s < 4; s++) {
            skt = sk_map[SZ::from_q(SZ(-1, -sz[s & 1], ipg))];
            op_prims[s][OpNames::R].resize(get<0>(skt).shape()[0]);
            op_matmul(
                sk_map[SZ::from_q(SZ(0, sz_minus[(s >> 1) | (s & 2)], 0))],
                skd[s & 1], skt,
                op_prims[(s >> 1) | (s & 2)][OpNames::B].data(),
                op_prims[s & 1][OpNames::D].data(),
                op_prims[s][OpNames::R].data());
            skt = sk_map[SZ::from_q(SZ(1, sz[s & 1], ipg))];
            op_prims[s][OpNames::RD].resize(get<0>(skt).shape()[0]);
            op_matmul(
                skc[s & 1],
                sk_map[SZ::from_q(SZ(0, sz_minus[(s >> 1) | (s & 2)], 0))], skt,
                op_prims[s & 1][OpNames::C].data(),
                op_prims[(s >> 1) | (s & 2)][OpNames::B].data(),
                op_prims[s][OpNames::RD].data());
        }

        int m = ii, i, j, k, s;
        for (auto &p : mpo_qc->tensors[ii]->ops) {
            OpElement<SZ> &op = *dynamic_pointer_cast<OpElement<SZ>>(p.first);
            skt = sk_map.at(SZ::from_q(op.q_label));
            switch (op.name) {
            case OpNames::I:
            case OpNames::C:
            case OpNames::D:
            case OpNames::A:
            case OpNames::AD:
            case OpNames::B:
            case OpNames::BD:
                p.second = op_prims[op.site_index.ss()][op.name];
                break;
            case OpNames::H:
                p.second.resize(get<0>(skt).shape()[0]);
                p.second[idx(skt, SZ(0, 0, 0))] = 0.0;
                p.second[idx(skt, SZ(1, -1, ipg))] = fd->t(1, m, m);
                p.second[idx(skt, SZ(1, 1, ipg))] = fd->t(0, m, m);
                p.second[idx(skt, SZ(2, 0, 0))] =
                    fd->t(0, m, m) + fd->t(1, m, m) +
                    0.5 * (fd->v(0, 1, m, m, m, m) + fd->v(1, 0, m, m, m, m));
                break;
            case OpNames::R:
                i = op.site_index[0];
                s = op.site_index.ss();
                if (v_orb_sym[i] != v_orb_sym[m] ||
                    (abs(fd->t(s, i, m)) < TINY &&
                     abs(fd->v(s, 0, i, m, m, m)) < TINY &&
                     abs(fd->v(s, 1, i, m, m, m)) < TINY))
                    p.second.resize(0);
                else {
                    p.second = op_prims[s].at(OpNames::D);
                    for (int kk = 0; kk < (int)p.second.size(); kk++)
                        p.second[kk] *= fd->t(s, i, m) * 0.5;
                    for (uint8_t sp = 0; sp < 2; sp++) {
                        vector<double> &tmp =
                            op_prims[s + (sp << 1)].at(OpNames::R);
                        double f1 = fd->v(s, sp, i, m, m, m);
                        for (int kk = 0; kk < (int)tmp.size(); kk++)
                            p.second[kk] += f1 * tmp[kk];
                    }
                }
                break;
            case OpNames::RD:
                i = op.site_index[0];
                s = op.site_index.ss();
                if (v_orb_sym[i] != v_orb_sym[m] ||
                    (abs(fd->t(s, i, m)) < TINY &&
                     abs(fd->v(s, 0, i, m, m, m)) < TINY &&
                     abs(fd->v(s, 1, i, m, m, m)) < TINY))
                    p.second.resize(0);
                else {
                    p.second = op_prims[s].at(OpNames::C);
                    for (int kk = 0; kk < (int)p.second.size(); kk++)
                        p.second[kk] *= fd->t(s, i, m) * 0.5;
                    for (uint8_t sp = 0; sp < 2; sp++) {
                        vector<double> &tmp =
                            op_prims[s + (sp << 1)].at(OpNames::RD);
                        double f1 = fd->v(s, sp, i, m, m, m);
                        for (int kk = 0; kk < (int)tmp.size(); kk++)
                            p.second[kk] += f1 * tmp[kk];
                    }
                }
                break;
            case OpNames::P:
                i = op.site_index[0];
                k = op.site_index[1];
                s = op.site_index.ss();
                if (abs(fd->v(s & 1, s >> 1, i, m, k, m)) < TINY)
                    p.second.resize(0);
                else {
                    p.second = op_prims[s].at(OpNames::AD);
                    for (int kk = 0; kk < (int)p.second.size(); kk++)
                        p.second[kk] *= fd->v(s & 1, s >> 1, i, m, k, m);
                }
                break;
            case OpNames::PD:
                i = op.site_index[0];
                k = op.site_index[1];
                s = op.site_index.ss();
                if (abs(fd->v(s & 1, s >> 1, i, m, k, m)) < TINY)
                    p.second.resize(0);
                else {
                    p.second = op_prims[s].at(OpNames::A);
                    for (int kk = 0; kk < (int)p.second.size(); kk++)
                        p.second[kk] *= fd->v(s & 1, s >> 1, i, m, k, m);
                }
                break;
            case OpNames::Q:
                i = op.site_index[0];
                j = op.site_index[1];
                s = op.site_index.ss();
                switch (s) {
                case 0U:
                case 3U:
                    if (abs(fd->v(s & 1, s >> 1, i, m, m, j)) < TINY &&
                        abs(fd->v(s & 1, 0, i, j, m, m)) < TINY &&
                        abs(fd->v(s & 1, 1, i, j, m, m)) < TINY)
                        p.second.resize(0);
                    else {
                        p.second =
                            op_prims[(s >> 1) | ((s & 1) << 1)].at(OpNames::B);
                        for (int kk = 0; kk < (int)p.second.size(); kk++)
                            p.second[kk] *= -fd->v(s & 1, s >> 1, i, m, m, j);
                        for (uint8_t sp = 0; sp < 2; sp++) {
                            vector<double> &tmp =
                                op_prims[sp | (sp << 1)].at(OpNames::B);
                            double f1 = fd->v(s & 1, sp, i, j, m, m);
                            for (int kk = 0; kk < (int)tmp.size(); kk++)
                                p.second[kk] += f1 * tmp[kk];
                        }
                    }
                    break;
                case 1U:
                case 2U:
                    if (abs(fd->v(s & 1, s >> 1, i, m, m, j)) < TINY)
                        p.second.resize(0);
                    else {
                        p.second =
                            op_prims[(s >> 1) | ((s & 1) << 1)].at(OpNames::B);
                        for (int kk = 0; kk < (int)p.second.size(); kk++)
                            p.second[kk] *= -fd->v(s & 1, s >> 1, i, m, m, j);
                    }
                    break;
                }
                break;
            default:
                assert(false);
            }
        }
        auto &pmat = mpo_qc->tensors[ii]->lmat;
        for (int i = 0; i < pmat->data.size(); i++) {
            int il = 0, ir = 0;
            if (ii == 0)
                ir = i;
            else if (ii == n_sites - 1)
                il = i;
            else {
                auto x =
                    dynamic_pointer_cast<SymbolicMatrix<SZ>>(pmat)->indices[i];
                il = x.first;
                ir = x.second;
            }
            uint64_t ql = qmps[ii][il].first, qr = qmps[ii + 1][ir].first;
            int ipl = qmps[ii][il].second, ipr = qmps[ii + 1][ir].second;
            int npl = infos[ii].at((uint32_t)ql),
                npr = infos[ii + 1].at((uint32_t)qr);
            const int incx = 1;
            if (!rdt_map.count((ql << 32) | qr))
                continue;
            size_t pir = rdt_map.at((ql << 32) | qr);
            double *pr = SZ::to_q(ql).is_fermion() == SZ::to_q(qr).is_fermion()
                             ? pe
                             : po;
            shared_ptr<OpExpr<SZ>> x = pmat->data[i];
            switch (x->get_type()) {
            case OpTypes::Zero:
                break;
            case OpTypes::Elem: {
                shared_ptr<OpElement<SZ>> op =
                    dynamic_pointer_cast<OpElement<SZ>>(x);
                vector<double> &xv = mpo_qc->tensors[ii]->ops.at(abs_value(x));
                if (xv.size() != 0) {
                    double *term_data = xv.data();
                    double factor = op->factor;
                    skt = sk_map.at(SZ::from_q(op->q_label));
                    const int n_blocks = get<0>(skt).shape()[0];
                    const uint64_t *pb = get<2>(skt).data();
                    assert(pb[n_blocks] == xv.size());
                    for (int ib = 0; ib < n_blocks; ib++) {
                        int nb = pb[ib + 1] - pb[ib];
                        daxpy(&nb, &factor, term_data + pb[ib], &incx,
                              pr + pir + (size_t)pb[ib] * npl * npr +
                                  nb * ((size_t)npr * ipl + ipr),
                              &incx);
                    }
                }
            } break;
            case OpTypes::Sum:
                for (auto &r : dynamic_pointer_cast<OpSum<SZ>>(x)->strings) {
                    shared_ptr<OpElement<SZ>> op =
                        dynamic_pointer_cast<OpElement<SZ>>(r->get_op());
                    vector<double> &xv = mpo_qc->tensors[ii]->ops.at(
                        abs_value((shared_ptr<OpExpr<SZ>>)op));
                    if (xv.size() != 0) {
                        double *term_data = xv.data();
                        double factor = op->factor;
                        skt = sk_map.at(SZ::from_q(op->q_label));
                        const int n_blocks = get<0>(skt).shape()[0];
                        const uint64_t *pb = get<2>(skt).data();
                        assert(pb[n_blocks] == xv.size());
                        for (int ib = 0; ib < n_blocks; ib++) {
                            int nb = pb[ib + 1] - pb[ib];
                            daxpy(&nb, &factor, term_data + pb[ib], &incx,
                                  pr + pir + (size_t)pb[ib] * npl * npr +
                                      nb * ((size_t)npr * ipl + ipr),
                                  &incx);
                        }
                    }
                }
                break;
            default:
                assert(false);
            }
        }
        // transpose
        auto &todata = get<2>(rodd), &tedata = get<2>(reven);
        todata = py::array_t<double>(vector<ssize_t>{size_odd});
        tedata = py::array_t<double>(vector<ssize_t>{size_even});
        flat_sparse_tensor_transpose<SZ>(oshs, odata, oi, perm, todata);
        flat_sparse_tensor_transpose<SZ>(eshs, edata, ei, perm, tedata);
        for (int i = 0, iodd = 0, ieven = 0; i < n_total; i++)
            if (skf[i]) {
                for (int j = 0; j < 4; j++) {
                    poqs[iodd * 4 + j] = psklqs[i * sklqi + pperm[j] * sklqj];
                    poshs[iodd * 4 + j] = psklshs[i * sklqi + pperm[j] * sklqj];
                }
                iodd++;
            } else {
                for (int j = 0; j < 4; j++) {
                    peqs[ieven * 4 + j] = psklqs[i * sklqi + pperm[j] * sklqj];
                    peshs[ieven * 4 + j] =
                        psklshs[i * sklqi + pperm[j] * sklqj];
                }
                ieven++;
            }
    }
    return rr;
}
