
#include "flat_sparse.hpp"
#include "sz.hpp"
#include <algorithm>
#include <cstring>
#include <numeric>
#include <iostream>
#include <list>
#include <math.h>

void flat_fermion_tensor_transpose(const py::array_t<uint32_t> &aqs,
                                  const py::array_t<uint32_t> &ashs,
                                  const py::array_t<double> &adata,
                                  const py::array_t<uint32_t> &aidxs,
                                  const py::array_t<int32_t> &perm,
                                  py::array_t<double> &cdata) {
    int n_blocks_a = (int)ashs.shape()[0], ndima = (int)ashs.shape()[1];
    const ssize_t asi = ashs.strides()[0] / sizeof(uint32_t),
                  asj = ashs.strides()[1] / sizeof(uint32_t);
    const int *perma = (const int *)perm.data();
    const double *pa = adata.data();
    const uint32_t *pia = aidxs.data(), *psha = ashs.data();
    double *pc = cdata.mutable_data();

    const uint32_t *apqs = aqs.data();
    int phase_a[n_blocks_a];
    for (int ia = 0; ia < n_blocks_a; ia++) {
        int pnuma[ndima];
        for (int j=0; j<ndima; j++) {
            pnuma[j] = to_sz(apqs[ia * asi + j * asj]).n();
        }
        int aparity_counter = 0;
        list<int> acounted;
        for (int xid=0; xid<ndima; xid++) {
            int idx_a = perma[xid];
            for (int j=0; j<idx_a; j++) {
                bool a_not_counted = !(std::find(acounted.begin(), acounted.end(), j) != acounted.end());
                if (a_not_counted){
                    aparity_counter += pnuma[j] * pnuma[idx_a];
                }
            }
            acounted.push_back(idx_a);
        }
        phase_a[ia] = pow(-1.0, (double)aparity_counter);
    }

    for (int ia = 0; ia < n_blocks_a; ia++) {
        const double *a = pa + pia[ia];
        double *c = pc + pia[ia];
        int shape_a[ndima];
        for (int i = 0; i < ndima; i++)
            shape_a[i] = psha[ia * asi + i * asj];
        uint32_t size_a = pia[ia + 1] - pia[ia];
        tensor_transpose_impl(ndima, size_a, perma, shape_a, a, c, phase_a[ia], 0.0);
    }
}

tuple<py::array_t<uint32_t>, py::array_t<uint32_t>, py::array_t<double>,
      py::array_t<uint32_t>>
flat_fermion_tensor_tensordot(
    const py::array_t<uint32_t> &aqs, const py::array_t<uint32_t> &ashs,
    const py::array_t<double> &adata, const py::array_t<uint32_t> &aidxs,
    const py::array_t<uint32_t> &bqs, const py::array_t<uint32_t> &bshs,
    const py::array_t<double> &bdata, const py::array_t<uint32_t> &bidxs,
    const py::array_t<int> &idxa, const py::array_t<int> &idxb) {
    if (aqs.shape()[0] == 0)
        return std::make_tuple(aqs, ashs, adata, aidxs);
    else if (bqs.shape()[0] == 0)
        return std::make_tuple(bqs, bshs, bdata, bidxs);

    int n_blocks_a = (int)aqs.shape()[0], ndima = (int)aqs.shape()[1];
    int n_blocks_b = (int)bqs.shape()[0], ndimb = (int)bqs.shape()[1];
    int nctr = (int)idxa.shape()[0];
    int ndimc = ndima - nctr + ndimb - nctr;
    const ssize_t asi = aqs.strides()[0] / sizeof(uint32_t),
                  asj = aqs.strides()[1] / sizeof(uint32_t);
    const ssize_t bsi = bqs.strides()[0] / sizeof(uint32_t),
                  bsj = bqs.strides()[1] / sizeof(uint32_t);
    assert(memcmp(aqs.strides(), ashs.strides(), 2 * sizeof(ssize_t)) == 0);
    assert(memcmp(bqs.strides(), bshs.strides(), 2 * sizeof(ssize_t)) == 0);

    // sort contracted indices (for tensor a)
    int pidxa[nctr], pidxb[nctr], ctr_idx[nctr];
    const int *ppidxa = idxa.data(), *ppidxb = idxb.data();
    for (int i = 0; i < nctr; i++)
        ctr_idx[i] = i;
    sort(ctr_idx, ctr_idx + nctr,
         [ppidxa](int a, int b) { return ppidxa[a] < ppidxa[b]; });
    for (int i = 0; i < nctr; i++)
        pidxa[i] = ppidxa[ctr_idx[i]], pidxb[i] = ppidxb[ctr_idx[i]];

    // checking whether permute is necessary
    int trans_a = 0, trans_b = 0;
    if (nctr == 0)
        trans_a = 1;
    else if (pidxa[nctr - 1] - pidxa[0] == nctr - 1) {
        if (pidxa[0] == 0 ||
            is_shape_one(ashs.data(), n_blocks_a, pidxa[0], asi, asj))
            trans_a = 1;
        else if (pidxa[nctr - 1] == ndima - 1 ||
                 is_shape_one(ashs.data() + (pidxa[nctr - 1] + 1) * asj,
                              n_blocks_a, ndima - (pidxa[nctr - 1] + 1), asi,
                              asj))
            trans_a = -1;
    }

    if (nctr == 0)
        trans_b = 1;
    else if (is_sorted(pidxb, pidxb + nctr) &&
             pidxb[nctr - 1] - pidxb[0] == nctr - 1) {
        if (pidxb[0] == 0 ||
            is_shape_one(bshs.data(), n_blocks_b, pidxb[0], bsi, bsj))
            trans_b = 1;
        else if (pidxb[nctr - 1] == ndimb - 1 ||
                 is_shape_one(bshs.data() + (pidxb[nctr - 1] + 1) * bsj,
                              n_blocks_b, ndimb - (pidxb[nctr - 1] + 1), bsi,
                              bsj))
            trans_b = -1;
    }

    // free indices
    int maska[ndima], maskb[ndimb], outa[ndima - nctr], outb[ndimb - nctr];
    memset(maska, -1, ndima * sizeof(int));
    memset(maskb, -1, ndimb * sizeof(int));
    for (int i = 0; i < nctr; i++)
        maska[pidxa[i]] = i, maskb[pidxb[i]] = i;
    for (int i = 0, j = 0; i < ndima; i++)
        if (maska[i] == -1)
            outa[j++] = i;
    for (int i = 0, j = 0; i < ndimb; i++)
        if (maskb[i] == -1)
            outb[j++] = i;

    // permutation
    int perma[ndima + n_blocks_a + 1], permb[ndimb + n_blocks_b + 1];
    int *piatr = perma + ndima, *pibtr = permb + ndimb;
    if (trans_a == 0) {
        memset(perma, -1, sizeof(int) * (ndima + n_blocks_a + 1));
        for (int i = 0; i < nctr; i++)
            perma[i] = pidxa[i];
        for (int i = nctr; i < ndima; i++)
            perma[i] = outa[i - nctr];
    }
    if (trans_b == 0) {
        memset(permb, -1, sizeof(int) * (ndimb + n_blocks_b + 1));
        for (int i = 0; i < nctr; i++)
            permb[i] = pidxb[i];
        for (int i = nctr; i < ndimb; i++)
            permb[i] = outb[i - nctr];
    }

    // free and contracted dims
    int a_free_dim[n_blocks_a], b_free_dim[n_blocks_b], ctr_dim[n_blocks_a];
    int a_free_dims[n_blocks_a][ndima - nctr],
        b_free_dims[n_blocks_b][ndimb - nctr];
    const uint32_t *psh = ashs.data();
    for (int i = 0; i < n_blocks_a; i++) {
        a_free_dim[i] = ctr_dim[i] = 1;
        for (int j = 0; j < nctr; j++)
            ctr_dim[i] *= psh[i * asi + pidxa[j] * asj];
        for (int j = 0; j < ndima - nctr; j++)
            a_free_dim[i] *= (a_free_dims[i][j] = psh[i * asi + outa[j] * asj]);
    }
    psh = bshs.data();
    for (int i = 0; i < n_blocks_b; i++) {
        b_free_dim[i] = 1;
        for (int j = 0; j < ndimb - nctr; j++)
            b_free_dim[i] *= (b_free_dims[i][j] = psh[i * bsi + outb[j] * bsj]);
    }

    // contracted q_label hashs
    size_t ctrqas[n_blocks_a], ctrqbs[n_blocks_b], outqas[n_blocks_a],
        outqbs[n_blocks_b];
    double phase_a[n_blocks_a], phase_b[n_blocks_b];

    psh = aqs.data();
    const uint32_t *apqs = aqs.data();

    for (int i = 0; i < n_blocks_a; i++) {
        ctrqas[i] = q_labels_hash(psh + i * asi, nctr, pidxa, asj);
        outqas[i] = q_labels_hash(psh + i * asi, ndima - nctr, outa, asj);

        int pnuma[ndima];
        for (int j=0; j<ndima; j++) {
            pnuma[j] = to_sz(apqs[i * asi + j * asj]).n();
        }
        int aparity_counter = 0;
        list<int> acounted;
        for (int xid=0; xid<nctr; xid++) {
            int idx_a = ppidxa[xid];
            for (int j=idx_a+1; j<ndima; j++) {
                bool a_not_counted = !(std::find(acounted.begin(), acounted.end(), j) != acounted.end());
                if (a_not_counted){
                    aparity_counter += pnuma[j] * pnuma[idx_a];
                }
            }
            acounted.push_back(idx_a);
        }
        phase_a[i] = pow(-1.0, (double)aparity_counter);
    }

    psh = bqs.data();
    const uint32_t *bpqs = bqs.data();
    for (int i = 0; i < n_blocks_b; i++) {
        ctrqbs[i] = q_labels_hash(psh + i * bsi, nctr, pidxb, bsj);
        outqbs[i] = q_labels_hash(psh + i * bsi, ndimb - nctr, outb, bsj);

        int pnumb[ndimb];
        for (int j=0; j<ndimb; j++) {
            pnumb[j] = to_sz(bpqs[i * bsi + j * bsj]).n();
        }
        int bparity_counter = 0;
        list<int> bcounted;
        for (int xid=0; xid<nctr; xid++) {
            int idx_b = ppidxb[xid];
            for (int j=0; j<idx_b; j++) {
                bool b_not_counted = !(std::find(bcounted.begin(), bcounted.end(), j) != bcounted.end());
                if (b_not_counted){
                    bparity_counter += pnumb[j] * pnumb[idx_b];
                }
            }
            bcounted.push_back(idx_b);
        }
        phase_b[i] = pow(-1.0, (double)bparity_counter);
    }

    unordered_map<size_t, vector<int>> map_idx_b;
    for (int i = 0; i < n_blocks_b; i++)
        map_idx_b[ctrqbs[i]].push_back(i);

    unordered_map<size_t,
                  vector<pair<vector<uint32_t>, vector<pair<int, int>>>>>
        map_out_q;
    ssize_t csize = 0;
    int n_blocks_c = 0;
    for (int ia = 0; ia < n_blocks_a; ia++) {
        if (map_idx_b.count(ctrqas[ia])) {
            piatr[ia] = 0;
            const auto &vb = map_idx_b.at(ctrqas[ia]);
            vector<uint32_t> q_out(ndimc);
            psh = aqs.data() + ia * asi;
            for (int i = 0; i < ndima - nctr; i++)
                q_out[i] = psh[outa[i] * asj];
            for (int ib : vb) {
                pibtr[ib] = 0;
                size_t hout = outqas[ia];
                hout ^= outqbs[ib] + 0x9E3779B9 + (hout << 6) + (hout >> 2);
                psh = bqs.data() + ib * bsi;
                for (int i = 0; i < ndimb - nctr; i++)
                    q_out[i + ndima - nctr] = psh[outb[i] * bsj];
                int iq = 0;
                if (map_out_q.count(hout)) {
                    auto &vq = map_out_q.at(hout);
                    for (; iq < (int)vq.size() && q_out != vq[iq].first; iq++)
                        ;
                    if (iq == (int)vq.size()) {
                        vq.push_back(make_pair(
                            q_out, vector<pair<int, int>>{make_pair(ia, ib)}));
                        csize += (ssize_t)a_free_dim[ia] * b_free_dim[ib],
                            n_blocks_c++;
                    } else
                        vq[iq].second.push_back(make_pair(ia, ib));
                } else {
                    map_out_q[hout].push_back(make_pair(
                        q_out, vector<pair<int, int>>{make_pair(ia, ib)}));
                    csize += (ssize_t)a_free_dim[ia] * b_free_dim[ib],
                        n_blocks_c++;
                }
            }
        }
    }

    vector<ssize_t> sh = {n_blocks_c, ndimc};
    py::array_t<uint32_t> cqs(sh), cshs(sh),
        cidxs(vector<ssize_t>{n_blocks_c + 1});
    assert(cqs.strides()[1] == sizeof(uint32_t));
    assert(cshs.strides()[1] == sizeof(uint32_t));
    py::array_t<double> cdata(vector<ssize_t>{csize});
    uint32_t *pcqs = cqs.mutable_data(), *pcshs = cshs.mutable_data(),
             *pcidxs = cidxs.mutable_data();
    uint32_t psha[n_blocks_a * ndima], pshb[n_blocks_b * ndimb];
    for (int i = 0; i < n_blocks_a; i++)
        for (int j = 0; j < ndima; j++)
            psha[i * ndima + j] = ashs.data()[i * asi + j * asj];
    for (int i = 0; i < n_blocks_b; i++)
        for (int j = 0; j < ndimb; j++)
            pshb[i * ndimb + j] = bshs.data()[i * bsi + j * bsj];

    pcidxs[0] = 0;
    for (auto &mq : map_out_q) {
        for (auto &mmq : mq.second) {
            int xia = mmq.second[0].first, xib = mmq.second[0].second;
            memcpy(pcqs, mmq.first.data(), ndimc * sizeof(uint32_t));
            memcpy(pcshs, a_free_dims[xia], (ndima - nctr) * sizeof(uint32_t));
            memcpy(pcshs + (ndima - nctr), b_free_dims[xib],
                   (ndimb - nctr) * sizeof(uint32_t));
            pcidxs[1] = pcidxs[0] + (uint32_t)a_free_dim[xia] * b_free_dim[xib];
            pcqs += ndimc;
            pcshs += ndimc;
            pcidxs++;
        }
    }

    // transpose
    double *pa = (double *)adata.data(), *pb = (double *)bdata.data();
    uint32_t *pia = (uint32_t *)aidxs.data(), *pib = (uint32_t *)bidxs.data();
    if (trans_a == 0) {
        int iatr = 0;
        for (int ia = 0; ia < n_blocks_a; ia++)
            if (piatr[ia] != -1)
                piatr[ia] = iatr, iatr += pia[ia + 1] - pia[ia];
        double *new_pa = new double[iatr];
        int *new_pia = (int *)piatr;
        for (int ia = 0; ia < n_blocks_a; ia++)
            if (piatr[ia] != -1) {
                double *a = pa + pia[ia], *new_a = new_pa + new_pia[ia];
                const int *shape_a = (const int *)(psha + ia * ndima);
                uint32_t size_a = pia[ia + 1] - pia[ia];
                tensor_transpose_impl(ndima, size_a, perma, shape_a, a, new_a,
                                      1.0, 0.0);
            }
        trans_a = 1;
        pa = new_pa;
        pia = (uint32_t *)new_pia;
    }

    if (trans_b == 0) {
        int ibtr = 0;
        for (int ib = 0; ib < n_blocks_b; ib++)
            if (pibtr[ib] != -1)
                pibtr[ib] = ibtr, ibtr += pib[ib + 1] - pib[ib];
        double *new_pb = new double[ibtr];
        int *new_pib = (int *)pibtr;
        for (int ib = 0; ib < n_blocks_b; ib++)
            if (pibtr[ib] != -1) {
                double *b = pb + pib[ib], *new_b = new_pb + new_pib[ib];
                const int *shape_b = (const int *)(pshb + ib * ndimb);
                uint32_t size_b = pib[ib + 1] - pib[ib];
                tensor_transpose_impl(ndimb, size_b, permb, shape_b, b, new_b,
                                      1.0, 0.0);
            }
        trans_b = 1;
        pb = new_pb;
        pib = (uint32_t *)new_pib;
    }

    auto tra = trans_a == -1 ? "n" : "t";
    auto trb = trans_b == 1 ? "n" : "t";
    const double alpha = 1.0, beta = 0.0;

    double *pc = cdata.mutable_data();
    for (auto &mq : map_out_q) {
        for (auto &mmq : mq.second) {
            int xia = 0, xib = 0;
            for (size_t i = 0; i < mmq.second.size(); i++) {
                xia = mmq.second[i].first, xib = mmq.second[i].second;
                double phase = phase_a[xia] * phase_b[xib];
                int ldb = trans_b == 1 ? b_free_dim[xib] : ctr_dim[xia];
                int lda = trans_a == -1 ? ctr_dim[xia] : a_free_dim[xia];
                int ldc = b_free_dim[xib];
                dgemm(trb, tra, &b_free_dim[xib], &a_free_dim[xia],
                      &ctr_dim[xia], &phase, pb + pib[xib], &ldb, pa + pia[xia],
                      &lda, i == 0 ? &beta : &alpha, pc, &ldc);
            }
            pc += (uint32_t)a_free_dim[xia] * b_free_dim[xib];
        }
    }

    if (pa != adata.data())
        delete[] pa;
    if (pb != bdata.data())
        delete[] pb;

    return std::make_tuple(cqs, cshs, cdata, cidxs);
}
