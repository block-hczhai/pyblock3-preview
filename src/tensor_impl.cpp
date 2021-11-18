
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

// Dense tensor api with raw pointer interface

#include "tensor_impl.hpp"

int hptt_num_threads = 1;

template <typename FL>
void tensor_transpose_impl(int ndim, size_t size, const int *perm,
                           const int *shape, const FL *a, FL *c, const FL alpha,
                           const FL beta) {
#ifdef _HAS_HPTT
    x_tensor_transpose(perm, ndim, alpha, a, shape, nullptr, beta, c, nullptr,
                       hptt_num_threads, 1);
#else
    size_t oldacc[ndim], newacc[ndim];
    oldacc[ndim - 1] = 1;
    for (int i = ndim - 1; i >= 1; i--)
        oldacc[i - 1] = oldacc[i] * shape[perm[i]];
    for (int i = 0; i < ndim; i++)
        newacc[perm[i]] = oldacc[i];
    if (beta == 0.0)
        for (size_t i = 0; i < size; i++) {
            size_t j = 0, ii = i;
            for (int k = ndim - 1; k >= 0; k--)
                j += (ii % shape[k]) * newacc[k], ii /= shape[k];
            c[j] = alpha * a[i];
        }
    else
        for (size_t i = 0; i < size; i++) {
            size_t j = 0, ii = i;
            for (int k = ndim - 1; k >= 0; k--)
                j += (ii % shape[k]) * newacc[k], ii /= shape[k];
            c[j] = alpha * a[i] + beta * c[j];
        }
#endif
}

template <typename FL>
void tensordot_impl(const FL *a, const int ndima, const ssize_t *na,
                    const FL *b, const int ndimb, const ssize_t *nb,
                    const int nctr, const int *idxa, const int *idxb, FL *c,
                    const FL alpha, const FL beta) {
    int outa[ndima - nctr], outb[ndimb - nctr];
    int a_free_dim = 1, b_free_dim = 1, ctr_dim = 1;
    set<int> idxa_set(idxa, idxa + nctr);
    set<int> idxb_set(idxb, idxb + nctr);
    for (int i = 0, ioa = 0; i < ndima; i++)
        if (!idxa_set.count(i))
            outa[ioa] = i, a_free_dim *= na[i], ioa++;
    for (int i = 0, iob = 0; i < ndimb; i++)
        if (!idxb_set.count(i))
            outb[iob] = i, b_free_dim *= nb[i], iob++;
    int trans_a = 0, trans_b = 0;

    int ctr_idx[nctr];
    for (int i = 0; i < nctr; i++)
        ctr_idx[i] = i, ctr_dim *= na[idxa[i]];
    sort(ctr_idx, ctr_idx + nctr,
         [idxa](int a, int b) { return idxa[a] < idxa[b]; });

    // checking whether permute is necessary
    if (idxa[ctr_idx[0]] == 0 && idxa[ctr_idx[nctr - 1]] == nctr - 1)
        trans_a = 1;
    else if (idxa[ctr_idx[0]] == ndima - nctr &&
             idxa[ctr_idx[nctr - 1]] == ndima - 1)
        trans_a = -1;

    if (idxb[ctr_idx[0]] == 0 && idxb[ctr_idx[nctr - 1]] == nctr - 1)
        trans_b = 1;
    else if (idxb[ctr_idx[0]] == ndimb - nctr &&
             idxb[ctr_idx[nctr - 1]] == ndimb - 1)
        trans_b = -1;

    // permute or reshape
    FL *new_a = (FL *)a, *new_b = (FL *)b;
    if (trans_a == 0) {
        vector<int> perm_a(ndima), shape_a(ndima);
        size_t size_a = 1;
        for (int i = 0; i < ndima; i++)
            shape_a[i] = na[i], size_a *= na[i];
        for (int i = 0; i < nctr; i++)
            perm_a[i] = idxa[ctr_idx[i]];
        for (int i = nctr; i < ndima; i++)
            perm_a[i] = outa[i - nctr];
        new_a = new FL[size_a];
        tensor_transpose_impl<FL>(ndima, size_a, perm_a.data(), shape_a.data(),
                                  a, new_a, 1.0, 0.0);
        trans_a = 1;
    }

    if (trans_b == 0) {
        vector<int> perm_b(ndimb), shape_b(ndimb);
        size_t size_b = 1;
        for (int i = 0; i < ndimb; i++)
            shape_b[i] = nb[i], size_b *= nb[i];
        for (int i = 0; i < nctr; i++)
            perm_b[i] = idxb[ctr_idx[i]];
        for (int i = nctr; i < ndimb; i++)
            perm_b[i] = outb[i - nctr];
        new_b = new FL[size_b];
        tensor_transpose_impl<FL>(ndimb, size_b, perm_b.data(), shape_b.data(),
                                  b, new_b, 1.0, 0.0);
        trans_b = 1;
    }

    // n == a-free, m == b-free, k = cont
    // parameter order : m, n, k
    // trans == N -> mat = m x k (fort) k x m (c++)
    // trans == N -> mat = k x n (fort) n x k (c++)
    // matc = m x n (fort) n x m (c++)

    int ldb = trans_b == 1 ? b_free_dim : ctr_dim;
    int lda = trans_a == -1 ? ctr_dim : a_free_dim;
    int ldc = b_free_dim;
    xgemm<FL>(trans_b == 1 ? "n" : "t", trans_a == -1 ? "n" : "t", &b_free_dim,
              &a_free_dim, &ctr_dim, &alpha, new_b, &ldb, new_a, &lda, &beta, c,
              &ldc);

    if (new_a != a)
        delete[] new_a;
    if (new_b != b)
        delete[] new_b;
}

// explicit template instantiation
template void tensor_transpose_impl<double>(int ndim, size_t size,
                                            const int *perm, const int *shape,
                                            const double *a, double *c,
                                            const double alpha = 1.0,
                                            const double beta = 0.0);
template void tensor_transpose_impl<complex<double>>(
    int ndim, size_t size, const int *perm, const int *shape,
    const complex<double> *a, complex<double> *c,
    const complex<double> alpha = 1.0, const complex<double> beta = 0.0);
template void tensordot_impl<double>(const double *a, const int ndima,
                                     const ssize_t *na, const double *b,
                                     const int ndimb, const ssize_t *nb,
                                     const int nctr, const int *idxa,
                                     const int *idxb, double *c,
                                     const double alpha = 1.0,
                                     const double beta = 0.0);
template void tensordot_impl<complex<double>>(
    const complex<double> *a, const int ndima, const ssize_t *na,
    const complex<double> *b, const int ndimb, const ssize_t *nb,
    const int nctr, const int *idxa, const int *idxb, complex<double> *c,
    const complex<double> alpha = 1.0, const complex<double> beta = 0.0);