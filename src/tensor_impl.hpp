
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

#pragma once

#include <algorithm>
#include <cassert>
#include <complex>
#include <cstdint>
#include <set>
#include <vector>
#ifdef _HAS_INTEL_MKL
#define MKL_Complex16 std::complex<double>
#include "mkl.h"
#endif
#ifdef _HAS_HPTT
#include "hptt.h"
#endif
#ifdef I
#undef I
#endif

#ifndef _HAS_INTEL_MKL

using namespace std;

extern "C" {

// vector scale
// vector [sx] = double [sa] * vector [sx]
extern void dscal(const int *n, const double *sa, double *sx,
                  const int *incx) noexcept;

// vector copy
// vector [dy] = [dx]
extern void dcopy(const int *n, const double *dx, const int *incx, double *dy,
                  const int *incy) noexcept;

extern void zcopy(const int *n, const complex<double> *dx, const int *incx,
                  complex<double> *dy, const int *incy) noexcept;

// vector addition
// vector [sy] = vector [sy] + double [sa] * vector [sx]
extern void daxpy(const int *n, const double *sa, const double *sx,
                  const int *incx, double *sy, const int *incy) noexcept;

extern void zaxpy(const int *n, const complex<double> *sa,
                  const complex<double> *sx, const int *incx,
                  complex<double> *sy, const int *incy) noexcept;

// vector dot product
extern double ddot(const int *n, const double *dx, const int *incx,
                   const double *dy, const int *incy) noexcept;

// Euclidean norm of a vector
extern double dnrm2(const int *n, const double *x, const int *incx) noexcept;

// matrix multiplication
// mat [c] = double [alpha] * mat [a] * mat [b] + double [beta] * mat [c]
extern void dgemm(const char *transa, const char *transb, const int *m,
                  const int *n, const int *k, const double *alpha,
                  const double *a, const int *lda, const double *b,
                  const int *ldb, const double *beta, double *c,
                  const int *ldc) noexcept;

// matrix multiplication
// mat [c] = double [alpha] * mat [a] * mat [b] + double [beta] * mat [c]
extern void zgemm(const char *transa, const char *transb, const int *m,
                  const int *n, const int *k, const complex<double> *alpha,
                  const complex<double> *a, const int *lda,
                  const complex<double> *b, const int *ldb,
                  const complex<double> *beta, complex<double> *c,
                  const int *ldc) noexcept;

// matrix-vector multiplication
// vec [y] = double [alpha] * mat [a] * vec [x] + double [beta] * vec [y]
extern void dgemv(const char *trans, const int *m, const int *n,
                  const double *alpha, const double *a, const int *lda,
                  const double *x, const int *incx, const double *beta,
                  double *y, const int *incy) noexcept;

// linear system a * x = b
extern void dgesv(const int *n, const int *nrhs, double *a, const int *lda,
                  int *ipiv, double *b, const int *ldb, int *info);

// QR factorization
extern void dgeqrf(const int *m, const int *n, double *a, const int *lda,
                   double *tau, double *work, const int *lwork, int *info);
extern void zgeqrf(const int *m, const int *n, complex<double> *a,
                   const int *lda, complex<double> *tau, complex<double> *work,
                   const int *lwork, int *info);
extern void dorgqr(const int *m, const int *n, const int *k, double *a,
                   const int *lda, const double *tau, double *work,
                   const int *lwork, int *info);
extern void zungqr(const int *m, const int *n, const int *k, complex<double> *a,
                   const int *lda, const complex<double> *tau,
                   complex<double> *work, const int *lwork, int *info);

// LQ factorization
extern void dgelqf(const int *m, const int *n, double *a, const int *lda,
                   double *tau, double *work, const int *lwork, int *info);
extern void zgelqf(const int *m, const int *n, complex<double> *a,
                   const int *lda, complex<double> *tau, complex<double> *work,
                   const int *lwork, int *info);
extern void dorglq(const int *m, const int *n, const int *k, double *a,
                   const int *lda, const double *tau, double *work,
                   const int *lwork, int *info);
extern void zunglq(const int *m, const int *n, const int *k, complex<double> *a,
                   const int *lda, const complex<double> *tau,
                   complex<double> *work, const int *lwork, int *info);

// eigenvalue problem
extern void dsyev(const char *jobz, const char *uplo, const int *n, double *a,
                  const int *lda, double *w, double *work, const int *lwork,
                  int *info);
extern void zheev(const char *jobz, const char *uplo, const int *n,
                  complex<double> *a, const int *lda, double *w,
                  complex<double> *work, const int *lwork, double *rwork,
                  int *info);

// SVD
// mat [a] = mat [u] * vector [sigma] * mat [vt]
extern void dgesvd(const char *jobu, const char *jobvt, const int *m,
                   const int *n, double *a, const int *lda, double *s,
                   double *u, const int *ldu, double *vt, const int *ldvt,
                   double *work, const int *lwork, int *info);
extern void zgesvd(const char *jobu, const char *jobvt, const int *m,
                   const int *n, complex<double> *a, const int *lda, double *s,
                   complex<double> *u, const int *ldu, complex<double> *vt,
                   const int *ldvt, complex<double> *work, const int *lwork,
                   double *rwork, int *info);

// matrix copy
// mat [b] = mat [a]
extern void dlacpy(const char *uplo, const int *m, const int *n,
                   const double *a, const int *lda, double *b, const int *ldb);
extern void zlacpy(const char *uplo, const int *m, const int *n,
                   const complex<double> *a, const int *lda, complex<double> *b,
                   const int *ldb);
}

typedef enum { CblasRowMajor = 101, CblasColMajor = 102 } CBLAS_LAYOUT;
typedef enum {
    CblasNoTrans = 111,
    CblasTrans = 112,
    CblasConjTrans = 113
} CBLAS_TRANSPOSE;

inline void cblas_dgemm_batch(
    const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE *TransA_Array,
    const CBLAS_TRANSPOSE *TransB_Array, const int *M_Array, const int *N_Array,
    const int *K_Array, const double *alpha_Array, const double **A_Array,
    const int *lda_Array, const double **B_Array, const int *ldb_Array,
    const double *beta_Array, double **C_Array, const int *ldc_Array,
    const int group_count, const int *group_size) {
    assert(Layout == CblasRowMajor);
    for (int ig = 0, i = 0; ig < group_count; ig++) {
        const char *tra = TransA_Array[ig] == CblasNoTrans ? "n" : "t";
        const char *trb = TransB_Array[ig] == CblasNoTrans ? "n" : "t";
        const int m = M_Array[ig], n = N_Array[ig], k = K_Array[ig];
        const double alpha = alpha_Array[ig], beta = beta_Array[ig];
        const int lda = lda_Array[ig], ldb = ldb_Array[ig], ldc = ldc_Array[ig];
        const int gsize = group_size[ig];
        for (int j = 0; j < gsize; j++, i++)
            dgemm(trb, tra, &n, &m, &k, &alpha, B_Array[i], &ldb, A_Array[i],
                  &lda, &beta, C_Array[i], &ldc);
    }
}

inline void cblas_zgemm_batch_impl(
    const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE *TransA_Array,
    const CBLAS_TRANSPOSE *TransB_Array, const int *M_Array, const int *N_Array,
    const int *K_Array, const complex<double> *alpha_Array,
    const complex<double> **A_Array, const int *lda_Array,
    const complex<double> **B_Array, const int *ldb_Array,
    const complex<double> *beta_Array, complex<double> **C_Array,
    const int *ldc_Array, const int group_count, const int *group_size) {
    assert(Layout == CblasRowMajor);
    for (int ig = 0, i = 0; ig < group_count; ig++) {
        const char *tra = TransA_Array[ig] == CblasNoTrans ? "n" : "t";
        const char *trb = TransB_Array[ig] == CblasNoTrans ? "n" : "t";
        const int m = M_Array[ig], n = N_Array[ig], k = K_Array[ig];
        const complex<double> alpha = alpha_Array[ig], beta = beta_Array[ig];
        const int lda = lda_Array[ig], ldb = ldb_Array[ig], ldc = ldc_Array[ig];
        const int gsize = group_size[ig];
        for (int j = 0; j < gsize; j++, i++)
            zgemm(trb, tra, &n, &m, &k, &alpha, B_Array[i], &ldb, A_Array[i],
                  &lda, &beta, C_Array[i], &ldc);
    }
}

inline void cblas_zgemm_batch(
    const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE *TransA_Array,
    const CBLAS_TRANSPOSE *TransB_Array, const int *M_Array, const int *N_Array,
    const int *K_Array, const void *alpha_Array, const void **A_Array,
    const int *lda_Array, const void **B_Array, const int *ldb_Array,
    const void *beta_Array, void **C_Array, const int *ldc_Array,
    const int group_count, const int *group_size) {
    cblas_zgemm_batch_impl(
        Layout, TransA_Array, TransB_Array, M_Array, N_Array, K_Array,
        (const complex<double> *)alpha_Array, (const complex<double> **)A_Array,
        lda_Array, (const complex<double> **)B_Array, ldb_Array,
        (const complex<double> *)beta_Array, (complex<double> **)C_Array,
        ldc_Array, group_count, group_size);
}

#endif

using namespace std;

extern int hptt_num_threads;

template <typename FL>
inline void xgemm(const char *transa, const char *transb, const int *m,
                  const int *n, const int *k, const FL *alpha, const FL *a,
                  const int *lda, const FL *b, const int *ldb, const FL *beta,
                  FL *c, const int *ldc) noexcept;

template <>
inline void xgemm<double>(const char *transa, const char *transb, const int *m,
                          const int *n, const int *k, const double *alpha,
                          const double *a, const int *lda, const double *b,
                          const int *ldb, const double *beta, double *c,
                          const int *ldc) noexcept {
    return dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <>
inline void xgemm<complex<double>>(
    const char *transa, const char *transb, const int *m, const int *n,
    const int *k, const complex<double> *alpha, const complex<double> *a,
    const int *lda, const complex<double> *b, const int *ldb,
    const complex<double> *beta, complex<double> *c, const int *ldc) noexcept {
    return zgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <typename FL>
inline void xcopy(const int *n, const FL *dx, const int *incx, FL *dy,
                  const int *incy) noexcept;

template <>
inline void xcopy<double>(const int *n, const double *dx, const int *incx,
                          double *dy, const int *incy) noexcept {
    dcopy(n, dx, incx, dy, incy);
}

template <>
inline void xcopy<complex<double>>(const int *n, const complex<double> *dx,
                                   const int *incx, complex<double> *dy,
                                   const int *incy) noexcept {
    zcopy(n, dx, incx, dy, incy);
}

template <typename FL>
inline void xaxpy(const int *n, const FL *sa, const FL *sx, const int *incx,
                  FL *sy, const int *incy) noexcept;

template <>
inline void xaxpy<double>(const int *n, const double *sa, const double *sx,
                          const int *incx, double *sy,
                          const int *incy) noexcept {
    daxpy(n, sa, sx, incx, sy, incy);
}

template <>
inline void xaxpy<complex<double>>(const int *n, const complex<double> *sa,
                                   const complex<double> *sx, const int *incx,
                                   complex<double> *sy,
                                   const int *incy) noexcept {
    zaxpy(n, sa, sx, incx, sy, incy);
}

template <typename FL>
inline void xlacpy(const char *uplo, const int *m, const int *n, const FL *a,
                   const int *lda, FL *b, const int *ldb);

template <>
inline void xlacpy(const char *uplo, const int *m, const int *n,
                   const double *a, const int *lda, double *b, const int *ldb) {
    dlacpy(uplo, m, n, a, lda, b, ldb);
}
template <>
inline void xlacpy(const char *uplo, const int *m, const int *n,
                   const complex<double> *a, const int *lda, complex<double> *b,
                   const int *ldb) {
    zlacpy(uplo, m, n, a, lda, b, ldb);
}

template <typename FL>
inline void xgeqrf(const int *m, const int *n, FL *a, const int *lda, FL *tau,
                   FL *work, const int *lwork, int *info);
template <>
inline void xgeqrf(const int *m, const int *n, double *a, const int *lda,
                   double *tau, double *work, const int *lwork, int *info) {
    dgeqrf(m, n, a, lda, tau, work, lwork, info);
}
template <>
inline void xgeqrf(const int *m, const int *n, complex<double> *a,
                   const int *lda, complex<double> *tau, complex<double> *work,
                   const int *lwork, int *info) {
    zgeqrf(m, n, a, lda, tau, work, lwork, info);
}

template <typename FL>
inline void xungqr(const int *m, const int *n, const int *k, FL *a,
                   const int *lda, const FL *tau, FL *work, const int *lwork,
                   int *info);
template <>
inline void xungqr(const int *m, const int *n, const int *k, double *a,
                   const int *lda, const double *tau, double *work,
                   const int *lwork, int *info) {
    dorgqr(m, n, k, a, lda, tau, work, lwork, info);
}
template <>
inline void xungqr(const int *m, const int *n, const int *k, complex<double> *a,
                   const int *lda, const complex<double> *tau,
                   complex<double> *work, const int *lwork, int *info) {
    zungqr(m, n, k, a, lda, tau, work, lwork, info);
}

template <typename FL>
inline void xgelqf(const int *m, const int *n, FL *a, const int *lda, FL *tau,
                   FL *work, const int *lwork, int *info);
template <>
inline void xgelqf(const int *m, const int *n, double *a, const int *lda,
                   double *tau, double *work, const int *lwork, int *info) {
    dgelqf(m, n, a, lda, tau, work, lwork, info);
}
template <>
inline void xgelqf(const int *m, const int *n, complex<double> *a,
                   const int *lda, complex<double> *tau, complex<double> *work,
                   const int *lwork, int *info) {
    zgelqf(m, n, a, lda, tau, work, lwork, info);
}

template <typename FL>
inline void xunglq(const int *m, const int *n, const int *k, FL *a,
                   const int *lda, const FL *tau, FL *work, const int *lwork,
                   int *info);
template <>
inline void xunglq(const int *m, const int *n, const int *k, double *a,
                   const int *lda, const double *tau, double *work,
                   const int *lwork, int *info) {
    dorglq(m, n, k, a, lda, tau, work, lwork, info);
}
template <>
inline void xunglq(const int *m, const int *n, const int *k, complex<double> *a,
                   const int *lda, const complex<double> *tau,
                   complex<double> *work, const int *lwork, int *info) {
    zunglq(m, n, k, a, lda, tau, work, lwork, info);
}

template <typename FL>
inline void xgesvd(const char *jobu, const char *jobvt, const int *m,
                   const int *n, FL *a, const int *lda, double *s, FL *u,
                   const int *ldu, FL *vt, const int *ldvt, FL *work,
                   const int *lwork, int *info);
template <>
inline void xgesvd(const char *jobu, const char *jobvt, const int *m,
                   const int *n, double *a, const int *lda, double *s,
                   double *u, const int *ldu, double *vt, const int *ldvt,
                   double *work, const int *lwork, int *info) {
    dgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info);
}
template <>
inline void xgesvd(const char *jobu, const char *jobvt, const int *m,
                   const int *n, complex<double> *a, const int *lda, double *s,
                   complex<double> *u, const int *ldu, complex<double> *vt,
                   const int *ldvt, complex<double> *work, const int *lwork,
                   int *info) {
    double *rwork = new double[5 * min(*m, *n)];
    zgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork,
           info);
    delete[] rwork;
}

template <typename FL>
inline void cblas_xgemm_batch(
    const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE *TransA_Array,
    const CBLAS_TRANSPOSE *TransB_Array, const int *M_Array, const int *N_Array,
    const int *K_Array, const FL *alpha_Array, const FL **A_Array,
    const int *lda_Array, const FL **B_Array, const int *ldb_Array,
    const FL *beta_Array, FL **C_Array, const int *ldc_Array,
    const int group_count, const int *group_size);
template <>
inline void cblas_xgemm_batch(
    const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE *TransA_Array,
    const CBLAS_TRANSPOSE *TransB_Array, const int *M_Array, const int *N_Array,
    const int *K_Array, const double *alpha_Array, const double **A_Array,
    const int *lda_Array, const double **B_Array, const int *ldb_Array,
    const double *beta_Array, double **C_Array, const int *ldc_Array,
    const int group_count, const int *group_size) {
    cblas_dgemm_batch(Layout, TransA_Array, TransB_Array, M_Array, N_Array,
                      K_Array, alpha_Array, A_Array, lda_Array, B_Array,
                      ldb_Array, beta_Array, C_Array, ldc_Array, group_count,
                      group_size);
}
template <>
inline void cblas_xgemm_batch(
    const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE *TransA_Array,
    const CBLAS_TRANSPOSE *TransB_Array, const int *M_Array, const int *N_Array,
    const int *K_Array, const complex<double> *alpha_Array,
    const complex<double> **A_Array, const int *lda_Array,
    const complex<double> **B_Array, const int *ldb_Array,
    const complex<double> *beta_Array, complex<double> **C_Array,
    const int *ldc_Array, const int group_count, const int *group_size) {
    cblas_zgemm_batch(Layout, TransA_Array, TransB_Array, M_Array, N_Array,
                      K_Array, (const void *)alpha_Array,
                      (const void **)A_Array, lda_Array, (const void **)B_Array,
                      ldb_Array, (const void *)beta_Array, (void **)C_Array,
                      ldc_Array, group_count, group_size);
}

template <typename FL>
inline void x_tensor_transpose(const int *perm, const int dim, const FL alpha,
                               const FL *A, const int *sizeA,
                               const int *outerSizeA, const FL beta, FL *B,
                               const int *outerSizeB, const int numThreads,
                               const int useRowMajor);

template <>
inline void x_tensor_transpose<double>(
    const int *perm, const int dim, const double alpha, const double *A,
    const int *sizeA, const int *outerSizeA, const double beta, double *B,
    const int *outerSizeB, const int numThreads, const int useRowMajor) {
#ifdef _HAS_HPTT
    dTensorTranspose(perm, dim, alpha, A, sizeA, outerSizeA, beta, B,
                     outerSizeB, numThreads, useRowMajor);
#endif
}

template <>
inline void x_tensor_transpose<complex<double>>(
    const int *perm, const int dim, const complex<double> alpha,
    const complex<double> *A, const int *sizeA, const int *outerSizeA,
    const complex<double> beta, complex<double> *B, const int *outerSizeB,
    const int numThreads, const int useRowMajor) {
#ifdef _HAS_HPTT
    zTensorTranspose(perm, dim, (double _Complex &)alpha, false,
                     (const double _Complex *)A, sizeA, outerSizeA,
                     (double _Complex &)beta, (double _Complex *)B, outerSizeB,
                     numThreads, useRowMajor);
#endif
}

template <typename FL>
void tensor_transpose_impl(int ndim, size_t size, const int *perm,
                           const int *shape, const FL *a, FL *c,
                           const FL alpha = 1.0, const FL beta = 0.0);

template <typename FL>
void tensordot_impl(const FL *a, const int ndima, const ssize_t *na,
                    const FL *b, const int ndimb, const ssize_t *nb,
                    const int nctr, const int *idxa, const int *idxb, FL *c,
                    const FL alpha = 1.0, const FL beta = 0.0);

// explicit template instantiation
extern template void tensor_transpose_impl<double>(
    int ndim, size_t size, const int *perm, const int *shape, const double *a,
    double *c, const double alpha = 1.0, const double beta = 0.0);
extern template void tensor_transpose_impl<complex<double>>(
    int ndim, size_t size, const int *perm, const int *shape,
    const complex<double> *a, complex<double> *c,
    const complex<double> alpha = 1.0, const complex<double> beta = 0.0);
extern template void tensordot_impl<double>(const double *a, const int ndima,
                                            const ssize_t *na, const double *b,
                                            const int ndimb, const ssize_t *nb,
                                            const int nctr, const int *idxa,
                                            const int *idxb, double *c,
                                            const double alpha = 1.0,
                                            const double beta = 0.0);
extern template void tensordot_impl<complex<double>>(
    const complex<double> *a, const int ndima, const ssize_t *na,
    const complex<double> *b, const int ndimb, const ssize_t *nb,
    const int nctr, const int *idxa, const int *idxb, complex<double> *c,
    const complex<double> alpha = 1.0, const complex<double> beta = 0.0);
