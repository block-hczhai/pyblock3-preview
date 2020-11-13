
#pragma once

#include <algorithm>
#include <cstdint>
#include <set>
#include <vector>
#include <cassert>
#ifdef _HAS_INTEL_MKL
#include "mkl.h"
#endif
#ifdef _HAS_HPTT
#include "hptt.h"
#endif

#ifndef _HAS_INTEL_MKL

extern "C" {

// vector scale
// vector [sx] = double [sa] * vector [sx]
extern void dscal(const int *n, const double *sa, double *sx,
                  const int *incx) noexcept;

// vector copy
// vector [dy] = [dx]
extern void dcopy(const int *n, const double *dx, const int *incx, double *dy,
                  const int *incy) noexcept;

// vector addition
// vector [sy] = vector [sy] + double [sa] * vector [sx]
extern void daxpy(const int *n, const double *sa, const double *sx,
                  const int *incx, double *sy, const int *incy) noexcept;

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
extern void dorgqr(const int *m, const int *n, const int *k, double *a,
                   const int *lda, const double *tau, double *work,
                   const int *lwork, int *info);

// LQ factorization
extern void dgelqf(const int *m, const int *n, double *a, const int *lda,
                   double *tau, double *work, const int *lwork, int *info);
extern void dorglq(const int *m, const int *n, const int *k, double *a,
                   const int *lda, const double *tau, double *work,
                   const int *lwork, int *info);

// eigenvalue problem
extern void dsyev(const char *jobz, const char *uplo, const int *n, double *a,
                  const int *lda, double *w, double *work, const int *lwork,
                  int *info);

// SVD
// mat [a] = mat [u] * vector [sigma] * mat [vt]
extern void dgesvd(const char *jobu, const char *jobvt, const int *m,
                   const int *n, double *a, const int *lda, double *s,
                   double *u, const int *ldu, double *vt, const int *ldvt,
                   double *work, const int *lwork, int *info);
// matrix copy
// mat [b] = mat [a]
extern void dlacpy(const char *uplo, const int *m, const int *n,
                   const double *a, const int *lda, double *b, const int *ldb);
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

#endif

using namespace std;

extern int hptt_num_threads;

void tensor_transpose_impl(int ndim, size_t size, const int *perm,
                           const int *shape, const double *a, double *c,
                           const double alpha = 1.0, const double beta = 0.0);

void tensordot_impl(const double *a, const int ndima, const ssize_t *na,
                    const double *b, const int ndimb, const ssize_t *nb,
                    const int nctr, const int *idxa, const int *idxb, double *c,
                    const double alpha = 1.0, const double beta = 0.0);
