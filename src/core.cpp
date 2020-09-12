
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <typeinfo>
#include <unordered_map>
#include <vector>
#ifdef _HAS_INTEL_MKL
#include "mkl.h"
#endif
#ifdef _HAS_HPTT
#include "hptt.h"
#endif

namespace py = pybind11;
using namespace std;

PYBIND11_MAKE_OPAQUE(vector<unordered_map<uint32_t, uint32_t>>);
PYBIND11_MAKE_OPAQUE(unordered_map<uint32_t, uint32_t>);

// Wall time recorder
struct Timer {
    double current;
    Timer() : current(0) { get_time(); }
    double get_time() {
        struct timeval t;
        gettimeofday(&t, NULL);
        double previous = current;
        current = t.tv_sec + 1E-6 * t.tv_usec;
        return current - previous;
    }
};

double tx = 0.0;
size_t tc = 0;

// Quantum number with particle number, projected spin
// and point group irreducible representation (non-spin-adapted)
// N and 2S must be of the same odd/even property (not checked)
// N/2S = -16384 ~ 16383
// (N: 14bits) - (2S: 14bits) - (fermion: 1bit) - (pg: 3bits)
struct SZLong {
    typedef void is_sz_t;
    uint32_t data;
    // S(invalid) must have maximal particle number n
    const static uint32_t invalid = 0x7FFFFFFFU;
    SZLong() : data(0) {}
    SZLong(uint32_t data) : data(data) {}
    SZLong(int n, int twos, int pg)
        : data((((uint32_t)n >> 1) << 18) |
               ((uint32_t)((twos & 0x7FFFU) << 3) | pg)) {}
    int n() const {
        return (int)(((((int32_t)data) >> 18) << 1) | ((data >> 3) & 1));
    }
    int twos() const { return (int)((int16_t)(data >> 2) >> 1); }
    int pg() const { return (int)(data & 0x7U); }
    void set_n(int n) {
        data = (data & 0x3FFF7U) | (((uint32_t)n >> 1) << 18) | ((n & 1) << 3);
    }
    void set_twos(int twos) {
        data = (data & 0xFFFC0007U) | ((uint32_t)((twos & 0x7FFFU) << 3));
    }
    void set_pg(int pg) { data = (data & (~0x7U)) | ((uint32_t)pg); }
    int multiplicity() const noexcept { return 1; }
    bool is_fermion() const noexcept { return (data >> 3) & 1; }
    bool operator==(SZLong other) const noexcept { return data == other.data; }
    bool operator!=(SZLong other) const noexcept { return data != other.data; }
    bool operator<(SZLong other) const noexcept { return data < other.data; }
    SZLong operator-() const noexcept {
        return SZLong((data & 0xFU) | (((~data) + (1 << 3)) & 0x3FFF8U) |
                      (((~data) + (((~data) & 0x8U) << 15)) & 0xFFFC0000U));
    }
    SZLong operator-(SZLong other) const noexcept { return *this + (-other); }
    SZLong operator+(SZLong other) const noexcept {
        return SZLong(
            ((data & 0xFFFC0000U) + (other.data & 0xFFFC0000U) +
             (((data & other.data) & 0x8U) << 15)) |
            (((data & 0x3FFF8U) + (other.data & 0x3FFF8U)) & 0x3FFF8U) |
            ((data ^ other.data) & 0xFU));
    }
    SZLong operator[](int i) const noexcept { return *this; }
    SZLong get_ket() const noexcept { return *this; }
    SZLong get_bra(SZLong dq) const noexcept { return *this + dq; }
    SZLong combine(SZLong bra, SZLong ket) const {
        return ket + *this == bra ? ket : SZLong(invalid);
    }
    size_t hash() const noexcept { return (size_t)data; }
    int count() const noexcept { return 1; }
    string to_str() const {
        stringstream ss;
        ss << "< N=" << n() << " SZ=";
        if (twos() & 1)
            ss << twos() << "/2";
        else
            ss << (twos() >> 1);
        ss << " PG=" << pg() << " >";
        return ss.str();
    }
    friend ostream &operator<<(ostream &os, SZLong c) {
        os << c.to_str();
        return os;
    }
};

namespace std {

template <> struct hash<SZLong> {
    size_t operator()(const SZLong &s) const noexcept { return s.hash(); }
};

template <> struct less<SZLong> {
    bool operator()(const SZLong &lhs, const SZLong &rhs) const noexcept {
        return lhs < rhs;
    }
};

} // namespace std

typedef SZLong SZ;

py::array_t<double> tensor_transpose(const py::array_t<double> &x,
                                     const py::array_t<int> &perm,
                                     const double alpha, const double beta) {
    int ndim = (int)x.ndim();
    vector<int> shape(ndim);
    for (int i = 0; i < ndim; i++)
        shape[i] = (int)x.shape()[i];
    vector<ssize_t> shape_y(ndim);
    for (int i = 0; i < ndim; i++)
        shape_y[i] = x.shape()[perm.data()[i]];
    py::array_t<double> c(shape_y);
#ifdef _HAS_HPTT
    dTensorTranspose(perm.data(), ndim, alpha, x.data(), shape.data(), nullptr,
                     beta, c.mutable_data(), nullptr, 1, 1);
#else
    size_t oldacc[ndim], newacc[ndim];
    oldacc[ndim - 1] = 1;
    for (int i = ndim - 1; i >= 1; i--)
        oldacc[i - 1] = oldacc[i] * shape_y[i];
    for (int i = 0; i < ndim; i++)
        newacc[perm.data()[i]] = oldacc[i];
    for (size_t i = 0; i < x.size(); i++) {
        size_t j = 0, ii = i;
        for (int k = ndim - 1; k >= 0; k--)
            j += (ii % x.shape()[k]) * newacc[k], ii /= x.shape()[k];
        c.mutable_data()[j] = alpha * x.data()[i] + beta * c.mutable_data()[j];
    }
#endif
    return c;
}

void tensordot_internal(const double *a, const int ndima, const int *shape_a,
                        const int *perm_a, int trans_a, const int a_free_dim,
                        const double *b, const int ndimb, const int *shape_b,
                        const int *perm_b, int trans_b, const int b_free_dim,
                        const int ctr_dim, double *c, const double alpha = 1.0,
                        const double beta = 0.0, int num_threads = 1) {
    // permute or reshape
    double *new_a = (double *)a, *new_b = (double *)b;
    if (trans_a == 0) {
        new_a = new double[(size_t)a_free_dim * ctr_dim];
#ifdef _HAS_HPTT
        dTensorTranspose(perm_a, ndima, 1.0, a, shape_a, nullptr, 0.0, new_a,
                         nullptr, num_threads, 1);
#else
        size_t oldacc[ndima], newacc[ndima];
        oldacc[ndima - 1] = 1;
        for (int i = ndima - 1; i >= 1; i--)
            oldacc[i - 1] = oldacc[i] * shape_a[perm_a[i]];
        for (int i = 0; i < ndima; i++)
            newacc[perm_a[i]] = oldacc[i];
        for (size_t i = 0; i < (size_t)a_free_dim * ctr_dim; i++) {
            size_t j = 0, ii = i;
            for (int k = ndima - 1; k >= 0; k--)
                j += (ii % shape_a[k]) * newacc[k], ii /= shape_a[k];
            new_a[j] = a[i];
        }
#endif
        trans_a = 1;
    }

    if (trans_b == 0) {
        new_b = new double[(size_t)b_free_dim * ctr_dim];
#ifdef _HAS_HPTT
        dTensorTranspose(perm_b, ndimb, 1.0, b, shape_b, nullptr, 0.0, new_b,
                         nullptr, num_threads, 1);
#else
        size_t oldacc[ndimb], newacc[ndimb];
        oldacc[ndimb - 1] = 1;
        for (int i = ndimb - 1; i >= 1; i--)
            oldacc[i - 1] = oldacc[i] * shape_b[perm_b[i]];
        for (int i = 0; i < ndimb; i++)
            newacc[perm_b[i]] = oldacc[i];
        for (size_t i = 0; i < (size_t)b_free_dim * ctr_dim; i++) {
            size_t j = 0, ii = i;
            for (int k = ndimb - 1; k >= 0; k--)
                j += (ii % shape_b[k]) * newacc[k], ii /= shape_b[k];
            new_b[j] = b[i];
        }
#endif
        trans_b = 1;
    }

#ifdef _HAS_INTEL_MKL
    mkl_set_num_threads(num_threads);
    mkl_set_dynamic(0);
#endif
    int ldb = trans_b == 1 ? b_free_dim : ctr_dim;
    int lda = trans_a == -1 ? ctr_dim : a_free_dim;
    int ldc = b_free_dim;
    dgemm(trans_b == 1 ? "n" : "t", trans_a == -1 ? "n" : "t", &b_free_dim,
          &a_free_dim, &ctr_dim, &alpha, new_b, &ldb, new_a, &lda, &beta, c,
          &ldc);

    if (new_a != a)
        delete[] new_a;
    if (new_b != b)
        delete[] new_b;
}

void tensordot(const double *a, const int ndima, const ssize_t *na,
               const double *b, const int ndimb, const ssize_t *nb,
               const int nctr, const int *idxa, const int *idxb, double *c,
               const double alpha = 1.0, const double beta = 0.0,
               const int num_threads = 1) {
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
    double *new_a = (double *)a, *new_b = (double *)b;
    if (trans_a == 0) {
        vector<int> perm_a(ndima), shape_a(ndima);
        size_t size_a = 1;
        for (int i = 0; i < ndima; i++)
            shape_a[i] = na[i], size_a *= na[i];
        for (int i = 0; i < nctr; i++)
            perm_a[i] = idxa[ctr_idx[i]];
        for (int i = nctr; i < ndima; i++)
            perm_a[i] = outa[i - nctr];
        new_a = new double[size_a];
#ifdef _HAS_HPTT
        dTensorTranspose(perm_a.data(), ndima, 1.0, a, shape_a.data(), nullptr,
                         0.0, new_a, nullptr, num_threads, 1);
#else
        size_t oldacc[ndima], newacc[ndima];
        oldacc[ndima - 1] = 1;
        for (int i = ndima - 1; i >= 1; i--)
            oldacc[i - 1] = oldacc[i] * shape_a[perm_a[i]];
        for (int i = 0; i < ndima; i++)
            newacc[perm_a[i]] = oldacc[i];
        for (size_t i = 0; i < size_a; i++) {
            size_t j = 0, ii = i;
            for (int k = ndima - 1; k >= 0; k--)
                j += (ii % shape_a[k]) * newacc[k], ii /= shape_a[k];
            new_a[j] = a[i];
        }
#endif
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
        new_b = new double[size_b];
#ifdef _HAS_HPTT
        dTensorTranspose(perm_b.data(), ndimb, 1.0, b, shape_b.data(), nullptr,
                         0.0, new_b, nullptr, num_threads, 1);
#else
        size_t oldacc[ndimb], newacc[ndimb];
        oldacc[ndimb - 1] = 1;
        for (int i = ndimb - 1; i >= 1; i--)
            oldacc[i - 1] = oldacc[i] * shape_b[perm_b[i]];
        for (int i = 0; i < ndimb; i++)
            newacc[perm_b[i]] = oldacc[i];
        for (size_t i = 0; i < size_b; i++) {
            size_t j = 0, ii = i;
            for (int k = ndimb - 1; k >= 0; k--)
                j += (ii % shape_b[k]) * newacc[k], ii /= shape_b[k];
            new_b[j] = b[i];
        }
#endif
        trans_b = 1;
    }

    // n == a-free, m == b-free, k = cont
    // parameter order : m, n, k
    // trans == N -> mat = m x k (fort) k x m (c++)
    // trans == N -> mat = k x n (fort) n x k (c++)
    // matc = m x n (fort) n x m (c++)

#ifdef _HAS_INTEL_MKL
    mkl_set_num_threads(num_threads);
    mkl_set_dynamic(0);
#endif
    int ldb = trans_b == 1 ? b_free_dim : ctr_dim;
    int lda = trans_a == -1 ? ctr_dim : a_free_dim;
    int ldc = b_free_dim;
    dgemm(trans_b == 1 ? "n" : "t", trans_a == -1 ? "n" : "t", &b_free_dim,
          &a_free_dim, &ctr_dim, &alpha, new_b, &ldb, new_a, &lda, &beta, c,
          &ldc);

    if (new_a != a)
        delete[] new_a;
    if (new_b != b)
        delete[] new_b;
}

py::array_t<double> tensor_tensordot(const py::array_t<double> &a,
                                     const py::array_t<double> &b,
                                     const py::array_t<int> &idxa,
                                     const py::array_t<int> &idxb, double alpha,
                                     double beta) {
    int ndima = (int)a.ndim(), ndimb = (int)b.ndim(), nctr = (int)idxa.size();
    vector<ssize_t> shapec;
    shapec.reserve(ndima + ndimb);
    shapec.insert(shapec.end(), a.shape(), a.shape() + ndima);
    shapec.insert(shapec.end(), b.shape(), b.shape() + ndima);
    for (int i = 0; i < nctr; i++)
        shapec[idxa.data()[i]] = -1, shapec[idxb.data()[i] + ndima] = -1;
    shapec.resize(
        distance(shapec.begin(), remove(shapec.begin(), shapec.end(), -1)));
    py::array_t<double> c(shapec);
    tensordot(a.data(), ndima, a.shape(), b.data(), ndimb, b.shape(), nctr,
              idxa.data(), idxb.data(), c.mutable_data(), alpha, beta, 1);
    return c;
}

inline size_t q_labels_hash(const uint32_t *qs, int nctr, const int *idxs,
                            const int inc) noexcept {
    size_t h = 0;
    for (int i = 0; i < nctr; i++)
        h ^= (size_t)qs[idxs[i] * inc] + 0x9E3779B9 + (h << 6) + (h >> 2);
    return h;
}

inline size_t q_labels_hash(const uint32_t *qs, int nctr,
                            const int inc) noexcept {
    size_t h = 0;
    for (int i = 0; i < nctr; i++)
        h ^= (size_t)qs[i * inc] + 0x9E3779B9 + (h << 6) + (h >> 2);
    return h;
}

inline bool is_shape_one(const uint32_t *shs, int n, int nfree, const int inci,
                         const int incj) noexcept {
    for (int j = 0; j < nfree * incj; j += incj)
        for (int i = 0; i < n * inci; i += inci)
            if (shs[i + j] != 1)
                return false;
    return true;
}

template <typename T> void print_array(T *x, int n, string name) {
    cout << name << " : ";
    for (int i = 0; i < n; i++)
        cout << x[i] << " ";
    cout << endl;
}

tuple<py::array_t<uint32_t>, py::array_t<uint32_t>, py::array_t<double>,
      py::array_t<uint32_t>>
flat_sparse_tensor_tensordot(
    const py::array_t<uint32_t> &aqs, const py::array_t<uint32_t> &ashs,
    const py::array_t<double> &adata, const py::array_t<uint32_t> &aidxs,
    const py::array_t<uint32_t> &bqs, const py::array_t<uint32_t> &bshs,
    const py::array_t<double> &bdata, const py::array_t<uint32_t> &bidxs,
    const py::array_t<int> &idxa, const py::array_t<int> &idxb) {
    if (aqs.shape()[0] == 0)
        return std::make_tuple(aqs, ashs, adata, aidxs);
    else if (bqs.shape()[0] == 0)
        return std::make_tuple(bqs, bshs, bdata, bidxs);
    Timer t;
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
    int perma[ndima], permb[ndimb];
    memset(perma, -1, sizeof(int) * ndima);
    memset(permb, -1, sizeof(int) * ndimb);
    if (trans_a == 0) {
        for (int i = 0; i < nctr; i++)
            perma[i] = pidxa[i];
        for (int i = nctr; i < ndima; i++)
            perma[i] = outa[i - nctr];
    }
    if (trans_b == 0) {
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
    psh = aqs.data();
    for (int i = 0; i < n_blocks_a; i++) {
        ctrqas[i] = q_labels_hash(psh + i * asi, nctr, pidxa, asj);
        outqas[i] = q_labels_hash(psh + i * asi, ndima - nctr, outa, asj);
    }
    psh = bqs.data();
    for (int i = 0; i < n_blocks_b; i++) {
        ctrqbs[i] = q_labels_hash(psh + i * bsi, nctr, pidxb, bsj);
        outqbs[i] = q_labels_hash(psh + i * bsi, ndimb - nctr, outb, bsj);
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
            const auto &vb = map_idx_b.at(ctrqas[ia]);
            vector<uint32_t> q_out(ndimc);
            psh = aqs.data() + ia * asi;
            for (int i = 0; i < ndima - nctr; i++)
                q_out[i] = psh[outa[i] * asj];
            for (int ib : vb) {
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
    double *pc = cdata.mutable_data();
    const double *pa = adata.data(), *pb = bdata.data();
    uint32_t psha[n_blocks_a * ndima], pshb[n_blocks_b * ndimb];
    for (int i = 0; i < n_blocks_a; i++)
        for (int j = 0; j < ndima; j++)
            psha[i * ndima + j] = ashs.data()[i * asi + j * asj];
    for (int i = 0; i < n_blocks_b; i++)
        for (int j = 0; j < ndimb; j++)
            pshb[i * ndimb + j] = bshs.data()[i * bsi + j * bsj];
    tx += t.get_time();
    tc++;

    const uint32_t *pia = aidxs.data(), *pib = bidxs.data();
    pcidxs[0] = 0;
    for (auto &mq : map_out_q) {
        for (auto &mmq : mq.second) {
            int xia = mmq.second[0].first, xib = mmq.second[0].second;
            memcpy(pcqs, mmq.first.data(), ndimc * sizeof(uint32_t));
            memcpy(pcshs, a_free_dims[xia], (ndima - nctr) * sizeof(uint32_t));
            memcpy(pcshs + (ndima - nctr), b_free_dims[xib],
                   (ndimb - nctr) * sizeof(uint32_t));
            pcidxs[1] = pcidxs[0] + (uint32_t)a_free_dim[xia] * b_free_dim[xib];
            for (size_t i = 0; i < mmq.second.size(); i++) {
                xia = mmq.second[i].first, xib = mmq.second[i].second;
                tensordot_internal(
                    pa + pia[xia], ndima, (const int *)(psha + xia * ndima),
                    perma, trans_a, a_free_dim[xia], pb + pib[xib], ndimb,
                    (const int *)(pshb + xib * ndimb), permb, trans_b,
                    b_free_dim[xib], ctr_dim[xia], pc, 1.0, i == 0 ? 0.0 : 1.0);
            }
            pcqs += ndimc;
            pcshs += ndimc;
            pcidxs++;
            pc += (uint32_t)a_free_dim[xia] * b_free_dim[xib];
        }
    }
    return std::make_tuple(cqs, cshs, cdata, cidxs);
}

tuple<py::array_t<uint32_t>, py::array_t<uint32_t>, py::array_t<double>,
      py::array_t<uint32_t>>
flat_sparse_tensor_tensordot_fast(
    const py::array_t<uint32_t> &aqs, const py::array_t<uint32_t> &ashs,
    const py::array_t<double> &adata, const py::array_t<uint32_t> &aidxs,
    const py::array_t<uint32_t> &bqs, const py::array_t<uint32_t> &bshs,
    const py::array_t<double> &bdata, const py::array_t<uint32_t> &bidxs,
    const py::array_t<int> &idxa, const py::array_t<int> &idxb) {
    if (aqs.shape()[0] == 0)
        return std::make_tuple(aqs, ashs, adata, aidxs);
    else if (bqs.shape()[0] == 0)
        return std::make_tuple(bqs, bshs, bdata, bidxs);
    Timer t;
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
    psh = aqs.data();
    for (int i = 0; i < n_blocks_a; i++) {
        ctrqas[i] = q_labels_hash(psh + i * asi, nctr, pidxa, asj);
        outqas[i] = q_labels_hash(psh + i * asi, ndima - nctr, outa, asj);
    }
    psh = bqs.data();
    for (int i = 0; i < n_blocks_b; i++) {
        ctrqbs[i] = q_labels_hash(psh + i * bsi, nctr, pidxb, bsj);
        outqbs[i] = q_labels_hash(psh + i * bsi, ndimb - nctr, outb, bsj);
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
    tx += t.get_time();
    tc++;

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
#ifdef _HAS_HPTT
                dTensorTranspose(perma, ndima, 1.0, a, shape_a, nullptr, 0.0,
                                 new_a, nullptr, 1, 1);
#else
                size_t oldacc[ndima], newacc[ndima];
                oldacc[ndima - 1] = 1;
                for (int i = ndima - 1; i >= 1; i--)
                    oldacc[i - 1] = oldacc[i] * shape_a[perma[i]];
                for (int i = 0; i < ndima; i++)
                    newacc[perma[i]] = oldacc[i];
                for (size_t i = 0; i < (size_t)size_a; i++) {
                    size_t j = 0, ii = i;
                    for (int k = ndima - 1; k >= 0; k--)
                        j += (ii % shape_a[k]) * newacc[k], ii /= shape_a[k];
                    new_a[j] = a[i];
                }
#endif
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
#ifdef _HAS_HPTT
                dTensorTranspose(permb, ndimb, 1.0, b, shape_b, nullptr, 0.0,
                                 new_b, nullptr, 1, 1);
#else
                size_t oldacc[ndimb], newacc[ndimb];
                oldacc[ndimb - 1] = 1;
                for (int i = ndimb - 1; i >= 1; i--)
                    oldacc[i - 1] = oldacc[i] * shape_b[permb[i]];
                for (int i = 0; i < ndimb; i++)
                    newacc[permb[i]] = oldacc[i];
                for (size_t i = 0; i < (size_t)size_b; i++) {
                    size_t j = 0, ii = i;
                    for (int k = ndimb - 1; k >= 0; k--)
                        j += (ii % shape_b[k]) * newacc[k], ii /= shape_b[k];
                    new_b[j] = b[i];
                }
#endif
            }
        trans_b = 1;
        pb = new_pb;
        pib = (uint32_t *)new_pib;
    }

#ifdef _HAS_INTEL_MKL
    mkl_set_num_threads(1);
    mkl_set_dynamic(0);
#endif

    auto tra = trans_a == -1 ? "n" : "t";
    auto trb = trans_b == 1 ? "n" : "t";
    const double alpha = 1.0, beta = 0.0;

    double *pc = cdata.mutable_data();
    for (auto &mq : map_out_q) {
        for (auto &mmq : mq.second) {
            int xia = 0, xib = 0;
            for (size_t i = 0; i < mmq.second.size(); i++) {
                xia = mmq.second[i].first, xib = mmq.second[i].second;
                int ldb = trans_b == 1 ? b_free_dim[xib] : ctr_dim[xia];
                int lda = trans_a == -1 ? ctr_dim[xia] : a_free_dim[xia];
                int ldc = b_free_dim[xib];
                dgemm(trb, tra, &b_free_dim[xib], &a_free_dim[xia],
                      &ctr_dim[xia], &alpha, pb + pib[xib], &ldb, pa + pia[xia],
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

tuple<py::array_t<uint32_t>, py::array_t<uint32_t>, py::array_t<double>,
      py::array_t<uint32_t>>
flat_sparse_tensor_add(
    const py::array_t<uint32_t> &aqs, const py::array_t<uint32_t> &ashs,
    const py::array_t<double> &adata, const py::array_t<uint32_t> &aidxs,
    const py::array_t<uint32_t> &bqs, const py::array_t<uint32_t> &bshs,
    const py::array_t<double> &bdata, const py::array_t<uint32_t> &bidxs) {

    const int n_blocks_a = (int)aqs.shape()[0], ndima = (int)aqs.shape()[1];
    const int n_blocks_b = (int)bqs.shape()[0], ndimb = (int)bqs.shape()[1];
    const ssize_t asi = aqs.strides()[0] / sizeof(uint32_t),
                  asj = aqs.strides()[1] / sizeof(uint32_t);
    const ssize_t bsi = bqs.strides()[0] / sizeof(uint32_t),
                  bsj = bqs.strides()[1] / sizeof(uint32_t);
    assert(ndima == ndimb);

    // q_label hashs
    size_t hqas[n_blocks_a], hqbs[n_blocks_b];
    vector<uint32_t> qs(ndima * (n_blocks_a + n_blocks_b));
    const uint32_t *psh = aqs.data();
    for (int i = 0; i < n_blocks_a; i++) {
        uint32_t *pq = qs.data() + i * ndima;
        for (int j = 0; j < ndima; j++)
            pq[j] = psh[i * asi + j * asj];
        hqas[i] = q_labels_hash(pq, ndima, 1);
    }
    psh = bqs.data();
    for (int i = 0; i < n_blocks_b; i++) {
        uint32_t *pq = qs.data() + (i + n_blocks_a) * ndimb;
        for (int j = 0; j < ndimb; j++)
            pq[j] = psh[i * bsi + j * bsj];
        hqbs[i] = q_labels_hash(pq, ndimb, 1);
    }

    const uint32_t *pia = aidxs.data(), *pib = bidxs.data();
    unordered_map<size_t, vector<int>> map_idx;
    ssize_t csize = 0;
    int n_blocks_c = 0;
    int iab_mp[n_blocks_a + n_blocks_b];
    for (int i = 0; i < n_blocks_a; i++) {
        map_idx[hqas[i]].push_back(i);
        csize += (ssize_t)pia[i + 1] - pia[i];
        n_blocks_c++;
        iab_mp[i] = i;
    }
    for (int i = 0; i < n_blocks_b; i++)
        if (map_idx.count(hqbs[i]) == 0) {
            map_idx[hqbs[i]].push_back(i + n_blocks_a);
            csize += (ssize_t)pib[i + 1] - pib[i];
            n_blocks_c++;
            iab_mp[i + n_blocks_a] = i + n_blocks_a;
        } else {
            vector<int> vi(ndimb);
            int iq = 0;
            auto &vq = map_idx.at(hqbs[i]);
            for (; iq < (int)vq.size() &&
                   memcmp(qs.data() + vq[iq] * ndima,
                          qs.data() + (i + n_blocks_a) * ndima,
                          sizeof(uint32_t) * ndima) != 0;
                 iq++)
                ;
            if (iq == (int)vq.size()) {
                vq.push_back(i + n_blocks_a);
                csize += (ssize_t)pib[i + 1] - pib[i];
                n_blocks_c++;
            }
            iab_mp[i + n_blocks_a] = vq[iq];
        }

    vector<ssize_t> sh = {n_blocks_c, ndima};
    py::array_t<uint32_t> cqs(sh), cshs(sh),
        cidxs(vector<ssize_t>{n_blocks_c + 1});
    assert(cqs.strides()[1] == sizeof(uint32_t));
    assert(cshs.strides()[1] == sizeof(uint32_t));
    uint32_t *pcqs = cqs.mutable_data(), *pcshs = cshs.mutable_data(),
             *pcidxs = cidxs.mutable_data();
    int ic = 0;
    pcidxs[0] = 0;
    int pic[n_blocks_a + n_blocks_b];
    for (auto &p : map_idx) {
        for (int iab : p.second) {
            pic[iab] = ic;
            if (iab < n_blocks_a) {
                for (int j = 0; j < ndima; j++) {
                    pcqs[ic * ndima + j] = aqs.data()[iab * asi + j * asj];
                    pcshs[ic * ndima + j] = ashs.data()[iab * asi + j * asj];
                }
                pcidxs[ic + 1] =
                    pcidxs[ic] + ((uint32_t)pia[iab + 1] - pia[iab]);
            } else {
                iab -= n_blocks_a;
                for (int j = 0; j < ndimb; j++) {
                    pcqs[ic * ndima + j] = bqs.data()[iab * bsi + j * bsj];
                    pcshs[ic * ndima + j] = bshs.data()[iab * bsi + j * bsj];
                }
                pcidxs[ic + 1] =
                    pcidxs[ic] + ((uint32_t)pib[iab + 1] - pib[iab]);
            }
            ic++;
        }
    }

    py::array_t<double> cdata(vector<ssize_t>{csize});
    const double *pa = adata.data(), *pb = bdata.data();
    double *pc = cdata.mutable_data();
    const int inc = 1;
    const double alpha = 1.0;
    for (int i = 0; i < n_blocks_a; i++) {
        int iab = iab_mp[i], ic = pic[iab];
        int n = pcidxs[ic + 1] - pcidxs[ic];
        if (iab == i)
            dcopy(&n, pa + pia[i], &inc, pc + pcidxs[ic], &inc);
        else
            daxpy(&n, &alpha, pa + pia[i], &inc, pc + pcidxs[ic], &inc);
    }
    for (int i = 0; i < n_blocks_b; i++) {
        int iab = iab_mp[i + n_blocks_a], ic = pic[iab];
        int n = pcidxs[ic + 1] - pcidxs[ic];
        if (iab == i + n_blocks_a)
            dcopy(&n, pb + pib[i], &inc, pc + pcidxs[ic], &inc);
        else
            daxpy(&n, &alpha, pb + pib[i], &inc, pc + pcidxs[ic], &inc);
    }

    return std::make_tuple(cqs, cshs, cdata, cidxs);
}

void flat_sparse_tensor_transpose(const py::array_t<uint32_t> &ashs,
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
    for (int ia = 0; ia < n_blocks_a; ia++) {
        const double *a = pa + pia[ia];
        double *c = pc + pia[ia];
        int shape_a[ndima];
        for (int i = 0; i < ndima; i++)
            shape_a[i] = psha[ia * asi + i * asj];
        uint32_t size_a = pia[ia + 1] - pia[ia];
#ifdef _HAS_HPTT
        dTensorTranspose(perma, ndima, 1.0, a, shape_a, nullptr, 0.0, c,
                         nullptr, 1, 1);
#else
        size_t oldacc[ndima], newacc[ndima];
        oldacc[ndima - 1] = 1;
        for (int i = ndima - 1; i >= 1; i--)
            oldacc[i - 1] = oldacc[i] * shape_a[perma[i]];
        for (int i = 0; i < ndima; i++)
            newacc[perma[i]] = oldacc[i];
        for (size_t i = 0; i < (size_t)size_a; i++) {
            size_t j = 0, ii = i;
            for (int k = ndima - 1; k >= 0; k--)
                j += (ii % shape_a[k]) * newacc[k], ii /= shape_a[k];
            new_a[j] = a[i];
        }
#endif
    }
}

void flat_sparse_tensor_matmul(const py::array_t<int32_t> &plan,
                               const py::array_t<double> &adata,
                               const py::array_t<double> &bdata,
                               py::array_t<double> &cdata) {
    int n_blocks_p = (int)plan.shape()[0], ndimp = (int)plan.shape()[1];
    assert(plan.strides()[1] == sizeof(int32_t));
    assert(ndimp == 9);
    const double alpha = 1.0;
    const double *pa = adata.data(), *pb = bdata.data();
    double *pc = cdata.mutable_data();
    for (int i = 0; i < n_blocks_p; i++) {
        const int32_t *pp = plan.data() + ndimp * i;
        const int trans_b = pp[0], trans_a = pp[1];
        const int m = pp[2], n = pp[3], k = pp[4];
        const int pib = pp[5], pia = pp[6], pic = pp[7];
        const double factor = alpha * pp[8];
        const auto tra = trans_a == -1 ? "n" : "t";
        const auto trb = trans_b == 1 ? "n" : "t";
        int ldb = trans_b == 1 ? m : k;
        int lda = trans_a == -1 ? k : n;
        int ldc = m;
        dgemm(trb, tra, &m, &n, &k, &factor, pb + pib, &ldb, pa + pia, &lda,
              &alpha, pc + pic, &ldc);
    }
}

inline SZLong to_sz(uint32_t x) {
    return SZLong((int)((x >> 17) & 16383) - 8192,
                  (int)((x >> 3) & 16383) - 8192, x & 7);
}

inline uint32_t from_sz(SZLong x) {
    return ((((uint32_t)(x.n() + 8192U) << 14) + (uint32_t)(x.twos() + 8192U))
            << 3) +
           (uint32_t)x.pg();
}

tuple<py::array_t<uint32_t>, py::array_t<uint32_t>, py::array_t<uint32_t>>
flat_sparse_tensor_skeleton(
    const vector<unordered_map<uint32_t, uint32_t>> &infos,
    const string &pattern, uint32_t fdq) {
    int ndim = (int)infos.size();
    size_t nx = 1;
    for (int i = 0; i < ndim; i++)
        nx *= infos[i].size();
    vector<uint32_t> qs, shs;
    vector<uint32_t> idxs(1, 0);
    vector<vector<pair<SZLong, uint32_t>>> infox(infos.size());
    for (int i = 0; i < ndim; i++) {
        infox[i].resize(infos[i].size());
        if (pattern[i] == '+') {
            int j = 0;
            for (auto &mr : infos[i]) {
                infox[i][j].first = to_sz(mr.first);
                infox[i][j].second = mr.second;
                j++;
            }
        } else {
            int j = 0;
            for (auto &mr : infos[i]) {
                infox[i][j].first = -to_sz(mr.first);
                infox[i][j].second = mr.second;
                j++;
            }
        }
    }
    SZLong dq = to_sz(fdq);
    vector<uint32_t> qk(ndim), shk(ndim);
    for (size_t x = 0; x < nx; x++) {
        size_t xp = x;
        SZLong xq = SZLong(0);
        for (int i = 0; i < ndim; xp /= infox[i].size(), i++)
            xq = xq + infox[i][xp % infox[i].size()].first;
        if (xq == dq) {
            uint32_t sz = 1;
            xp = x;
            for (int i = 0; i < ndim; xp /= infox[i].size(), i++) {
                auto &r = infox[i][xp % infox[i].size()];
                qk[i] =
                    pattern[i] == '+' ? from_sz(r.first) : from_sz(-r.first),
                shk[i] = r.second, sz *= r.second;
            }
            qs.insert(qs.end(), qk.begin(), qk.end());
            shs.insert(shs.end(), shk.begin(), shk.end());
            idxs.push_back(idxs.back() + sz);
        }
    }
    vector<ssize_t> sh = {(ssize_t)qs.size() / ndim, ndim};
    py::array_t<uint32_t> cqs(sh), cshs(sh),
        cidxs(vector<ssize_t>{(ssize_t)idxs.size()});
    assert(cqs.strides()[1] == sizeof(uint32_t));
    assert(cshs.strides()[1] == sizeof(uint32_t));
    memcpy(cqs.mutable_data(), qs.data(), qs.size() * sizeof(uint32_t));
    memcpy(cshs.mutable_data(), shs.data(), shs.size() * sizeof(uint32_t));
    memcpy(cidxs.mutable_data(), idxs.data(), idxs.size() * sizeof(uint32_t));
    return std::make_tuple(cqs, cshs, cidxs);
}

tuple<int, int, vector<unordered_map<uint32_t, uint32_t>>,
      vector<unordered_map<uint32_t, uint32_t>>>
flat_sparse_tensor_matmul_init(
    const py::array_t<uint32_t> &loqs, const py::array_t<uint32_t> &loshs,
    const py::array_t<uint32_t> &leqs, const py::array_t<uint32_t> &leshs,
    const py::array_t<uint32_t> &roqs, const py::array_t<uint32_t> &roshs,
    const py::array_t<uint32_t> &reqs, const py::array_t<uint32_t> &reshs) {
    int n_blocks_l = (int)loqs.shape()[0] + (int)leqs.shape()[0];
    int ndiml =
        loqs.shape()[0] != 0 ? (int)loqs.shape()[1] : (int)leqs.shape()[1];
    int n_blocks_r = (int)roqs.shape()[0] + (int)reqs.shape()[0];
    int ndimr =
        roqs.shape()[0] != 0 ? (int)roqs.shape()[1] : (int)reqs.shape()[1];
    vector<unordered_map<uint32_t, uint32_t>> lqs(ndiml);
    vector<unordered_map<uint32_t, uint32_t>> rqs(ndimr);
    if (loqs.shape()[0] != 0) {
        const ssize_t asi = loqs.strides()[0] / sizeof(uint32_t),
                      asj = loqs.strides()[1] / sizeof(uint32_t);
        uint32_t n = loqs.shape()[0];
        for (int i = 0; i < ndiml; i++)
            for (int j = 0; j < n; j++)
                lqs[i][loqs.data()[j * asi + i * asj]] =
                    loshs.data()[j * asi + i * asj];
    }
    if (leqs.shape()[0] != 0) {
        const ssize_t asi = leqs.strides()[0] / sizeof(uint32_t),
                      asj = leqs.strides()[1] / sizeof(uint32_t);
        uint32_t n = leqs.shape()[0];
        for (int i = 0; i < ndiml; i++)
            for (int j = 0; j < n; j++)
                lqs[i][leqs.data()[j * asi + i * asj]] =
                    leshs.data()[j * asi + i * asj];
    }
    if (roqs.shape()[0] != 0) {
        const ssize_t asi = roqs.strides()[0] / sizeof(uint32_t),
                      asj = roqs.strides()[1] / sizeof(uint32_t);
        uint32_t n = roqs.shape()[0];
        for (int i = 0; i < ndimr; i++)
            for (int j = 0; j < n; j++)
                rqs[i][roqs.data()[j * asi + i * asj]] =
                    roshs.data()[j * asi + i * asj];
    }
    if (reqs.shape()[0] != 0) {
        const ssize_t asi = reqs.strides()[0] / sizeof(uint32_t),
                      asj = reqs.strides()[1] / sizeof(uint32_t);
        uint32_t n = reqs.shape()[0];
        for (int i = 0; i < ndimr; i++)
            for (int j = 0; j < n; j++)
                rqs[i][reqs.data()[j * asi + i * asj]] =
                    reshs.data()[j * asi + i * asj];
    }
    int dl = (ndiml - 2) / 2, dr = (ndimr - 2) / 2;
    vector<unordered_map<uint32_t, uint32_t>> vinfos(dl + dr + 2);
    vector<unordered_map<uint32_t, uint32_t>> winfos(1);
    vinfos[0] = lqs[0];
    for (int i = 0; i < dl; i++) {
        vinfos[i + 1] = lqs[i + 1];
        vinfos[i + 1].insert(lqs[i + 1 + dl].begin(), lqs[i + 1 + dl].end());
    }
    for (int i = 0; i < dr; i++) {
        vinfos[i + 1 + dl] = rqs[i + 1];
        vinfos[i + 1 + dl].insert(rqs[i + 1 + dr].begin(),
                                  rqs[i + 1 + dr].end());
    }
    vinfos[dl + dr + 1] = rqs[ndimr - 1];
    winfos[0] = lqs[ndiml - 1];
    winfos[0].insert(rqs[0].begin(), rqs[0].end());
    winfos.insert(winfos.end(), rqs.begin() + 1, rqs.begin() + 1 + dr);
    winfos.insert(winfos.end(), lqs.begin() + (1 + dl),
                  lqs.begin() + (1 + dl + dl));
    return std::make_tuple(dl, dr, vinfos, winfos);
}

py::array_t<int32_t> flat_sparse_tensor_matmul_plan(
    const py::array_t<uint32_t> &aqs, const py::array_t<uint32_t> &ashs,
    const py::array_t<uint32_t> &aidxs, const py::array_t<uint32_t> &bqs,
    const py::array_t<uint32_t> &bshs, const py::array_t<uint32_t> &bidxs,
    const py::array_t<int> &idxa, const py::array_t<int> &idxb,
    const py::array_t<uint32_t> &cqs, const py::array_t<uint32_t> &cidxs,
    bool ferm_op) {

    assert(bqs.shape()[0] != 0);
    if (aqs.shape()[0] == 0)
        return py::array_t<int32_t>(vector<ssize_t>{0, 9});

    Timer t;
    int n_blocks_a = (int)aqs.shape()[0], ndima = (int)aqs.shape()[1];
    int n_blocks_b = (int)bqs.shape()[0], ndimb = (int)bqs.shape()[1];
    int n_blocks_c = (int)cqs.shape()[0], ndimc = (int)cqs.shape()[1];
    int nctr = (int)idxa.shape()[0];
    assert(ndimc == ndima - nctr + ndimb - nctr);
    const ssize_t asi = aqs.strides()[0] / sizeof(uint32_t),
                  asj = aqs.strides()[1] / sizeof(uint32_t);
    const ssize_t bsi = bqs.strides()[0] / sizeof(uint32_t),
                  bsj = bqs.strides()[1] / sizeof(uint32_t);
    const ssize_t csi = cqs.strides()[0] / sizeof(uint32_t),
                  csj = cqs.strides()[1] / sizeof(uint32_t);
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

    assert(trans_a != 0 && trans_b != 0);

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
        outqbs[n_blocks_b], allqcs[n_blocks_c];
    psh = aqs.data();
    for (int i = 0; i < n_blocks_a; i++) {
        ctrqas[i] = q_labels_hash(psh + i * asi, nctr, pidxa, asj);
        outqas[i] = q_labels_hash(psh + i * asi, ndima - nctr, outa, asj);
    }
    psh = bqs.data();
    for (int i = 0; i < n_blocks_b; i++) {
        ctrqbs[i] = q_labels_hash(psh + i * bsi, nctr, pidxb, bsj);
        outqbs[i] = q_labels_hash(psh + i * bsi, ndimb - nctr, outb, bsj);
    }

    psh = cqs.data();
    for (int i = 0; i < n_blocks_c; i++) {
        size_t hl = q_labels_hash(psh + i * csi, ndima - nctr, csj);
        size_t hr = q_labels_hash(psh + i * csi + (ndima - nctr) * csj,
                                  ndimb - nctr, csj);
        allqcs[i] = hl ^ (hr + 0x9E3779B9 + (hl << 6) + (hl >> 2));
    }

    unordered_map<size_t, vector<int>> map_idx_b, map_idx_c;
    vector<vector<uint32_t>> vqcs(n_blocks_c);
    for (int i = 0; i < n_blocks_b; i++)
        map_idx_b[ctrqbs[i]].push_back(i);
    for (int i = 0; i < n_blocks_c; i++) {
        map_idx_c[allqcs[i]].push_back(i);
        vqcs[i].resize(ndimc);
        psh = cqs.data() + i * csi;
        for (int j = 0; j < ndimc; j++)
            vqcs[i][j] = psh[j * csj];
    }

    uint32_t *pia = (uint32_t *)aidxs.data(), *pib = (uint32_t *)bidxs.data();
    uint32_t *pic = (uint32_t *)cidxs.data();
    vector<vector<int>> r;
    for (int ia = 0; ia < n_blocks_a; ia++) {
        if (map_idx_b.count(ctrqas[ia])) {
            const auto &vb = map_idx_b.at(ctrqas[ia]);
            vector<uint32_t> q_out(ndimc);
            psh = aqs.data() + ia * asi;
            for (int i = 0; i < ndima - nctr; i++)
                q_out[i] = psh[outa[i] * asj];
            for (int ib : vb) {
                size_t hout = outqas[ia];
                hout ^= outqbs[ib] + 0x9E3779B9 + (hout << 6) + (hout >> 2);
                psh = bqs.data() + ib * bsi;
                int xferm = 0;
                for (int i = 0; i < ndimb - nctr; i++)
                    q_out[i + ndima - nctr] = psh[outb[i] * bsj];
                if (ferm_op)
                    for (int i = 0; i < pidxb[0]; i++)
                        xferm ^= (psh[i * bsj] & 8) != 0;
                assert(map_idx_c.count(hout));
                auto &vq = map_idx_c.at(hout);
                int iq = 0;
                for (; iq < (int)vq.size() && q_out != vqcs[vq[iq]]; iq++)
                    ;
                assert(iq < (int)vq.size());
                int ic = vq[iq];
                r.push_back(vector<int>{trans_b, trans_a, b_free_dim[ib],
                                        a_free_dim[ia], ctr_dim[ia],
                                        (int)pib[ib], (int)pia[ia],
                                        (int)pic[ic], xferm ? -1 : 1});
            }
        }
    }

    assert(r.size() != 0);
    ssize_t rz = (ssize_t)r[0].size();
    vector<ssize_t> sh = {(ssize_t)r.size(), rz};
    py::array_t<int32_t> ar(sh);
    assert(ar.strides()[1] == sizeof(int32_t));
    for (size_t i = 0; i < r.size(); i++)
        memcpy(ar.mutable_data() + i * rz, r[i].data(), sizeof(int32_t) * rz);

    return ar;
}

PYBIND11_MODULE(block3, m) {

    m.doc() = "python extension part for pyblock3.";

    py::bind_map<unordered_map<uint32_t, uint32_t>>(m, "MapUIntUInt");
    py::bind_vector<vector<unordered_map<uint32_t, uint32_t>>>(
        m, "VectorMapUIntUInt");

    py::module tensor = m.def_submodule("tensor", "Tensor");
    tensor.def("transpose", &tensor_transpose, py::arg("x"), py::arg("perm"),
               py::arg("alpha") = 1.0, py::arg("beta") = 0.0);
    tensor.def("tensordot", &tensor_tensordot, py::arg("a"), py::arg("b"),
               py::arg("idxa"), py::arg("idxb"), py::arg("alpha") = 1.0,
               py::arg("beta") = 0.0);

    py::module flat_sparse_tensor =
        m.def_submodule("flat_sparse_tensor", "FlatSparseTensor");
    flat_sparse_tensor.def("skeleton", &flat_sparse_tensor_skeleton,
                           py::arg("infos"), py::arg("pattern"), py::arg("dq"));
    flat_sparse_tensor.def("tensordot", &flat_sparse_tensor_tensordot,
                           py::arg("aqs"), py::arg("ashs"), py::arg("adata"),
                           py::arg("aidxs"), py::arg("bqs"), py::arg("bshs"),
                           py::arg("bdata"), py::arg("bidxs"), py::arg("idxa"),
                           py::arg("idxb"));
    flat_sparse_tensor.def("tensordot_fast", &flat_sparse_tensor_tensordot_fast,
                           py::arg("aqs"), py::arg("ashs"), py::arg("adata"),
                           py::arg("aidxs"), py::arg("bqs"), py::arg("bshs"),
                           py::arg("bdata"), py::arg("bidxs"), py::arg("idxa"),
                           py::arg("idxb"));
    flat_sparse_tensor.def("add", &flat_sparse_tensor_add, py::arg("aqs"),
                           py::arg("ashs"), py::arg("adata"), py::arg("aidxs"),
                           py::arg("bqs"), py::arg("bshs"), py::arg("bdata"),
                           py::arg("bidxs"));
    flat_sparse_tensor.def("transpose", &flat_sparse_tensor_transpose,
                           py::arg("ashs"), py::arg("adata"), py::arg("aidxs"),
                           py::arg("perm"), py::arg("cdata"));
    flat_sparse_tensor.def("matmul", &flat_sparse_tensor_matmul,
                           py::arg("plan"), py::arg("adata"), py::arg("bdata"),
                           py::arg("cdata"));
    flat_sparse_tensor.def("matmul_init", &flat_sparse_tensor_matmul_init,
                           py::arg("loqs"), py::arg("loshs"), py::arg("leqs"),
                           py::arg("leshs"), py::arg("roqs"), py::arg("roshs"),
                           py::arg("reqs"), py::arg("reshs"));
    flat_sparse_tensor.def("matmul_plan", &flat_sparse_tensor_matmul_plan,
                           py::arg("aqs"), py::arg("ashs"), py::arg("aidxs"),
                           py::arg("bqs"), py::arg("bshs"), py::arg("bidxs"),
                           py::arg("idxa"), py::arg("idxb"), py::arg("cqs"),
                           py::arg("cidxs"), py::arg("ferm_op"));

    m.def("time_cost", []() { return py::make_tuple(tx, tc); });

    py::class_<SZ>(m, "SZ")
        .def(py::init<>())
        .def(py::init<uint32_t>())
        .def(py::init<int, int, int>())
        .def_readwrite("data", &SZ::data)
        .def_property("n", &SZ::n, &SZ::set_n)
        .def_property("twos", &SZ::twos, &SZ::set_twos)
        .def_property("pg", &SZ::pg, &SZ::set_pg)
        .def_property_readonly("multiplicity", &SZ::multiplicity)
        .def_property_readonly("is_fermion", &SZ::is_fermion)
        .def_property_readonly("count", &SZ::count)
        .def("combine", &SZ::combine)
        .def("__getitem__", &SZ::operator[])
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::self < py::self)
        .def(-py::self)
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def("get_ket", &SZ::get_ket)
        .def("get_bra", &SZ::get_bra, py::arg("dq"))
        .def("__hash__", &SZ::hash)
        .def("__repr__", &SZ::to_str);
}
