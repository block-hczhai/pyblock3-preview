
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

py::object sub_tensor_tensordot(py::object a, py::object b, py::object axes) {
    Timer t;
    py::object np = py::module::import("numpy");
    py::object cls = a.attr("__class__");
    py::array_t<double> ta = a.cast<py::array_t<double>>(),
                        tb = b.cast<py::array_t<double>>();
    py::tuple qa = a.attr("q_labels").cast<py::tuple>(),
              qb = b.attr("q_labels").cast<py::tuple>();

    vector<int> idxa, idxb;
    int n_ctr = 0;
    if (py::isinstance<py::int_>(axes)) {
        n_ctr = axes.cast<int>();
        for (int i = 0; i < n_ctr; i++)
            idxa.push_back(i - n_ctr), idxb.push_back(i);
    } else if (py::isinstance<py::tuple>(axes)) {
        py::tuple t = axes.cast<py::tuple>();
        py::list ta = t[0].cast<py::list>(), tb = t[1].cast<py::list>();
        assert(ta.size() == tb.size());
        n_ctr = (int)ta.size();
        for (int i = 0; i < n_ctr; i++) {
            idxa.push_back(ta[i].cast<int>());
            idxb.push_back(tb[i].cast<int>());
        }
    } else
        throw runtime_error("wrong type for axes!");

    vector<int> out_idxa(ta.ndim()), out_idxb(tb.ndim());
    for (int i = 0; i < ta.ndim(); i++)
        out_idxa[i] = i;
    for (int i = 0; i < tb.ndim(); i++)
        out_idxb[i] = i;

    for (int i = 0; i < n_ctr; i++) {
        if (idxa[i] < 0)
            idxa[i] += ta.ndim();
        if (idxb[i] < 0)
            idxb[i] += tb.ndim();
        out_idxa[idxa[i]] = -1;
        out_idxb[idxb[i]] = -1;
    }

    py::list out_q_labels;
    for (int i = 0; i < ta.ndim(); i++)
        if (out_idxa[i] != -1)
            out_q_labels.append(qa[out_idxa[i]]);
    for (int i = 0; i < tb.ndim(); i++)
        if (out_idxb[i] != -1)
            out_q_labels.append(qb[out_idxb[i]]);

    for (int i = 0; i < n_ctr; i++)
        assert(py::hash(qa[idxa[i]]) == py::hash(qb[idxb[i]]));

    // tx += t.get_time();
    // tc++;
    py::object r = np.attr("tensordot")(ta, tb, axes).attr("view")(cls);
    r.attr("q_labels") = py::tuple(move(out_q_labels));

    return r;
}

py::object sparse_tensor_add(py::object a, py::object b) {
    Timer t;
    py::object num = py::module::import("numbers").attr("Number");
    py::object cls = a.attr("__class__");
    py::list blocks;
    if (py::isinstance(a, num)) {
        py::list bblocks = b.attr("blocks").cast<py::list>();
        int nb = bblocks.size();
        for (int i = 0; i < nb; i++)
            blocks.append(a + (py::object)bblocks[i]);
    } else if (py::isinstance(b, num)) {
        py::list ablocks = a.attr("blocks").cast<py::list>();
        int na = ablocks.size();
        for (int i = 0; i < na; i++)
            blocks.append((py::object)ablocks[i] + b);
    } else {
        unordered_map<size_t, py::object> blocks_map;
        py::list ablocks = a.attr("blocks").cast<py::list>(),
                 bblocks = b.attr("blocks").cast<py::list>();
        int na = ablocks.size(), nb = bblocks.size();
        for (int ia = 0; ia < na; ia++) {
            py::object block = ablocks[ia];
            py::tuple qs = block.attr("q_labels").cast<py::tuple>();
            blocks_map[py::hash(qs)] = block;
        }
        for (int ib = 0; ib < nb; ib++) {
            py::object block = bblocks[ib];
            py::tuple qs = block.attr("q_labels").cast<py::tuple>();
            size_t h = py::hash(qs);
            if (blocks_map.count(h))
                blocks_map[h] += block;
            else
                blocks_map[h] = block;
        }
        for (auto &rr : blocks_map)
            blocks.append(rr.second);
    }
    tx += t.get_time();
    tc++;
    return cls(blocks);
}

py::object sparse_tensor_tensordot(py::object a, py::object b,
                                   py::object axes) {
    Timer t;
    py::object np = py::module::import("numpy");
    py::object cls = a.attr("__class__");
    int na = a.attr("ndim").cast<int>(), nb = b.attr("ndim").cast<int>();

    vector<int> idxa, idxb;
    int n_ctr = 0;
    if (py::isinstance<py::int_>(axes)) {
        n_ctr = axes.cast<int>();
        for (int i = 0; i < n_ctr; i++)
            idxa.push_back(i - n_ctr), idxb.push_back(i);
    } else if (py::isinstance<py::tuple>(axes)) {
        py::tuple t = axes.cast<py::tuple>();
        py::list ta = t[0].cast<py::list>(), tb = t[1].cast<py::list>();
        assert(ta.size() == tb.size());
        n_ctr = (int)ta.size();
        for (int i = 0; i < n_ctr; i++) {
            idxa.push_back(ta[i].cast<int>());
            idxb.push_back(tb[i].cast<int>());
        }
    } else
        throw runtime_error("wrong type for axes!");

    for (int i = 0; i < n_ctr; i++) {
        if (idxa[i] < 0)
            idxa[i] += na;
        if (idxb[i] < 0)
            idxb[i] += nb;
    }

    unordered_map<size_t, py::list> map_idx_b;
    py::list ablocks = a.attr("blocks").cast<py::list>(),
             bblocks = b.attr("blocks").cast<py::list>();

    for (auto &block : bblocks) {
        py::list subg;
        py::tuple qs = block.attr("q_labels").cast<py::tuple>();
        for (int i = 0; i < n_ctr; i++)
            subg.append(qs[idxb[i]]);
        py::tuple sub_qs(move(subg));
        map_idx_b[py::hash(sub_qs)].append(block);
    }

    unordered_map<size_t, py::object> blocks_map;
    int nba = ablocks.size();
    for (int ia = 0; ia < nba; ia++) {
        py::object block_a = ablocks[ia];
        py::list subg;
        py::tuple qs = block_a.attr("q_labels").cast<py::tuple>();
        for (int i = 0; i < n_ctr; i++)
            subg.append(qs[idxa[i]]);
        py::tuple sub_qs(move(subg));
        size_t h = py::hash(sub_qs);
        if (map_idx_b.count(h)) {
            py::list mblocks = map_idx_b.at(h);
            int nbb = mblocks.size();
            for (int ib = 0; ib < nbb; ib++) {
                py::object block_b = mblocks[ib];
                py::object mat = sub_tensor_tensordot(block_a, block_b, axes);
                size_t h2 = py::hash(mat.attr("q_labels"));
                if (blocks_map.count(h2))
                    blocks_map[h2] += mat;
                else
                    blocks_map[h2] = mat;
            }
        }
    }

    py::list r;
    for (auto &rr : blocks_map)
        r.append(rr.second);
    tx += t.get_time();
    tc++;
    return cls(r);
}

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
    int n_blocks_a = (int)aqs.shape()[0], ndima = (int)aqs.shape()[1];
    int n_blocks_b = (int)bqs.shape()[0], ndimb = (int)bqs.shape()[1];
    int nctr = (int)idxa.shape()[0];
    int ndimc = ndima - nctr + ndimb - nctr;

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
    if (nctr == 0 || (pidxa[0] == 0 && pidxa[nctr - 1] == nctr - 1))
        trans_a = 1;
    else if (pidxa[0] == ndima - nctr && pidxa[nctr - 1] == ndima - 1)
        trans_a = -1;

    if (nctr == 0 || (pidxb[0] == 0 && pidxb[nctr - 1] == nctr - 1))
        trans_b = 1;
    else if (pidxb[0] == ndimb - nctr && pidxb[nctr - 1] == ndimb - 1)
        trans_b = -1;

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
    const ssize_t asi = aqs.strides()[0] / sizeof(uint32_t),
                  asj = aqs.strides()[1] / sizeof(uint32_t);
    const ssize_t bsi = bqs.strides()[0] / sizeof(uint32_t),
                  bsj = bqs.strides()[1] / sizeof(uint32_t);
    assert(memcmp(aqs.strides(), ashs.strides(), 2 * sizeof(ssize_t)) == 0);
    assert(memcmp(bqs.strides(), bshs.strides(), 2 * sizeof(ssize_t)) == 0);
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
    int ic = 0;
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
            ic++;
        }
    }
    return std::make_tuple(cqs, cshs, cdata, cidxs);
}

PYBIND11_MODULE(block3, m) {

    m.doc() = "python extension part for pyblock3.";

    py::module tensor = m.def_submodule("tensor", "Tensor");
    tensor.def("transpose", &tensor_transpose, py::arg("x"), py::arg("perm"),
               py::arg("alpha") = 1.0, py::arg("beta") = 0.0);
    tensor.def("tensordot", &tensor_tensordot, py::arg("a"), py::arg("b"),
               py::arg("idxa"), py::arg("idxb"), py::arg("alpha") = 1.0,
               py::arg("beta") = 0.0);

    py::module sub_tensor = m.def_submodule("sub_tensor", "SubTensor");
    sub_tensor.def("tensordot", &sub_tensor_tensordot, py::arg("a"),
                   py::arg("b"), py::arg("axes") = py::cast(2));

    py::module sparse_tensor = m.def_submodule("sparse_tensor", "SparseTensor");
    sparse_tensor.def("tensordot", &sparse_tensor_tensordot, py::arg("a"),
                      py::arg("b"), py::arg("axes") = py::cast(2));
    sparse_tensor.def("add", &sparse_tensor_add, py::arg("a"), py::arg("b"));

    py::module flat_sparse_tensor =
        m.def_submodule("flat_sparse_tensor", "FlatSparseTensor");
    flat_sparse_tensor.def("tensordot", &flat_sparse_tensor_tensordot,
                           py::arg("aqs"), py::arg("ashs"), py::arg("adata"),
                           py::arg("aidxs"), py::arg("bqs"), py::arg("bshs"),
                           py::arg("bdata"), py::arg("bidxs"), py::arg("idxa"),
                           py::arg("idxb"));

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
