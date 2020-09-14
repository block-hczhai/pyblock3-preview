
#pragma once

#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace std;

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

namespace std {

template <> struct hash<SZLong> {
    size_t operator()(const SZLong &s) const noexcept { return s.hash(); }
};

template <> struct hash<vector<uint32_t>> {
    size_t operator()(const vector<uint32_t> &s) const noexcept {
        return q_labels_hash(s.data(), s.size(), 1);
    }
};

template <> struct less<SZLong> {
    bool operator()(const SZLong &lhs, const SZLong &rhs) const noexcept {
        return lhs < rhs;
    }
};

} // namespace std

inline SZLong to_sz(uint32_t x) {
    return SZLong((int)((x >> 17) & 16383) - 8192,
                  (int)((x >> 3) & 16383) - 8192, x & 7);
}

inline uint32_t from_sz(SZLong x) {
    return ((((uint32_t)(x.n() + 8192U) << 14) + (uint32_t)(x.twos() + 8192U))
            << 3) +
           (uint32_t)x.pg();
}

inline bool less_sz(SZLong x, SZLong y) noexcept {
    return x.n() != y.n()
               ? x.n() < y.n()
               : (x.twos() != y.twos() ? x.twos() < y.twos() : x.pg() < y.pg());
}

inline bool less_psz(const pair<SZLong, uint32_t> &x,
                     const pair<SZLong, uint32_t> &y) noexcept {
    return less_sz(x.first, y.first);
}

inline bool is_shape_one(const uint32_t *shs, int n, int nfree, const int inci,
                         const int incj) noexcept {
    for (int j = 0; j < nfree * incj; j += incj)
        for (int i = 0; i < n * inci; i += inci)
            if (shs[i + j] != 1)
                return false;
    return true;
}
