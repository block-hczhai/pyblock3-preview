
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

#pragma once

#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace std;

struct U1 {
    int _n;
    U1() : _n(0) {}
    U1(int _n) : _n(_n) {}
    int n() const { return _n; }
    int parity() const { return _n & 1; }
    void set_n(int _n) { this->_n = _n; }
    int multiplicity() const noexcept { return 1; }
    int is_fermion() const { return parity(); }
    bool operator==(U1 other) const noexcept {
        return _n == other._n;
    }
    bool operator!=(U1 other) const noexcept {
        return _n != other._n;
    }
    bool operator<(U1 other) const noexcept {
        return _n < other._n;
    }
    U1 operator-() const noexcept { return U1(-_n); }
    U1 operator-(U1 other) const noexcept { return *this + (-other); }
    U1 operator+(U1 other) const noexcept {
        return U1(_n + other._n);
    }
    U1 operator[](int i) const noexcept { return *this; }
    size_t hash() const noexcept {
        return (size_t)((size_t)_n << 24);
    }
    int count() const noexcept { return 1; }
    static U1 to_q(uint32_t x) noexcept {
        return U1((int)((x >> 17) & 16383) - 8192);
    }
    static uint32_t from_q(U1 x) {
        return ((uint32_t)(x.n() + 8192U) << 17);
    }
    string to_str() const {
        stringstream ss;
        ss << "< N=" << n() << " >";
        return ss.str();
    }
    friend ostream &operator<<(ostream &os, U1 c) {
        os << c.to_str();
        return os;
    }
};

namespace std {

template <> struct hash<U1> {
    size_t operator()(const U1 &s) const noexcept { return s.hash(); }
};

template <> struct less<U1> {
    bool operator()(const U1 &lhs, const U1 &rhs) const noexcept {
        return lhs < rhs;
    }
};

}

struct U11 {
    int _n, _sz;
    U11() : _n(0), _sz(0) {}
    U11(int _n, int _sz) : _n(_n), _sz(_sz) {}
    int n() const { return _n; }
    int sz() const { return _sz; }
    int parity() const { return _n & 1; }
    void set_n(int _n) { this->_n = _n; }
    void set_sz(int _sz) { this->_sz = _sz; }
    int multiplicity() const noexcept { return 1; }
    int is_fermion() const { return parity(); }
    bool operator==(U11 other) const noexcept {
        return _n == other._n && _sz == other._sz;
    }
    bool operator!=(U11 other) const noexcept {
        return _n != other._n || _sz != other._sz;
    }
    bool operator<(U11 other) const noexcept {
        if (_n != other._n)
            return _n < other._n;
        else
            return _sz < other._sz;
    }
    U11 operator-() const noexcept { return U11(-_n, -_sz); }
    U11 operator-(U11 other) const noexcept { return *this + (-other); }
    U11 operator+(U11 other) const noexcept {
        return U11(_n + other._n, _sz + other._sz);
    }
    U11 operator[](int i) const noexcept { return *this; }
    size_t hash() const noexcept {
        return (size_t)(((size_t)_n << 24) | ((size_t)_sz << 8));
    }
    int count() const noexcept { return 1; }
    static U11 to_q(uint32_t x) noexcept {
        return U11((int)((x >> 17) & 16383) - 8192,
                  (int)((x >> 3) & 16383) - 8192);
    }
    static uint32_t from_q(U11 x) {
        return ((((uint32_t)(x.n() + 8192U) << 14) +
                 (uint32_t)(x.sz() + 8192U))
                << 3);
    }
    string to_str() const {
        stringstream ss;
        ss << "< N=" << n() << " ZN=" << sz() << ">";
        return ss.str();
    }
    friend ostream &operator<<(ostream &os, U11 c) {
        os << c.to_str();
        return os;
    }
};

namespace std {

template <> struct hash<U11> {
    size_t operator()(const U11 &s) const noexcept { return s.hash(); }
};

template <> struct less<U11> {
    bool operator()(const U11 &lhs, const U11 &rhs) const noexcept {
        return lhs < rhs;
    }
};

} // namespace std

struct ZN {
    int modulus() const { return 1024;}
    int _n;
    ZN() : _n(0) {}
    ZN(int _n) : _n(_n%modulus()) {}
    int n() const { return _n; }
    int parity() const { return _n % 2; }
    void set_n(int _n) { this->_n = (_n % modulus()); }
    int multiplicity() const noexcept { return 1; }
    int is_fermion() const { return parity(); }
    bool operator==(ZN other) const noexcept {
        return _n == other._n;
    }
    bool operator!=(ZN other) const noexcept {
        return _n != other._n;
    }
    bool operator<(ZN other) const noexcept {
        return _n < other._n;
    }
    ZN operator-() const noexcept { return ZN(modulus() - _n); }
    ZN operator-(ZN other) const noexcept { return *this + (-other); }
    ZN operator+(ZN other) const noexcept {
        return ZN(_n + other._n);
    }
    ZN operator[](int i) const noexcept { return *this; }
    size_t hash() const noexcept {
        return (size_t)((size_t)_n << 24);
    }
    int count() const noexcept { return 1; }
    static ZN to_q(uint32_t x) noexcept {
        return ZN((int) x);
    }
    static uint32_t from_q(ZN x) {
        return (uint32_t) x.n();
    }
    string to_str() const {
        stringstream ss;
        ss << "< N=" << n() << " >";
        return ss.str();
    }
    friend ostream &operator<<(ostream &os, ZN c) {
        os << c.to_str();
        return os;
    }
};

namespace std {

template <> struct hash<ZN> {
    size_t operator()(const ZN &s) const noexcept { return s.hash(); }
};

template <> struct less<ZN> {
    bool operator()(const ZN &lhs, const ZN &rhs) const noexcept {
        return lhs < rhs;
    }
};

}

struct Z2:ZN {
    int modulus() const { return 2;}
    int _n;
    Z2() : _n(0) {}
    Z2(int _n) : _n(_n%modulus()) {}
    int n() const { return _n; }
    int parity() const { return _n % 2; }
    bool operator==(Z2 other) const noexcept {
        return _n == other._n;
    }
    bool operator!=(Z2 other) const noexcept {
        return _n != other._n;
    }
    bool operator<(Z2 other) const noexcept {
        return _n < other._n;
    }
    Z2 operator-() const noexcept { return Z2(modulus() - _n); }
    Z2 operator-(Z2 other) const noexcept { return *this + (-other); }
    Z2 operator+(Z2 other) const noexcept {
        return Z2(_n + other._n);
    }
    Z2 operator[](int i) const noexcept { return *this; }
    static Z2 to_q(uint32_t x) noexcept {
        return Z2((int) x);
    }
    static uint32_t from_q(Z2 x) {
        return (uint32_t) x.n();
    }
    friend ostream &operator<<(ostream &os, Z2 c) {
        os << c.to_str();
        return os;
    }
};

namespace std {

template <> struct hash<Z2> {
    size_t operator()(const Z2 &s) const noexcept { return s.hash(); }
};

template <> struct less<Z2> {
    bool operator()(const Z2 &lhs, const Z2 &rhs) const noexcept {
        return lhs < rhs;
    }
};

}

struct Z4: ZN {
    int modulus() const { return 4;}
    int _n;
    Z4() : _n(0) {}
    Z4(int _n) : _n(_n%modulus()) {}
    int n() const { return _n; }
    int parity() const { return _n % 2; }
    bool operator==(Z4 other) const noexcept {
        return _n == other._n;
    }
    bool operator!=(Z4 other) const noexcept {
        return _n != other._n;
    }
    bool operator<(Z4 other) const noexcept {
        return _n < other._n;
    }
    Z4 operator-() const noexcept { return Z4(modulus() - _n); }
    Z4 operator-(Z4 other) const noexcept { return *this + (-other); }
    Z4 operator+(Z4 other) const noexcept {
        return Z4(_n + other._n);
    }
    Z4 operator[](int i) const noexcept { return *this; }
    static Z4 to_q(uint32_t x) noexcept {
        return Z4((int) x);
    }
    static uint32_t from_q(Z4 x) {
        return (uint32_t) x.n();
    }
    friend ostream &operator<<(ostream &os, Z4 c) {
        os << c.to_str();
        return os;
    }
};

namespace std {

template <> struct hash<Z4> {
    size_t operator()(const Z4 &s) const noexcept { return s.hash(); }
};

template <> struct less<Z4> {
    bool operator()(const Z4 &lhs, const Z4 &rhs) const noexcept {
        return lhs < rhs;
    }
};

}

struct Z22:U11 {
    int _n, _sz;
    Z22() : _n(0), _sz(0) {}
    Z22(int _n, int _sz) : _n(_n % 2), _sz(_sz% 2) {}
    int n() const { return _n; }
    int sz() const { return _sz; }
    int parity() const { return _n % 2; }
    bool operator==(Z22 other) const noexcept {
        return _n == other._n && _sz == other._sz;
    }
    bool operator!=(Z22 other) const noexcept {
        return _n != other._n || _sz != other._sz;
    }
    bool operator<(Z22 other) const noexcept {
        if (_n != other._n)
            return _n < other._n;
        else
            return _sz < other._sz;
    }
    Z22 operator-() const noexcept { return Z22(2 -_n, 2 -_sz); }
    Z22 operator-(Z22 other) const noexcept { return *this + (-other); }
    Z22 operator+(Z22 other) const noexcept {
        return Z22(_n + other._n, _sz + other._sz);
    }
    Z22 operator[](int i) const noexcept { return *this; }
    size_t hash() const noexcept {
        return (size_t)(((size_t)_n << 24) | ((size_t)_sz << 8));
    }
    static Z22 to_q(uint32_t x) noexcept {
        return Z22((int)(x/2), (int)(x % 2));
    }
    static uint32_t from_q(Z22 x) {
        return (uint32_t)(x.n()*2 + x.sz()) ;
    }

    friend ostream &operator<<(ostream &os, Z22 c) {
        os << c.to_str();
        return os;
    }
};

namespace std {

template <> struct hash<Z22> {
    size_t operator()(const Z22 &s) const noexcept { return s.hash(); }
};

template <> struct less<Z22> {
    bool operator()(const Z22 &lhs, const Z22 &rhs) const noexcept {
        return lhs < rhs;
    }
};

} // namespace std
