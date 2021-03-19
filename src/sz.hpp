
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

struct SZ {
    int _n, _twos, _pg;
    SZ() : _n(0), _twos(0), _pg(0) {}
    SZ(int _n, int _twos, int _pg) : _n(_n), _twos(_twos), _pg(_pg) {}
    int n() const { return _n; }
    int twos() const { return _twos; }
    int pg() const { return _pg; }
    int parity() const { return _n & 1; }
    void set_n(int _n) { this->_n = _n; }
    void set_twos(int _twos) { this->_twos = _twos; }
    void set_pg(int _pg) { this->_pg = _pg; }
    int multiplicity() const noexcept { return 1; }
    int is_fermion() const { return parity(); }
    bool operator==(SZ other) const noexcept {
        return _n == other._n && _twos == other._twos && _pg == other._pg;
    }
    bool operator!=(SZ other) const noexcept {
        return _n != other._n || _twos != other._twos || _pg != other._pg;
    }
    bool operator<(SZ other) const noexcept {
        if (_n != other._n)
            return _n < other._n;
        else if (_twos != other._twos)
            return _twos < other._twos;
        else
            return _pg < other._pg;
    }
    SZ operator-() const noexcept { return SZ(-_n, -_twos, _pg); }
    SZ operator-(SZ other) const noexcept { return *this + (-other); }
    SZ operator+(SZ other) const noexcept {
        return SZ(_n + other._n, _twos + other._twos, _pg ^ other._pg);
    }
    SZ operator[](int i) const noexcept { return *this; }
    size_t hash() const noexcept {
        return (size_t)(((size_t)_n << 24) | ((size_t)_twos << 8) | _pg);
    }
    int count() const noexcept { return 1; }
    static SZ to_q(uint32_t x) noexcept {
        return SZ((int)((x >> 17) & 16383) - 8192,
                  (int)((x >> 3) & 16383) - 8192, x & 7);
    }
    static uint32_t from_q(SZ x) {
        return ((((uint32_t)(x.n() + 8192U) << 14) +
                 (uint32_t)(x.twos() + 8192U))
                << 3) +
               (uint32_t)x.pg();
    }
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
    friend ostream &operator<<(ostream &os, SZ c) {
        os << c.to_str();
        return os;
    }
};

namespace std {

template <> struct hash<SZ> {
    size_t operator()(const SZ &s) const noexcept { return s.hash(); }
};

template <> struct less<SZ> {
    bool operator()(const SZ &lhs, const SZ &rhs) const noexcept {
        return lhs < rhs;
    }
};

} // namespace std
