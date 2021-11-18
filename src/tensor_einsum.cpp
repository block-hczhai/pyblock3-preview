
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

#include "tensor_einsum.hpp"
#include <cstring>
#include <iostream>
#include <stdexcept>

inline string &string_trim(string &x) {
    if (x.empty())
        return x;
    x.erase(0, x.find_first_not_of(" \t"));
    x.erase(x.find_last_not_of(" \t") + 1);
    return x;
}

template <typename T> inline string to_string(const T &i) {
    stringstream ss;
    ss << i;
    return ss.str();
}

template <typename FL>
py::array_t<FL> array_view(const vector<ssize_t> &shape,
                           const vector<ssize_t> &strides,
                           py::array_t<FL> arr) {
    return py::array_t<FL>(shape, strides, arr.mutable_data(), arr);
}

template <typename FL>
py::array_t<FL> tensor_einsum(const string &script,
                              const vector<py::array_t<FL>> &arrs) {
    // separate scripts
    bool explicit_mode = false;
    string result;
    vector<string> operands;
    int idx = 0;
    for (int i = 0; i < script.length(); i++)
        if (script[i] == ',') {
            operands.push_back(script.substr(idx, i - idx));
            operands.back() = string_trim(operands.back());
            idx = i + 1;
        } else if (i < script.length() - 1 && script[i] == '-' &&
                   script[i + 1] == '>') {
            operands.push_back(script.substr(idx, i - idx));
            operands.back() = string_trim(operands.back());
            idx = i + 2;
            explicit_mode = true;
        }
    if (explicit_mode) {
        result = script.substr(idx);
        result = string_trim(result);
    } else {
        operands.push_back(script.substr(idx));
        operands.back() = string_trim(operands.back());
    }
    if (operands.size() != arrs.size())
        throw runtime_error("number of scripts " + to_string(operands.size()) +
                            " does match number of arrays " +
                            to_string(arrs.size()));
    // first pass
    const int _ELLIP = 1, _EMPTY = 2, _MAX_CHAR = 256;
    int char_map[_MAX_CHAR], char_count[_MAX_CHAR];
    // feature of operator
    // 1 = has_ellipsis; 2 = has_empty
    vector<int> op_features(operands.size() + 1, 0);
    idx = 0;
    memset(char_map, -1, sizeof(int) * _MAX_CHAR);
    memset(char_count, 0, sizeof(int) * _MAX_CHAR);
    // empty characters
    char_map['\t'] = char_map[' '] = char_map['\n'] = char_map['\r'] = -3;
    // illegal characters
    char_map['-'] = char_map['.'] = char_map['>'] = char_map['\0'] = -2;
    // first reserved character for ellipsis
    char nxt = '!';
    string ellip = "";
    bool ellip_determined = false;
    for (int iop = 0; iop < operands.size(); iop++) {
        int iellip = -1;
        for (int j = 0; j < operands[iop].length(); j++)
            if (j < operands[iop].length() - 2 && operands[iop][j] == '.' &&
                operands[iop][j + 1] == '.' && operands[iop][j + 2] == '.') {
                if (op_features[iop] & _ELLIP)
                    throw runtime_error("Multiple ellipses found in script " +
                                        operands[iop]);
                iellip = j;
                j += 2;
                op_features[iop] |= _ELLIP;
            } else if (char_map[operands[iop][j]] == -3)
                op_features[iop] |= _EMPTY;
            else if (char_map[operands[iop][j]] == -2)
                throw runtime_error("Illegal character " +
                                    string(1, operands[iop][j]) +
                                    " found in script " + operands[iop]);
            else {
                if (char_map[operands[iop][j]] == -1)
                    char_map[operands[iop][j]] = idx++;
                char_count[operands[iop][j]]++;
            }
        // remove empty characters inside script
        if (op_features[iop] & _EMPTY) {
            stringstream ss;
            for (int j = 0; j < operands[iop].length(); j++)
                if (char_map[operands[iop][j]] != -3)
                    ss << operands[iop][j];
            operands[iop] = ss.str();
            op_features[iop] ^= _EMPTY;
        }
        // handle ellipses of operands
        if (op_features[iop] & _ELLIP) {
            int nchar = arrs[iop].ndim() - (operands[iop].length() - 3);
            if (!ellip_determined) {
                stringstream ss;
                for (int j = 0; j < nchar; j++) {
                    while (char_map[nxt] != -1)
                        nxt++;
                    char_map[nxt] = -2;
                    ss << nxt;
                }
                ellip = ss.str();
                ellip_determined = true;
            }
            if (nchar != ellip.length())
                throw runtime_error("Length of ellipses does not match in " +
                                    operands[iop]);
            operands[iop] = operands[iop].replace(iellip, 3, ellip);
            op_features[iop] ^= _ELLIP;
        }
    }
    if (!explicit_mode) {
        // handle implicit mode
        stringstream ss;
        // if there is ellipsis, put it before all other indices
        if (ellip_determined)
            ss << ellip;
        // for all other indices that appearing only once, put them
        // according to alphabetically
        // do not use char type here, as it may exceed _MAX_CHAR
        for (int k = 0; k < _MAX_CHAR; k++)
            if (char_count[k] == 1)
                ss << (char)k;
        result = ss.str();
    } else {
        // handle ellipsis / empty in result script in explicit mode
        stringstream ss;
        for (int j = 0; j < result.length(); j++)
            if (j < result.length() - 2 && result[j] == '.' &&
                result[j + 1] == '.' && result[j + 2] == '.') {
                if (op_features.back() & _ELLIP)
                    throw runtime_error(
                        "Multiple ellipses found in output script " + result);
                // it is okay ellipsis does not appear in any operands
                // in that case it works as if there is no ellipsis in the
                // result
                ss << ellip;
                j += 2;
                op_features.back() |= _ELLIP;
            } else if (char_map[result[j]] == -4)
                throw runtime_error("Repeated character " +
                                    string(1, result[j]) +
                                    " found in output script " + result);
            else if (char_map[result[j]] == -3)
                continue;
            else if (char_map[result[j]] == -2)
                throw runtime_error("Illegal character " +
                                    string(1, result[j]) + " found in script " +
                                    result);
            else if (char_count[result[j]] == 0)
                throw runtime_error(
                    "character " + string(1, result[j]) +
                    " found in output script did not appear in an input");
            else {
                ss << result[j];
                char_map[result[j]] = -4;
            }
        result = ss.str();
    }
    for (int iop = 0; iop < operands.size() - 1; iop++)
        cout << operands[iop] << ",";
    cout << operands.back() << "->" << result << endl;
    // allow possible reorder in future
    vector<string> gscripts = operands;
    vector<py::array_t<FL>> garrs = arrs;
    // now char_map representes the last time one index
    // has been seen (in which op)
    memset(char_map, -1, sizeof(int) * _MAX_CHAR);
    for (int i = 0; i < (int)gscripts.size(); i++)
        for (int j = 0; j < gscripts[i].length(); j++)
            char_map[gscripts[i][j]] = i;
    for (int j = 0; j < (int)result.length(); j++)
        char_map[result[j]] = (int)gscripts.size();
    // reorder strides
    // fix internal repeated script
    // put broadcast indices at the beginning
    // (only when broadcast indices appear in both)
    // do internal sum (using axpy)
    return arrs[0];
}

template py::array_t<double>
tensor_einsum(const string &script, const vector<py::array_t<double>> &arrs);

template py::array_t<complex<double>>
tensor_einsum(const string &script,
              const vector<py::array_t<complex<double>>> &arrs);