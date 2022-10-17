
#  pyblock3: An Efficient python MPS/DMRG Library
#  Copyright (C) 2020 The pyblock3 developers. All Rights Reserved.
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program. If not, see <https://www.gnu.org/licenses/>.
#
#


def einsum(script, *arrs, tensordot=None, transpose=None):
    from collections import Counter
    idx = 0
    operands = []
    explicit_mode = False
    for i, ic in enumerate(script):
        if ic == ',':
            operands.append(script[idx:i].strip())
            idx = i + 1
        elif script[i:i + 2] == '->':
            operands.append(script[idx:i].strip())
            idx = i + 2
            explicit_mode = True
    assert explicit_mode
    result = script[idx:].strip()
    if len(operands) != len(arrs):
        raise RuntimeError("number of scripts", len(
            operands), "does not match number of arrays", len(arrs))
    # first pass
    _ELLIP, _EMPTY = 1, 2
    char_map = {}
    char_count = Counter()
    # feature of operator
    # 1 = has_ellipsis; 2 = has_empty
    op_features = [0] * (len(operands) + 1)
    idx = 0
    # empty characters
    char_map['\t'] = char_map[' '] = char_map['\n'] = char_map['\r'] = -3
    # illegal characters
    char_map['-'] = char_map['.'] = char_map['>'] = char_map['\0'] = -2
    # first reserved character for ellipsis
    for iop, opr in enumerate(operands):
        jskip = 0
        for j, jop in enumerate(opr):
            if jskip != 0:
                jskip -= 1
                continue
            if jop not in char_map:
                char_map[jop] = -1
            if opr[j:j + 3] == '...':
                if op_features[iop] & _ELLIP:
                    raise RuntimeError(
                        "Multiple ellipses found in script", opr)
                jskip = 2
                op_features[iop] |= _ELLIP
            elif char_map[jop] == -3:
                op_features[iop] |= _EMPTY
            elif char_map[jop] == -2:
                raise RuntimeError("Illegal character", jop,
                                   "found in script", opr)
            else:
                if char_map[jop] == -1:
                    char_map[jop] = idx
                    idx += 1
                char_count[jop] += 1
        # remove empty characters inside script
        if op_features[iop] & _EMPTY:
            operands[iop] = ''.join(
                jop for j, jop in enumerate(opr) if char_map[jop] != -3)
            op_features[iop] ^= _EMPTY
        # handle ellipses of operands
        assert not (op_features[iop] & _ELLIP)
    # handle implicit mode
    assert explicit_mode
    # handle ellipsis / empty in result script in explicit mode
    gscripts = operands.copy()
    garrs = [x for x in arrs]
    # now char_count representes the count of each index
    for kk in char_count:
        char_count[kk] = 0
    for i in gscripts:
        for j in i:
            char_count[j] += 1
    for j in result:
        char_count[j] += 1
    perm, sum_idx = [], []
    # handle internal sum and repeated indices
    for i, ig in enumerate(gscripts):
        for kk in char_map:
            char_map[kk] = -1
        perm = [0] * len(ig)
        k = 0
        newss = ""
        for j, jg in enumerate(ig):
            if char_map[jg] == -1:
                char_map[jg] = k
                k += 1
                newss += jg
            perm[j] = char_map[jg]
        for kk in char_map:
            char_map[kk] = 0
        for jg in ig:
            char_map[jg] += 1
        # handle repeated indices
        assert k == len(ig)
        sum_idx_num = 0
        sum_idx = [0] * k
        newss = ""
        for j in range(k):
            if char_map[ig[j]] == char_count[ig[j]]:
                sum_idx[j] = -1
                sum_idx_num += 1
            else:
                newss += ig[j]
        for j in range(k):
            if char_map[ig[j]] > 1:
                char_count[ig[j]] -= char_map[ig[j]] - 1
        # handle internal sum
        assert sum_idx_num == 0
    # perform tensordot
    for i, ig in enumerate(gscripts):
        if i == 0:
            continue
        for kk in char_map:
            char_map[kk] = 0
        for jg in gscripts[0]:
            char_map[jg] += 1
        for jg in gscripts[i]:
            char_map[jg] += 1
        newss, newsr = "", ""
        idxa, idxb = [], []
        br_idxa, br_idxb = [], []
        for j, jg in enumerate(gscripts[0]):
            if char_map[jg] > 1:
                if char_map[jg] == char_count[jg]:
                    idxa.append(j)
                else:
                    br_idxa.append(j)
                    newsr += jg
            else:
                newss += jg
        for j, jg in enumerate(gscripts[i]):
            if char_map[jg] > 1:
                if char_map[jg] == char_count[jg]:
                    idxb.append(j)
                else:
                    br_idxb.append(j)
            else:
                newss += jg
        for kk in char_map:
            char_map[kk] = -1
        for j, jx in enumerate(idxa):
            char_map[gscripts[0][jx]] = j
        assert len(br_idxa) == 0
        assert len(br_idxb) == 0
        idxb.sort(key=lambda x, char_map=char_map, ig=ig: char_map[ig[x]])
        # print('tensordot : 0 x i : axes =', idxa, idxb)
        tmp = tensordot(garrs[0], garrs[i], axes=(idxa, idxb))
        # remove contracted and broadcast index count
        for x in idxa:
            char_count[gscripts[0][x]] -= 2
        garrs[0] = tmp
        gscripts[0] = newsr + newss
    # final transpose (no copy)
    assert len(gscripts[0]) == len(result)
    if gscripts[0] != result:
        for kk in char_map:
            char_map[kk] = -1
        for j, jg in enumerate(gscripts[0]):
            char_map[jg] = j
        perm = [0] * len(gscripts[0])
        for j, jg in enumerate(result):
            perm[j] = char_map[result[j]]
        # print('transpose : 0 : axes =', perm)
        garrs[0] = transpose(garrs[0], axes=perm)
    return garrs[0]
