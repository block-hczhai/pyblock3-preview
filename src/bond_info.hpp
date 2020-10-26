
#pragma once

#include "sz.hpp"
#include <algorithm>
#include <unordered_map>

using namespace std;

typedef unordered_map<
    uint32_t, pair<uint32_t, unordered_map<vector<uint32_t>,
                                           pair<uint32_t, vector<uint32_t>>>>>
    map_fusing;

void bond_info_trans_to_sz(
    const vector<unordered_map<uint32_t, uint32_t>> &infos,
    const string &pattern, vector<vector<pair<SZ, uint32_t>>> &infox,
    bool sorted = false);

map_fusing
bond_info_fusing_product(const vector<unordered_map<uint32_t, uint32_t>> &infos,
                         const string &pattern);

pair<vector<unordered_map<uint32_t, uint32_t>>,
     vector<unordered_map<uint32_t, uint32_t>>>
bond_info_set_bond_dimension_occ(
    const vector<unordered_map<uint32_t, uint32_t>> &basis,
    vector<unordered_map<uint32_t, uint32_t>> &left_dims,
    vector<unordered_map<uint32_t, uint32_t>> &right_dims, uint32_t vacuum,
    uint32_t target, int m, const vector<double> &occ, double bias);
