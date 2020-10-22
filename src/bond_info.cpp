
#include "bond_info.hpp"

void bond_info_trans_to_sz(
    const vector<unordered_map<uint32_t, uint32_t>> &infos,
    const string &pattern, vector<vector<pair<SZ, uint32_t>>> &infox,
    bool sorted) {
    int ndim = (int)infos.size();
    infox.resize(ndim);
    for (int i = 0; i < ndim; i++) {
        infox[i].resize(infos[i].size());
        int j = 0;
        for (auto &mr : infos[i]) {
            infox[i][j].first = to_sz(mr.first);
            infox[i][j].second = mr.second;
            j++;
        }
        if (sorted)
            sort(infox[i].begin(), infox[i].end(), less_psz);
        if (pattern[i] == '-')
            for (j = 0; j < (int)infox[i].size(); j++)
                infox[i][j].first = -infox[i][j].first;
    }
}

map_fusing
bond_info_fusing_product(const vector<unordered_map<uint32_t, uint32_t>> &infos,
                         const string &pattern) {
    int ndim = (int)infos.size();
    size_t nx = 1;
    for (int i = 0; i < ndim; i++)
        nx *= infos[i].size();
    vector<vector<pair<SZ, uint32_t>>> infox;
    bond_info_trans_to_sz(infos, pattern, infox, true);
    map_fusing r;
    vector<uint32_t> qk(ndim), shk(ndim);
    for (size_t x = 0; x < nx; x++) {
        uint32_t sz = 1;
        size_t xp = x;
        SZ xq;
        for (int i = ndim - 1; i >= 0; xp /= infox[i].size(), i--) {
            auto &r = infox[i][xp % infox[i].size()];
            xq = xq + r.first;
            qk[i] = pattern[i] == '+' ? from_sz(r.first) : from_sz(-r.first);
            shk[i] = r.second;
            sz *= r.second;
        }
        auto &rr = r[from_sz(xq)];
        rr.second[qk] = make_pair(rr.first, shk);
        rr.first += sz;
    }
    return r;
}
