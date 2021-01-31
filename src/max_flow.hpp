
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
#include <cstring>
#include <unordered_map>
#include <vector>

using namespace std;

// Max flow (DINIC)
struct FLOW {
    typedef unordered_map<int, int>::const_iterator mci;
    vector<unordered_map<int, int>> resi;
    vector<int> dist;
    vector<mci> rix;
    int n, nfs;
    FLOW(int n) : n(n) {
        resi.resize(n + 2);
        dist.resize(n + 2);
        rix.resize(n + 2);
        nfs = 0;
    }
    void MVC_DFS(int i, uint8_t *vis) {
        vis[i] = 1;
        for (const auto &ri : resi[i])
            if (ri.second != 0 && !vis[ri.first])
                MVC_DFS(ri.first, vis);
    }
    // Minimum Vertex Cover (bipartite graph)
    void MVC(int xi, int yi, int xn, int yn, vector<int> &vx, vector<int> &vy) {
        DINIC();
        vector<uint8_t> vis;
        vis.reserve(n + 2);
        memset(vis.data(), 0, (n + 2) * sizeof(uint8_t));
        MVC_DFS(n, vis.data());
        vx.reserve(xn);
        vy.reserve(yn);
        for (int i = 0; i < xn; i++)
            if (resi[n][xi + i] == 0 && !vis[xi + i])
                vx.push_back(i);
        for (int i = 0; i < yn; i++)
            if (resi[yi + i][n + 1] == 0 && vis[yi + i])
                vy.push_back(i);
    }
    // DINIC Algorithm (O(n^2m). for capacity=1, O(m sqrt(n)))
    int DINIC() {
        int inf = 0;
        for (int i = 0; i < n + 2; i++)
            for (const auto &ri : resi[i])
                if (ri.second != 0)
                    resi[ri.first][i] = 0, inf++;
        int rs = 0;
        for (nfs = 0; DBFS(); rs += DDFS(n, inf), nfs++)
            ;
        return rs;
    }
    // DINIC DFS
    int DDFS(int x, int flow) {
        if (x == n + 1)
            return flow;
        int used = 0;
        for (mci ri = rix[x]; ri != resi[x].end(); ri++) {
            int i = (rix[x] = ri)->first, fx;
            if (ri->second != 0 && dist[i] == dist[x] + 1) {
                fx = DDFS(i, min(flow - used, ri->second));
                resi[x][i] -= fx;
                resi[i][x] += fx;
                used += fx;
                if (used == flow)
                    return used;
            }
        }
        if (!used)
            dist[x] = -1;
        return used;
    }
    // DINIC BFS
    bool DBFS() {
        int nn = n + 2;
        for (int i = 0; i < nn; i++)
            rix[i] = resi[i].begin();
        memset(dist.data(), -1, nn * sizeof(int));
        dist[n] = 0;
        int h = 0, r = 1, x;
        vector<int> q;
        q.reserve(nn);
        q[0] = n;
        while (h < r) {
            if ((x = q[h++]) == n + 1)
                return true;
            for (const auto &ri : resi[x])
                if (ri.second != 0 && dist[ri.first] == -1)
                    dist[ri.first] = dist[x] + 1, q[r++] = ri.first;
        }
        return dist[n + 1] != -1;
    }
};
