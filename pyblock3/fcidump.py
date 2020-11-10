
import numpy as np

PointGroup = {
    "d2h": [8, 0, 7, 6, 1, 5, 2, 3, 4],
    "c2v": [8, 0, 2, 3, 1],
    "c2h": [8, 0, 2, 3, 1],
    "d2": [8, 0, 3, 2, 1],
    "cs": [8, 0, 1],
    "c2": [8, 0, 1],
    "ci": [8, 0, 1],
    "c1": [8, 0]
}


class FCIDUMP:

    def __init__(self, pg='c1', n_sites=0, n_elec=0, twos=0, ipg=0, uhf=False,
                 h1e=None, g2e=None, orb_sym=None, const_e=0):
        self.pg = pg
        self.n_sites = n_sites
        self.n_elec = n_elec
        self.twos = twos
        self.ipg = ipg
        self.h1e = h1e
        self.g2e = g2e  # aa, ab, bb
        self.const_e = const_e
        self.uhf = uhf
        self.general = False
        self.orb_sym = [0] * self.n_sites if orb_sym is None else orb_sym

    def t(self, s, i, j):
        return self.h1e[s][i, j] if self.uhf else self.h1e[i, j]

    def v(self, sij, skl, i, j, k, l):
        if self.uhf:
            if sij == skl:
                return self.g2e[sij + sij][i, j, k, l]
            elif sij == 0 and skl == 1:
                return self.g2e[1][i, j, k, l]
            else:
                return self.g2e[1][k, l, i, j]
        else:
            return self.g2e[i, j, k, l]

    def read(self, filename):
        """
        Read FCI options and integrals from FCIDUMP file.

        Args:
            filename : str
        """
        with open(filename, 'r') as f:
            ff = f.read().lower()
            if '/' in ff:
                pars, ints = ff.split('/')
            elif '&end' in ff:
                pars, ints = ff.split('&end')

        cont = ','.join(pars.split()[1:])
        cont = cont.split(',')
        cont_dict = {}
        p_key = None
        for c in cont:
            if '=' in c or p_key is None:
                p_key, b = c.split('=')
                cont_dict[p_key.strip().lower()] = b.strip()
            elif len(c.strip()) != 0:
                if len(cont_dict[p_key.strip().lower()]) != 0:
                    cont_dict[p_key.strip().lower()] += ',' + c.strip()
                else:
                    cont_dict[p_key.strip().lower()] = c.strip()

        for k, v in cont_dict.items():
            if ',' in v:
                v = cont_dict[k] = v.split(',')

        self.n_sites = int(cont_dict.get('norb'))
        self.twos = int(cont_dict.get('ms2', 0))
        self.ipg = PointGroup[self.pg][int(cont_dict.get('isym', 0))]
        self.n_elec = int(cont_dict.get('nelec', 0))
        self.uhf = int(cont_dict.get('iuhf', 0)) != 0
        self.general = int(cont_dict.get('igeneral', 0)) != 0
        self.orb_sym = [PointGroup[self.pg]
                        [int(i)] for i in cont_dict.get('orbsym')]

        data = []
        for l in ints.split('\n'):
            ll = l.strip()
            if len(ll) == 0 or ll.strip()[0] == '!':
                continue
            ll = ll.split()
            d = float(ll[0])
            i, j, k, l = [int(x) for x in ll[1:]]
            data.append((i, j, k, l, d))
        if not self.uhf:
            self.h1e = np.zeros((self.n_sites, self.n_sites))
            self.g2e = np.zeros((self.n_sites, self.n_sites,
                                 self.n_sites, self.n_sites))
            for i, j, k, l, d in data:
                if i + j + k + l == 0:
                    self.const_e = d
                elif k + l == 0:
                    self.h1e[i - 1, j - 1] = self.h1e[j - 1, i - 1] = d
                else:
                    self.g2e[i - 1, j - 1, k - 1, l - 1] = d
                    if not self.general:
                        self.g2e[j - 1, i - 1, k - 1, l - 1] = d
                        self.g2e[j - 1, i - 1, l - 1, k - 1] = d
                        self.g2e[i - 1, j - 1, l - 1, k - 1] = d
                        self.g2e[k - 1, l - 1, i - 1, j - 1] = d
                        self.g2e[k - 1, l - 1, j - 1, i - 1] = d
                        self.g2e[l - 1, k - 1, j - 1, i - 1] = d
                        self.g2e[l - 1, k - 1, i - 1, j - 1] = d
        else:
            self.h1e = (None, None)
            self.h1e[0] = np.zeros((self.n_sites, self.n_sites))
            self.h1e[1] = np.zeros((self.n_sites, self.n_sites))
            self.g2e = (None, None, None)  # aa, ab, bb
            self.g2e[0] = np.zeros(
                (self.n_sites, self.n_sites, self.n_sites, self.n_sites))
            self.g2e[1] = np.zeros(
                (self.n_sites, self.n_sites, self.n_sites, self.n_sites))
            self.g2e[2] = np.zeros(
                (self.n_sites, self.n_sites, self.n_sites, self.n_sites))
            ip = 0
            for i, j, k, l, d in data:
                if i + j + k + l == 0:
                    ip += 1
                    if ip == 6:
                        self.const_e = d
                elif k + l == 0:
                    assert ip == 3 or ip == 4
                    if ip == 3:
                        self.h1e[0][i - 1, j -
                                    1] = self.h1e[0][j - 1, i - 1] = d
                    elif ip == 4:
                        self.h1e[1][i - 1, j -
                                    1] = self.h1e[1][j - 1, i - 1] = d
                else:
                    ig = [0, 3, 1][ip]
                    self.g2e[ig][i - 1, j - 1, k - 1, l - 1] = d
                    if not self.general:
                        self.g2e[ig][j - 1, i - 1, k - 1, l - 1] = d
                        self.g2e[ig][j - 1, i - 1, l - 1, k - 1] = d
                        self.g2e[ig][i - 1, j - 1, l - 1, k - 1] = d
                    if not self.general and (ip == 0 or ip == 1):
                        self.g2e[ig][k - 1, l - 1, i - 1, j - 1] = d
                        self.g2e[ig][k - 1, l - 1, j - 1, i - 1] = d
                        self.g2e[ig][l - 1, k - 1, j - 1, i - 1] = d
                        self.g2e[ig][l - 1, k - 1, i - 1, j - 1] = d
        return self
