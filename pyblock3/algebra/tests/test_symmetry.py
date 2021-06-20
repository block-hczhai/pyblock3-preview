import unittest
import numpy as np
from pyblock3.algebra.fermion_symmetry import U11, U1, Z4, Z2, Z22
from itertools import product

class TestAlgebras(unittest.TestCase):
    def test_U11_Z22_numerics(self):
        for na in range(-5,5):
            for sa in range(-na, na):
                syma = U11(na, sa)
                z22a = Z22(na, sa)
                for nb in range(-5,5):
                    for sb in range(-nb, nb):
                        symb = U11(nb, sb)
                        z22b = Z22(nb, sb)
                        out_add = syma + symb
                        z22_add = z22a + z22b
                        assert out_add.n == (na+nb)
                        assert out_add.sz == (sa+sb)
                        assert z22_add.n == (na+nb) % 2
                        assert z22_add.sz == (sa+sb) % 2
                        out_sub = syma - symb
                        z22_sub = z22a - z22b
                        assert out_sub.n == (na-nb)
                        assert out_sub.sz == (sa-sb)
                        assert z22_sub.n == (na-nb) % 2
                        assert z22_sub.sz == (sa-sb) %2

    def test_z4_z2_u1_numerics(self):
        for za in range(-10,10):
            z4a = Z4(za)
            z2a = Z2(za)
            u1a = U1(za)
            for zb in range(-5,5):
                z4b = Z4(zb)
                z2b = Z2(zb)
                u1b = U1(zb)
                z4_add = z4a + z4b
                z2_add = z2a + z2b
                u1_add = u1a + u1b
                assert z4_add.n == (za+zb) % 4
                assert z2_add.n == (za+zb) % 2
                assert u1_add.n == (za+zb)
                z4_sub = z4a - z4b
                z2_sub = z2a - z2b
                u1_sub = u1a - u1b
                assert z4_sub.n == (za-zb) % 4
                assert z2_sub.n == (za-zb) % 2
                assert u1_sub.n == (za-zb)

    def test_U11_Z22_flat(self):
        for n in range(-10,10):
            for sz in range(-n, n):
                sym1 = U11(n, sz)
                sym2 = U11.from_flat(sym1.to_flat())
                assert sym1==sym2
                sym1 = Z22(n, sz)
                sym2 = Z22.from_flat(sym1.to_flat())
                assert sym1==sym2

    def test_z4_z2_u1_flat(self):
        for z in range(-10,10):
            z4a = Z4(z)
            z4b = Z4.from_flat(z4a.to_flat())
            assert z4a==z4b
            z2a = Z2(z)
            z2b = Z2.from_flat(z2a.to_flat())
            assert z2a==z2b
            u1a = U1(z)
            u1b = U1.from_flat(u1a.to_flat())
            assert u1a==u1b

    def test_U11_compute(self):
        ndim = 10
        sign_array = np.random.randint(0,2,ndim)
        sign_map = {0:"+", 1:"-"}
        pattern = "".join([sign_map[ix] for ix in sign_array])
        na = np.random.randint(0,10,ndim)
        sa = np.random.randint(-5,5,ndim)
        U11arrs = [U11(n, s) for n, s in zip(na, sa)]
        out = U11._compute(pattern, U11arrs)

        outn = 0
        outs = 0
        for p, n, s in zip(sign_array, na, sa):
            if p==0:
                outn += n
                outs += s
            else:
                outn -= n
                outs -= s
        assert out.n == outn
        assert out.sz == outs

    def test_Z22_compute(self):
        ndim = 10
        sign_array = np.random.randint(0,2,ndim)
        sign_map = {0:"+", 1:"-"}
        pattern = "".join([sign_map[ix] for ix in sign_array])
        na = np.random.randint(0,10,ndim)
        sa = np.random.randint(-5,5,ndim)
        Z22arrs = [Z22(n, s) for n, s in zip(na, sa)]
        out = Z22._compute(pattern, Z22arrs)

        outn = 0
        outs = 0
        for p, n, s in zip(sign_array, na, sa):
            if p==0:
                outn += n
                outs += s
            else:
                outn -= n
                outs -= s
        assert out.n == outn % 2
        assert out.sz == outs % 2

    def test_z4_z2_u1_compute(self):
        ndim = 10
        sign_array = np.random.randint(0,2,ndim)
        sign_map = {0:"+", 1:"-"}
        pattern = "".join([sign_map[ix] for ix in sign_array])
        na = np.random.randint(0,10,ndim)
        z4arrs = [Z4(n) for n in na]
        z4out = Z4._compute(pattern, z4arrs)
        z2arrs = [Z2(n) for n in na]
        z2out = Z2._compute(pattern, z2arrs)
        u1arrs = [U1(n) for n in na]
        u1out = U1._compute(pattern, u1arrs)

        z4outn = 0
        z2outn = 0
        u1outn = 0

        for p, n in zip(sign_array, na):
            if p==0:
                z4outn += n
                z2outn += n
                u1outn += n
            else:
                z4outn -= n
                z2outn -= n
                u1outn -= n
        assert z4out.n == z4outn % 4
        assert z2out.n == z2outn % 2
        assert u1out.n == u1outn

    def test_U11_compute_flat(self):
        nblks = 30
        ndim = 5
        sign_array = np.random.randint(0,2,ndim)
        sign_map = {0:"+", 1:"-"}
        pattern = "".join([sign_map[ix] for ix in sign_array])

        na = np.random.randint(0,10,nblks*ndim).reshape(nblks, ndim)
        sa = np.random.randint(-5,5,nblks*ndim).reshape(nblks, ndim)

        U11lsts = [[U11(na[nb,nd], sa[nb,nd]) for nd in range(ndim)] for nb in range(nblks)]
        U11arrs = [[U11lsts[nb][nd].to_flat() for nd in range(ndim)] for nb in range(nblks)]
        U11arrs = np.asarray(U11arrs, dtype=int)
        out = U11._compute(pattern, U11arrs)
        outneg = U11._compute(pattern, U11arrs, offset=("-", U11(1,-1)), neg=True)
        for i in range(nblks):
            outa = U11._compute(pattern, U11lsts[i])
            outb = U11.from_flat(out[i])
            assert outa == outb
            assert -(outa - U11(1,-1)) == U11.from_flat(outneg[i])

    def test_Z22_compute_flat(self):
        nblks = 30
        ndim = 5
        sign_array = np.random.randint(0,2,ndim)
        sign_map = {0:"+", 1:"-"}
        pattern = "".join([sign_map[ix] for ix in sign_array])

        na = np.random.randint(0,10,nblks*ndim).reshape(nblks, ndim)
        sa = np.random.randint(-5,5,nblks*ndim).reshape(nblks, ndim)

        Z22lsts = [[Z22(na[nb,nd], sa[nb,nd]) for nd in range(ndim)] for nb in range(nblks)]
        Z22arrs = [[Z22lsts[nb][nd].to_flat() for nd in range(ndim)] for nb in range(nblks)]
        Z22arrs = np.asarray(Z22arrs, dtype=int)
        out = Z22._compute(pattern, Z22arrs)
        outneg = Z22._compute(pattern, Z22arrs, offset=("-", Z22(1,-1)), neg=True)
        for i in range(nblks):
            outa = Z22._compute(pattern, Z22lsts[i])
            outb = Z22.from_flat(out[i])
            assert outa == outb
            assert -(outa - Z22(1,-1)) == Z22.from_flat(outneg[i])

    def test_z4_z2_compute_flat(self):
        nblks = 30
        ndim = 5
        sign_array = np.random.randint(0,2,ndim)
        sign_map = {0:"+", 1:"-"}
        pattern = "".join([sign_map[ix] for ix in sign_array])

        na = np.random.randint(0,10,nblks*ndim).reshape(nblks, ndim)

        z4lsts = [[Z4(na[nb,nd]) for nd in range(ndim)] for nb in range(nblks)]
        z4arrs = [[z4lsts[nb][nd].to_flat() for nd in range(ndim)] for nb in range(nblks)]
        z4arrs = np.asarray(z4arrs, dtype=int)
        z4out = Z4._compute(pattern, z4arrs)
        z4neg = Z4._compute(pattern, z4arrs, offset=("+", Z4(3)), neg=True)

        z2lsts = [[Z2(na[nb,nd]) for nd in range(ndim)] for nb in range(nblks)]
        z2arrs = [[z2lsts[nb][nd].to_flat() for nd in range(ndim)] for nb in range(nblks)]
        z2arrs = np.asarray(z2arrs, dtype=int)
        z2out = Z2._compute(pattern, z2arrs)
        z2neg = Z2._compute(pattern, z2arrs, offset=("-", Z2(1)), neg=True)

        u1lsts = [[U1(na[nb,nd]) for nd in range(ndim)] for nb in range(nblks)]
        u1arrs = [[u1lsts[nb][nd].to_flat() for nd in range(ndim)] for nb in range(nblks)]
        u1arrs = np.asarray(u1arrs, dtype=int)
        u1out = U1._compute(pattern, u1arrs)
        u1neg = U1._compute(pattern, u1arrs, offset=("-", U1(1)), neg=True)

        for i in range(nblks):
            z4outa = Z4._compute(pattern, z4lsts[i])
            z4outb = Z4.from_flat(z4out[i])
            assert z4outa == z4outb
            assert -(z4outa + Z4(3)) == Z4.from_flat(z4neg[i])

            z2outa = Z2._compute(pattern, z2lsts[i])
            z2outb = Z2.from_flat(z2out[i])
            assert z2outa == z2outb
            assert -(z2outa - Z2(1)) == Z2.from_flat(z2neg[i])

            u1outa = U1._compute(pattern, u1lsts[i])
            u1outb = U1.from_flat(u1out[i])
            assert u1outa == u1outb
            assert -(u1outa - U1(1)) == U1.from_flat(u1neg[i])


if __name__ == "__main__":
    print("Full Tests for Fermionic Symmetries")
    unittest.main()
