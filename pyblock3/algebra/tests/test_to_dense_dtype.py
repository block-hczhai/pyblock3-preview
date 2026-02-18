import unittest

import numpy as np

from pyblock3.algebra.ad.core import SparseTensor as ADSparseTensor
from pyblock3.algebra.ad.core import SubTensor as ADSubTensor
from pyblock3.algebra.core import SliceableTensor, SparseTensor, SubTensor
from pyblock3.algebra.symmetry import BondInfo, SZ


class TestToDenseDtype(unittest.TestCase):
    @staticmethod
    def _simple_info():
        return BondInfo({SZ(0): 1, SZ(1): 1})

    def test_sliceable_to_dense_promotes_dtype(self):
        info = self._simple_info()
        reduced = np.array([np.array([1.0]), np.array([2.0 + 3.0j])], dtype=object)
        tensor = SliceableTensor(reduced=reduced, infos=(info,))

        dense = tensor.to_dense()

        self.assertTrue(np.issubdtype(dense.dtype, np.complexfloating))
        self.assertEqual(dense[1], 2.0 + 3.0j)

    def test_sparse_to_dense_promotes_dtype(self):
        info = self._simple_info()
        q0, q1 = sorted(info)
        tensor = SparseTensor(
            blocks=[
                SubTensor(reduced=np.array([1.0]), q_labels=(q0,)),
                SubTensor(reduced=np.array([2.0 + 3.0j]), q_labels=(q1,)),
            ]
        )

        dense = tensor.to_dense(infos=(info,))

        self.assertTrue(np.issubdtype(dense.dtype, np.complexfloating))
        self.assertEqual(dense[1], 2.0 + 3.0j)

    def test_ad_sparse_to_dense_promotes_dtype(self):
        info = self._simple_info()
        q0, q1 = sorted(info)
        tensor = ADSparseTensor(
            blocks=[
                ADSubTensor(data=np.array([1.0]), q_labels=(q0,)),
                ADSubTensor(data=np.array([2.0 + 3.0j]), q_labels=(q1,)),
            ],
            pattern="+",
        )

        dense = np.asarray(tensor.to_dense(infos=(info,)))

        self.assertTrue(np.issubdtype(dense.dtype, np.complexfloating))
        self.assertEqual(dense[1], 2.0 + 3.0j)


if __name__ == "__main__":
    unittest.main()
