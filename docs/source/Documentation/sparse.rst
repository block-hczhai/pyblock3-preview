
Block-Sparse Tensor
===================

A ``SubTensor`` is a ``numpy.ndarray`` with a ``tuple`` of ``SZ`` for quantum labels associated.
It can be initialized using a ``numpy.ndarray.shape`` and a ``tuple`` of ``SZ``.

.. code:: python

   >>> from pyblock3.algebra.core import SubTensor
   >>> from pyblock3.algebra.symmetry import SZ
   >>> x = SubTensor.zeros((4,3), q_labels=(SZ(0, 0, 0), SZ(1, 1, 2)))
   >>> x
   (Q=) (< N=0 SZ=0 PG=0 >, < N=1 SZ=1/2 PG=2 >) (R=) array([[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]])
   >>> x[1,:] = 1
   >>> x
   (Q=) (< N=0 SZ=0 PG=0 >, < N=1 SZ=1/2 PG=2 >) (R=) array([[0., 0., 0.],
         [1., 1., 1.],
         [0., 0., 0.],
         [0., 0., 0.]])
   >>> x.q_labels
   (< N=0 SZ=0 PG=0 >, < N=1 SZ=1/2 PG=2 >)

A ``SparseTensor`` represents a block-sparse tensor, which contains a list of ``SubTensor``.
A quantum-number conserving ``SparseTensor`` can be initialized using a ``BondInfo``, ``pattern`` and ``dq``.
``pattern`` is a string of '+' or '-', indicating how to combine ``SZ`` to get ``dq``.
``dq`` is the conserved quantum number.
For 1D ``SparseTensor``, the initialization method does not require quantum-number conservation.

.. code:: python

   >>> from pyblock3.algebra.core import SparseTensor
   >>> from pyblock3.algebra.symmetry import SZ, BondInfo
   >>> x = BondInfo({SZ(0, 0, 0): 1, SZ(1, 1, 2): 2, SZ(-1, -1, 2): 2})
   >>> SparseTensor.random((x, x), pattern='++', dq=SZ(0, 0, 0))
   0 (Q=) (< N=-1 SZ=-1/2 PG=2 >, < N=1 SZ=1/2 PG=2 >) (R=) array([[0.89718406, 0.85419892],
         [0.65863698, 0.98023596]])
   1 (Q=) (< N=0 SZ=0 PG=0 >, < N=0 SZ=0 PG=0 >) (R=) array([[0.69742141]])
   2 (Q=) (< N=1 SZ=1/2 PG=2 >, < N=-1 SZ=-1/2 PG=2 >) (R=) array([[0.50722408, 0.34099007],
         [0.40760832, 0.8430552 ]])

Note that the resulting tensor has three non-zero blocks, for each block, the quantum numbers adds to ``dq``, which is ``SZ(0, 0, 0)``.
So this is a quantum-number-conserving block-sparse tensor.

``SparseTensor`` supports most common `numpy.ndarray` operations:

.. code:: python

   >>> import numpy as np
   >>> x = SparseTensor.random((x, x), pattern='++', dq=SZ(0, 0, 0))
   >>> y = 2 * x
   >>> np.linalg.norm(y)
   3.386356824229238
   >>> np.linalg.norm(x)
   1.693178412114619
   >>> np.tensordot(x, y, axes=1)
   0 (Q=) (< N=-1 SZ=-1/2 PG=2 >, < N=-1 SZ=-1/2 PG=2 >) (R=) array([[0.37106833, 0.92381267],
         [0.23117763, 1.27553566]])
   1 (Q=) (< N=0 SZ=0 PG=0 >, < N=0 SZ=0 PG=0 >) (R=) array([[1.25199691]])
   2 (Q=) (< N=1 SZ=1/2 PG=2 >, < N=1 SZ=1/2 PG=2 >) (R=) array([[0.66650452, 0.63606196],
         [0.61864202, 0.98009947]])

A ``FermionTensor`` contains two ``SparseTensor``, including blocks with odd ``dq`` and even ``dq``, respectively.

``FlatSparseTensor`` is another representation of block-sparse tensor, where quantum-number are combined together to a single integer,
and floating-point contents of all blocks are merged to one single "flattened" ``numpy.ndarray``.

``FlatSparseTensor`` has the same interface as ``SparseTensor``, but ``FlatSparseTensor`` provides much faster C++ implemntation
for functions like `tensordot` and `tranpose`. For debugging purpose, ``FlatSparseTensor`` also has pure python implementation,
which can be activated by setting ``ENABLE_FAST_IMPLS = False`` in ``flat.py``.
