
Symmetry Labels
===============

`SZ` represents a collection of three quantum numbers (particle number, projected spin, point group irreducible representation).

The group algebra for `SZ` is also defined:

.. code:: python

    >>> from pyblock3.algebra.symmetry import SZ
    >>> a = SZ(0, 0, 0)
    >>> b = SZ(1, 1, 2)
    >>> a + b
    < N=1 SZ=1/2 PG=2 >
    >>> b + b
    < N=2 SZ=1 PG=0 >
    >>> -b
    < N=-1 SZ=-1/2 PG=2 >

`BondInfo` represents a map from `SZ` to number of states. The union (`__or__`), intersection (`__and__`), addition (`__add__`)
and tensor product (`__xor__`) of two `BondInfo` are also defined:

.. code:: python

    >>> from pyblock3.algebra.symmetry import SZ, BondInfo
    >>> bi = BondInfo({SZ(0, 0, 0): 1, SZ(1, 1, 2): 2})
    >>> ci = BondInfo({SZ(1, 1, 2): 2, SZ(-1, -1, 2): 2})
    >>> bi | ci
    < N=-1 SZ=-1/2 PG=2 > = 2 < N=0 SZ=0 PG=0 > = 1 < N=1 SZ=1/2 PG=2 > = 2
    >>> bi & ci
    < N=1 SZ=1/2 PG=2 > = 2
    >>> bi + ci
    < N=-1 SZ=-1/2 PG=2 > = 2 < N=0 SZ=0 PG=0 > = 1 < N=1 SZ=1/2 PG=2 > = 4
    >>> bi ^ ci
    < N=-1 SZ=-1/2 PG=2 > = 2 < N=0 SZ=0 PG=0 > = 4 < N=1 SZ=1/2 PG=2 > = 2 < N=2 SZ=1 PG=0 > = 4
