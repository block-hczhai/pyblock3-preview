# pyblock3

Still under construction!!

## Installation

Dependence `mkl`, `block2` (optional), and `hptt` (optional).

`cmake` (version >= 3.0) can be used to compile C++ part of the code (for better performance), as follows:

    mkdir build
    cd build
    cmake .. -DUSE_MKL=ON -DUSE_HPTT=ON
    make

