#!/bin/bash

set -e -x

cd /github/workspace

PYTHON_VERSION=$1
PARALLEL=$2

if [ "${PYTHON_VERSION}" = "3.6" ]; then
    PY_VER=cp36-cp36m
elif [ "${PYTHON_VERSION}" = "3.7" ]; then
    PY_VER=cp37-cp37m
elif [ "${PYTHON_VERSION}" = "3.8" ]; then
    PY_VER=cp38-cp38
elif [ "${PYTHON_VERSION}" = "3.9" ]; then
    PY_VER=cp39-cp39
elif [ "${PYTHON_VERSION}" = "3.10" ]; then
    PY_VER=cp310-cp310
fi

PY_EXE=/opt/python/"${PY_VER}"/bin/python3
sed -i "/DPYTHON_EXECUTABLE/a \                '-DPYTHON_EXECUTABLE=${PY_EXE}'," setup.py

/opt/python/"${PY_VER}"/bin/pip install --upgrade --no-cache-dir pip
/opt/python/"${PY_VER}"/bin/pip install --no-cache-dir mkl==2021.4 mkl-include intel-openmp 'numpy<1.23.0' psutil 'cmake>=3.19' pybind11==2.10.1

sed -i '/new_soname = src_name/a \    if any(x in src_name for x in ["libmkl_avx2", "libmkl_avx512"]): new_soname = src_name' \
    $($(cat $(which auditwheel) | head -1 | awk -F'!' '{print $2}') -c "from auditwheel import repair;print(repair.__file__)")

/opt/python/"${PY_VER}"/bin/pip wheel . -w ./dist --no-deps

find . -type f -iname "*-linux*.whl" -exec sh -c "auditwheel repair '{}' -w \$(dirname '{}') --plat '${PLAT}'" \;
find . -type f -iname "*-linux*.whl" -exec rm {} \;
find . -type f -iname "*-manylinux*.whl"
