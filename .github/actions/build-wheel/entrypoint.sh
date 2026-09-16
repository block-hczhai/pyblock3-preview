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
elif [ "${PYTHON_VERSION}" = "3.11" ]; then
    PY_VER=cp311-cp311
elif [ "${PYTHON_VERSION}" = "3.12" ]; then
    PY_VER=cp312-cp312
elif [ "${PYTHON_VERSION}" = "3.13" ]; then
    PY_VER=cp313-cp313
elif [ "${PYTHON_VERSION}" = "3.13t" ]; then
    PY_VER=cp313-cp313t
elif [ "${PYTHON_VERSION}" = "3.14" ]; then
    PY_VER=cp314-cp314
elif [ "${PYTHON_VERSION}" = "3.14t" ]; then
    PY_VER=cp314-cp314t
elif [ "${PYTHON_VERSION}" = "3.15-dev" ]; then
    PY_VER=cp315-cp315
elif [ "${PYTHON_VERSION}" = "3.15t-dev" ]; then
    PY_VER=cp315-cp315t
else
    echo "Unsupported Python version: ${PYTHON_VERSION}" >&2
    exit 1
fi

PY_EXE=/opt/python/"${PY_VER}"/bin/python3
sed -i "/DPYTHON_EXECUTABLE/a \                '-DPYTHON_EXECUTABLE=${PY_EXE}'," setup.py

ls -l /opt/python
/opt/python/"${PY_VER}"/bin/pip install --upgrade --no-cache-dir pip setuptools
/opt/python/"${PY_VER}"/bin/pip install --no-cache-dir 'cmake>=3.19' pybind11==3.1.0
/opt/python/"${PY_VER}"/bin/pip install --no-cache-dir mkl==2024.2.2 mkl-include intel-openmp
$(cat $(which auditwheel) | head -1 | awk -F'!' '{print $2}') -m pip install auditwheel==5.1.2
$(cat $(which auditwheel) | head -1 | awk -F'!' '{print $2}') -m pip install setuptools


sed -i '/for soname, src_path/a \                if any(x in soname for x in ["libmkl"]): continue' \
    $($(cat $(which auditwheel) | head -1 | awk -F'!' '{print $2}') -c "from auditwheel import repair;print(repair.__file__)")
${PY_EXE} -c 'import site; x = site.getsitepackages(); x += [xx.replace("site-packages", "dist-packages") for xx in x]; print("*".join(x))' > /tmp/ptmp
sed -i '/rpath_set\[rpath\]/a \    import site\n    for x in set(["../lib" + p.split("lib")[-1] for p in open("/tmp/ptmp").read().strip().split("*")]): rpath_set[rpath.replace("../..", x)] = ""' \
    $($(cat $(which auditwheel) | head -1 | awk -F'!' '{print $2}') -c "from auditwheel import repair;print(repair.__file__)")
sed -i '/rpath_set\[rpath\]/a \    rpath_set["$ORIGIN/../.."] = ""' \
    $($(cat $(which auditwheel) | head -1 | awk -F'!' '{print $2}') -c "from auditwheel import repair;print(repair.__file__)")

cmake --version
/opt/python/"${PY_VER}"/bin/pip wheel . -w ./dist --no-deps

find . -type f -iname "*-linux*.whl" -exec sh -c "auditwheel repair '{}' -w \$(dirname '{}') --plat '${PLAT}'" \;
find . -type f -iname "*-linux*.whl" -exec rm {} \;
find . -type f -iname "*-manylinux*.whl"

rm /tmp/ptmp

cd /tmp
PYTHONPATH= ${PY_EXE} -m pip install --no-deps --no-cache-dir /github/workspace/dist/*-manylinux*.whl
PYTHONPATH= ${PY_EXE} -c "import block3; print(block3.__file__)"
cd /github/workspace
