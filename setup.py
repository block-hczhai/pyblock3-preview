#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(name='pyblock3',
      version='0.1.0',
      packages=find_packages(),
      license='LICENSE',
      description='An Efficient python block-sparse tensor and MPS/DMRG Library.',
      long_description=open('README.md').read(),
      author='Huanchen Zhai, Yang Gao, and Garnet K.-L. Chan',
      author_email='hczhai@ucla.edu',
      url='https://github.com/block-hczhai/pyblock3-preview',
      install_requires=[
          "numpy",
          "numba",
          "psutil"
      ]
      )
