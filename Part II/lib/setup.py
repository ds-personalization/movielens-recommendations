#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 17:01:29 2018

@author: red
"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy

ext = Extension("MF_cython", ["MF_python.pyx"],
    include_dirs = [numpy.get_include()])

setup(
    ext_modules=[ext], cmdclass = {'build_ext': build_ext}
)