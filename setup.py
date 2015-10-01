#!/usr/bin/env python

# This file is part of thermotools.
#
# Copyright 2015 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
#
# thermotools is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

r"""
thermotools is a lowlevel implementation toolbox for the analyis of free energy calculations
"""

from setuptools import setup
from distutils.core import Extension
from sys import exit as sys_exit
import versioneer

try:
    from Cython.Distutils import build_ext
except ImportError:
    print("ERROR - please install the cython dependency manually:")
    print("pip install cython")
    sys_exit(1)

try:
    from numpy import get_include
except ImportError:
    print("ERROR - please install the numpy dependency manually:")
    print("pip install numpy")
    sys_exit(1)

ext_lse = Extension(
    "thermotools.lse",
    sources=["ext/lse/lse.pyx", "ext/lse/_lse.c"],
    include_dirs=[get_include()],
    extra_compile_args=["-O3"])
ext_bar = Extension(
    "thermotools.bar",
    sources=["ext/bar/bar.pyx", "ext/bar/_bar.c", "ext/lse/_lse.c"],
    include_dirs=[get_include()],
    extra_compile_args=["-O3"])
ext_wham = Extension(
    "thermotools.wham",
    sources=["ext/wham/wham.pyx", "ext/wham/_wham.c", "ext/lse/_lse.c"],
    include_dirs=[get_include()],
    extra_compile_args=["-O3"])
ext_mbar = Extension(
    "thermotools.mbar",
    sources=["ext/mbar/mbar.pyx", "ext/mbar/_mbar.c", "ext/lse/_lse.c"],
    include_dirs=[get_include()],
    extra_compile_args=["-O3"])
# ext_tram = Extension(
#     "thermotools.tram",
#     sources=["ext/tram/tram.pyx", "ext/tram/_tram.c", "ext/lse/_lse.c"],
#     include_dirs=[get_include()],
#     extra_compile_args=["-O3"])
ext_dtram = Extension(
    "thermotools.dtram",
    sources=["ext/dtram/dtram.pyx", "ext/dtram/_dtram.c", "ext/lse/_lse.c"],
    include_dirs=[get_include()],
    extra_compile_args=["-O3"])
# ext_xtram = Extension(
#     "thermotools.xtram",
#     sources=["ext/xtram/xtram.pyx", "ext/xtram/_xtram.c", "ext/lse/_lse.c"],
#     include_dirs=[get_include()],
#     extra_compile_args=["-O3"])
ext_util = Extension(
    "thermotools.util",
    sources=["ext/util/util.pyx", "ext/util/_util.c"],
    include_dirs=[get_include()],
    extra_compile_args=["-O3"])

cmd_class = versioneer.get_cmdclass()
cmd_class.update({'build_ext': build_ext})

setup(
    cmdclass=cmd_class,
    ext_modules=[
        ext_lse,
        ext_bar,
        ext_wham,
        ext_mbar,
        #ext_tram,
        ext_dtram,
        #ext_xtram,
        ext_util,
        ],
    name='thermotools',
    version=versioneer.get_version(),
    description='Lowlevel implementation of free energy estimators',
    long_description='Lowlevel implementation of free energy estimators',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
        'Natural Language :: English',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: C',
        'Programming Language :: Cython',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Physics'],
    keywords=[
        'free energy',
        'Markov state model',
        'BAR',
        'WHAM',
        'MBAR',
        'TRAM',
        'dTRAM',
        'xTRAM'],
    url='https://github.com/markovmodel/thermotools',
    maintainer='Christoph Wehmeyer',
    maintainer_email='christoph.wehmeyer@fu-berlin.de',
    license='LGPLv3+',
    setup_requires=[
        'numpy>=1.7.1',
        'cython>=0.20',
        'setuptools>=0.6'],
    tests_require=[
        'numpy>=1.7.1',
        'scipy>=0.11',
        'msmtools>=1.1',
        'nose>=1.3'],
    install_requires=[
        'numpy>=1.7.1',
        'scipy>=0.11',
        'msmtools>=1.1'],
    packages=['thermotools'],
    test_suite='nose.collector',
    scripts=[]
)
