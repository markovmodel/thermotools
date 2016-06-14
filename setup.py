#!/usr/bin/env python

# This file is part of thermotools.
#
# Copyright 2015, 2016 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
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
thermotools is a lowlevel implementation toolbox for the analyis of multi-ensemble calculations.
"""

from setuptools import setup, Extension
import versioneer

def extensions():
    from numpy import get_include
    from Cython.Build import cythonize
    extra_compile_args = ["-O3", "-std=c99"]
    ext_bar = Extension(
        "thermotools.bar",
        sources=["ext/bar/bar.pyx", "ext/bar/_bar.c", "ext/util/_util.c"],
        include_dirs=[get_include()],
        extra_compile_args=extra_compile_args)
    ext_wham = Extension(
        "thermotools.wham",
        sources=["ext/wham/wham.pyx", "ext/wham/_wham.c", "ext/util/_util.c"],
        include_dirs=[get_include()],
        extra_compile_args=extra_compile_args)
    ext_mbar = Extension(
        "thermotools.mbar",
        sources=["ext/mbar/mbar.pyx", "ext/mbar/_mbar.c", "ext/util/_util.c"],
        include_dirs=[get_include()],
        extra_compile_args=extra_compile_args)
    ext_tram = Extension(
        "thermotools.tram",
        sources=["ext/tram/tram.pyx", "ext/tram/_tram.c", "ext/util/_util.c"],
        include_dirs=[get_include()],
        extra_compile_args=extra_compile_args)
    ext_dtram = Extension(
        "thermotools.dtram",
        sources=["ext/dtram/dtram.pyx", "ext/dtram/_dtram.c", "ext/util/_util.c"],
        include_dirs=[get_include()],
        extra_compile_args=extra_compile_args)
    ext_trammbar = Extension(
        "thermotools.trammbar",
        sources=["ext/trammbar/trammbar.pyx", "ext/tram/_tram.c", "ext/util/_util.c"],
        include_dirs=[get_include()],
        extra_compile_args=extra_compile_args + ["-DTRAMMBAR"])
    ext_mbar_direct = Extension(
        "thermotools.mbar_direct",
        sources=["ext/mbar_direct/mbar_direct.pyx", "ext/mbar_direct/_mbar_direct.c", "ext/util/_util.c"],
        include_dirs=[get_include()],
        extra_compile_args=extra_compile_args)
    ext_tram_direct = Extension(
        "thermotools.tram_direct",
        sources=["ext/tram_direct/tram_direct.pyx", "ext/tram_direct/_tram_direct.c", "ext/util/_util.c"],
        include_dirs=[get_include()],
        extra_compile_args=extra_compile_args)
    ext_trammbar_direct = Extension(
        "thermotools.trammbar_direct",
        sources=["ext/trammbar_direct/trammbar_direct.pyx", "ext/tram_direct/_tram_direct.c", "ext/util/_util.c"],
        include_dirs=[get_include()],
        extra_compile_args=extra_compile_args + ["-DTRAMMBAR"])
    ext_util = Extension(
        "thermotools.util",
        sources=["ext/util/util.pyx", "ext/util/_util.c"],
        include_dirs=[get_include()],
        extra_compile_args=extra_compile_args)
    exts = [
        ext_bar,
        ext_wham,
        ext_mbar,
        ext_tram,
        ext_dtram,
        ext_trammbar,
        ext_mbar_direct,
        ext_tram_direct,
        ext_trammbar_direct,
        ext_util]
    return cythonize(exts)

class lazy_cythonize(list):
    """evaluates extension list lazyly.
    pattern taken from http://tinyurl.com/qb8478q"""
    def __init__(self, callback):
        self._list, self.callback = None, callback
    def c_list(self):
        if self._list is None: self._list = self.callback()
        return self._list
    def __iter__(self):
        for e in self.c_list(): yield e
    def __getitem__(self, ii): return self.c_list()[ii]
    def __len__(self): return len(self.c_list())

long_description = 'The thermotools package is a lowlevel implementation toolbox for the analyis \
of multi-ensemble calculations. It contains estimators for the state-continuous transition-based \
reweighting analysis method (TRAM) and its state-discrete variant (dTRAM), Bennet acceptance ratio \
(BAR) and its multi-state variant (MBAR), and the weighted histogram analysis method (WHAM). \
While you can use thermotools on its own, we recommend to use it in combination with PyEMMA.'

setup(
    cmdclass=versioneer.get_cmdclass(),
    ext_modules=lazy_cythonize(extensions),
    name='thermotools',
    version=versioneer.get_version(),
    description='lowlevel implementation toolbox for the analyis of multi-ensemble calculations',
    long_description=long_description,
    classifiers=[
        'Development Status :: 4 - Beta',
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
        'MEMM',
        'multi ensemble',
        'free energy',
        'Markov state model',
        'BAR',
        'WHAM',
        'MBAR',
        'TRAM',
        'dTRAM'],
    url='https://github.com/markovmodel/thermotools',
    maintainer='Christoph Wehmeyer',
    maintainer_email='christoph.wehmeyer@fu-berlin.de',
    license='LGPLv3+',
    setup_requires=[
        'numpy>=1.7',
        'cython>=0.20',
        'setuptools>=0.6'],
    tests_require=[
        'numpy>=1.7',
        'scipy>=0.11',
        'msmtools>=1.1',
        'nose'],
    install_requires=[
        'numpy>=1.7',
        'scipy>=0.11',
        'msmtools>=1.1'],
    packages=['thermotools'],
    test_suite='nose.collector',
    scripts=[]
)
