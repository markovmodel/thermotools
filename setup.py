from setuptools import setup
from distutils.core import Extension
from sys import exit as sys_exit

try:
    from Cython.Distutils import build_ext
except ImportError:
    print "ERROR - please install the cython dependency manually:"
    print "pip install cython"
    sys_exit(1)

try:
    from numpy import get_include
except ImportError:
    print "ERROR - please install the numpy dependency manually:"
    print "pip install numpy"
    sys_exit(1)

ext_lse = Extension(
    "thermotools.lse",
    sources=["ext/lse/lse.pyx", "ext/lse/_lse.c"],
    include_dirs=[get_include()],
    extra_compile_args=["-O3"])
ext_wham = Extension(
    "thermotools.wham",
    sources=["ext/wham/wham.pyx", "ext/wham/_wham.c", "ext/lse/_lse.c"],
    include_dirs=[get_include()],
    extra_compile_args=["-O3"])
ext_dtram = Extension(
    "thermotools.dtram",
    sources=["ext/dtram/dtram.pyx", "ext/dtram/_dtram.c", "ext/lse/_lse.c"],
    include_dirs=[get_include()],
    extra_compile_args=["-O3"])

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[
        ext_lse,
        ext_wham,
        ext_dtram],
    name='thermotools',
    version='0.0.0',
    description='Lowlevel implementation of free energy estimators',
    long_description='Lowlevel implementation of free energy estimators',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: C',
        'Programming Language :: Cython',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Physics'],
    keywords=[
        'free energy',
        'Markov state model',
        'WHAM',
        'dTRAM'],
    url='',
    author='Christoph Wehmeyer',
    author_email='christoph.wehmeyer@fu-berlin.de',
    license='Simplified BSD License',
    setup_requires=[
        'numpy>=1.7.1',
        'cython>=0.15',
        'setuptools>=0.6'],
    tests_require=['numpy>=1.7.1', 'nose>=1.3'],
    install_requires=['numpy>=1.7.1'],
    packages=['thermotools'],
    test_suite='nose.collector',
    scripts=[]
)
