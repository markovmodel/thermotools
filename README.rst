***********
thermotools
***********

.. image:: https://travis-ci.org/markovmodel/thermotools.svg?branch=devel
   :target: https://travis-ci.org/markovmodel/thermotools
.. image:: https://badge.fury.io/py/thermotools.svg
   :target: https://pypi.python.org/pypi/thermotools
.. image:: https://binstar.org/omnia/thermotools/badges/installer/conda.svg
   :target: https://conda.binstar.org/omnia
.. image:: https://binstar.org/omnia/thermotools/badges/version.svg
   :target: https://binstar.org/omnia/thermotools

This Python package provides a lowlevel implementation of (transition-based) reweighting analyis
methods.


Installation
============

Using conda (recommended)::

   conda install -c omnia thermotools

Using pip from PyPI::

   pip install thermotools

Using pip from github (this will install the latest development version)::

   pip install git+https://github.com/markovmodel/thermotools.git@devel

If the pip installation fails due to missing packages, you need to install (or upgrade) them manually::

   pip install --upgrade cython
   pip install --upgrade numpy
