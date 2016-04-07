******************************
thermotools - developers guide
******************************

Branches
========

  * **master** should always be stable and pointing at the latest conda/PyPI release.
  * **devel** contains the latest changes/fixes/features which will converege at some point to the next release. This branch should also be kept stable. Please fork from this branch if you want to contribute.
  * All other branches should be considered experimental, temporary, and not necessarily stable. Use them to introduce new features to thermotools and merge them into **devel** once stable and accepted.

How to contribute
=================

If you want to contribute to thermotools, please adhere to the following procedure:

  #. Create a feature branch via fork (or branch if you have the necessary permissions) from the latest commit to **devel**.
  #. Work exclusively on this local version.
  #. Provide unit tests for new features and amend unit tests if you change existing functions.
  #. Also provide/amend docstrings for propper documentation.
  #. Push this feature branch to github and create a pull request to merge your feature branch into **devel**.

How is the code organised
=========================

This repository is organised as follows:

  * **ext** contains the C/Cython implementations of the estimators as well as convenience functions, where each subdirectory contains a C header file, a C file with the actual implementation, and a pyx file with the Python API to the C functions and docstrings.
  * **test** contains unit tests.
  * **thermotools** contains an ``__init__.py`` file which imports the external modules onto the package level namespace.
  * **setup.py** collects all external modules' source (from **ext**) and builds shared objects during the installation.
