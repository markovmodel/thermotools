=====
Usage
=====
 
.. toctree::
   :maxdepth: 1

Package layout
==============

thermotools provides Python functions for several estimators, e.g., WHAM or dTRAM. Each estimator has its own module which provides a function to perform a full estimation; it also provides access to functions for all steps involved in the estimation.

As an example, the estimation function for WHAM is available via::

   from thermotools.wham import estimate

The estimators are mostly written C, which is compiled at installation time via ``cython``. The C code can be found in the github repository in the folder ``ext``.

Besides the estimators, thermotools provides Python wrappers to the C tools that are used in all the estimators. The wrappers can be accessed via the ``util`` module.

How to run an estimation
========================

Once you have decided on an estimator, you have to prepare the data and pass it to the respective estimator's ``estimate`` function. For WHAM, the required input consists of two matrices (numpy.ndarray) ``state_counts`` and  ``bias_energies``. The first matrix counts how often each discrete state has been observed in each thermodynamic state; the second matrix contains the reduced bias energies in all thermodynamic and discrete states. The call::

   from thermotools.wham import estimate
   therm_energies, conf_energies, = estimate(state_counts, bias_matrix)

returns two numpy.ndarrays with the reduced free energies of the thermodynamic states and configurational states. Please note that the estimation process can be controlled by additional parameters of the ``estimate`` function; more information on this can be found in the docstrings and API documentation.
