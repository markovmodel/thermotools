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

import thermotools.wham as wham
import thermotools.mbar as mbar
import thermotools.dtram as dtram
import numpy as np
from numpy.testing import assert_allclose
from nose.tools import assert_true

from thermotools.callback import CallbackInterrupt, generic_callback_stop

#   ************************************************************************************************
#   test generic_callback_stop
#   ************************************************************************************************

def test_dtram_stop():
    T = 5
    M = 10
    therm_energies, conf_energies, log_lagrangian_mult, err_traj, lll_traj = dtram.estimate(
        np.ones(shape=(T, M, M), dtype=np.intc),
        np.zeros(shape=(T, M), dtype=np.float64),
        maxiter=10, maxerr=-1.0, err_out=1, lll_out=2,
        callback=generic_callback_stop)
    assert_allclose(therm_energies, 0.0, atol=1.0E-15)
    assert_allclose(conf_energies, np.log(M), atol=1.0E-15)
    assert_allclose(log_lagrangian_mult, np.log(M + dtram.get_prior()), atol=1.0E-15)
    assert_true(err_traj.shape[0] == 1)
    assert_true(lll_traj.shape[0] == 0)

def test_wham_stop():
    T = 5
    M = 10
    therm_energies, conf_energies, err_traj, lll_traj = wham.estimate(
        np.ones(shape=(T, M), dtype=np.intc),
        np.zeros(shape=(T, M), dtype=np.float64),
        maxiter=10, maxerr=-1.0, err_out=1, lll_out=2,
        callback=generic_callback_stop)
    assert_allclose(therm_energies, 0.0, atol=1.0E-15)
    assert_allclose(conf_energies, np.log(M), atol=1.0E-15)
    assert_true(err_traj.shape[0] == 1)
    assert_true(lll_traj.shape[0] == 0)


