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

import thermotools.mbar as mbar
import numpy as np
from numpy.testing import assert_allclose

ca = np.ascontiguousarray

def test_mbar_fk_with_zeros():
    T = 5
    X = 10
    log_therm_state_counts = np.zeros(shape=(T,), dtype=np.float64)
    bias_energies = np.zeros(shape=(T, X), dtype=np.float64)
    scratch = np.zeros(shape=(T,), dtype=np.float64)
    therm_energies = np.zeros(shape=(T,), dtype=np.float64)
    new_therm_energies = np.zeros(shape=(T,), dtype=np.float64)
    mbar.update_therm_energies(
        log_therm_state_counts, therm_energies, [ca(bias_energies.T)], scratch, new_therm_energies)
    assert_allclose(new_therm_energies, 0.0, atol=1.0E-15)

def test_mbar_fk_with_ascending_bias():
    T = 5
    X = 10
    log_therm_state_counts = np.zeros(shape=(T,), dtype=np.float64)
    bias_energies = np.array([[K]*X for K in range(T)], dtype=np.float64)
    scratch = np.zeros(shape=(T,), dtype=np.float64)
    therm_energies = np.zeros(shape=(T,), dtype=np.float64)
    new_therm_energies = np.zeros(shape=(T,), dtype=np.float64)
    ref = np.array([float(K) - np.log(X) for K in range(T)], dtype=np.float64)
    ref -= ref.min()
    mbar.update_therm_energies(
        log_therm_state_counts, therm_energies, [ca(bias_energies.T)], scratch, new_therm_energies)
    assert_allclose(new_therm_energies, ref, atol=1.0E-15)
