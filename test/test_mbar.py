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

from thermotools.mbar import iterate_fk
from thermotools.lse import logsumexp
import numpy as np
from numpy.testing import assert_allclose

def test_mbar_fk_with_zeros():
    T = 5
    X = 10
    log_N_K = np.zeros(shape=(T,), dtype=np.float64)
    b_K_x = np.zeros(shape=(T, X), dtype=np.float64)
    scratch = np.zeros(shape=(T,), dtype=np.float64)
    f_K = np.zeros(shape=(T,), dtype=np.float64)
    new_f_K = np.zeros(shape=(T,), dtype=np.float64)
    ref = 0.0
    iterate_fk(log_N_K, f_K, b_K_x, scratch, new_f_K)
    assert_allclose(new_f_K, ref, atol=1.0E-15)

def test_mbar_fk_with_ascending_bias():
    T = 5
    X = 10
    log_N_K = np.zeros(shape=(T,), dtype=np.float64)
    b_K_x = np.array([[K]*X for K in range(T)], dtype=np.float64)
    scratch = np.zeros(shape=(T,), dtype=np.float64)
    f_K = np.zeros(shape=(T,), dtype=np.float64)
    new_f_K = np.zeros(shape=(T,), dtype=np.float64)
    ref = np.array([float(K) - np.log(X) for K in range(T)], dtype=np.float64)
    ref -= ref.min()
    iterate_fk(log_N_K, f_K, b_K_x, scratch, new_f_K)
    assert_allclose(new_f_K, ref, atol=1.0E-15)
