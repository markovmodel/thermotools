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

from thermotools.wham import iterate_fi, iterate_fk
from thermotools.lse import logsumexp
import numpy as np
from numpy.testing import assert_allclose

def test_wham_fk_with_zeros():
    T = 5
    M = 10
    f_i = np.zeros(shape=(M,), dtype=np.float64)
    b_K_i = np.zeros(shape=(T, M), dtype=np.float64)
    scratch = np.zeros(shape=(M,), dtype=np.float64)
    f_K = np.zeros(shape=(T,), dtype=np.float64)
    ref = -np.log(M)
    iterate_fk(f_i, b_K_i, scratch, f_K)
    assert_allclose(f_K, ref, atol=1.0E-15)

def test_wham_fk_with_ascending_K():
    T = 5
    M = 10
    f_i = np.zeros(shape=(M,), dtype=np.float64)
    b_K_i = -np.log(np.array([[K]*M for K in range(1, T + 1)], dtype=np.float64)/float(M))
    scratch = np.zeros(shape=(M,), dtype=np.float64)
    f_K = np.zeros(shape=(T,), dtype=np.float64)
    ref = -np.log(np.arange(T).astype(np.float64) + 1)
    iterate_fk(f_i, b_K_i, scratch, f_K)
    assert_allclose(f_K, ref, atol=1.0E-15)

def test_wham_fi_with_zeros():
    T = 5
    M = 10
    N_K_i = np.ones(shape=(T, M), dtype=np.float64)
    log_N_K = np.log(N_K_i.sum(axis=1))
    log_N_i = np.log(N_K_i.sum(axis=0))
    f_i = np.zeros(shape=(M,), dtype=np.float64)
    b_K_i = np.zeros(shape=(T, M), dtype=np.float64)
    scratch_T = np.zeros(shape=(T,), dtype=np.float64)
    f_K = np.zeros(shape=(T,), dtype=np.float64)
    ref = 0.0
    iterate_fi(log_N_K, log_N_i, f_K, b_K_i, scratch_T, f_i)
    assert_allclose(f_i, ref, atol=1.0E-15)

def test_wham_fi_with_ascending_K():
    T = 5
    M = 10
    N_K_i = np.ones(shape=(T, M), dtype=np.float64)
    log_N_K = np.log(N_K_i.sum(axis=1))
    log_N_i = np.log(N_K_i.sum(axis=0))
    f_i = np.zeros(shape=(M,), dtype=np.float64)
    b_K_i = np.log(np.array([[i*M for i in range(1, M + 1)] for K in range(T)], dtype=np.float64))
    scratch_T = np.zeros(shape=(T,), dtype=np.float64)
    f_K = np.zeros(shape=(T,), dtype=np.float64)
    ref = -np.log((np.arange(M).astype(np.float64) + 1) / (0.5 * M * (M + 1)))
    ref -= ref.min() 
    iterate_fi(log_N_K, log_N_i, f_K, b_K_i, scratch_T, f_i)
    assert_allclose(f_i, ref, atol=1.0E-15)
