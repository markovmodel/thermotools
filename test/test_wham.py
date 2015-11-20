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
import numpy as np
from numpy.testing import assert_allclose

def test_wham_fk_with_zeros():
    T = 5
    M = 10
    conf_energies = np.zeros(shape=(M,), dtype=np.float64)
    bias_energies = np.zeros(shape=(T, M), dtype=np.float64)
    scratch = np.zeros(shape=(M,), dtype=np.float64)
    therm_energies = np.zeros(shape=(T,), dtype=np.float64)
    ref = -np.log(M)
    wham.update_therm_energies(conf_energies, bias_energies, scratch, therm_energies)
    assert_allclose(therm_energies, ref, atol=1.0E-15)

def test_wham_fk_with_ascending_K():
    T = 5
    M = 10
    conf_energies = np.zeros(shape=(M,), dtype=np.float64)
    bias_energies = -np.log(np.array([[K]*M for K in range(1, T + 1)], dtype=np.float64)/float(M))
    scratch = np.zeros(shape=(M,), dtype=np.float64)
    therm_energies = np.zeros(shape=(T,), dtype=np.float64)
    ref = -np.log(np.arange(T).astype(np.float64) + 1)
    wham.update_therm_energies(conf_energies, bias_energies, scratch, therm_energies)
    assert_allclose(therm_energies, ref, atol=1.0E-15)

def test_wham_fi_with_zeros():
    T = 5
    M = 10
    N_K_i = np.ones(shape=(T, M), dtype=np.float64)
    log_therm_state_counts = np.log(N_K_i.sum(axis=1))
    log_conf_state_counts = np.log(N_K_i.sum(axis=0))
    therm_energies = np.zeros(shape=(T,), dtype=np.float64)
    bias_energies = np.zeros(shape=(T, M), dtype=np.float64)
    conf_energies = np.zeros(shape=(M,), dtype=np.float64)
    scratch_T = np.zeros(shape=(T,), dtype=np.float64)
    wham.update_conf_energies(
        log_therm_state_counts, log_conf_state_counts, therm_energies, bias_energies,
        scratch_T, conf_energies)
    assert_allclose(conf_energies, np.log(M), atol=1.0E-15)

def test_wham_fi_with_ascending_K():
    T = 5
    M = 10
    N_K_i = np.ones(shape=(T, M), dtype=np.float64)
    log_therm_state_counts = np.log(N_K_i.sum(axis=1))
    log_conf_state_counts = np.log(N_K_i.sum(axis=0))
    conf_energies = np.zeros(shape=(M,), dtype=np.float64)
    bias_energies = np.log(
        np.array([[i*M for i in range(1, M + 1)] for K in range(T)], dtype=np.float64))
    scratch_T = np.zeros(shape=(T,), dtype=np.float64)
    therm_energies = np.zeros(shape=(T,), dtype=np.float64)
    ref = -np.log((np.arange(M).astype(np.float64) + 1))
    wham.update_conf_energies(
        log_therm_state_counts, log_conf_state_counts, therm_energies, bias_energies,
        scratch_T, conf_energies)
    assert_allclose(conf_energies, ref, atol=1.0E-15)
