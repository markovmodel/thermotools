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

from thermotools.dtram import iterate_lognu, iterate_fi, get_pk
import numpy as np
from numpy.testing import assert_allclose

def test_lognu_zero_counts():
    nm = 200
    nt = 100
    log_nu_K_i = np.zeros(shape=(nt, nm), dtype=np.float64)
    b_K_i = np.zeros(shape=(nt, nm), dtype=np.float64)
    f_i = np.zeros(shape=(nm,), dtype=np.float64)
    C_K_ij = np.zeros(shape=(nt, nm, nm), dtype=np.intc) # C_K_ii = 1.0E-10 (internal prior)
    scratch_i = np.zeros(shape=(nm,), dtype=np.float64)
    new_log_nu_K_i = np.zeros(shape=(nt, nm), dtype=np.float64)
    ref_log_nu_K_i = np.log(1.0E-10*np.ones(shape=(nt, nm), dtype=np.float64)) # (prior)
    iterate_lognu(log_nu_K_i, b_K_i, f_i, C_K_ij, scratch_i, new_log_nu_K_i)
    assert_allclose(new_log_nu_K_i, ref_log_nu_K_i, atol=1.0E-16)

def test_lognu_all_factors_unity():
    nm = 200
    nt = 100
    log_nu_K_i = np.zeros(shape=(nt, nm), dtype=np.float64)
    b_K_i = np.zeros(shape=(nt, nm), dtype=np.float64)
    f_i = np.zeros(shape=(nm,), dtype=np.float64)
    C_K_ij = np.ones(shape=(nt, nm, nm), dtype=np.intc)
    scratch_i = np.zeros(shape=(nm,), dtype=np.float64)
    new_log_nu_K_i = np.zeros(shape=(nt, nm), dtype=np.float64)
    ref_log_nu_K_i = np.log(nm*np.ones(shape=(nt, nm), dtype=np.float64))
    iterate_lognu(log_nu_K_i, b_K_i, f_i, C_K_ij, scratch_i, new_log_nu_K_i)
    assert_allclose(new_log_nu_K_i, ref_log_nu_K_i, atol=1.0E-16)

def test_lognu_K_range():
    nm = 200
    nt = 100
    log_nu_K_i = np.zeros(shape=(nt, nm), dtype=np.float64)
    for K in range(nt):
        log_nu_K_i[K, :] = np.log(K + 1.0)
    b_K_i = np.zeros(shape=(nt, nm), dtype=np.float64)
    f_i = np.zeros(shape=(nm,), dtype=np.float64)
    C_K_ij = np.ones(shape=(nt, nm, nm), dtype=np.intc)
    scratch_i = np.zeros(shape=(nm,), dtype=np.float64)
    new_log_nu_K_i = np.zeros(shape=(nt, nm), dtype=np.float64)
    ref_log_nu_K_i = np.log(nm*np.ones(shape=(nt, nm), dtype=np.float64))
    iterate_lognu(log_nu_K_i, b_K_i, f_i, C_K_ij, scratch_i, new_log_nu_K_i)
    assert_allclose(new_log_nu_K_i, ref_log_nu_K_i, atol=1.0E-16)


def test_fi_zero_counts():
    nm = 200
    nt = 100
    log_nu_K_i = np.zeros(shape=(nt, nm), dtype=np.float64)
    b_K_i = np.zeros(shape=(nt, nm), dtype=np.float64)
    f_i = np.zeros(shape=(nm,), dtype=np.float64)
    C_K_ij = np.zeros(shape=(nt, nm, nm), dtype=np.intc)
    scratch_TM = np.zeros(shape=(nt, nm), dtype=np.float64)
    scratch_M = np.zeros(shape=(nm,), dtype=np.float64)
    new_f_i = np.zeros(shape=(nm,), dtype=np.float64)
    ref_f_i = 0.0
    iterate_fi(log_nu_K_i, b_K_i, f_i, C_K_ij, scratch_TM, new_f_i)
    assert_allclose(new_f_i, ref_f_i, atol=1.0E-16)

def test_fi_all_factors_unity():
    nm = 200
    nt = 100
    log_nu_K_i = np.zeros(shape=(nt, nm), dtype=np.float64)
    b_K_i = np.zeros(shape=(nt, nm), dtype=np.float64)
    f_i = np.zeros(shape=(nm,), dtype=np.float64)
    C_K_ij = np.ones(shape=(nt, nm, nm), dtype=np.intc)
    scratch_TM = np.zeros(shape=(nt, nm), dtype=np.float64)
    scratch_M = np.zeros(shape=(nm,), dtype=np.float64)
    new_f_i = np.zeros(shape=(nm,), dtype=np.float64)
    ref_f_i = 0.0
    iterate_fi(log_nu_K_i, b_K_i, f_i, C_K_ij, scratch_TM, new_f_i)
    assert_allclose(new_f_i, ref_f_i, atol=1.0E-16)


def test_pij_zero_counts():
    nm = 200
    nt = 100
    log_nu_K_i = np.zeros(shape=(nt, nm), dtype=np.float64)
    b_K_i = np.zeros(shape=(nt, nm), dtype=np.float64)
    f_i = np.zeros(shape=(nm,), dtype=np.float64)
    C_K_ij = np.zeros(shape=(nt, nm, nm), dtype=np.intc)
    scratch_M = np.zeros(shape=(nm,), dtype=np.float64)
    p_K_ij = get_pk(log_nu_K_i, b_K_i, f_i, C_K_ij, scratch_M)
    ref_p_ij = np.eye(nm, dtype=np.float64)
    for K in range(nt):
        assert_allclose(p_K_ij[K, :, :], ref_p_ij, atol=1.0E-16)

def test_pij_all_factors_unity():
    nm = 200
    nt = 100
    log_nu_K_i = np.zeros(shape=(nt, nm), dtype=np.float64)
    b_K_i = np.zeros(shape=(nt, nm), dtype=np.float64)
    f_i = np.zeros(shape=(nm,), dtype=np.float64)
    C_K_ij = np.ones(shape=(nt, nm, nm), dtype=np.intc)
    scratch_M = np.zeros(shape=(nm,), dtype=np.float64)
    p_K_ij = get_pk(log_nu_K_i, b_K_i, f_i, C_K_ij, scratch_M)
    ref_p_ij = np.ones(shape=(nm, nm), dtype=np.float64) + np.eye(nm, dtype=np.float64)*1.0E-10
    ref_p_ij /= ref_p_ij.sum(axis=1)[:, np.newaxis]
    for K in range(nt):
        assert_allclose(p_K_ij[K, :, :], ref_p_ij, atol=1.0E-16)
