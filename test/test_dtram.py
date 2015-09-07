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

from thermotools import dtram_set_lognu, dtram_lognu, dtram_fi, dtram_p, dtram_fk
import numpy as np
from numpy.testing import assert_allclose

def run_dtram(C_K_ij, b_K_i, log_nu_K_i, f_K, f_i, maxiter, ftol):
    dtram_set_lognu(log_nu_K_i, C_K_ij)
    scratch_K_i = np.zeros(shape=b_K_i.shape, dtype=np.float64)
    scratch_i = np.zeros(shape=f_i.shape, dtype=np.float64)
    old_f_K = f_K.copy()
    for m in range(maxiter):
        tmp_log_nu_K_i = np.copy(log_nu_K_i)
        dtram_lognu(tmp_log_nu_K_i, b_K_i, f_i, C_K_ij, scratch_i, log_nu_K_i)
        tmp_f_i = np.copy(f_i)
        dtram_fi(log_nu_K_i, b_K_i, tmp_f_i, C_K_ij, scratch_K_i, scratch_i, f_i)
        dtram_fk(b_K_i, f_i, scratch_i, f_K)
        nz = (old_f_K != 0.0)
        if (nz.sum() > 0) and (np.max(np.abs((f_K[nz] - old_f_K[nz])/old_f_K[nz])) < ftol):
            break
        else:
            old_f_K[:] = f_K[:]



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
    dtram_lognu(log_nu_K_i, b_K_i, f_i, C_K_ij, scratch_i, new_log_nu_K_i)
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
    dtram_lognu(log_nu_K_i, b_K_i, f_i, C_K_ij, scratch_i, new_log_nu_K_i)
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
    dtram_lognu(log_nu_K_i, b_K_i, f_i, C_K_ij, scratch_i, new_log_nu_K_i)
    assert_allclose(new_log_nu_K_i, ref_log_nu_K_i, atol=1.0E-16)

def test_dtram_with_toy_model():
    C_K_ij = np.array([
        [[2358, 29, 0], [29, 0, 32], [0, 32, 197518]],
        [[16818, 16763, 0], [16763, 0, 16510], [0, 16510, 16635]]], dtype=np.intc)
    b_K_i = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 8.0]], dtype=np.float64)
    f_i = np.zeros(shape=(b_K_i.shape[1],), dtype=np.float64)
    f_K = np.zeros(shape=(b_K_i.shape[0],), dtype=np.float64)
    log_nu_K_i = np.zeros(shape=b_K_i.shape, dtype=np.float64)
    run_dtram(C_K_ij, b_K_i, log_nu_K_i, f_K, f_i, 10000, 1.0E-15)
    pi = np.array([1.82026887e-02, 3.30458960e-04, 9.81466852e-01], dtype=np.float64)
    assert_allclose(np.exp(-f_i), pi, atol=1.0E-8)
    T = np.array([
        [9.90504397e-01, 9.49560284e-03, 0.0],
        [5.23046803e-01, 0.0, 4.76953197e-01],
        [0.0, 1.60589690e-04, 9.99839410e-01]], dtype=np.float64)
    scratch_i = np.zeros(shape=f_i.shape, dtype=np.float64)
    C_0_ij = np.ascontiguousarray(C_K_ij[0, :, :])
    b_0_i = np.ascontiguousarray(b_K_i[0, :])
    log_nu_0_i = np.ascontiguousarray(log_nu_K_i[0, :])
    p_ij = np.zeros(shape=C_0_ij.shape, dtype=np.float64)
    dtram_p(log_nu_0_i, b_0_i, f_i, C_0_ij, scratch_i, p_ij)
    assert_allclose(p_ij, T, atol=1.0E-8)
