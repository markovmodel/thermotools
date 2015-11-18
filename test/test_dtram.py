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

import thermotools.dtram as dtram
import numpy as np
from numpy.testing import assert_allclose

def test_prior():
    assert_allclose(np.log(dtram.get_prior()), dtram.get_log_prior(), atol=1.0E-16)

def test_lognu_zero_counts():
    nm = 200
    nt = 100
    log_lagrangian_mult = np.zeros(shape=(nt, nm), dtype=np.float64)
    bias_energies = np.zeros(shape=(nt, nm), dtype=np.float64)
    conf_energies = np.zeros(shape=(nm,), dtype=np.float64)
    C_K_ij = np.zeros(shape=(nt, nm, nm), dtype=np.intc) # C_K_ii may have an internal prior
    scratch_i = np.zeros(shape=(nm,), dtype=np.float64)
    new_log_lagrangian_mult = np.zeros(shape=(nt, nm), dtype=np.float64)
    ref_log_lagrangian_mult = np.log(dtram.get_prior() * np.ones(shape=(nt, nm), dtype=np.float64))
    dtram.update_log_lagrangian_mult(
        log_lagrangian_mult, bias_energies, conf_energies, C_K_ij,
        scratch_i, new_log_lagrangian_mult)
    assert_allclose(new_log_lagrangian_mult, ref_log_lagrangian_mult, atol=1.0E-16)

def test_lognu_all_factors_unity():
    nm = 200
    nt = 100
    log_lagrangian_mult = np.zeros(shape=(nt, nm), dtype=np.float64)
    bias_energies = np.zeros(shape=(nt, nm), dtype=np.float64)
    conf_energies = np.zeros(shape=(nm,), dtype=np.float64)
    C_K_ij = np.ones(shape=(nt, nm, nm), dtype=np.intc)
    scratch_i = np.zeros(shape=(nm,), dtype=np.float64)
    new_log_lagrangian_mult = np.zeros(shape=(nt, nm), dtype=np.float64)
    ref_log_lagrangian_mult = np.log(nm*np.ones(shape=(nt, nm), dtype=np.float64))
    dtram.update_log_lagrangian_mult(
        log_lagrangian_mult, bias_energies, conf_energies, C_K_ij,
        scratch_i, new_log_lagrangian_mult)
    assert_allclose(new_log_lagrangian_mult, ref_log_lagrangian_mult, atol=1.0E-16)

def test_lognu_K_range():
    nm = 200
    nt = 100
    log_lagrangian_mult = np.zeros(shape=(nt, nm), dtype=np.float64)
    for K in range(nt):
        log_lagrangian_mult[K, :] = np.log(K + 1.0)
    bias_energies = np.zeros(shape=(nt, nm), dtype=np.float64)
    conf_energies = np.zeros(shape=(nm,), dtype=np.float64)
    C_K_ij = np.ones(shape=(nt, nm, nm), dtype=np.intc)
    scratch_i = np.zeros(shape=(nm,), dtype=np.float64)
    new_log_lagrangian_mult = np.zeros(shape=(nt, nm), dtype=np.float64)
    ref_log_lagrangian_mult = np.log(nm*np.ones(shape=(nt, nm), dtype=np.float64))
    dtram.update_log_lagrangian_mult(
        log_lagrangian_mult, bias_energies, conf_energies, C_K_ij,
        scratch_i, new_log_lagrangian_mult)
    assert_allclose(new_log_lagrangian_mult, ref_log_lagrangian_mult, atol=1.0E-16)


def test_fi_zero_counts():
    nm = 200
    nt = 100
    log_lagrangian_mult = np.zeros(shape=(nt, nm), dtype=np.float64)
    bias_energies = np.zeros(shape=(nt, nm), dtype=np.float64)
    conf_energies = np.zeros(shape=(nm,), dtype=np.float64)
    C_K_ij = np.zeros(shape=(nt, nm, nm), dtype=np.intc)
    scratch_TM = np.zeros(shape=(nt, nm), dtype=np.float64)
    scratch_M = np.zeros(shape=(nm,), dtype=np.float64)
    new_conf_energies = np.zeros(shape=(nm,), dtype=np.float64)
    dtram.update_conf_energies(
        log_lagrangian_mult, bias_energies, conf_energies, C_K_ij,
        scratch_TM, new_conf_energies)
    assert_allclose(new_conf_energies, 0.0, atol=1.0E-16)

def test_fi_all_factors_unity():
    nm = 200
    nt = 100
    log_lagrangian_mult = np.zeros(shape=(nt, nm), dtype=np.float64)
    bias_energies = np.zeros(shape=(nt, nm), dtype=np.float64)
    conf_energies = np.zeros(shape=(nm,), dtype=np.float64)
    C_K_ij = np.ones(shape=(nt, nm, nm), dtype=np.intc)
    scratch_TM = np.zeros(shape=(nt, nm), dtype=np.float64)
    scratch_M = np.zeros(shape=(nm,), dtype=np.float64)
    new_conf_energies = np.zeros(shape=(nm,), dtype=np.float64)
    dtram.update_conf_energies(
        log_lagrangian_mult, bias_energies, conf_energies, C_K_ij,
        scratch_TM, new_conf_energies)
    assert_allclose(new_conf_energies, 0.0, atol=1.0E-16)


def test_pij_zero_counts():
    nm = 200
    nt = 100
    log_lagrangian_mult = np.zeros(shape=(nt, nm), dtype=np.float64)
    bias_energies = np.zeros(shape=(nt, nm), dtype=np.float64)
    conf_energies = np.zeros(shape=(nm,), dtype=np.float64)
    C_K_ij = np.zeros(shape=(nt, nm, nm), dtype=np.intc)
    scratch_M = np.zeros(shape=(nm,), dtype=np.float64)
    p_K_ij = dtram.estimate_transition_matrices(
        log_lagrangian_mult, bias_energies, conf_energies, C_K_ij, scratch_M)
    ref_p_ij = np.eye(nm, dtype=np.float64)
    for K in range(nt):
        assert_allclose(p_K_ij[K, :, :], ref_p_ij, atol=1.0E-16)

def test_pij_all_factors_unity():
    nm = 200
    nt = 100
    log_lagrangian_mult = np.zeros(shape=(nt, nm), dtype=np.float64)
    bias_energies = np.zeros(shape=(nt, nm), dtype=np.float64)
    conf_energies = np.zeros(shape=(nm,), dtype=np.float64)
    C_K_ij = np.ones(shape=(nt, nm, nm), dtype=np.intc)
    scratch_M = np.zeros(shape=(nm,), dtype=np.float64)
    p_K_ij = dtram.estimate_transition_matrices(
        log_lagrangian_mult, bias_energies, conf_energies, C_K_ij, scratch_M)
    ref_p_ij = np.ones(shape=(nm, nm), dtype=np.float64) + \
        np.eye(nm, dtype=np.float64) * dtram.get_prior()
    ref_p_ij /= ref_p_ij.sum(axis=1)[:, np.newaxis]
    for K in range(nt):
        assert_allclose(p_K_ij[K, :, :], ref_p_ij, atol=1.0E-16)
