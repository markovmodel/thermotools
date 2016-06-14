# This file is part of thermotools.
#
# Copyright 2015, 2016 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
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

import numpy as np
import msmtools
import thermotools.tram as tram
import thermotools.tram_direct as tram_direct
import thermotools.trammbar as trammbar
import thermotools.trammbar_direct as trammbar_direct
import sys
from numpy.testing import assert_allclose

def tower_sample(distribution):
    cdf = np.cumsum(distribution)
    rnd = np.random.rand() * cdf[-1]
    return np.searchsorted(cdf, rnd)

def T_matrix(energy):
    n = energy.shape[0]
    metropolis = energy[np.newaxis, :] - energy[:, np.newaxis]
    metropolis[(metropolis < 0.0)] = 0.0
    selection = np.zeros((n, n))
    selection += np.diag(np.ones(n - 1) * 0.5, k=1)
    selection += np.diag(np.ones(n - 1) * 0.5, k=-1)
    selection[0, 0] = 0.5
    selection[-1, -1] = 0.5
    metr_hast = selection * np.exp(-metropolis)
    for i in range(metr_hast.shape[0]):
        metr_hast[i, i] = 0.0
        metr_hast[i, i] = 1.0 - metr_hast[i, :].sum()
    return metr_hast

def draw_transition_counts(transition_matrices, n_samples, x0):
    """generates a discrete Markov chain"""
    count_matrices = np.zeros(shape=transition_matrices.shape, dtype=np.intc)
    conf_state_sequence = np.zeros(
        shape=(transition_matrices.shape[0] * (n_samples + 1),), dtype=np.intc)
    state_counts = np.zeros(shape=transition_matrices.shape[0:2], dtype=np.intc)
    h = 0
    for K in range(transition_matrices.shape[0]):
        x = x0
        state_counts[K, x] += 1
        conf_state_sequence[h] = x
        h += 1
        for s in range(n_samples):
            x_new = tower_sample(transition_matrices[K, x, :])
            count_matrices[K, x, x_new] += 1
            x = x_new
            state_counts[K, x] += 1
            conf_state_sequence[h] = x
            h += 1
    return count_matrices, conf_state_sequence, state_counts

class TestRandom(object):
    @classmethod
    def setup_class(cls):
        n_therm_states = 4
        n_conf_states = 4
        n_samples = 10000
        cls.bias_energies = np.zeros(shape=(n_therm_states, n_conf_states), dtype=np.float64)
        cls.T = np.zeros(shape=(n_therm_states, n_conf_states, n_conf_states), dtype=np.float64)
        while True:
            # generate random stionary distributions
            for k in range(n_therm_states):
                cls.bias_energies[k, :] = -np.log(np.random.rand(n_conf_states))
                if k > 0:
                    cls.bias_energies[k, :] += np.random.rand()
            # generate transition matrices
            for k in range(n_therm_states):
                cls.T[k, :, :] = T_matrix(cls.bias_energies[k, :])
            cls.count_matrices, cls.conf_state_sequence, cls.state_counts = draw_transition_counts(
                cls.T, n_samples, 0)
            if msmtools.analysis.is_connected(cls.count_matrices.sum(axis=0), directed=True):
                break
        cls.bias_energies_sh = cls.bias_energies - cls.bias_energies[0, :] # hide pi^{0}_i
        cls.bias_energies_sh = np.ascontiguousarray(cls.bias_energies_sh[:, cls.conf_state_sequence])
        cls.n_therm_states = n_therm_states
        cls.n_conf_states = n_conf_states
        cls.n_samples = n_samples
    @classmethod
    def teardown_class(cls):
        pass
    def setup(self):
        pass
    def teardown(self):
        pass
    def test_tram(self):
        self.helper_tram(False, 0, False)
    def test_tram_direct(self):
        self.helper_tram(True, 0, False)
    def test_tram_direct_with_dTRAM_acceleration(self):
        self.helper_tram(True, 1, False)
    def test_trammbar_as_tram(self):
        self.helper_tram(False, 0, True)
    def test_trammbar_direct_as_tram(self):
        self.helper_tram(True, 0, True)
    def helper_tram(self, direct_space, N_dtram_accelerations, use_trammbar):
        if direct_space:
            _tram = tram_direct
        else:
            _tram = tram
        if use_trammbar:
            if direct_space:
                _tram = trammbar_direct
            else:
                _tram = trammbar
        ca = np.ascontiguousarray
        biased_conf_energies, conf_energies, therm_energies, log_lagrangian_mult, error_history, logL_history = _tram.estimate(
            self.count_matrices, self.state_counts,
            [ca(self.bias_energies_sh[:, 0:self.n_samples//2].T), ca(self.bias_energies_sh[:, self.n_samples//2:].T)],
            [self.conf_state_sequence[0:self.n_samples//2], self.conf_state_sequence[self.n_samples//2:]],
            maxiter=1000000, maxerr=1.0E-10, save_convergence_info=10, N_dtram_accelerations=N_dtram_accelerations)
        transition_matrices = _tram.estimate_transition_matrices(
            log_lagrangian_mult, biased_conf_energies, self.count_matrices, None)
        # check expectations (do a trivial test: recompute conf_energies with different functions)
        mu = np.zeros(shape=self.conf_state_sequence.shape[0], dtype=np.float64)
        _tram.get_pointwise_unbiased_free_energies(None, log_lagrangian_mult, biased_conf_energies,
            therm_energies, self.count_matrices,
            [ca(self.bias_energies_sh[:, 0:self.n_samples//2].T), ca(self.bias_energies_sh[:, self.n_samples//2:].T)],
            [self.conf_state_sequence[0:self.n_samples//2], self.conf_state_sequence[self.n_samples//2:]],
            self.state_counts, None, None,
            [mu[0:self.n_samples//2], mu[self.n_samples//2:]])
        counts,_ = np.histogram(
            self.conf_state_sequence, weights=np.exp(-mu), bins=self.n_conf_states)
        pmf = -np.log(counts)
        assert_allclose(pmf, conf_energies)
        biased_conf_energies -= np.min(biased_conf_energies)
        bias_energies =  self.bias_energies - np.min(self.bias_energies)
        nz = np.where(self.state_counts > 0)
        assert not np.any(np.isinf(log_lagrangian_mult[nz]))
        assert_allclose(biased_conf_energies, bias_energies, atol=0.1)
        assert_allclose(transition_matrices, self.T, atol=0.1)
        assert np.all(logL_history[-1] + 1.E-5 >= np.array(logL_history[0:-1]))
        # check exact identities of TRAM
        # (1) sum_j v_j T_ji + v_i = sum_j c_ij + sum_j c_ji
        for k in range(self.n_therm_states):
            lagrangian_mult = np.exp(log_lagrangian_mult[k, :])
            assert_allclose(
                lagrangian_mult.T.dot(transition_matrices[k, :, :]) + lagrangian_mult,
                self.count_matrices[k, :, :].sum(axis=0) + self.count_matrices[k, :, :].sum(axis=1))
        # (2) sum_jk v^k_j T^k_ji = sum_jk c^k_ji
        total = np.zeros(self.n_conf_states)
        for k in range(self.n_therm_states):
            lagrangian_mult = np.exp(log_lagrangian_mult[k, :])
            total += lagrangian_mult.T.dot(transition_matrices[k, :, :])
        assert_allclose(total, self.count_matrices.sum(axis=0).sum(axis=0))
