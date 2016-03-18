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

import thermotools.wham as wham
import thermotools.tram as tram
import thermotools.tram_direct as tram_direct
import thermotools.mbar as mbar
import thermotools.mbar_direct as mbar_direct
import thermotools.dtram as dtram
import thermotools.tram as tram
import numpy as np
from numpy.testing import assert_allclose

#   ************************************************************************************************
#   data generation functions
#   ************************************************************************************************

def tower_sample(distribution):
    cdf = np.cumsum(distribution)
    rnd = np.random.rand() * cdf[-1]
    return np.searchsorted(cdf, rnd)

def draw_independent_samples(biased_stationary_distribution, n_samples):
    state_counts = np.zeros(shape=biased_stationary_distribution.shape, dtype=np.intc)
    conf_state_sequence = np.zeros(
        biased_stationary_distribution.shape[0]*n_samples, dtype=np.intc)
    for K in range(biased_stationary_distribution.shape[0]):
        for s in range(n_samples):
            i = tower_sample(biased_stationary_distribution[K, :])
            state_counts[K, i] += 1
            conf_state_sequence[K * n_samples + s] = i
    return conf_state_sequence, state_counts

def draw_transition_counts(transition_matrices, n_samples, x0):
    """generates a discrete Markov chain"""
    count_matrices = np.zeros(shape=transition_matrices.shape, dtype=np.intc)
    conf_state_sequence = np.zeros(
        shape=(transition_matrices.shape[0]*(n_samples+1),), dtype=np.intc)
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

#   ************************************************************************************************
#   test class
#   ************************************************************************************************

class TestThreeTwoModel(object):
    @classmethod
    def setup_class(cls):
        cls.energy = np.array([1.0, 2.0, 0.0], dtype=np.float64)
        cls.bias_energies = np.array([[0.0, 0.0, 0.0], 2.0 - cls.energy], dtype=np.float64)
        cls.stationary_distribution = np.exp(-cls.energy) / np.exp(-cls.energy).sum()
        cls.conf_energies = -np.log(cls.stationary_distribution)
        cls.biased_conf_energies = cls.conf_energies[np.newaxis, :] + cls.bias_energies
        cls.conf_partition_function = np.exp(-cls.biased_conf_energies)
        cls.partition_function = cls.conf_partition_function.sum(axis=1)
        cls.therm_energies = -np.log(cls.partition_function)
        cls.biased_stationary_distribution = np.exp(-cls.bias_energies) * \
            cls.stationary_distribution[np.newaxis, :] / cls.partition_function[:, np.newaxis]
        metropolis = cls.energy[np.newaxis, :] - cls.energy[:, np.newaxis]
        metropolis[(metropolis < 0.0)] = 0.0
        selection = np.array([[0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]], dtype=np.float64)
        metr_hast = selection * np.exp(-metropolis)
        for i in range(metr_hast.shape[0]):
            metr_hast[i, i] = 0.0
            metr_hast[i, i] = 1.0 - metr_hast[i, :].sum()
        cls.transition_matrices = np.array([metr_hast, selection])
        cls.n_samples = 10000
        cls.conf_state_sequence_ind, cls.state_counts_ind = draw_independent_samples(
            cls.biased_stationary_distribution, cls.n_samples)
        cls.count_matrices, cls.conf_state_sequence, cls.state_counts = draw_transition_counts(
            cls.transition_matrices, cls.n_samples, 0)
    @classmethod
    def teardown_class(cls):
        pass
    def setup(self):
        pass
    def teardown(self):
        pass
    def test_wham(self):
        therm_energies, conf_energies, increments, loglikelihoods = \
            wham.estimate(
                self.state_counts_ind, self.bias_energies, maxiter=50000, maxerr=1.0E-15)
        atol = 1.0E-1
        assert_allclose(therm_energies, self.therm_energies, atol=atol)
        assert_allclose(conf_energies, self.conf_energies, atol=atol)
    def test_mbar(self):
        bias_energy_sequence = np.ascontiguousarray(
            self.bias_energies[:, self.conf_state_sequence_ind].T)
        therm_energies, conf_energies, biased_conf_energies, err_history = mbar.estimate(
            self.state_counts_ind.sum(axis=1),
            [bias_energy_sequence],
            [self.conf_state_sequence_ind],
            maxiter=10000, maxerr=1.0E-15)
        maxerr = 1.0E-1
        assert_allclose(biased_conf_energies, self.biased_conf_energies, atol=maxerr)
        assert_allclose(conf_energies, self.conf_energies, atol=maxerr)
        assert_allclose(therm_energies, self.therm_energies, atol=maxerr)
    def test_mbar_direct(self):
        bias_energy_sequence = np.ascontiguousarray(
            self.bias_energies[:, self.conf_state_sequence_ind].T)
        therm_energies, conf_energies, biased_conf_energies, err_history = mbar_direct.estimate(
            self.state_counts_ind.sum(axis=1),
            [bias_energy_sequence],
            [self.conf_state_sequence_ind],
            maxiter=10000, maxerr=1.0E-15)
        maxerr = 1.0E-1
        assert_allclose(biased_conf_energies, self.biased_conf_energies, atol=maxerr)
        assert_allclose(conf_energies, self.conf_energies, atol=maxerr)
        assert_allclose(therm_energies, self.therm_energies, atol=maxerr)
    def test_dtram(self):
        therm_energies, conf_energies, log_lagrangian_mult, increments, loglikelihoods = \
            dtram.estimate(
                self.count_matrices, self.bias_energies, maxiter=10000, maxerr=1.0E-15)
        transition_matrices = dtram.estimate_transition_matrices(
            log_lagrangian_mult, self.bias_energies, conf_energies, self.count_matrices,
            np.zeros(shape=conf_energies.shape, dtype=np.float64))
        maxerr = 1.0E-1
        assert_allclose(therm_energies, self.therm_energies, atol=maxerr)
        assert_allclose(conf_energies, self.conf_energies, atol=maxerr)
        assert_allclose(transition_matrices, self.transition_matrices, atol=maxerr)
    def test_tram(self):
        bias_energies = np.ascontiguousarray(self.bias_energies[:,self.conf_state_sequence].T)
        biased_conf_energies, conf_energies, therm_energies, log_lagrangian_mult, error_history, logL_history = tram.estimate(
            self.count_matrices, self.state_counts, [bias_energies], [self.conf_state_sequence],
            maxiter=10000, maxerr=1.0E-12, save_convergence_info=10)
        transition_matrices = tram.estimate_transition_matrices(
            log_lagrangian_mult, biased_conf_energies, self.count_matrices, None)
        maxerr = 1.0E-1
        assert_allclose(biased_conf_energies, self.biased_conf_energies, atol=maxerr)
        assert_allclose(conf_energies, self.conf_energies, atol=maxerr)
        assert_allclose(therm_energies, self.therm_energies, atol=maxerr)
        assert_allclose(transition_matrices, self.transition_matrices, atol=maxerr)
        # lower bound on the log-likelihood must be maximal at convergence
        assert np.all(logL_history[-1]+1.E-5>=logL_history[0:-1])
    def test_tram_direct(self):
        bias_energies = np.ascontiguousarray(self.bias_energies[:,self.conf_state_sequence].T)
        biased_conf_energies, conf_energies, therm_energies, log_lagrangian_mult, error_history, logL_history = tram_direct.estimate(
            self.count_matrices, self.state_counts, [bias_energies], [self.conf_state_sequence],
            maxiter=10000, maxerr=1.0E-12, save_convergence_info=10)
        transition_matrices = tram.estimate_transition_matrices(
            log_lagrangian_mult, biased_conf_energies, self.count_matrices, None)
        maxerr = 1.0E-1
        assert_allclose(biased_conf_energies, self.biased_conf_energies, atol=maxerr)
        assert_allclose(conf_energies, self.conf_energies, atol=maxerr)
        assert_allclose(therm_energies, self.therm_energies, atol=maxerr)
        assert_allclose(transition_matrices, self.transition_matrices, atol=maxerr)
        # lower bound on the log-likelihood must be maximal at convergence
        assert np.all(logL_history[-1]+1.E-5>=logL_history[0:-1])        
