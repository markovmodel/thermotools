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
import thermotools.dtram as dtram
import numpy as np
from numpy.testing import assert_allclose

#   ************************************************************************************************
#   fixed-point iterations
#   ************************************************************************************************

#   ************************************************************************************************
#   data generation functions
#   ************************************************************************************************

def tower_sample(distribution):
    cdf = np.cumsum(distribution)
    rnd = np.random.rand() * cdf[-1]
    ind = (cdf > rnd)
    idx = np.where(ind == True)
    return np.min(idx)

def draw_independent_samples(biased_stationary_distribution, n_samples):
    state_counts = np.zeros(shape=biased_stationary_distribution.shape, dtype=np.intc)
    for K in range(biased_stationary_distribution.shape[0]):
        for s in range(n_samples):
            state_counts[K, tower_sample(biased_stationary_distribution[K, :])] += 1
    return state_counts

def draw_transition_counts(transition_matrices, n_samples, x0):
    """generates a discrete Markov chain"""
    count_matrices = np.zeros(transition_matrices.shape, dtype=np.intc)
    for K in range(transition_matrices.shape[0]):
        x = x0
        for s in range(n_samples):
            x_new = tower_sample(transition_matrices[K, x, :])
            count_matrices[K, x, x_new] += 1
            x = x_new
    return count_matrices

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
        cls.biased_stationary_distribution = np.exp(-cls.bias_energies) * cls.stationary_distribution[np.newaxis, :] / cls.partition_function[:, np.newaxis]
        metropolis = cls.energy[np.newaxis, :] - cls.energy[:, np.newaxis]
        metropolis[(metropolis < 0.0)] = 0.0
        selection = np.array([[0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]], dtype=np.float64)
        metr_hast = selection * np.exp(-metropolis)
        for i in range(metr_hast.shape[0]):
            metr_hast[i, i] = 0.0
            metr_hast[i, i] = 1.0 - metr_hast[i, :].sum()
        cls.transition_matrices = np.array([metr_hast, selection])
        cls.n_samples = 10000
        cls.state_counts = draw_independent_samples(cls.biased_stationary_distribution, cls.n_samples)
        cls.count_matrices = draw_transition_counts(cls.transition_matrices, cls.n_samples, 0)
    @classmethod
    def teardown_class(cls):
        pass
    def setup(self):
        pass
    def teardown(self):
        pass
    def test_wham(self):
        therm_energies, conf_energies = wham.estimate(self.state_counts, self.bias_energies, maxiter=50000, maxerr=1.0E-15)
        atol = 1.0E-1
        assert_allclose(therm_energies, self.therm_energies, atol=atol)
        assert_allclose(conf_energies, self.conf_energies, atol=atol)
    def test_dtram(self):
        therm_energies, conf_energies, log_lagrangian_mult = dtram.estimate(self.count_matrices, self.bias_energies, 10000, 1.0E-15)
        transition_matrices = dtram.estimate_transition_matrices(
            log_lagrangian_mult, self.bias_energies, conf_energies, self.count_matrices, np.zeros(shape=conf_energies.shape, dtype=np.float64))
        maxerr = 1.0E-1
        assert_allclose(therm_energies, self.therm_energies, atol=maxerr)
        assert_allclose(conf_energies, self.conf_energies, atol=maxerr)
        assert_allclose(transition_matrices, self.transition_matrices, atol=maxerr)
