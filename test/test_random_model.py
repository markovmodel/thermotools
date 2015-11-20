from __future__ import absolute_import
import unittest
from six.moves import range

import numpy as np
import msmtools
import thermotools.tram as tram
import thermotools.tram_direct as tram_direct

def tower_sample(distribution):
    cdf = np.cumsum(distribution)
    rnd = np.random.rand() * cdf[-1]
    return np.searchsorted(cdf, rnd)

def T_matrix(energy):
    n = energy.shape[0]
    metropolis = energy[np.newaxis, :] - energy[:, np.newaxis]
    metropolis[(metropolis < 0.0)] = 0.0
    selection = np.zeros((n,n))
    selection += np.diag(np.ones(n-1)*0.5,k=1)
    selection += np.diag(np.ones(n-1)*0.5,k=-1)
    selection[0,0] = 0.5
    selection[-1,-1] = 0.5
    metr_hast = selection * np.exp(-metropolis)
    for i in range(metr_hast.shape[0]):
        metr_hast[i, i] = 0.0
        metr_hast[i, i] = 1.0 - metr_hast[i, :].sum()
    return metr_hast

def draw_transition_counts(transition_matrices, n_samples, x0):
    """generates a discrete Markov chain"""
    count_matrices = np.zeros(shape=transition_matrices.shape, dtype=np.intc)
    conf_state_sequence = np.zeros(shape=(transition_matrices.shape[0]*(n_samples+1),), dtype=np.intc)
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

class TestRandom(unittest.TestCase):
    def test_tram(self):
        
        n_therm_states = 4
        n_conf_states = 4
        n_samples = 10000

        bias_energies = np.zeros(shape=(n_therm_states, n_conf_states), dtype=np.float64)
        T = np.zeros(shape=(n_therm_states, n_conf_states, n_conf_states), dtype=np.float64)
        while True:
            # generate two random stionary distributions
            for k in range(n_therm_states):
                bias_energies[k,:] = -np.log(np.random.rand(n_conf_states))
                if k>0:
                    bias_energies[k,:] += np.random.rand()

            # generate transition matrices
            for k in range(n_therm_states):
                T[k,:,:] = T_matrix(bias_energies[k,:])

            count_matrices, conf_state_sequence, state_counts = draw_transition_counts(T, n_samples, 0)

            if msmtools.analysis.is_connected(count_matrices.sum(axis=0), directed=True):
                break
                
        bias_energies_sh = bias_energies - bias_energies[0,:]
        bias_energies_sh = np.ascontiguousarray(bias_energies_sh[:,conf_state_sequence])

        biased_conf_energies, conf_energies, therm_energies, log_lagrangian_mult, error_history, logL_history = tram.estimate(
            count_matrices, state_counts, bias_energies_sh, conf_state_sequence,
            maxiter=1000000, maxerr=1.0E-10, lll_out=10)
        transition_matrices = tram.estimate_transition_matrices(
        log_lagrangian_mult, biased_conf_energies, count_matrices, None)
        
        biased_conf_energies -= np.min(biased_conf_energies)
        bias_energies -=  np.min(bias_energies)

        nz = np.where(state_counts>0)
        assert not np.any(np.isinf(log_lagrangian_mult[nz]))
        assert np.allclose(biased_conf_energies, bias_energies, atol=0.1)
        assert np.allclose(transition_matrices, T, atol=0.1)
        assert np.all(logL_history[-1]+1.E-5>=np.array(logL_history[0:-1]))

if __name__ == "__main__":
    unittest.main()
