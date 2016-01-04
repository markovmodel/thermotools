from __future__ import absolute_import
import unittest
from six.moves import range

import numpy as np
import thermotools.cset as cset

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

class TestCset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        n_therm_states = 4
        n_conf_states = 4
        n_samples = 10000
        bias_energies = np.zeros((n_therm_states,n_conf_states))
        bias_energies[0,:] = np.array([0.0, 15.0, 15.0, 0.0 ]) 
        bias_energies[1,:] = np.array([15.0, 0.0, 15.0, 20.0 ])+4  
        bias_energies[2,:] = np.array([15.0,15.0,  0.0, 20.0 ])+8
        bias_energies[3,:] = np.array([15.0, 0.0, 15.0, 20.0 ])+12
        bias_energies_sh = bias_energies - bias_energies[0,:]

        T = np.zeros(shape=(n_therm_states, n_conf_states, n_conf_states), dtype=np.float64)
        tramtrajs = [np.zeros((n_samples, 2+n_therm_states)) for k in range(n_therm_states)]
        for k in range(n_therm_states):
            T[k,:,:] = T_matrix(bias_energies[k,:])
            count_matrices, conf_state_sequence, state_counts = draw_transition_counts(T, n_samples, 0) # ??? retuned data format
            tramtrajs[k][:,0] = k
            tramtrajs[k][:,1] = conf_state_sequence
            tramtrajs[k][:,2:] = bias_energies_sh[:,conf_state_sequence]

        cls.count_matrices = count_matrices
        cls.state_counts = cls.state_counts
        cls.tramtrajs = tramtrajs
        cls.tram_traj = np.concatenate(self.tram_trajs)

    def test_cset_post_hoc_RE(self):
        csets, projected_cset = cset.compute_csets_TRAM('neighbors', self.state_counts, self.count_matrices, self.tram_traj, nn=1)
        assert np.equal(projected_cset, np.array([0,1,2]))

    def test_cset_post_hoc_RE(self):
        csets, projected_cset = cset.compute_csets_TRAM('post_hoc_RE', self.state_counts, self.count_matrices, self.tram_traj)
        assert np.equal(projected_cset, np.array([0,1,2]))

if __name__ == "__main__":
    unittest.main()
