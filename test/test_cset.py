from __future__ import absolute_import
import unittest
from six.moves import range

import numpy as np
import thermotools.cset as cset
import thermotools.util as util

class TestCset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        m1 = 10
        s1 = 1
        m2 = 20
        s2 = 1
        n_samples = 1000

        def u(x):
            return np.array([0.5*((x-m1)/s1)**2, 0.5*((x-m2)/s2)**2]).T

        x1 = np.random.randn(n_samples)*s1 + m1
        tramtraj1 = np.zeros((n_samples,4))
        tramtraj1[:,0]  = 0
        tramtraj1[:,1]  = x1<20
        tramtraj1[:,2:] = u(x1)

        tramtraj2 = np.zeros((n_samples,4))
        x2 = np.random.randn(n_samples)*s2 + m2
        tramtraj2[:,0]  = 1
        tramtraj2[:,1]  = x2<20
        tramtraj2[:,2:] = u(x2)
        
        tramtrajs = [tramtraj1, tramtraj2]
        tram_sequence = np.concatenate(tramtrajs)
        state_counts = util.state_counts(tramtrajs)
        count_matrices = util.count_matrices(
            [np.ascontiguousarray(t[:, :2]).astype(np.intc) for t in tramtrajs], 1,
            sliding='sliding', sparse_return=False, nstates=2)
        cls.tram_sequence = tram_sequence
        cls.state_counts = state_counts
        cls.count_matrices = count_matrices

    def test_summed_count_matrix(self):
        csets, projected_cset = cset.compute_csets_TRAM('summed_count_matrix', self.state_counts, self.count_matrices, self.tram_sequence)
        np.testing.assert_allclose(csets[0], np.array([1]))
        np.testing.assert_allclose(csets[1], np.array([0, 1]))
        np.testing.assert_allclose(projected_cset, np.array([0,1]))

    #def test_strong_in_every_ensemble(self):
    #    csets, projected_cset = cset.compute_csets_TRAM('strong_in_every_ensemble', self.state_counts, self.count_matrices, self.tram_sequence)
    #    np.testing.assert_allclose(csets[0], np.array([1]))
    #    np.testing.assert_allclose(csets[1], np.array([0, 1]))
    #    np.testing.assert_allclose(projected_cset, np.array([0,1]))

    def test_cset_neighbors(self):
        csets, projected_cset = cset.compute_csets_TRAM('neighbors', self.state_counts, self.count_matrices, self.tram_sequence, nn=1)
        np.testing.assert_allclose(csets[0], np.array([1]))
        np.testing.assert_allclose(csets[1], np.array([0, 1]))
        np.testing.assert_allclose(projected_cset, np.array([0,1]))

    def test_cset_post_hoc_RE(self):
        csets, projected_cset = cset.compute_csets_TRAM('post_hoc_RE', self.state_counts, self.count_matrices, self.tram_sequence)
        np.testing.assert_allclose(csets[0], np.array([]))
        np.testing.assert_allclose(csets[1], np.array([0, 1]))
        np.testing.assert_allclose(projected_cset, np.array([0, 1]))

    def test_cset_BAR_variance(self):
        csets, projected_cset = cset.compute_csets_TRAM('BAR_variance', self.state_counts, self.count_matrices, self.tram_sequence)
        np.testing.assert_allclose(csets[0], np.array([]))
        np.testing.assert_allclose(csets[1], np.array([0, 1]))
        np.testing.assert_allclose(projected_cset, np.array([0, 1]))


if __name__ == "__main__":
    unittest.main()
