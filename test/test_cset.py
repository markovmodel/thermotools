# This file is part of thermotools.
#
# Copyright 2016 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
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
import thermotools.cset as cset
import thermotools.util as util

class TestCset(object):
    @classmethod
    def setup_class(cls):
        m1 = 10
        s1 = 1
        m2 = 20
        s2 = 1
        n_samples = 1000
        def u(x):
            return np.array([0.5 * ((x - m1) / s1)**2, 0.5 * ((x - m2) / s2)**2]).T
        x1 = np.random.randn(n_samples) * s1 + m1
        ttraj1 = np.zeros(n_samples, dtype=np.intc)
        dtraj1 = (x1 < 20).astype(np.intc)
        bias_traj1 = u(x1)
        x2 = np.random.randn(n_samples) * s2 + m2
        ttraj2 = np.ones(n_samples, dtype=np.intc)
        dtraj2 = (x2 < 20).astype(np.intc)
        bias_traj2 = u(x2)
        ttrajs = [ttraj1, ttraj2]
        dtrajs = [dtraj1, dtraj2]
        bias_trajs = [bias_traj1, bias_traj2]
        state_counts = util.state_counts(ttrajs, dtrajs)
        count_matrices = util.count_matrices(
            ttrajs, dtrajs, 1, sliding='sliding', sparse_return=False, nstates=2)
        cls.ttrajs = ttrajs
        cls.dtrajs = dtrajs
        cls.bias_trajs = bias_trajs
        cls.state_counts = state_counts
        cls.count_matrices = count_matrices
    @classmethod
    def teardown_class(cls):
        pass
    def setup(self):
        pass
    def teardown(self):
        pass
    def test_summed_count_matrix(self):
        csets, projected_cset = cset.compute_csets_TRAM(
            'summed_count_matrix', self.state_counts, self.count_matrices,
            ttrajs=self.ttrajs, dtrajs=self.dtrajs, bias_trajs=self.bias_trajs)
        np.testing.assert_allclose(csets[0], np.array([1]))
        np.testing.assert_allclose(csets[1], np.array([0, 1]))
        np.testing.assert_allclose(projected_cset, np.array([0,1]))
    # def test_strong_in_every_ensemble(self):
    #     csets, projected_cset = cset.compute_csets_TRAM(
    #         'strong_in_every_ensemble', self.state_counts, self.count_matrices, self.tram_sequence)
    #     np.testing.assert_allclose(csets[0], np.array([1]))
    #     np.testing.assert_allclose(csets[1], np.array([0, 1]))
    #     np.testing.assert_allclose(projected_cset, np.array([0,1]))
    def test_cset_neighbors(self):
        csets, projected_cset = cset.compute_csets_TRAM(
            'neighbors', self.state_counts, self.count_matrices, ttrajs=self.ttrajs,
            dtrajs=self.dtrajs, bias_trajs=self.bias_trajs, nn=1)
        np.testing.assert_allclose(csets[0], np.array([1]))
        np.testing.assert_allclose(csets[1], np.array([0, 1]))
        np.testing.assert_allclose(projected_cset, np.array([0,1]))
    def test_cset_post_hoc_RE(self):
        csets, projected_cset = cset.compute_csets_TRAM(
            'post_hoc_RE', self.state_counts, self.count_matrices, ttrajs=self.ttrajs,
            dtrajs=self.dtrajs, bias_trajs=self.bias_trajs)
        np.testing.assert_allclose(csets[0], np.array([]))
        np.testing.assert_allclose(csets[1], np.array([0, 1]))
        np.testing.assert_allclose(projected_cset, np.array([0, 1]))
    def test_cset_BAR_variance(self):
        csets, projected_cset = cset.compute_csets_TRAM(
            'BAR_variance', self.state_counts, self.count_matrices, ttrajs=self.ttrajs,
            dtrajs=self.dtrajs, bias_trajs=self.bias_trajs)
        np.testing.assert_allclose(csets[0], np.array([]))
        np.testing.assert_allclose(csets[1], np.array([0, 1]))
        np.testing.assert_allclose(projected_cset, np.array([0, 1]))
    def test_restrict(self):
        csets, projected_cset = cset.compute_csets_TRAM(
            'summed_count_matrix', self.state_counts, self.count_matrices,
            ttrajs=self.ttrajs, dtrajs=self.dtrajs, bias_trajs=self.bias_trajs)
        new_state_counts, new_count_matrices, new_dtrajs, new_bias_trajs = cset.restrict_to_csets(
            csets, self.state_counts, self.count_matrices, self.ttrajs,
            self.dtrajs, self.bias_trajs)
        np.testing.assert_allclose(new_count_matrices, self.count_matrices)
        np.testing.assert_allclose(new_state_counts, self.state_counts)
        for x,y in zip(self.bias_trajs, new_bias_trajs):
            np.testing.assert_allclose(x, y)
        for x,y in zip(self.dtrajs, new_dtrajs):
            np.testing.assert_allclose(x, y)
        csets, projected_cset = cset.compute_csets_TRAM(
            'post_hoc_RE', self.state_counts, self.count_matrices, ttrajs=self.ttrajs,
            dtrajs=self.dtrajs, bias_trajs=self.bias_trajs)
        new_state_counts, new_count_matrices, new_dtrajs, new_bias_trajs = cset.restrict_to_csets(
            csets, self.state_counts, self.count_matrices, self.ttrajs,
            self.dtrajs, self.bias_trajs)
        np.testing.assert_allclose(new_state_counts[0,:], 0)
        np.testing.assert_allclose(new_state_counts[1,:], self.state_counts[1,:])
        np.testing.assert_allclose(new_count_matrices[0,:,:], 0)
        np.testing.assert_allclose(new_count_matrices[1,:,:], self.count_matrices[1,:,:])
        assert len(new_bias_trajs[0])==0
        np.testing.assert_allclose(new_bias_trajs[1], self.bias_trajs[1])
