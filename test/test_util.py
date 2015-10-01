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

import thermotools.util as util
import numpy as np
from nose.tools import assert_true
from numpy.testing import assert_array_equal

def test_break_points_us_like_trajs():
    X = 2000
    T = 100
    for K in range(T):
        bp = util.get_therm_state_break_points(np.ones(shape=(X,), dtype=np.intc) * K)
        assert_true(bp.shape[0] == 1)
        assert_true(bp[0] == 0)

def test_break_points_st_like_trajs():
    bp = util.get_therm_state_break_points(np.arange(1000).astype(np.intc))
    assert_true(bp.shape[0] == 1000)
    assert_array_equal(bp, np.array(range(1000), dtype=np.intc))
    bp = util.get_therm_state_break_points(
        np.array([0] * 10 + [1] * 20 + [0] * 30 + [1], dtype=np.intc))
    assert_true(bp.shape[0] == 4)
    assert_array_equal(bp, np.array([0, 10, 30, 60], dtype=np.intc))

def test_count_matrices_single_counts():
    dtraj = [
        np.array(
            [[0, 0], [0, 0], [0, 1], [0, 1], [0, 2], [0, 2], [0, 0], [0, 2], [0, 1], [0, 0]],
            dtype=np.intc),
        np.array(
            [[1, 0], [1, 0], [1, 1], [1, 1], [1, 2], [1, 2], [1, 0], [1, 2], [1, 1], [1, 0]],
            dtype=np.intc)]
    C_K = util.count_matrices(dtraj, 1, sparse_return=False)
    ref = np.ones(shape=(2, 3, 3), dtype=np.intc)
    assert_array_equal(C_K, ref)

def test_count_matrices_st_traj():
    dtraj = [np.array([
        [0, 0], [0, 0],
        [1, 0], [1, 1], [1, 0],
        [0, 1], [0, 1],
        [2, 1], [2, 2], [2, 1],
        [0, 2], [0, 2]], dtype=np.intc)]
    C_K = util.count_matrices(dtraj, 1, sliding=True, sparse_return=False, nthermo=4, nstates=4)
    ref = np.zeros(shape=(4, 4, 4), dtype=np.intc)
    ref[0, 0, 0] = 1
    ref[0, 1, 1] = 1
    ref[0, 2, 2] = 1
    ref[1, 0, 1] = 1
    ref[1, 1, 0] = 1
    ref[2, 1, 2] = 1
    ref[2, 2, 1] = 1
    assert_array_equal(C_K, ref)

def test_state_counts():
    dtraj = [np.zeros(shape=(10, 2), dtype=np.intc), 2 * np.ones(shape=(20, 2), dtype=np.intc)]
    ref = np.array([[10, 0, 0, 0], [0] * 4, [0, 0, 20, 0], [0] * 4, [0] * 4], dtype=np.intc)
    N = util.state_counts(dtraj, nthermo=5, nstates=4)
    assert_array_equal(N, ref)
