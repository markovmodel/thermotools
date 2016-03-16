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

import thermotools.util as util
import numpy as np
from nose.tools import assert_true
from numpy.testing import assert_array_equal, assert_almost_equal

####################################################################################################
#   sorting
####################################################################################################

def test_mixed_sort_reverse():
    # testing against numpy.sort()
    x = np.ascontiguousarray(np.arange(1000)[::-1].astype(np.float64))
    y = np.sort(x)
    util.mixed_sort(x)
    assert_array_equal(x, y)

def test_mixed_sort_random():
    # testing against numpy.sort()
    x = np.random.rand(1000).astype(np.float64)
    y = np.sort(x)
    util.mixed_sort(x)
    assert_array_equal(x, y)

####################################################################################################
#   direct summation schemes
####################################################################################################

def test_kahan_summation():
    # np.sum() fails for this array when unsorted
    array = np.array([1.0E-8, 1.0, 1.0E+8] * 100000, dtype=np.float64)
    result = util.kahan_summation(array, sort_array=False)
    assert_true(result == 10000000100000.001)
    result = util.kahan_summation(array, sort_array=True, inplace=False)
    assert_true(result == 10000000100000.001)
    result = util.kahan_summation(array, sort_array=True, inplace=True)
    assert_true(result == 10000000100000.001)

####################################################################################################
#   logspace summation schemes
####################################################################################################

def test_logsumexp_zeros():
    N = 10000
    data = np.zeros(shape=(N,), dtype=np.float64)
    assert_almost_equal(util.logsumexp(data, inplace=False), np.log(N), decimal=15)
    assert_almost_equal(util.logsumexp(-data, inplace=False), np.log(N), decimal=15)

def test_logsumexp_converged_geometric_series():
    data = np.ascontiguousarray(np.arange(10000)[::-1].astype(np.float64))
    assert_almost_equal(
        util.logsumexp(-data, inplace=False, sort_array=False, use_kahan=False),
        0.45867514538708193, decimal=15)
    assert_almost_equal(
        util.logsumexp(-data, inplace=False, sort_array=False, use_kahan=True),
        0.45867514538708193, decimal=15)
    assert_almost_equal(
        util.logsumexp(-data, inplace=False, sort_array=True, use_kahan=False),
        0.45867514538708193, decimal=15)
    assert_almost_equal(
        util.logsumexp(-data, inplace=False, sort_array=True, use_kahan=True),
        0.45867514538708193, decimal=15)
    assert_almost_equal(
        util.logsumexp(-data, inplace=True, sort_array=True, use_kahan=True),
        0.45867514538708193, decimal=15)
    
def test_logsumexp_truncated_diverging_geometric_series():
    data = np.ascontiguousarray(np.arange(10000)[::-1].astype(np.float64))
    assert_almost_equal(
        util.logsumexp(data, inplace=False, sort_array=False, use_kahan=False),
        9999.4586751453862, decimal=15)
    assert_almost_equal(
        util.logsumexp(data, inplace=False, sort_array=False, use_kahan=True),
        9999.4586751453862, decimal=15)
    assert_almost_equal(
        util.logsumexp(data, inplace=False, sort_array=True, use_kahan=False),
        9999.4586751453862, decimal=15)
    assert_almost_equal(
        util.logsumexp(data, inplace=False, sort_array=True, use_kahan=True),
        9999.4586751453862, decimal=15)
    assert_almost_equal(
        util.logsumexp(data, inplace=True, sort_array=True, use_kahan=True),
        9999.4586751453862, decimal=15)

def test_logsumexp_pair():
    assert_almost_equal(util.logsumexp_pair(0.0, 0.0), np.log(2.0), decimal=15)
    assert_almost_equal(util.logsumexp_pair(1.0, 1.0), 1.0 + np.log(2.0), decimal=15)
    assert_almost_equal(util.logsumexp_pair(10.0, 10.0), 10.0 + np.log(2.0), decimal=15)
    assert_almost_equal(util.logsumexp_pair(100.0, 100.0), 100.0 + np.log(2.0), decimal=15)
    assert_almost_equal(util.logsumexp_pair(1000.0, 1000.0), 1000.0 + np.log(2.0), decimal=15)
    assert_almost_equal(util.logsumexp_pair(10.0, 0.0), 10.000045398899218, decimal=15)
    assert_almost_equal(util.logsumexp_pair(0.0, 10.0), 10.000045398899218, decimal=15)
    assert_almost_equal(util.logsumexp_pair(100.0, 0.0), 100.0, decimal=15)
    assert_almost_equal(util.logsumexp_pair(0.0, 100.0), 100.0, decimal=15)
    assert_almost_equal(util.logsumexp_pair(1000.0, 0.0), 1000.0, decimal=15)
    assert_almost_equal(util.logsumexp_pair(0.0, 1000.0), 1000.0, decimal=15)

####################################################################################################
#   counting states and transitions
####################################################################################################

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
    dtrajs = [
        np.array([0, 0, 1, 1, 2, 2, 0, 2, 1, 0], dtype=np.intc),
        np.array([0, 0, 1, 1, 2, 2, 0, 2, 1, 0], dtype=np.intc)]
    ttrajs = [np.array([0] * 10, dtype=np.intc), np.array([1] * 10, dtype=np.intc)]
    # dtraj = [
    #     np.array(
    #         [[0, 0], [0, 0], [0, 1], [0, 1], [0, 2], [0, 2], [0, 0], [0, 2], [0, 1], [0, 0]],
    #         dtype=np.intc),
    #     np.array(
    #         [[1, 0], [1, 0], [1, 1], [1, 1], [1, 2], [1, 2], [1, 0], [1, 2], [1, 1], [1, 0]],
    #         dtype=np.intc)]
    C_K = util.count_matrices(ttrajs, dtrajs, 1, sparse_return=False)
    ref = np.ones(shape=(2, 3, 3), dtype=np.intc)
    assert_array_equal(C_K, ref)

def test_count_matrices_st_traj():
    ttraj = [np.array([0, 0, 1, 1, 1, 0, 0, 2, 2, 2, 0, 0], dtype=np.intc)]
    dtraj = [np.array([0, 0, 0, 1, 0, 1, 1, 1, 2, 1, 2, 2], dtype=np.intc)]
    C_K = util.count_matrices(ttraj, dtraj, 1, sliding=True, sparse_return=False, nthermo=4, nstates=4)
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
    ttrajs = [np.zeros(shape=(10,), dtype=np.intc), 2 * np.ones(shape=(20,), dtype=np.intc)]
    dtrajs = [np.zeros(shape=(10,), dtype=np.intc), 2 * np.ones(shape=(20,), dtype=np.intc)]
    ref = np.array([[10, 0, 0, 0], [0] * 4, [0, 0, 20, 0], [0] * 4, [0] * 4], dtype=np.intc)
    N = util.state_counts(ttrajs, dtrajs, nthermo=5, nstates=4)
    assert_array_equal(N, ref)

def test_restriction():
    T = 10
    M = 100
    X = 1000
    state_sequence = np.array([[0, i] for i in range(M)] * 10, dtype=np.intc)
    bias_energy_sequence = np.ascontiguousarray(
        np.array([[i] * T for i in range(X)], dtype=np.float64).transpose())
    cset = [i for i in range(M) if i % 2 == 0]
    ref_state_sequence = np.array([[0, i] for i in range(int(M / 2))] * 10, dtype=np.intc)
    ref_bias_energy_sequence = np.ascontiguousarray(
        np.array([[i] * T for i in range(X) if i % 2 == 0], dtype=np.float64).transpose())
    new_state_sequence, new_bias_energy_sequence = util.restrict_samples_to_cset(
        state_sequence, bias_energy_sequence, cset)
    assert_array_equal(new_state_sequence, ref_state_sequence)
    assert_array_equal(new_bias_energy_sequence, ref_bias_energy_sequence)

####################################################################################################
#   bias calculation tools
####################################################################################################

def test_get_umbrella_bias_binary():
    nsamples = 100
    nthermo = 2
    ndim = 3
    traj = np.linspace(0.0, 2.0, nsamples)
    for _i in range(1, ndim):
        traj = np.vstack((traj, np.linspace(0.0, 2.0, nsamples)))
    traj = np.ascontiguousarray(traj.T, dtype=np.float64)
    umbrella_centers = np.zeros(shape=(nthermo, ndim), dtype=np.float64)
    umbrella_centers[1, :] = 1.0
    force_constants = np.array([
        np.zeros(shape=(ndim, ndim), dtype=np.float64), np.eye(ndim, dtype=np.float64)])
    bias = util.get_umbrella_bias(traj, umbrella_centers, force_constants)
    ref = np.vstack((
        np.zeros(shape=(nsamples)),
        0.5 * ndim * np.linspace(-1.0, 1.0, nsamples)**2)).T.astype(np.float64)
    assert_almost_equal(bias, ref, decimal=15)

####################################################################################################
#   transition matrix renormalization
####################################################################################################
