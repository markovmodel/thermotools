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

from nose.tools import assert_true
from thermotools.lse import logsumexp, logsumexp_pair
import numpy as np

def test_logsumexp_zeros_short():
    N = 5
    data = np.zeros(shape=(N,), dtype=np.float64)
    assert_true(np.abs(logsumexp(data) - np.log(N)) < 1.0E-15)
    assert_true(np.abs(logsumexp(-data) - np.log(N)) < 1.0E-15)

def test_logsumexp_zeros_long():
    N = 10000
    data = np.zeros(shape=(N,), dtype=np.float64)
    assert_true(np.abs(logsumexp(data) - np.log(N)) < 1.0E-15)
    assert_true(np.abs(logsumexp(-data) - np.log(N)) < 1.0E-15)

def test_logsumexp_converged_geometric_series():
    data = np.arange(10000).astype(np.float64)
    assert_true(np.abs(logsumexp(-data) - 0.45867514538708193) < 1.0E-15)
    
def test_logsumexp_truncated_diverging_geometric_series():
    data = np.arange(10000).astype(np.float64)
    assert_true(np.abs(logsumexp(data) - 9999.4586751453862) < 1.0E-15)

def test_logsumexp_pair():
    assert_true(np.abs(logsumexp_pair(0.0, 0.0) - np.log(2.0)) < 1.0E-15)
    assert_true(np.abs(logsumexp_pair(1.0, 1.0) - (1.0 + np.log(2.0))) < 1.0E-15)
    assert_true(np.abs(logsumexp_pair(10.0, 10.0) - (10.0 + np.log(2.0))) < 1.0E-15)
    assert_true(np.abs(logsumexp_pair(100.0, 100.0) - (100.0 + np.log(2.0))) < 1.0E-15)
    assert_true(np.abs(logsumexp_pair(1000.0, 1000.0) - (1000.0 + np.log(2.0))) < 1.0E-15)
    assert_true(np.abs(logsumexp_pair(10.0, 0.0) - 10.000045398899218) < 1.0E-15)
    assert_true(np.abs(logsumexp_pair(0.0, 10.0) - 10.000045398899218) < 1.0E-15)
    assert_true(np.abs(logsumexp_pair(100.0, 0.0) - 100.0) < 1.0E-15)
    assert_true(np.abs(logsumexp_pair(0.0, 100.0) - 100.0) < 1.0E-15)
    assert_true(np.abs(logsumexp_pair(1000.0, 0.0) - 1000.0) < 1.0E-15)
    assert_true(np.abs(logsumexp_pair(0.0, 1000.0) - 1000.0) < 1.0E-15)
