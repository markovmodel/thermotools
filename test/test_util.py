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
    assert_true(np.all(bp == range(1000)))
    bp = util.get_therm_state_break_points(
        np.array([0] * 10 + [1] * 20 + [0] * 30 + [1], dtype=np.intc))
    assert_true(bp.shape[0] == 4)
    assert_true(np.all(bp == [0, 10, 30, 60]))
