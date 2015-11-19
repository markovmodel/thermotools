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
import thermotools.mbar as mbar
import thermotools.dtram as dtram
import numpy as np
from nose.tools import assert_raises

from thermotools.callback import CallbackInterrupt, generic_callback_stop

#   ************************************************************************************************
#   test generic_callback_stop
#   ************************************************************************************************

def test_dtram_stop():
    T = 5
    M = 10
    assert_raises(
        CallbackInterrupt,
        dtram.estimate,
        np.ones(shape=(T, M, M), dtype=np.intc),
        np.zeros(shape=(T, M), dtype=np.float64),
        callback=generic_callback_stop)

