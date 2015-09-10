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
from thermotools.bar import df
import scipy.integrate
import numpy as np

def delta_f_gaussian():
    f1 = scipy.integrate.quad(make_gauss(N=100000, sigma=1, mu=0), -np.inf, np.inf)
    f2 = scipy.integrate.quad(make_gauss(N=100000, sigma=5, mu=0), -np.inf, np.inf)
    return np.log(f2[0]) - np.log(f1[0])

def make_gauss(N, sigma, mu):
    k = 1
    s = -1.0 / (2 * sigma**2)
    def f(x):
        return k * np.exp(s * (x - mu)**2)
    return f

def test_bar():
    x1 = np.random.normal(loc=0, scale=1.0, size=10000)
    x2 = np.random.normal(loc=0, scale=12.5, size=10050)
    u_x1_x1 = 0.5 * x1**2
    u_x2_x2 = 12.5 * x2**2
    u_x1_x2 = 0.5 * x2**2
    u_x2_x1 = 12.5 * x1**2
    dbIJ = u_x1_x1 - u_x2_x1
    dbJI = u_x2_x2 - u_x1_x2
    bar = df(dbIJ, dbJI, np.zeros(dbJI.shape[0]))
    assert_true(np.fabs(delta_f_gaussian() - bar) < 0.05)

