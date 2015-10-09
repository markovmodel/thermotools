# This file is part of PyEMMA.
#
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
#
# PyEMMA is free software: you can redistribute it and/or modify
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

from __future__ import absolute_import
import unittest
from six.moves import range
import numpy as np
import thermotools.tram

class TestTRAM(unittest.TestCase):
    def test_log_likelihood(self):
        nu = np.random.rand(2,2)*100.0
        CKij = np.random.randint(1,10,size=(2,2,2)).astype(np.intc)
        Z = np.random.rand(2,2)
        state_counts = np.maximum(CKij.sum(axis=0), CKij.sum(axis=1)).astype(np.intc)
        bias_energy_sequence = np.random.rand(2,10)
        Markov_state_sequence = np.random.randint(0,2,size=10).astype(np.intc)

        scratch_M = np.zeros(2)
        scratch_T = np.zeros(2)

        # regular case
        T = np.zeros((2,2,2))
        for k in range(2):
            for i in range(2):
                for j in range(2):
                    T[k,i,j] = (CKij[k,i,j]+CKij[k,j,i])*Z[k,j] / (Z[k,i]*nu[k,j]+Z[k,j]*nu[k,i])
        for k in range(2):
            for i in range(2):
                T[k,i,i] = 0
        for k in range(2):
            for i in range(2):
                T[k,i,i] = 1.0-T[k,i,:].sum()

        R = np.zeros((2,2))
        for k in range(2):
            for i in range(2):
                for j in range(2):
                    R[k,i] += (CKij[k,i,j]+CKij[k,j,i])*nu[k,j] / (Z[k,i]*nu[k,j]+Z[k,j]*nu[k,i])
                R[k,i] += (state_counts[k,i]-CKij[k,:,i].sum()) / Z[k,i]

        fKi = -np.log(Z)

        a = (CKij*np.log(T)).sum()
        b = (state_counts*fKi).sum()
        tmp = (R[:,Markov_state_sequence]*np.exp(-bias_energy_sequence)).sum(axis=0)
        c = -np.log(tmp).sum()

        assert np.all(Markov_state_sequence<2)
        assert np.all(Markov_state_sequence>=0)

        reference = a+b+c

        log_lagrangian_mult = np.log(nu)
        biased_conf_energies = fKi
        count_matrices = CKij
        state_sequence = Markov_state_sequence
        log_R_K_i = np.log(R)

        log_R_K_i_compare = np.zeros_like(log_R_K_i)
        new_biased_conf_energies = np.zeros_like(biased_conf_energies)
        thermotools.tram.update_biased_conf_energies(log_lagrangian_mult, biased_conf_energies, count_matrices, bias_energy_sequence, state_sequence,
            state_counts, log_R_K_i_compare, scratch_M, scratch_T, new_biased_conf_energies)

        assert np.allclose(log_R_K_i, log_R_K_i_compare)

        compare = thermotools.tram.log_likelihood(log_lagrangian_mult, biased_conf_energies, count_matrices, bias_energy_sequence, state_sequence,
                        state_counts, log_R_K_i_compare, scratch_M, scratch_T)

        assert np.allclose(reference, compare)

if __name__ == "__main__":
    unittest.main()
