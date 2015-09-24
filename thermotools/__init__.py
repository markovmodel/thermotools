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

r"""
thermotools is a lowlevel implementation toolbox for the analyis of free energy calculations
"""

from . import lse
from . import bar
from . import wham
from . import mbar
#from . import tram
from . import dtram
#from . import xtram
from ._version import get_versions

__author__ = "Christoph Wehmeyer, Antonia Mey"
__copyright__ = "Copyright 2015 Computational Molecular Biology Group, FU-Berlin"
__credits__ = [
    "Christoph Wehmeyer",
    "Antonia Mey",
    "Fabian Paul",
    "Benjamin Trendelkamp-Schroer",
    "Martin Scherer",
    "Hao Wu",
    "John D. Chodera",
    "Frank Noe"],
__license__ = "LGPLv3+"
__version__ = get_versions()['version']
__email__ = "christoph.wehmeyer@fu-berlin.de"

del get_versions
