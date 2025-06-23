#
# Copyright (C) 2025 Jacob Davis
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
import jax
import jax.numpy as jnp
import interp

def bsm_jax(min_distance, max_distance, prior_distance_power, 
            cosmology, gmst, sample_rate, toas, snrs, responses, 
            locations, horizons, rescale_loglikelihood):
    return [jnp.zeros(12*16*16), 0, 0]