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

import numpy as np
from jax import jit, vmap
import jax
import jax.numpy as jnp
import time
from functools import partial
from interp import *
SQRT_2 = jnp.sqrt(2)

class log_radial_integrator:
    def __init__(self, r1, r2, k, cosmology, pmax, size):
        alpha = 4
        p0 = 0.5 * r2 if k >= 0 else 0.5 * r1
        xmax = jnp.log(pmax)
        x0 = jnp.min(jnp.log(p0), xmax)
        xmin = x0 - (1 + SQRT_2) * alpha
        ymax = x0 + alpha
        ymin = 2 * x0 - SQRT_2 * alpha - xmax
        d = (xmax - xmin) / (size - 1)
        umin = - (1 + 1/SQRT_2) * alpha
        vmax = x0 - (1/SQRT_2) * alpha
        k1 = k + 1
        p0_limit = jnp.log(jnp.log(r2/r1)) if k == -1 else jnp.log((jnp.power(r2,k1)-jnp.power(r1,k1))/(k1))


# typedef struct {
#     bicubic_interp *region0;
#     cubic_interp *region1;
#     cubic_interp *region2;
#     double ymax, vmax, p0_limit;
# } log_radial_integrator;