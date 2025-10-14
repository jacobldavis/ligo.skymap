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

import jax.numpy as jnp
import pytest

from ligo.skymap.jaxcore.integrate import integrator_eval, integrator_init

# --- TEST SUITE ---


@pytest.mark.parametrize(
    "expected, tol, r1, r2, p2, b, k",
    [
        (0, 0, 0, 1, 0, 0, 0),
        (0, 0, jnp.exp(1), jnp.exp(2), 0, 0, -1),
        (jnp.log(63), 0, 3, 6, 0, 0, 2),
        (-0.480238, 1e-3, 1, 2, 1, 0, 0),
        (0.432919, 1e-3, 1, 2, 1, 0, 2),
        (-2.76076, 1e-3, 0, 1, 1, 0, 2),
        (61.07118, 1e-3, 0, 1e9, 1, 0, 2),
        (-jnp.inf, 5e-2, 0, 0.1, 1, 0, 2),
        (-jnp.inf, 1e-3, 0, 1e-3, 1, 0, 2),
        (2.94548, 1e-4, 0, 4, 1, 1, 2),
        (2.94545, 1e-4, 0.5, 4, 1, 1, 2),
        (2.94085, 1e-4, 1, 4, 1, 1, 2),
        (-2.43264, 1e-5, 0, 1, 1, 1, 2),
        (-2.43808, 1e-5, 0.5, 1, 1, 1, 2),
        (-0.707038, 1e-5, 1, 1.5, 1, 1, 2),
    ],
)
def test_log_radial_integral(expected, tol, r1, r2, p2, b, k):
    """Test scalar integrator evaluation against known reference values."""
    p = jnp.sqrt(p2)

    regions, limits = integrator_init(
        r1, r2, k, cosmology=False, pmax=p + 0.5, size=400
    )
    # regions = ((r0_fx, r0_x0, r0_xlen, r0_a), (r1_f, r1_t0, r1_len, r1_a), (r2_f, r2_t0, r2_len, r2_a))
    # limits = (p0_limit, vmax, ymax)

    result_jax = integrator_eval(
        regions[0],  # region0
        regions[1],  # region1
        regions[2],  # region2
        limits,
        p,
        b,
        jnp.log(p),
        jnp.log(b),
    )

    assert result_jax == pytest.approx(expected, abs=tol)
