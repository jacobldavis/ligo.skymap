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

from ligo.skymap.jaxcore.integrate import log_radial_integrator

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
    p = jnp.sqrt(p2)

    integrator = log_radial_integrator(r1, r2, k, 0, p + 0.5, 400)

    region0 = (
        integrator.region0.fx,
        integrator.region0.x0,
        integrator.region0.xlength,
        integrator.region0.a,
    )
    region1 = (
        integrator.region1.f,
        integrator.region1.t0,
        integrator.region1.length,
        integrator.region1.a,
    )
    region2 = (
        integrator.region2.f,
        integrator.region2.t0,
        integrator.region2.length,
        integrator.region2.a,
    )
    limits = (integrator.p0_limit, integrator.vmax, integrator.ymax)

    result_jax = integrator.integrator_eval(
        region0, region1, region2, limits, p, b, jnp.log(p), jnp.log(b)
    )

    assert result_jax == pytest.approx(expected, abs=tol)


def test_log_radial_integral_vectorized():
    # Test data as arrays
    expected = jnp.array(
        [
            0,
            0,
            jnp.log(63),
            -0.480238,
            0.432919,
            -2.76076,
            61.07118,
            -jnp.inf,
            -jnp.inf,
            2.94548,
            2.94545,
            2.94085,
            -2.43264,
            -2.43808,
            -0.707038,
        ]
    )
    tol = jnp.array(
        [
            0,
            0,
            0,
            1e-3,
            1e-3,
            1e-3,
            1e-3,
            5e-2,
            1e-3,
            1e-4,
            1e-4,
            1e-4,
            1e-5,
            1e-5,
            1e-5,
        ]
    )
    r1 = jnp.array([0, jnp.exp(1), 3, 1, 1, 0, 0, 0, 0, 0, 0.5, 1, 0, 0.5, 1])
    r2 = jnp.array([1, jnp.exp(2), 6, 2, 2, 1, 1e9, 0.1, 1e-3, 4, 4, 4, 1, 1, 1.5])
    p2 = jnp.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    b = jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    k = jnp.array([0, -1, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

    p = jnp.sqrt(p2)

    # Create integrators for each test case
    integrators = [
        log_radial_integrator(r1[i], r2[i], k[i], 0, p[i] + 0.5, 400)
        for i in range(len(expected))
    ]

    # Extract regions and limits
    region0_list = [
        (integ.region0.fx, integ.region0.x0, integ.region0.xlength, integ.region0.a)
        for integ in integrators
    ]
    region1_list = [
        (integ.region1.f, integ.region1.t0, integ.region1.length, integ.region1.a)
        for integ in integrators
    ]
    region2_list = [
        (integ.region2.f, integ.region2.t0, integ.region2.length, integ.region2.a)
        for integ in integrators
    ]
    limits_list = [(integ.p0_limit, integ.vmax, integ.ymax) for integ in integrators]

    result_jax = jnp.array(
        [
            integrators[i].integrator_eval(
                region0_list[i],
                region1_list[i],
                region2_list[i],
                limits_list[i],
                p[i],
                b[i],
                jnp.log(p[i]),
                jnp.log(b[i]),
            )
            for i in range(len(expected))
        ]
    )

    # Check each result against expected with its tolerance
    for i in range(len(expected)):
        assert result_jax[i] == pytest.approx(expected[i], abs=tol[i])
