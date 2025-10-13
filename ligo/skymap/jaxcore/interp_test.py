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

from ligo.skymap.jaxcore.interp import cubic_interp_init, cubic_interp_eval, bicubic_interp_init, bicubic_interp_eval

# --- TEST SUITE ---


def test_cubic_interp_zero_output():
    t = jnp.arange(-10.0, 10.0 + 0.01, 0.01)
    f, t0, length, a = cubic_interp_init(jnp.array([0, 0, 0, 0]), 4, -1, 1)
    result = cubic_interp_eval(t, f, t0, length, a)
    assert jnp.allclose(result, 0)


def test_cubic_interp_quadratic():
    t = jnp.arange(0, 2 + 0.01, 0.01)
    f, t0, length, a = cubic_interp_init(jnp.array([1, 0, 1, 4]), 4, -1, 1)
    result = cubic_interp_eval(t, f, t0, length, a)

    assert jnp.isfinite(result).all()
    assert result[0] == pytest.approx(0, abs=1e-2)
    assert result[-1] == pytest.approx(4, abs=1e-2)


def test_bicubic_interp_flat_plane():
    s = jnp.arange(0, 2 + 0.01, 0.01)
    t = jnp.arange(0, 2 + 0.01, 0.01)
    fx, x0, xlength, a = bicubic_interp_init(
        jnp.array([-1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]),
        4,
        4,
        -1,
        -1,
        1,
        1,
    )

    result = bicubic_interp_eval(s, t, fx, x0, xlength, a)
    assert jnp.isfinite(result).all()
    assert result[0] == pytest.approx(0, abs=1e-2)
    assert result[-1] == pytest.approx(2, abs=1e-2)


def test_bicubic_interp_ramp():
    s = jnp.arange(0, 2 + 0.01, 0.01)
    t = jnp.arange(0, 2 + 0.01, 0.01)
    fx, x0, xlength, a = bicubic_interp_init(
        jnp.array([-1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1, 8, 8, 8, 8]),
        4,
        4,
        -1,
        -1,
        1,
        1,
    )

    result = bicubic_interp_eval(s, t, fx, x0, xlength, a)
    assert jnp.isfinite(result).all()
    assert result[0] == pytest.approx(0, abs=1e-2)
    assert result[-1] == pytest.approx(8, abs=1e-1)
