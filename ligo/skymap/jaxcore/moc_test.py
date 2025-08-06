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
import pytest  # type: ignore

from ligo.skymap.jaxcore.moc import nest2uniq64, uniq2ang64, uniq2nest64

# --- TEST SUITE ---


@pytest.mark.parametrize(
    "order, nest, uniq",
    [
        (0, 0, 4),
        (0, 1, 5),
        (0, 2, 6),
        (0, 3, 7),
        (0, 4, 8),
        (0, 5, 9),
        (0, 6, 10),
        (0, 7, 11),
        (0, 8, 12),
        (0, 9, 13),
        (0, 10, 14),
        (0, 11, 15),
        (1, 0, 16),
        (1, 1, 17),
        (1, 2, 18),
        (1, 47, 63),
        (12, 0, 0x4000000),
        (12, 1, 0x4000001),
    ],
)
def test_nest2uniq64(order, nest, uniq):
    assert nest2uniq64(order, nest) == uniq
    order_result, nest_result = uniq2nest64(uniq)
    assert order_result == order
    assert nest_result == nest


def test_uniq2ang64_known_values():
    order0 = 4
    test_pixels = [0, 1, 2, 10, 100]

    for ipix in test_pixels:
        uniq = nest2uniq64(order0, ipix)
        theta, phi = uniq2ang64(uniq)

        # Assert values are in valid angular range
        assert jnp.isfinite(theta)
        assert jnp.isfinite(phi)
        assert 0 <= theta <= jnp.pi
        assert 0 <= phi < 2 * jnp.pi


def test_uniq2ang64_consecutive_distinction():
    order0 = 4
    angles = []
    for ipix in range(10):
        uniq = nest2uniq64(order0, ipix)
        theta, phi = uniq2ang64(uniq)
        angles.append((float(theta), float(phi)))

    # Ensure that angles are not all identical
    unique_angles = set(angles)
    assert len(unique_angles) > 1
