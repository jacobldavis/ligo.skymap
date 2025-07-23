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

import math
from jax import jit, lax
import jax
import jax.numpy as jnp
import numpy as np

default_log_radial_integrator_size = 400
M_PI_2 = jnp.pi / 2
M_LN2 = jnp.log(2)
ntwopsi = 10
nu = 10
# --- MOC AND HEALPIX FUNCTIONS ---

def ang2vec(theta, phi):
    """Convert spherical coordinates to 3D Cartesian vector.

    Parameters
    ----------
    theta : float
        Polar angle in radians.
    phi : float
        Azimuthal angle in radians.

    Returns
    -------
    array_like
        3D unit vector corresponding to (theta, phi).
    """
    sz = jnp.sin(theta)
    return jnp.array([sz * jnp.cos(phi), sz * jnp.sin(phi), jnp.cos(theta)])

@jit
def nest2uniq64(order, nest):
    """Convert NESTED pixel index to UNIQ64 format.

    Parameters
    ----------
    order : int
        HEALPix resolution order.
    nest : int
        Pixel index in NESTED scheme.

    Returns
    -------
    int
        Unique pixel index in UNIQ64 format.
    """
    return jnp.where(nest < 0, -1, nest + (2 ** (2 * (order + 1))))

@jit
def uniq2order64(uniq):
    """Extract the HEALPix order from a UNIQ64 pixel index.

    Parameters
    ----------
    uniq : int
        Pixel index in UNIQ64 format.

    Returns
    -------
    int
        HEALPix resolution order.
    """
    safe_uniq = jnp.where(uniq >= 4, uniq, jnp.uint32(4))
    log2u = jnp.floor(jnp.log2(safe_uniq.astype(jnp.float32)))
    order = (log2u.astype(jnp.uint32) >> 1) - 1
    return jnp.where(uniq < 4, -1, order.astype(jnp.int32))

@jit
def uniq2pixarea64(uniq):
    """Compute pixel area in steradians from a UNIQ64 index.

    Parameters
    ----------
    uniq : int
        UNIQ64 pixel index.

    Returns
    -------
    float
        Pixel area in steradians.
    """
    order = uniq2order64(uniq)
    return jnp.where(order < 0, jnp.nan, jnp.ldexp(jnp.pi / 3, -2 * order))

@jit
def uniq2nest64(uniq):
    """Convert UNIQ64 index to (order, NESTED index).

    Parameters
    ----------
    uniq : int
        UNIQ64 pixel index.

    Returns
    -------
    tuple
        order : int
            HEALPix resolution order.
        nest : int
            Pixel index in NESTED scheme.
    """
    order = uniq2order64(uniq)
    two_pow = 2 ** (2 * (order + 1))
    nest = jnp.where(order < 0, -1, uniq - two_pow)
    return order, nest

@jit
def build_ctab():
    """Build the compression lookup table for bit interleaving.

    Returns
    -------
    list
        Lookup table used to compress 64-bit values to HEALPix xy-faces.
    """
    def Z(a): return jnp.array([a, a+1, a+256, a+257])
    def Y(a): return jnp.concatenate([Z(a), Z(a+2), Z(a+512), Z(a+514)])
    def X(a): return jnp.concatenate([Y(a), Y(a+4), Y(a+1024), Y(a+1028)])
    return jnp.concatenate([X(0), X(8), X(2048), X(2056)])

@jit
def split_64bit(val):
    """Split a 64-bit integer into two 32-bit parts.

    Parameters
    ----------
    val : uint64
        64-bit input value.

    Returns
    -------
    tuple
        low : uint32
            Lower 32 bits.
        high : uint32
            Upper 32 bits.
    """
    val = jnp.uint64(val)
    low = jnp.uint32(val & jnp.uint64(0xFFFFFFFF))
    high = jnp.uint32((val >> 32) & jnp.uint64(0xFFFFFFFF))
    return low, high

@jit
def compress_bits64(v, ctab):
    """Compress bits from a 64-bit integer using a lookup table.

    Parameters
    ----------
    v : uint64
        Pixel index.
    ctab : array_like
        Compression table built from `build_ctab`.

    Returns
    -------
    int
        Compressed value with interleaved bits.
    """
    v_low, v_high = split_64bit(v)
    # Process lower 32 bits
    mask32 = jnp.uint32(0x55555555)
    raw_low = v_low & mask32
    raw_low |= raw_low >> jnp.uint32(15)
    
    # Process upper 32 bits  
    raw_high = v_high & mask32
    raw_high |= raw_high >> jnp.uint32(15)
    
    # Combine results
    b0 = ctab[raw_low & 0xff]
    b1 = ctab[(raw_low >> 8) & 0xff] << 4
    b2 = ctab[raw_high & 0xff] << 16  
    b3 = ctab[(raw_high >> 8) & 0xff] << 20
    return b0 | b1 | b2 | b3

@jit
def nest2xyf64(nside, pix, ctab):
    """Convert NESTED pixel index to (face, x, y) on the HEALPix grid.

    Parameters
    ----------
    nside : int
        HEALPix resolution parameter.
    pix : int
        Pixel index in NESTED scheme.
    ctab : array_like
        Compression lookup table.

    Returns
    -------
    tuple
        face_num : int
            HEALPix face index.
        ix, iy : int
            x and y coordinates within the face.
    """
    npface = nside * nside
    pix = jnp.asarray(pix, dtype=jnp.int32)
    face_num = pix // npface
    pix = pix & (npface - 1)
    ix = compress_bits64(pix, ctab)
    iy = compress_bits64(pix // 2, ctab)
    return face_num, ix, iy

@jit
def pix2ang_nest_z_phi64(nside, pix, ctab, jrll, jpll):
    """Convert NESTED pixel index to z and phi coordinates.

    Parameters
    ----------
    nside : int
        HEALPix resolution parameter.
    pix : int
        NESTED pixel index.
    ctab : array_like
        Bit compression table.
    jrll, jpll : array_like
        Face row/column metadata.

    Returns
    -------
    tuple
        z : float
            Cos(theta) coordinate.
        s : float
            Auxiliary sine projection value.
        phi : float
            Azimuthal angle in radians.
    """
    nl4 = nside * 4
    fact2 = 4.0 / (12.0 * nside * nside)

    face_num, ix, iy = nest2xyf64(nside, pix, ctab)

    jr = jrll[face_num] * nside - ix - iy - 1

    z = jnp.where(
        jr < nside,
        1.0 - (jr * jr * fact2),
        jnp.where(
            jr > 3 * nside,
            (((nl4 - jr) ** 2) * fact2) - 1.0,
            (2 * nside - jr) * (2 * nside * fact2)
        )
    )

    tmp = jnp.where(jr < nside, jr * jr * fact2, ((nl4 - jr) ** 2) * fact2)
    valid_tmp = tmp * (2.0 - tmp)
    s = jnp.where((z > 0.99) | (z < -0.99), jnp.sqrt(valid_tmp), -5.0)

    nr = jnp.where((jr < nside) | (jr > 3 * nside),
                   jnp.where(jr < nside, jr, nl4 - jr),
                   nside)

    kshift = jnp.where((jr >= nside) & (jr <= 3 * nside), (jr - nside) & 1, 0)

    jp = (jpll[face_num] * nr + ix - iy + 1 + kshift) // 2
    jp = jnp.where(jp > nl4, jp - nl4, jp)
    jp = jnp.where(jp < 1, jp + nl4, jp)

    phi = (jp - (kshift + 1) * 0.5) * ((jnp.pi / 2) / nr)

    return z, s, phi

@jit
def pix2ang_nest64(nside, ipix, ctab, jrll, jpll):
    """Convert NESTED pixel index to (theta, phi) angular coordinates.

    Parameters
    ----------
    nside : int
        HEALPix resolution.
    ipix : int
        Pixel index in NESTED ordering.
    ctab : array_like
        Compression table.
    jrll, jpll : array_like
        Lookup tables for face geometry.

    Returns
    -------
    tuple
        theta : float
            Polar angle.
        phi : float
            Azimuthal angle.
    """
    z, s, phi = pix2ang_nest_z_phi64(nside, ipix, ctab, jrll, jpll)
    z_clamped = jnp.clip(z, -1.0, 1.0)
    theta = jnp.where(s < -2.0, jnp.acos(z_clamped), jnp.atan2(s, z_clamped))
    return theta, phi

ctab = jnp.array(build_ctab())
jrll = jnp.array([2,2,2,2,3,3,3,3,4,4,4,4])
jpll = jnp.array([1,3,5,7,0,2,4,6,1,3,5,7])

@jit
def uniq2ang64(uniq):
    """Convert a UNIQ64 index to spherical (theta, phi) coordinates.

    Parameters
    ----------
    uniq : int
        Pixel index in UNIQ64 format.

    Returns
    -------
    tuple
        theta : float
            Polar angle in radians.
        phi : float
            Azimuthal angle in radians.
    """
    order, nest = uniq2nest64(uniq)
    valid = order >= 0
    nside = 2 ** order
    theta, phi = pix2ang_nest64(nside, nest, ctab, jrll, jpll)
    return jnp.where(valid, theta, 0.0), jnp.where(valid, phi, 0.0)

# --- TEST SUITE ---

def test_nest2uniq64(order, nest, uniq):
    uniq_result = nest2uniq64(order, nest)
    print(f"nest2uniq64 Expected: {uniq}, Result: {uniq_result}")
    order_result, nest_result = uniq2nest64(uniq)
    print(f"uniq2nest64 Expected (O/N): {order} {nest}, Result: {order_result} {nest_result}")

# test_nest2uniq64(0, 0, 4)
# test_nest2uniq64(0, 1, 5)
# test_nest2uniq64(0, 2, 6)
# test_nest2uniq64(0, 3, 7)
# test_nest2uniq64(0, 4, 8)
# test_nest2uniq64(0, 5, 9)
# test_nest2uniq64(0, 6, 10)
# test_nest2uniq64(0, 7, 11)
# test_nest2uniq64(0, 8, 12)
# test_nest2uniq64(0, 9, 13)
# test_nest2uniq64(0, 10, 14)
# test_nest2uniq64(0, 11, 15)
# test_nest2uniq64(1, 0, 16)
# test_nest2uniq64(1, 1, 17)
# test_nest2uniq64(1, 2, 18)
# test_nest2uniq64(1, 47, 63)
# test_nest2uniq64(12, 0, 0x4000000)
# test_nest2uniq64(12, 1, 0x4000001)
# test_nest2uniq64(29, 0, 0x1000000000000000)
# test_nest2uniq64(29, 1, 0x1000000000000001)
# test_nest2uniq64(29, 0x2FFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF)

def test_uniq2ang64():
    # Test with some known values
    order0 = 4
    test_pixels = [0, 1, 2, 10, 100]
    
    for ipix in test_pixels:
        uniq = nest2uniq64(order0, ipix)
        theta, phi = uniq2ang64(uniq)
        print(f"Pixel {ipix}: UNIQ={uniq}, theta={theta:.6f}, phi={phi:.6f}")
    
    # Check if consecutive pixels give different angles
    consecutive_angles = []
    for ipix in range(10):
        uniq = nest2uniq64(order0, ipix)
        theta, phi = uniq2ang64(uniq)
        consecutive_angles.append((theta, phi))
    
    print("\nConsecutive pixel angles:")
    for i, (theta, phi) in enumerate(consecutive_angles):
        print(f"  Pixel {i}: theta={theta:.6f}, phi={phi:.6f}")
