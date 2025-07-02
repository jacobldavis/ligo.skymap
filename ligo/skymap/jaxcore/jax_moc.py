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
from jax import jit
import jax.numpy as jnp
import numpy as np

default_log_radial_integrator_size = 400
M_PI_2 = jnp.pi / 2
M_LN2 = jnp.log(2)
ntwopsi = 10
nu = 10

def ang2vec(theta, phi):
    sz = jnp.sin(theta)
    return jnp.array([sz * jnp.cos(phi), sz * jnp.sin(phi), jnp.cos(theta)])

def nest2uniq64(order, nest):
    return jnp.where(nest < 0, -1, nest + (1 << 2 * (order + 1)))

def uniq2order64(uniq):
    order = jnp.floor(jnp.log2(uniq)).astype(jnp.int8)
    return jnp.where(uniq < 4, -1, (order >> 1) - 1)

def uniq2pixarea64(uniq):
    order = uniq2order64(uniq)
    if (order < 0):
        return math.nan
    else:
        return math.ldexp(math.pi / 3, -2 * order)

def uniq2nest64(uniq):
    order = uniq2order64(uniq)
    nest = jnp.where(order < 0, -1, uniq - (1 << 2 * (order + 1)))
    return order, nest

def build_ctab() -> list[int]:
    def Z(a): return [a, a+1, a+256, a+257]
    def Y(a): return Z(a) + Z(a+2) + Z(a+512) + Z(a+514)
    def X(a): return Y(a) + Y(a+4) + Y(a+1024) + Y(a+1028)

    return X(0) + X(8) + X(2048) + X(2056)

halfpi=1.570796326794896619231321691639751442099

@jit
def compress_bits64(v, ctab):
    v = jnp.asarray(v, dtype=jnp.uint64)
    mask = jnp.array(np.uint64(0x5555555555555555), dtype=jnp.uint64)
    raw = v & mask
    raw |= raw >> jnp.uint64(15)

    b0 = ctab[raw & 0xff]
    b1 = ctab[(raw >> 8) & 0xff] << 4
    b2 = ctab[(raw >> 32) & 0xff] << 16
    b3 = ctab[(raw >> 40) & 0xff] << 20

    return b0 | b1 | b2 | b3

@jit
def nest2xyf64(nside, pix, ctab):
    npface = nside * nside
    pix = jnp.asarray(pix, dtype=jnp.int64)
    face_num = pix // npface
    pix = pix & (npface - 1)
    ix = compress_bits64(pix, ctab)
    iy = compress_bits64(pix >> 1, ctab)
    return face_num, ix, iy

@jit
def pix2ang_nest_z_phi64(nside, pix, ctab, jrll, jpll):
    nl4 = nside * 4
    npix = 12 * nside * nside
    fact2 = 4.0 / npix

    face_num, ix, iy = nest2xyf64(nside, pix, ctab)

    jr = jrll[face_num] * nside - ix - iy - 1

    z = jnp.where(
        jr < nside,
        1.0 - (jr * jr * fact2),
        jnp.where(
            jr > 3 * nside,
            1.0 - ((nl4 - jr) ** 2 * fact2),
            (2 * nside - jr) * (2 * nside * fact2)
        )
    )

    tmp = jnp.where(jr < nside, jr * jr * fact2, (nl4 - jr) ** 2 * fact2)
    s = jnp.where((jr < nside) | (jr > 3 * nside) & (z > 0.99),
                  jnp.sqrt(tmp * (2.0 - tmp)),
                  -5.0)

    nr = jnp.where((jr < nside) | (jr > 3 * nside),
                   jnp.where(jr < nside, jr, nl4 - jr),
                   nside)

    kshift = jnp.where((jr >= nside) & (jr <= 3 * nside), (jr - nside) & 1, 0)

    jp = (jpll[face_num] * nr + ix - iy + 1 + kshift) // 2
    jp = jnp.where(jp > nl4, jp - nl4, jp)
    jp = jnp.where(jp < 1, jp + nl4, jp)

    halfpi = jnp.pi / 2
    phi = (jp - (kshift + 1) * 0.5) * (halfpi / nr)

    return z, s, phi

@jit
def pix2ang_nest64(nside, ipix, ctab, jrll, jpll):
    z, s, phi = pix2ang_nest_z_phi64(nside, ipix, ctab, jrll, jpll)
    theta = jnp.where(s < -2.0, jnp.arccos(z), jnp.arctan2(s, z))
    return theta, phi

@jit
def uniq2ang64(uniq):
    ctab = jnp.array(build_ctab())
    jrll = jnp.array([2,2,2,2,3,3,3,3,4,4,4,4])
    jpll = jnp.array([1,3,5,7,0,2,4,6,1,3,5,7])
    order, nest = uniq2nest64(uniq)
    valid = order >= 0
    nside = 1 << order
    theta, phi = pix2ang_nest64(nside, nest, ctab, jrll, jpll)
    return jnp.where(valid, theta, 0.0), jnp.where(valid, phi, 0.0)
