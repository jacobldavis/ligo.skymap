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

def ang2vec(theta, phi):
    sz = math.sin(theta)
    return [sz * math.cos(phi), sz * math.sin(phi), math.cos(theta)]

def nest2uniq64(order, nest):
    return -1 if nest < 0 else nest + (1 << 2 * (order + 1))

def uniq2order64(uniq):
    if uniq < 4:
        return -1

    order = uniq.bit_length() - 1

    return (order >> 1) - 1

def uniq2nest64(uniq):
    order = uniq2order64(uniq)
    if (order < 0): nest = -1 
    else: nest = uniq - (1 << 2 * (order + 1))
    return order, nest

def build_ctab() -> list[int]:
    def Z(a): return [a, a+1, a+256, a+257]
    def Y(a): return Z(a) + Z(a+2) + Z(a+512) + Z(a+514)
    def X(a): return Y(a) + Y(a+4) + Y(a+1024) + Y(a+1028)

    return X(0) + X(8) + X(2048) + X(2056)

ctab = build_ctab()
jrll = [2,2,2,2,3,3,3,3,4,4,4,4]
jpll = [1,3,5,7,0,2,4,6,1,3,5,7]
halfpi=1.570796326794896619231321691639751442099

def compress_bits64(v):
    raw = v & 0x5555555555555555
    raw |= raw >> 15
    result = (ctab[ raw        & 0xff]      |
              (ctab[(raw >>  8) & 0xff] << 4) |
              (ctab[(raw >> 32) & 0xff] << 16) |
              (ctab[(raw >> 40) & 0xff] << 20))
    return result

def nest2xyf64(nside, pix):
    npface = nside * nside
    face_num = pix/npface 
    pix &= npface-1 
    ix = compress_bits64(pix)  
    iy = compress_bits64(pix >> 1)

    return face_num, ix, iy

def pix2ang_nest_z_phi64(nside, pix):
    nl4 = nside*4
    npix = 12*nside*nside 
    fact2 = 4.0/npix 
    s = -5

    face_num, ix, iy = nest2xyf64(nside,pix)
    jr = (jrll[face_num]*nside) - ix - iy - 1

    if (jr<nside):
        nr = jr 
        tmp = (nr*nr)*fact2 
        z = 1 - tmp 
        if (z > 0.99): s = math.sqrt(tmp*(2.0-tmp))
        kshift = 0
    elif (jr > 3 * nside):
        nr = nl4-jr
        tmp = (nr*nr)*fact2 
        z = 1 - tmp 
        if (z > 0.99): s = math.sqrt(tmp*(2.0-tmp))
        kshift = 0
    else:
        fact1 = (nside<<1)*fact2
        nr = nside 
        z = (2*nside-jr)*fact1 
        kshift = (jr-nside)&1
    
    jp = (jpll[face_num]*nr + ix - iy + 1 + kshift) / 2
    if (jp>nl4): jp0-=nl4
    if (jp < 1): jp += nl4

    phi = (jp-(kshift+1)*0.5)*(halfpi/nr)

    return z, s, phi

def pix2ang_nest64(nside, ipix):
    z, s, phi = pix2ang_nest_z_phi64(nside, ipix)
    theta = math.acos(z) if (s < -2) else math.atan2(s,z)
    return theta, phi

def uniq2ang64(uniq):
    order, nest = uniq2nest64(uniq)
    if (order < 0): return 0, 0
    nside = 1 << order 
    return pix2ang_nest64(nside, nest)
