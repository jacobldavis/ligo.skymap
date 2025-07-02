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
ARANGE4 = jnp.arange(4)

class cubic_interp:
    def __init__(self, data, n, tmin, dt):
        self.f = jnp.float32(1 / dt)
        self.t0 = 3 - jnp.float32(self.f * tmin)
        self.length = jnp.int32(n + 6)
        idx = np.arange(n + 6).astype(np.int32)
        self.a = self.compute_coeffs(data, n, idx)

    @staticmethod
    @jit
    def compute_coeffs(data, n, idx):
        # Clip indices and build z
        z = jnp.stack([
            data[jnp.clip(idx - 4, 0, n - 1)],
            data[jnp.clip(idx - 3, 0, n - 1)],
            data[jnp.clip(idx - 2, 0, n - 1)],
            data[jnp.clip(idx - 1, 0, n - 1)],
        ], axis=1)

        nan_or_inf = lambda x: jnp.isnan(x) | jnp.isinf(x)
        bad_12 = nan_or_inf(z[:,1]) | nan_or_inf(z[:,2])
        bad_03 = nan_or_inf(z[:,0]) | nan_or_inf(z[:,3])

        # Compute coefficients
        a0 = 1.5 * (z[:,1] - z[:,2]) + 0.5 * (z[:,3] - z[:,0])
        a1 = z[:,0] - 2.5 * z[:,1] + 2 * z[:,2] - 0.5 * z[:,3]
        a2 = 0.5 * (z[:,2] - z[:,0])
        a3 = z[:,1]

        # Initialize coefficients
        out = jnp.stack([
            jnp.where(bad_12, 0.0, jnp.where(bad_03, 0.0, a0)),
            jnp.where(bad_12, 0.0, jnp.where(bad_03, 0.0, a1)),
            jnp.where(bad_12, 0.0, jnp.where(bad_03, a2, a2)),
            jnp.where(bad_12, z[:,1], a3),
        ], axis=1)

        return out

    @staticmethod
    @jit
    def cubic_interp_eval_jax(data, f, t0, length, a):
        x = jnp.clip(data * f + t0, 0.0, length - 1.0)
        ix = x.astype(int)
        x -= ix
        
        a0 = a[ix, 0]
        a1 = a[ix, 1]
        a2 = a[ix, 2]
        a3 = a[ix, 3]

        return ((a0 * x + a1) * x + a2) * x + a3

@jit
def cubic_eval(coeffs, x):
    return ((coeffs[..., 0] * x + coeffs[..., 1]) * x + coeffs[..., 2]) * x + coeffs[..., 3]

@jit
def nan_or_inf(x):
    return jnp.isnan(x) | jnp.isinf(x)

@jit
def interpolate_1d(z):
    bad12 = nan_or_inf(z[1]) | nan_or_inf(z[2])
    bad03 = nan_or_inf(z[0]) | nan_or_inf(z[3])

    # Compute coefficients
    a0 = 1.5 * (z[1] - z[2]) + 0.5 * (z[3] - z[0])
    a1 = z[0] - 2.5 * z[1] + 2 * z[2] - 0.5 * z[3]
    a2 = 0.5 * (z[2] - z[0])
    a3 = z[1]

    # Initialize coefficients
    return jnp.stack([
        jnp.where(bad12, 0.0, jnp.where(bad03, 0.0, a0)),
        jnp.where(bad12, 0.0, jnp.where(bad03, 0.0, a1)),
        jnp.where(bad12, 0.0, jnp.where(bad03, a2, a2)),
        jnp.where(bad12, z[1], a3)
    ])

@partial(jit, static_argnames=['ns', 'nt'])
def compute_coeffs(data, ns, nt):
    length_s = ns + 6
    length_t = nt + 6

    def get_z(js, iss, itt):
        ks = jnp.clip(iss + js - 4, 0, ns - 1)

        def get_zt(jt):
            kt = jnp.clip(itt + jt - 4, 0, nt - 1)
            return data[ks * nt + kt]

        return vmap(get_zt)(ARANGE4)

    def compute_block(iss, itt):
        a_rows = vmap(lambda js: interpolate_1d(get_z(js, iss, itt)))(jnp.arange(4))
        return vmap(interpolate_1d)(a_rows.T)

    blocks = vmap(lambda iss: vmap(lambda itt: compute_block(iss, itt))(jnp.arange(length_t)))(jnp.arange(length_s))
    return blocks.reshape(length_s * length_t, 4, 4)

class bicubic_interp:
    def __init__(self, data, ns, nt, smin, tmin, ds, dt):
        self.fx = jnp.array([jnp.float32(1/ds), jnp.float32(1/dt)])
        self.x0 = jnp.array([jnp.float32(3 - self.fx[0] * smin), jnp.float32(3 - self.fx[1] * tmin)])
        self.xlength = jnp.array([jnp.int32(ns + 6), jnp.int32(nt + 6)])
        self.a = compute_coeffs(data,ns,nt)

    @staticmethod
    @jit
    def bicubic_interp_eval_jax(s, t, fx, x0, xlength, a):
        s = jnp.asarray(s)
        t = jnp.asarray(t)         
        x = jnp.stack([s, t], axis=-1)  

        def eval_point(x):
            x = jnp.atleast_1d(x)
            is_nan = jnp.isnan(x[0]) | jnp.isnan(x[1])

            x_scaled = x * fx + x0
            x_clipped = jnp.clip(x_scaled, 0.0, xlength - 1.0)

            ix = jnp.floor(x_clipped).astype(jnp.int32)
            x_frac = x_clipped - ix

            flat_idx = ix[0] * xlength[1] + ix[1]
            coeff = a[flat_idx] 

            b = cubic_eval(coeff.T, x_frac[1])
            result = cubic_eval(b, x_frac[0])  

            return jnp.where(is_nan, x[0] + x[1], result)

        return eval_point(x)

# --- TEST SUITE ---

def test_cubic_interp_0():
    t = jnp.arange(-10.0, 10.0 + 0.01, 0.01)
    test = cubic_interp(jnp.array([0,0,0,0]), 4, -1, 1)

    _ = test.cubic_interp_eval_jax(t,test.f,test.t0,test.length,test.a)

    start = time.perf_counter()
    result = test.cubic_interp_eval_jax(t,test.f,test.t0,test.length,test.a)
    end = time.perf_counter()
    print(end-start)
    print(result) # expected all 0s

def test_cubic_interp_1():
    t = jnp.arange(0, 2 + 0.01, 0.01)
    test = cubic_interp(jnp.array([1,0,1,4]), 4, -1, 1)

    _ = test.cubic_interp_eval_jax(t,test.f,test.t0,test.length,test.a)

    start = time.perf_counter()
    result = test.cubic_interp_eval_jax(t,test.f,test.t0,test.length,test.a)
    end = time.perf_counter()
    print(end-start)
    print(result) # expected all squared values

def test_bicubic_interp_0():
    s = jnp.arange(0, 2 + 0.01, 0.01)
    t = jnp.arange(0, 2 + 0.01, 0.01)
    test = bicubic_interp(jnp.array([-1,-1,-1,-1,0,0,0,0,1,1,1,1,2,2,2,2]),4,4,-1,-1,1,1)

    _ = test.bicubic_interp_eval_jax(s,t,test.fx,test.x0,test.xlength,test.a)

    start = time.perf_counter()
    result = test.bicubic_interp_eval_jax(s,t,test.fx,test.x0,test.xlength,test.a)
    print(jnp.shape(result))
    end = time.perf_counter()
    print(end-start)
    print(result) # expect values 0 - 2

def test_bicubic_interp_1():
    s = jnp.arange(0, 2 + 0.01, 0.01)
    t = jnp.arange(0, 2 + 0.01, 0.01)
    test = bicubic_interp(jnp.array([-1,-1,-1,-1,0,0,0,0,1,1,1,1,8,8,8,8]),4,4,-1,-1,1,1)

    _ = test.bicubic_interp_eval_jax(s,t,test.fx,test.x0,test.xlength,test.a)

    start = time.perf_counter()
    result = test.bicubic_interp_eval_jax(s,t,test.fx,test.x0,test.xlength,test.a)
    end = time.perf_counter()
    print(end-start)
    print(result) # expect values 0 - 8
