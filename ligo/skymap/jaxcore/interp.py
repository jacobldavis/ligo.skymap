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

class cubic_interp:
    def __init__(self, data, n, tmin, dt):
        self.f = jnp.float32(1 / dt)
        self.t0 = 3 - jnp.float32(self.f * tmin)
        self.length = jnp.int32(n + 6)
        self.a = []
        for i in range(self.length):
            z = [0,0,0,0]
            for j in range(len(z)):
                z[j] = data[min(max(i+j-4,0),n-1)]
            if np.isnan(z[1]) or np.isinf(z[1]) or np.isnan(z[2]) or np.isinf(z[2]):
                self.a.append([0,0,0,z[1]])
            elif np.isnan(z[0]) or np.isinf(z[0]) or np.isnan(z[3]) or np.isinf(z[3]):
                self.a.append([0,0,z[2]-z[1],z[1]])
            else:
                self.a.append([1.5 * (z[1] - z[2]) + 0.5 * (z[3] - z[0]),
                                z[0] - 2.5 * z[1] + 2 * z[2] - 0.5 * z[3],
                                0.5 * (z[2] - z[0]), z[1]])
        self.a = jnp.array(self.a)
    
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

class bicubic_interp:
    def __init__(self, data, ns, nt, smin, tmin, ds, dt):
        self.fx = jnp.array([jnp.float32(1/ds), jnp.float32(1/dt)])
        self.x0 = jnp.array([jnp.float32(3 - self.fx[0] * smin), jnp.float32(3 - self.fx[1] * tmin)])
        self.xlength = jnp.array([jnp.int32(ns + 6), jnp.int32(nt + 6)])
        self.a = np.zeros((self.xlength[0]*self.xlength[1], 4, 4), dtype=np.float64)
        for iss in range(self.xlength[0]):
            for itt in range(self.xlength[1]):
                a = np.zeros((4, 4), dtype=np.float64)
                a1 = np.zeros((4, 4), dtype=np.float64)
                for js in range(4):
                    z = np.zeros(4, dtype=np.float64)
                    ks = np.clip(iss + js - 4, 0, ns - 1)
                    for jt in range(4):
                        kt = np.clip(itt + jt - 4, 0, nt - 1)
                        z[jt] = data[ks * ns + kt]
                        if np.isnan(z[1]) or np.isinf(z[1]) or np.isnan(z[2]) or np.isinf(z[2]):
                            a[js] = [0,0,0,z[1]]
                        elif np.isnan(z[0]) or np.isinf(z[0]) or np.isnan(z[3]) or np.isinf(z[3]):
                            a[js] = [0,0,z[2]-z[1],z[1]]
                        else:
                            a[js] = [1.5 * (z[1] - z[2]) + 0.5 * (z[3] - z[0]),
                                            z[0] - 2.5 * z[1] + 2 * z[2] - 0.5 * z[3],
                                            0.5 * (z[2] - z[0]), z[1]]
                for js in range(4):
                    for jt in range(4):
                        a1[js][jt] = a[jt][js]
                for js in range(4):
                    if np.isnan(a1[3][1]) or np.isinf(a1[3][1]) or np.isnan(a1[3][2]) or np.isinf(a1[3][2]):
                        a[js] = [0,0,0,a1[js][1]]
                    elif np.isnan(a1[3][0]) or np.isinf(a1[3][0]) or np.isnan(a1[3][3]) or np.isinf(a1[3][3]):
                        a[js] = [0,0,a1[js][2]-a1[js][1],a1[js][1]]
                    else:
                        a[js] = [1.5 * (a1[js][1] - a1[js][2]) + 0.5 * (a1[js][3] - a1[js][0]),
                                        a1[js][0] - 2.5 * a1[js][1] + 2 * a1[js][2] - 0.5 * a1[js][3],
                                        0.5 * (a1[js][2] - a1[js][0]), a1[js][1]]
                self.a[iss * self.xlength[0] + itt] = a
        self.a = jnp.array(self.a)

    @staticmethod
    @jit
    def bicubic_interp_eval_jax(s, t, fx, x0, xlength, a):
        s = jnp.asarray(s)
        t = jnp.asarray(t)
        fx = jnp.asarray(fx)           
        x0 = jnp.asarray(x0)          
        xlength = jnp.asarray(xlength) 
        a = jnp.asarray(a)           
        x = jnp.stack([s, t], axis=-1)  

        def eval_point(x):
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

        return vmap(eval_point)(x)

def test_cubic_interp_0():
    t = jnp.arange(-10.0, 10.0 + 0.01, 0.01)
    test = cubic_interp([0,0,0,0], 4, -1, 1)

    _ = test.cubic_interp_eval_jax(t,test.f,test.t0,test.length,test.a)

    start = time.perf_counter()
    result = test.cubic_interp_eval_jax(t,test.f,test.t0,test.length,test.a)
    end = time.perf_counter()
    print(end-start)
    print(result) # expected all 0s

def test_cubic_interp_1():
    t = jnp.arange(0, 1 + 0.01, 0.01)
    test = cubic_interp([1,0,1,4], 4, -1, 1)

    _ = test.cubic_interp_eval_jax(t,test.f,test.t0,test.length,test.a)

    start = time.perf_counter()
    result = test.cubic_interp_eval_jax(t,test.f,test.t0,test.length,test.a)
    end = time.perf_counter()
    print(end-start)
    print(result) # expected all squared values

def test_bicubic_interp_0():
    s = jnp.arange(0, 2 + 0.01, 0.01)
    t = jnp.arange(0, 2 + 0.01, 0.01)
    test = bicubic_interp([-1,-1,-1,-1,0,0,0,0,1,1,1,1,2,2,2,2],4,4,-1,-1,1,1)

    _ = test.bicubic_interp_eval_jax(s,t,test.fx,test.x0,test.xlength,test.a)

    start = time.perf_counter()
    result = test.bicubic_interp_eval_jax(s,t,test.fx,test.x0,test.xlength,test.a)
    end = time.perf_counter()
    print(end-start)
    print(result) # expect values 0 - 2

def test_bicubic_interp_1():
    s = jnp.arange(0, 2 + 0.01, 0.01)
    t = jnp.arange(0, 2 + 0.01, 0.01)
    test = bicubic_interp([1,1,1,1,0,0,0,0,1,1,1,1,4,4,4,4],4,4,-1,-1,1,1)

    _ = test.bicubic_interp_eval_jax(s,t,test.fx,test.x0,test.xlength,test.a)

    start = time.perf_counter()
    result = test.bicubic_interp_eval_jax(s,t,test.fx,test.x0,test.xlength,test.a)
    end = time.perf_counter()
    print(end-start)
    print(result) # expect values 0 - 4

test_bicubic_interp_1()
