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
from jax import jit
import jax.numpy as jnp
import time

class cubic_interp:
    def __init__(self, data, n, tmin, dt):
        self.f = jnp.float32(1 / dt)
        self.t0 = 3 - jnp.float32(self.f * tmin)
        self.length = jnp.int32(n + 6)
        idx = jnp.arange(self.length)
        self.a = self.compute_coeffs(data, n, idx)

    @staticmethod
    @jit
    def compute_coeffs(data, n, idx):
        # Clip indices
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

class bicubic_interp:
    def __init__(self, data, ns, nt, smin, tmin, ds, dt):
        self.fx = np.array([1/ds, 1/dt])
        self.x0 = np.array([3 - self.fx[0] * smin, 3 - self.fx[1] * tmin])
        self.xlength = np.array([ns + 6, nt + 6])
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
    t = jnp.arange(0, 1 + 0.01, 0.01)
    test = cubic_interp(jnp.array([1,0,1,4]), 4, -1, 1)

    _ = test.cubic_interp_eval_jax(t,test.f,test.t0,test.length,test.a)

    start = time.perf_counter()
    result = test.cubic_interp_eval_jax(t,test.f,test.t0,test.length,test.a)
    end = time.perf_counter()
    print(end-start)
    print(result) # expected all squared values
