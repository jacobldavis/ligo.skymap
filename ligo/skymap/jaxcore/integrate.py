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
from numpy.polynomial.legendre import leggauss
from jax import jit, vmap, lax
from jax.scipy.special import i0
import jax
import jax.numpy as jnp
import time
import timeit
from functools import partial
from interp import *
from cosmology import *
from quadax import quadgk # type: ignore

SQRT_2 = jnp.sqrt(2)
ETA = 0.01

# --- COSMOLOGY BEGIN ---

# gsl_spline_init
@jit
def compute_natural_cubic_spline_coeffs(x, y):
    n = x.shape[0]
    h = x[1:] - x[:-1]
    alpha = 3 * (y[2:] - y[1:-1]) / h[1:] - 3 * (y[1:-1] - y[:-2]) / h[:-1]

    # Tridiagonal matrix system
    l = jnp.ones(n)
    mu = jnp.zeros(n)
    z = jnp.zeros(n)
    l = l.at[1:-1].set(2 * (x[2:] - x[:-2]) - h[:-1] * mu[1:-1])
    mu = mu.at[1:-1].set(h[1:] / l[1:-1])
    z = z.at[1:-1].set((alpha - h[:-1] * z[:-2]) / l[1:-1])

    # Back substitution
    c = jnp.zeros(n)
    b = jnp.zeros(n - 1)
    d = jnp.zeros(n - 1)

    for j in range(n - 2, -1, -1):
        c = c.at[j].set(z[j] - mu[j] * c[j + 1])
        b = b.at[j].set((y[j + 1] - y[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3)
        d = d.at[j].set((c[j + 1] - c[j]) / (3 * h[j]))

    a = y[:-1]
    return x, a, b, c[:-1], d  

# gsl_spline_eval
@jit
def evaluate_cubic_spline(x_knots, coeffs, x_eval):
    a, b, c, d = coeffs 
    def eval_one(x_val):
        i = jnp.clip(jnp.searchsorted(x_knots, x_val) - 1, 0, x_knots.shape[0] - 2)
        dx = x_val - x_knots[i]
        return (dx * (dx * (dx * d[i] + c[i]) + b[i]) + a[i])

    return vmap(eval_one)(x_eval)

# dVC_dVL_init
@jit
def init_log_dVC_dVL_spline():
    len_data = dVC_dVL_data.shape[0]
    x = dVC_dVL_tmin + dVC_dVL_dt * jnp.arange(len_data)
    return compute_natural_cubic_spline_coeffs(x, dVC_dVL_data)

x_knots, a_s, b_s, c_s, d_s = init_log_dVC_dVL_spline()
coeffs = (a_s, b_s, c_s, d_s)

# log_dVC_dVL
@jit
def log_dVC_dVL(DL, x_knots, coeffs):
    log_DL = jnp.log(DL)
    
    spline_val = evaluate_cubic_spline(x_knots, coeffs, jnp.array([log_DL]))[0]
    linear_val = dVC_dVL_high_z_slope * log_DL + dVC_dVL_high_z_intercept

    return jnp.where(
        log_DL <= dVC_dVL_tmin, 0.0,
        jnp.where(
            log_DL >= dVC_dVL_tmax, linear_val,
            spline_val
        )
    )

# --- COSMOLOGY END ---

# gsl_sf_bessel_I0_scaled
@jit
def bessel_I0_scaled(x):
    return jnp.exp(-jnp.abs(x)) * i0(x)

@jit 
def log_radial_integrand(r, p, b, k, cosmology, x_knots, coeffs, scale=0):
    r = jnp.maximum(r, 1e-10)
    ret = jnp.log(bessel_I0_scaled(b/r)*jnp.pow(r, k)) + scale - jnp.pow(p/r - 0.5 * b/p, 2)
    return jnp.where(cosmology, ret + log_dVC_dVL(r, x_knots, coeffs), ret)

@jit
def radial_integrand(r, p, b, k, cosmology, x_knots, coeffs, scale=0):
    x = p / r - 0.5 * b / p
    ret = scale - x**2

    multiplier = bessel_I0_scaled(b / r) * jnp.power(r, k)
    return jnp.where(cosmology, jnp.exp(ret + log_dVC_dVL(r, x_knots, coeffs)) * multiplier, jnp.exp(ret) * multiplier)

x64, w64 = leggauss(64)
x64 = jnp.array(x64)
w64 = jnp.array(w64)

@jit
def compute_breakpoints(p, b, r1, r2):
    eta = 0.01
    pinv = 1.0 / p
    log_eta = jnp.log(eta)

    middle = 2 * p**2 / b
    left = 1.0 / (1.0 / middle + jnp.sqrt(-log_eta) * pinv)
    right = 1.0 / (1.0 / middle - jnp.sqrt(-log_eta) * pinv)

    # Start with r1
    breakpoints = jnp.full((5,), jnp.nan)
    breakpoints = breakpoints.at[0].set(r1)
    n = 1

    def try_add(bp, x, n):
        cond = jnp.logical_and(x > bp[n-1], x < r2)
        bp = lax.cond(cond, lambda b: b.at[n].set(x), lambda b: b, bp)
        n = n + cond.astype(jnp.int32)
        return bp, n

    # Only do this branch if b != 0
    def with_b_nonzero(args):
        bp, n = args
        bp, n = try_add(bp, left, n)
        bp, n = try_add(bp, middle, n)
        bp, n = try_add(bp, right, n)
        bp = bp.at[n].set(r2)
        n = n + 1
        return bp, n

    def with_b_zero(args):
        bp, n = args
        bp = bp.at[n].set(r2)
        n = n + 1
        return bp, n

    breakpoints, nbreakpoints = lax.cond(
        b != 0,
        with_b_nonzero,
        with_b_zero,
        operand=(breakpoints, n)
    )

    return breakpoints, nbreakpoints

# gsl_integration_qagp
def gaussian_quad_integrate(func, a, b):
    t = 0.5 * (x64 + 1) * (b - a) + a
    w = 0.5 * (b - a) * w64
    return jnp.sum(w * func(t))                  

#@jit
def log_radial_integral(xmin, ymin, ix, iy, d, r1, r2, k, cosmology):
    # Determine p and b
    x = xmin + ix * d
    y = ymin + iy * d 
    p = jnp.exp(x)
    b = 2 * (p ** 2) / jnp.exp(y)
    log_offset = -float('inf')
    result = 0

    # Determine breakpoints
    breakpoints, nbreakpoints = compute_breakpoints(p, b, r1, r2)
    
    # Re-scale the integrand
    def log_integrand(r): return log_radial_integrand(r,p,b,k,cosmology,x_knots,coeffs)
    log_vals = vmap(log_integrand)(breakpoints)
    log_vals = jnp.where(jnp.isnan(log_vals), -jnp.inf, log_vals)
    log_offset = jnp.max(log_vals)
    log_offset = jnp.where(log_offset == -float('inf'), 0.0, log_offset)
    print(log_offset)

    MAX_BREAKPOINTS = 5
    breakpoints = jnp.pad(breakpoints, (0, MAX_BREAKPOINTS - breakpoints.shape[0]), constant_values=0.0)
    mask = jnp.arange(MAX_BREAKPOINTS - 1) < (nbreakpoints - 1)

    # Perform Gaussian quadrature
    @jit
    def integral_segment(i):
        a_b = breakpoints[i]
        b_b = breakpoints[i + 1]
        f = lambda r: radial_integrand(r, p, b, k, cosmology, x_knots, coeffs) - log_offset
        return gaussian_quad_integrate(f, a_b, b_b)

    segments = vmap(integral_segment)(jnp.arange(MAX_BREAKPOINTS - 1))
    result = jnp.sum(segments * mask)

    return jnp.where(result > 0, log_offset + jnp.log(result), jnp.nan)

class log_radial_integrator:
    def __init__(self, r1, r2, k, cosmology, pmax, size):
        # Initialize constant values
        alpha = 4
        p0 = 0.5 * r2 if k >= 0 else 0.5 * r1
        xmax = jnp.log(pmax)
        x0 = min(jnp.log(p0), xmax)
        xmin = x0 - (1 + SQRT_2) * alpha
        self.ymax = x0 + alpha
        ymin = 2 * x0 - SQRT_2 * alpha - xmax
        d = (xmax - xmin) / (size - 1)
        umin = - (1 + 1/SQRT_2) * alpha
        self.vmax = x0 - (1/SQRT_2) * alpha
        k1 = k + 1
        self.p0_limit = jnp.log(jnp.log(r2/r1)) if k == -1 else jnp.log((jnp.power(r2,k1)-jnp.power(r1,k1))/(k1))

        # Create data arrays for initializing interps
        z0 = vmap(lambda ix: vmap(lambda iy: log_radial_integral(xmin, ymin, ix, iy, d, r1, r2, k, cosmology))(jnp.arange(size)))(jnp.arange(size))
        z0_flat = jnp.ravel(z0)
        print(z0_flat)

        # Initialize the interps
        self.region0 = bicubic_interp(z0_flat, size, size, xmin, ymin, d, d)
        z1 = vmap(lambda i: z0[i][size-1])(jnp.arange(size))
        self.region1 = cubic_interp(z1, size, xmin, d)
        z2 = vmap(lambda i: z0[i][size - 1 - i])(jnp.arange(size))
        self.region2 = cubic_interp(z2, size, umin, d)
    
    # TODO: change the params to tuples if desired
    @staticmethod
    @jit
    def log_radial_integrator_eval(fx, x0, xlength, a, 
                                   f1, t01, length1, a1, 
                                   f2, t02, length2, a2,
                                   p0_limit, vmax, ymax, p, b, log_p, log_b):
        x = log_p 
        y = jnp.log(2) + 2 * log_p - log_b
        result = jnp.pow(0.5 * b / p, 2)
        result += jnp.where(y >= ymax, 
                            cubic_interp.cubic_interp_eval_jax(x,f1,t01,length1,a1),
                            jnp.where((0.5 * (x + y)) <= vmax,
                                      cubic_interp.cubic_interp_eval_jax(0.5 * (x-y),f2,t02,length2,a2),
                                      bicubic_interp.bicubic_interp_eval_jax(x,y,fx,x0,xlength,a)))

        return jnp.where(p > 0, result, p0_limit)

# --- QUADAX IMPLEMENTATION ---

class log_radial_integrator_quadax:
    def __init__(self, r1, r2, k):
        self.r1 = r1
        self.r2 = r2
        self.k = k
    
    @jit
    def log_radial_integrator_eval_quadax(r1, r2, p, b, k):
        def integrand(r):
            return radial_integrand(r, p, b, k, 0, x_knots, coeffs)
        result_quadax, err = quadgk(integrand, [r1,r2], epsabs=1e-8)
        return result_quadax
    
# --- TEST SUITE ---

def test_log_radial_integral(expected, tol, r1, r2, p2, b, k):
    p = jnp.sqrt(p2)

    print("==> JAX VERSION:")
    start = time.perf_counter()
    integrator = log_radial_integrator(r1, r2, k, 0, p + 0.5, 400)
    end = time.perf_counter()
    print(f"JAX init time: {end - start}")

    result_jax_compile = integrator.log_radial_integrator_eval(
        integrator.region0.fx, integrator.region0.x0, integrator.region0.xlength, integrator.region0.a,
        integrator.region1.f, integrator.region1.t0, integrator.region1.length, integrator.region1.a,
        integrator.region2.f, integrator.region2.t0, integrator.region2.length, integrator.region2.a,
        integrator.p0_limit, integrator.vmax, integrator.ymax,
        p, b, jnp.log(p), jnp.log(b)
    )
    start = time.perf_counter()
    result_jax = integrator.log_radial_integrator_eval(
        integrator.region0.fx, integrator.region0.x0, integrator.region0.xlength, integrator.region0.a,
        integrator.region1.f, integrator.region1.t0, integrator.region1.length, integrator.region1.a,
        integrator.region2.f, integrator.region2.t0, integrator.region2.length, integrator.region2.a,
        integrator.p0_limit, integrator.vmax, integrator.ymax,
        p, b, jnp.log(p), jnp.log(b)
    )
    end = time.perf_counter()
    print(f"JAX result: {result_jax}")
    print(f"JAX time: {end-start}")

    print("\n==> QUADAX VERSION:")
    a = log_radial_integrator_quadax(r1, r2, k)
    result = log_radial_integrator_quadax.log_radial_integrator_eval_quadax(a.r1, a.r2, p, b, a.k)
    start = time.perf_counter()
    result_quadax = log_radial_integrator_quadax.log_radial_integrator_eval_quadax(a.r1, a.r2, p, b, a.k)
    log_result_quadax = np.log(result_quadax) if result_quadax > 0 else np.nan
    end = time.perf_counter()
    print(f"QuadAx result: {log_result_quadax}")
    print(f"QuadAx time: {end-start}")

test_log_radial_integral(-0.480238, 1e-3, 1, 2, 1, 0, 0)

