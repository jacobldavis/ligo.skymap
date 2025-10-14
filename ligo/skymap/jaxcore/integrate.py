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
from jax import jit, lax, vmap
from jax.scipy.special import i0e
from quadax import quadgk

from ligo.skymap.jaxcore.cosmology import (
    dVC_dVL_data,
    dVC_dVL_dt,
    dVC_dVL_high_z_intercept,
    dVC_dVL_high_z_slope,
    dVC_dVL_tmax,
    dVC_dVL_tmin,
)
from ligo.skymap.jaxcore.interp import (
    bicubic_interp_eval,
    bicubic_interp_init,
    cubic_interp_eval,
    cubic_interp_init,
)

# --- COSMOLOGY ---


@jit
def compute_natural_cubic_spline_coeffs(x, y):
    """Compute natural cubic spline coefficients for 1D data.

    Parameters
    ----------
    x : array_like
        1D array of strictly increasing x-values (knots).
    y : array_like
        1D array of function values at corresponding x.

    Returns
    -------
    x_knots : array_like
        Original x-values.
    a, b, c, d : array_like
        Coefficients of the spline segments for evaluating cubic polynomials.
    """
    n = x.shape[0]
    h = x[1:] - x[:-1]
    alpha = 3 * (y[2:] - y[1:-1]) / h[1:] - 3 * (y[1:-1] - y[:-2]) / h[:-1]

    # Tridiagonal matrix system
    li = jnp.ones(n)
    mu = jnp.zeros(n)
    z = jnp.zeros(n)
    li = li.at[1:-1].set(2 * (x[2:] - x[:-2]) - h[:-1] * mu[1:-1])
    mu = mu.at[1:-1].set(h[1:] / li[1:-1])
    z = z.at[1:-1].set((alpha - h[:-1] * z[:-2]) / li[1:-1])

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


@jit
def evaluate_cubic_spline(x_knots, coeffs, x_eval):
    """Evaluate cubic spline at a set of points.

    Parameters
    ----------
    x_knots : array_like
        Knot locations (must be sorted in ascending order).
    coeffs : tuple
        Tuple of spline coefficients (a, b, c, d).
    x_eval : array_like
        Points at which to evaluate the spline.

    Returns
    -------
    y_eval : array_like
        Interpolated values at x_eval.
    """
    a, b, c, d = coeffs

    def eval_one(x_val):
        i = jnp.clip(jnp.searchsorted(x_knots, x_val) - 1, 0, x_knots.shape[0] - 2)
        dx = x_val - x_knots[i]
        return dx * (dx * (dx * d[i] + c[i]) + b[i]) + a[i]

    return vmap(eval_one)(x_eval)


@jit
def init_log_dVC_dVL_spline():
    len_data = dVC_dVL_data.shape[0]
    x = dVC_dVL_tmin + dVC_dVL_dt * jnp.arange(len_data)
    return compute_natural_cubic_spline_coeffs(x, dVC_dVL_data)


x_knots, a_s, b_s, c_s, d_s = init_log_dVC_dVL_spline()
coeffs = (a_s, b_s, c_s, d_s)


@jit
def log_dVC_dVL(DL, x_knots, coeffs):
    """Evaluate log(dVC/dVL) given luminosity distance.

    Parameters
    ----------
    DL : float
        Luminosity distance in Mpc.
    x_knots : array_like
        Knot positions in log(DL).
    coeffs : tuple
        Spline coefficients (a, b, c, d).

    Returns
    -------
    float
        log(dVC/dVL) evaluated at log(DL).
    """
    log_DL = jnp.log(DL)

    spline_val = evaluate_cubic_spline(x_knots, coeffs, jnp.array([log_DL]))[0]
    linear_val = dVC_dVL_high_z_slope * log_DL + dVC_dVL_high_z_intercept

    return jnp.where(
        log_DL <= dVC_dVL_tmin,
        0.0,
        jnp.where(log_DL >= dVC_dVL_tmax, linear_val, spline_val),
    )


# --- INTEGRATOR ---


@jit
def log_radial_integrand(r, p, b, k, cosmology, x_knots, coeffs, scale=0):
    """Logarithmic version of radial integrand.

    Parameters
    ----------
    r : float
        Luminosity distance.
    p, b, k : float
        Constants of the integrand.
    cosmology : bool
        Whether to include cosmology.
    x_knots : array_like
        Spline x-knots for dVC/dVL.
    coeffs : tuple
        Spline coefficients.
    scale : float, optional
        Constant added to stabilize log-domain integration.

    Returns
    -------
    float
        Log of the radial integrand.

    NOTE: cosmology is temporarily disabled to increase runtimes
    """
    ret = jnp.log(i0e(b / r) * jnp.pow(r, k)) + scale - jnp.pow(p / r - 0.5 * b / p, 2)
    return ret


@jit
def radial_integrand(r, p, b, k, cosmology, x_knots, coeffs, scale=0):
    """Radial integrand used for numerical integration.

    Parameters
    ----------
    r : float
        Radial distance (Mpc).
    p, b, k : float
        Model parameters.
    cosmology : bool
        Whether to include cosmology.
    x_knots, coeffs : array_like, tuple
        Spline knots and coefficients for log(dVC/dVL).
    scale : float, optional
        Scale for numerical stability in log-space.

    Returns
    -------
    float
        Value of the integrand at r.

    NOTE: cosmology is temporarily disabled to improve runtimes
    """
    ret = scale - jnp.pow(p / r - 0.5 * b / p, 2)
    multiplier = i0e(b / r) * jnp.power(r, k)
    return jnp.exp(ret) * multiplier


@jit
def compute_breakpoints(p, b, r1, r2):
    """Compute quadrature breakpoints for integration.

    Parameters
    ----------
    p, b : float
        Parameters of the integrand.
    r1, r2 : float
        Integration bounds.

    Returns
    -------
    breakpoints : array_like
        Array of breakpoints for adaptive integration.
    nbreakpoints : int
        Number of valid breakpoints.
    """
    eta = 0.01
    pinv = 1.0 / p
    log_eta = jnp.log(eta)

    middle = 2 * p**2 / b
    left = 1.0 / (1.0 / middle + jnp.sqrt(-log_eta) * pinv)
    right = 1.0 / (1.0 / middle - jnp.sqrt(-log_eta) * pinv)

    # Start with r1
    breakpoints = jnp.full((5,), 0)
    breakpoints = breakpoints.at[0].set(r1)
    n = 1

    def try_add(bp, x, n):
        cond = jnp.logical_and(x > bp[n - 1], x < r2)
        bp = lax.cond(cond, lambda b: b.at[n].set(x), lambda b: b, bp)
        n = n + cond
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
        b != 0, with_b_nonzero, with_b_zero, operand=(breakpoints, n)
    )

    return breakpoints, nbreakpoints


@jit
def log_radial_integral(xmin, ymin, ix, iy, d, r1, r2, k, cosmology):
    """Compute the log of the integral over the radial distance.

    Parameters
    ----------
    xmin, ymin : float
        Lower bounds of the log(p), log(b) grid.
    ix, iy : int
        Indices into the grid.
    d : float
        Grid step size in both x and y.
    r1, r2 : float
        Integration bounds.
    k : float
        Power-law exponent.
    cosmology : bool
        Whether to apply cosmological volume correction.

    Returns
    -------
    float
        Logarithm of the integral value.
    """
    # Compute p and b
    x = xmin + ix * d
    y = ymin + iy * d
    p = jnp.exp(x)
    b = 2 * (jnp.pow(p, 2)) / jnp.exp(y)

    # Determine breakpoints and log_offset
    breakpoints, nbreakpoints = compute_breakpoints(p, b, r1, r2)

    def log_integrand(r):
        return log_radial_integrand(r, p, b, k, cosmology, x_knots, coeffs)

    log_vals = vmap(log_integrand)(breakpoints)
    log_vals = jnp.where(jnp.isnan(log_vals), -jnp.inf, log_vals)
    log_offset = jnp.max(log_vals)
    log_offset = jnp.where(log_offset == -jnp.inf, 0.0, log_offset)

    def integrand(r):
        return radial_integrand(r, p, b, k, cosmology, x_knots, coeffs, -log_offset)

    # Adaptive quadrature
    result, _ = quadgk(integrand, [r1, r2], epsrel=1e-8)

    return jnp.log(result) + log_offset


def integrator_init(r1, r2, k, cosmology, pmax, size):
    """Adaptive log-domain radial integral evaluator using interpolation.

    Parameters
    ----------
    r1, r2 : float
        Integration bounds in Mpc.
    k : float
        Exponent for radial power-law.
    cosmology : bool
        If True, include comoving volume factor.
    pmax : float
        Maximum value of parameter p.
    size : int
        Number of grid points in interpolation domain.

    Attributes
    ----------
    region0 : bicubic_interp
        2D interpolation over (log(p), log(b)) domain.
    region1, region2 : cubic_interp
        1D interpolation over boundaries of region0.
    """
    # Initialize constant values
    alpha = 4
    p0 = jnp.where(k >= 0, 0.5 * r2, 0.5 * r1)
    xmax = jnp.log(pmax)
    x0 = jnp.where(jnp.log(p0) > xmax, jnp.log(p0), xmax)
    xmin = x0 - (1 + jnp.sqrt(2)) * alpha
    ymax = x0 + alpha
    ymin = 2 * x0 - jnp.sqrt(2) * alpha - xmax
    d = (xmax - xmin) / (size - 1)
    umin = -(1 + 1 / jnp.sqrt(2)) * alpha
    vmax = x0 - (1 / jnp.sqrt(2)) * alpha
    k1 = k + 1
    r2 = jnp.where(1e-12 > r2, 1e-12, r2)
    r1 = jnp.where(1e-12 > r1, 1e-12, r1)
    p0_limit = jnp.where(
        k == -1,
        jnp.log(jnp.log(r2 / r1)),
        jnp.log((jnp.power(r2, k1) - jnp.power(r1, k1)) / (k1)),
    )

    # Create data arrays for initializing interps
    z0 = vmap(
        lambda ix: vmap(
            lambda iy: log_radial_integral(xmin, ymin, ix, iy, d, r1, r2, k, cosmology)
        )(jnp.arange(size))
    )(jnp.arange(size))
    z0_flat = jnp.ravel(z0)

    # Initialize the interps
    region0 = bicubic_interp_init(z0_flat, size, size, xmin, ymin, d, d)
    z1 = vmap(lambda i: z0[i][size - 1])(jnp.arange(size))
    region1 = cubic_interp_init(z1, size, xmin, d)
    z2 = vmap(lambda i: z0[i][size - 1 - i])(jnp.arange(size))
    region2 = cubic_interp_init(z2, size, umin, d)

    # Return interpolants and limits
    regions = (region0, region1, region2)
    limits = (p0_limit, vmax, ymax)

    return regions, limits


@staticmethod
@jit
def integrator_eval(region0, region1, region2, limits, p, b, log_p, log_b):
    """Evaluate integral at (p, b) using precomputed interpolants.

    Parameters
    ----------
    region0, region1, region2 : tuple
        Interpolator parameters for interior and boundary.
    limits : tuple
        (p0_limit, vmax, ymax) bounding behavior for fallback cases.
    p, b : float
        Parameters.
    log_p, log_b : float
        Precomputed logarithms of p and b.

    Returns
    -------
    float
        Approximation of the log integral value.
    """
    # Unpack values
    fx, x0, xlength, a = region0
    f1, t01, length1, a1 = region1
    f2, t02, length2, a2 = region2
    p0_limit, vmax, ymax = limits

    # Evaluate interpolant
    x = log_p
    y = jnp.log(2) + 2 * log_p - log_b
    result = jnp.pow(0.5 * b / p, 2)
    result += jnp.where(
        y >= ymax,
        cubic_interp_eval(x, f1, t01, length1, a1),
        jnp.where(
            (0.5 * (x + y)) <= vmax,
            cubic_interp_eval(0.5 * (x - y), f2, t02, length2, a2),
            bicubic_interp_eval(x, y, fx, x0, xlength, a),
        ),
    )

    return jnp.where(p > 0, result, p0_limit)
