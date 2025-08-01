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
from jax import jit, vmap
import jax.numpy as jnp
from functools import partial
ARANGE4 = jnp.arange(4)
SQRT_2 = jnp.sqrt(2)
ETA = 0.01

# --- CUBIC INTERP ---


@jit
def nan_or_inf(x):
    return jnp.isnan(x) | jnp.isinf(x)


class cubic_interp:
    """1D cubic interpolation using precomputed coefficients.

    Parameters
    ----------
    data : array_like
        Input data array of shape (n,).
    n : int
        Number of input data samples.
    tmin : float
        Minimum t value corresponding to the first sample.
    dt : float
        Spacing between samples.

    Attributes
    ----------
    f : float
        Inverse of dt, used for scaling.
    t0 : float
        Precomputed shift constant.
    length : int
        Length of the padded coefficient array.
    a : jax.numpy.ndarray
        Array of shape (n+6, 4) containing cubic coefficients.
    """

    def __init__(self, data, n, tmin, dt):
        self.f = 1 / dt
        self.t0 = 3 - self.f * tmin
        self.length = n + 6
        self.a = vmap(lambda idx: self.compute_coeffs(
            idx, data, n))(jnp.arange(n + 6))

    @staticmethod
    @jit
    def compute_coeffs(idx, data, n):
        """Compute cubic interpolation coefficients.

        Parameters
        ----------
        data : array_like
            Input data.
        n : int
            Number of samples in data.
        idx : array_like
            Index array for interpolation positions.

        Returns
        -------
        coeffs : jax.numpy.ndarray
            Array of cubic coefficients, shape (4).
        """
        # Clip indices and build z
        z = jnp.array([
            data[jnp.clip(idx - 4, 0, n - 1)],
            data[jnp.clip(idx - 3, 0, n - 1)],
            data[jnp.clip(idx - 2, 0, n - 1)],
            data[jnp.clip(idx - 1, 0, n - 1)],
        ])
        bad12 = nan_or_inf(z[1] + z[2])
        bad03 = nan_or_inf(z[0] + z[3])

        # Compute coefficients
        a0 = 1.5 * (z[1] - z[2]) + 0.5 * (z[3] - z[0])
        a1 = z[0] - 2.5 * z[1] + 2 * z[2] - 0.5 * z[3]
        a2 = 0.5 * (z[2] - z[0])
        a3 = z[1]

        # Initialize coefficients
        return jnp.array([
            jnp.where(bad12, 0.0, jnp.where(bad03, 0.0, a0)),
            jnp.where(bad12, 0.0, jnp.where(bad03, 0.0, a1)),
            jnp.where(bad12, 0.0, jnp.where(bad03, z[2]-z[1], a2)),
            jnp.where(bad12, z[1], jnp.where(bad03, z[1], a3))
        ])

    @staticmethod
    @jit
    def cubic_interp_eval_jax(data, f, t0, length, a):
        """Evaluate interpolated values for input points using coefficients.

        Parameters
        ----------
        data : array_like
            Input t-values to evaluate.
        f : float
            Scaling factor (1/dt).
        t0 : float
            Offset for scaling.
        length : int
            Length of coefficient array.
        a : array_like
            Coefficient array.

        Returns
        -------
        result : jax.numpy.ndarray
            Interpolated values.
        """
        x = jnp.clip(data * f + t0, 0.0, length - 1.0)
        ix = x.astype(int)
        x -= ix

        a0 = a[ix, 0]
        a1 = a[ix, 1]
        a2 = a[ix, 2]
        a3 = a[ix, 3]

        return (x * (x * (x * a0 + a1) + a2) + a3)

# --- BICUBIC INTERP ---


@jit
def cubic_eval(coeffs, x):
    return ((coeffs[..., 0] * x + coeffs[..., 1])
            * x + coeffs[..., 2]) * x + coeffs[..., 3]


@jit
def interpolate_1d(z):
    """Compute cubic interpolation coefficients for 1D array z.

    Parameters
    ----------
    z : array_like
        Input array of 4 samples.

    Returns
    -------
    coeffs : jax.numpy.ndarray
        Coefficients for cubic interpolation.
    """
    bad12 = nan_or_inf(z[1] + z[2])
    bad03 = nan_or_inf(z[0] + z[3])

    # Compute coefficients
    a0 = 1.5 * (z[1] - z[2]) + 0.5 * (z[3] - z[0])
    a1 = z[0] - 2.5 * z[1] + 2 * z[2] - 0.5 * z[3]
    a2 = 0.5 * (z[2] - z[0])
    a3 = z[1]

    # Initialize coefficients
    return jnp.stack([
        jnp.where(bad12, 0.0, jnp.where(bad03, 0.0, a0)),
        jnp.where(bad12, 0.0, jnp.where(bad03, 0.0, a1)),
        jnp.where(bad12, 0.0, jnp.where(bad03, z[2]-z[1], a2)),
        jnp.where(bad12, z[1], a3)
    ])


@partial(jit, static_argnames=['ns', 'nt'])
def compute_coeffs(data, ns, nt):
    """Compute bicubic interpolation coefficients from 2D gridded data.

    Parameters
    ----------
    data : array_like
        Flattened input array of shape (ns * nt,).
    ns : int
        Grid size in first dimension (rows).
    nt : int
        Grid size in second dimension (columns).

    Returns
    -------
    coeffs : jax.numpy.ndarray
        Array of shape ((ns+6)*(nt+6), 4, 4) with interpolation coefficients.
    """
    length_s = ns + 6
    length_t = nt + 6

    def get_z(js, iss, itt):
        ks = jnp.clip(iss + js - 4, 0, ns - 1)

        def get_zt(jt):
            kt = jnp.clip(itt + jt - 4, 0, nt - 1)
            return data[ks * nt + kt]

        return vmap(get_zt)(ARANGE4)

    def compute_block(iss, itt):
        a_rows = vmap(lambda js: interpolate_1d(get_z(js, iss, itt)))(ARANGE4)
        return vmap(interpolate_1d)(a_rows.T)

    blocks = vmap(lambda iss: vmap(lambda itt: compute_block(iss, itt))(
        jnp.arange(length_t)))(jnp.arange(length_s))
    return blocks.reshape(length_s * length_t, 4, 4)


class bicubic_interp:
    """2D bicubic interpolation using precomputed coefficients.

    Parameters
    ----------
    data : array_like
        Flattened input array of shape (ns * nt,).
    ns, nt : int
        Grid sizes in each dimension.
    smin, tmin : float
        Minimum values for s and t.
    ds, dt : float
        Step sizes in s and t dimensions.

    Attributes
    ----------
    fx : jax.numpy.ndarray
        Inverse step sizes.
    x0 : jax.numpy.ndarray
        Precomputed shift constants.
    xlength : jax.numpy.ndarray
        Array dimensions with 6-element padding.
    a : jax.numpy.ndarray
        Precomputed bicubic coefficient tensor.
    """

    def __init__(self, data, ns, nt, smin, tmin, ds, dt):
        self.fx = jnp.array([1/ds, 1/dt])
        self.x0 = jnp.array([3 - self.fx[0] * smin, 3 - self.fx[1] * tmin])
        self.xlength = jnp.array([ns + 6, nt + 6])
        self.a = compute_coeffs(data, ns, nt)

    @staticmethod
    @jit
    def bicubic_interp_eval_jax(s, t, fx, x0, xlength, a):
        """Evaluate bicubic interpolation at point(s) (s, t).

        Parameters
        ----------
        s, t : float or array_like
            Evaluation coordinates.
        fx : array_like
            Inverse step sizes in each dimension.
        x0 : array_like
            Offset constants.
        xlength : array_like
            Sizes of each dimension.
        a : array_like
            Flattened coefficients of shape (ns*nt, 4, 4).

        Returns
        -------
        result : float
            Interpolated value at the given point(s).
        """
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
