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

from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import jit, vmap

from ligo.skymap.jaxcore.integrate import integrator_eval
from ligo.skymap.jaxcore.moc import ang2vec, ntwopsi, nu, uniq2ang64


@partial(jit, static_argnames=["nu"])
def u_points_weights_init(nu):
    """
    Compute Gaussian quadrature nodes and log-weights for u integration.

    Parameters
    ----------
    nu : int
        Number of quadrature nodes.

    Returns
    -------
    jnp.ndarray
        Array of shape (nu, 2)
    """
    points, weights = np.polynomial.legendre.leggauss(nu)

    u_points_weights = np.column_stack((points, np.log(weights)))
    return jnp.array(u_points_weights)


u_points_weights = u_points_weights_init(nu)


@jit
def compute_F(responses, horizons, phi, theta, gmst, nifos):
    """
    Compute antenna factors from the detector response tensor and source
    sky location, and return as complex number(s) F_plus + i F_cross.

    Parameters
    ----------
    responses : array_like
        Tensor responses of shape (nifo, 3, 3).
    horizons : array_like
        Detector horizon distances.
    phi : float
        Source azimuthal angle.
    theta : float
        Source polar angle.
    gmst : float
        Greenwich mean sidereal time.
    nifos : int
        Number of detectors.
    Returns
    -------
    array
        Complex response factors for each detector.
    """
    dec = jnp.pi / 2 - theta
    gha = gmst - phi
    cosgha = jnp.cos(gha)
    singha = jnp.sin(gha)
    cosdec = jnp.cos(dec)
    sindec = jnp.sin(dec)

    X = jnp.array([-singha, -cosgha, 0.0])
    Y = jnp.array([-cosgha * sindec, singha * sindec, cosdec])

    DX = jnp.sum(responses * X[None, None, :], axis=2)
    DY = jnp.sum(responses * Y[None, None, :], axis=2)

    vals = jnp.sum(
        (X[None, :] * DX - Y[None, :] * DY) + (X[None, :] * DY + Y[None, :] * DX) * 1j,
        axis=1,
    )

    vals *= horizons

    mask = jnp.arange(responses.shape[0]) < nifos
    return jnp.where(mask, vals, 0)


@jit
def toa_errors(theta, phi, gmst, locs, toas):
    """
    Compute time-of-arrival errors for a given sky location.

    Parameters
    ----------
    theta, phi : float
        Sky coordinates.
    gmst : float
        Greenwich mean sidereal time.
    locs : array_like
        Detector locations.
    toas : array_like
        Nominal arrival times.

    Returns
    -------
    array
        Time delay offsets.
    """
    n = ang2vec(theta, phi - gmst)
    dot = jnp.dot(locs, n)
    dt = toas + dot
    return dt


@jit
def catrom(x0, x1, x2, x3, t):
    return x1 + t * (
        -0.5 * x0
        + 0.5 * x2
        + t
        * (
            x0
            - 2.5 * x1
            + 2.0 * x2
            - 0.5 * x3
            + t * (-0.5 * x0 + 1.5 * x1 - 1.5 * x2 + 0.5 * x3)
        )
    )


@jit
def exp_i(phi):
    phi = jnp.float32(phi)
    return jnp.cos(phi) + 1j * jnp.sin(phi)


@jit
def compute_snrs_interp(snrs, dt, sample_rate, nsamples):
    """
    Evaluate interpolated complex SNR.
    Parameters
    ----------
    snrs : array_like
        Complex-valued SNR time series, shape (nifo, nsamples).
    dt : array_like
        Time delays for each detector, shape (nifo,).
    sample_rate : float
        Sampling rate in Hz.
    nsamples : int
        Number of samples in the SNR time series.
    Returns
    -------
    complex
        Interpolated SNR value, or 0 outside valid range.
    """
    isamples = jnp.arange(snrs.shape[1])

    tt = isamples[None, :] - dt[:, None] * sample_rate - 0.5 * (nsamples - 1)

    i = jnp.floor(tt).astype(jnp.int32)
    f = jnp.float32(tt - jnp.floor(tt))
    cond = jnp.logical_and(i >= 1, i < nsamples - 2)

    n_detectors = snrs.shape[0]
    detector_indices = jnp.arange(n_detectors)[:, None]
    x_im1 = snrs[detector_indices, i - 1]
    x_i = snrs[detector_indices, i]
    x_ip1 = snrs[detector_indices, i + 1]
    x_ip2 = snrs[detector_indices, i + 2]

    mag = catrom(x_im1[..., 0], x_i[..., 0], x_ip1[..., 0], x_ip2[..., 0], f)
    phase = catrom(x_im1[..., 1], x_i[..., 1], x_ip1[..., 1], x_ip2[..., 1], f)
    val = mag * exp_i(phase)

    result = jnp.where(cond, val, jnp.complex64(0.0 + 0.0j))
    return result.T


@jit
def compute_accum_logsumexp(accum_flat):
    """
    Compute log-sum-exp.

    Parameters
    ----------
    accum_flat : array_like
        Array of log-likelihood values to sum.

    Returns
    -------
    float
        log(sum(exp(accum_flat)))
    """
    max_val = jnp.max(accum_flat)
    return jnp.log(jnp.sum(jnp.exp(accum_flat - max_val))) + max_val


@partial(jit, static_argnames=["integrator_idx", "ntwopsi"])
def compute_pixel_core(
    integrators,
    F,
    snrs_interp,
    integrator_idx,
    ntwopsi,
    rescale_loglikelihood,
):
    """
     Compute likelihood for a single pixel.

     Parameters
     ----------
     integrators : list
         Tupled radial integrators (region functions and limits).
     F : array_like
         Complex antenna response factors for each detector, shape (nifo,).
     snrs_interp : array_like
         Interpolated SNR values at this sky position, shape (nsamples, nifo).s
     responses : array_like
         Detector tensor responses.
    integrator_idx : int
         Which integrator to use (0 for probability, 1-2 for distance moments).
     ntwopsi : int
         Number of polarization angles (typically 32).
     rescale_loglikelihood : float
         Factor for log-likelihood normalization.

     Returns
     -------
     array_like
         Log of the marginalized likelihood for this pixel.

    """
    twopsi_vals = (2 * jnp.pi / ntwopsi) * jnp.arange(ntwopsi)
    exp_i_twopsi_vals = jnp.exp(1j * twopsi_vals)

    u_vals = u_points_weights[:, 0]
    u2_vals = u_vals * u_vals
    u_log_weights = u_points_weights[:, 1]

    n_samples = snrs_interp.shape[0]

    # Pre-compute antenna-response for all twopsi values
    tmp = F[None, :] * jnp.conj(exp_i_twopsi_vals[:, None])

    integrator_funcs = integrators[0][integrator_idx]
    integrator_limits = integrators[1][integrator_idx]

    # Compute contribution for one (twopsi, u) pair across all samples
    def compute_for_twopsi_u(twopsi_idx, u_idx):
        """Compute contribution for one (twopsi, u) pair across all samples."""
        u = u_vals[u_idx]
        u2 = u2_vals[u_idx]

        z_times_r = 0.5 * (1 + u2) * jnp.real(tmp[twopsi_idx]) - 1j * u * jnp.imag(
            tmp[twopsi_idx]
        )

        p2 = jnp.sum(jnp.real(z_times_r) ** 2 + jnp.imag(z_times_r) ** 2)
        p2 *= 0.5 * rescale_loglikelihood**2
        p = jnp.sqrt(p2)
        logp = jnp.log(p)

        I0arg = jnp.sum(jnp.conj(z_times_r)[None, :] * snrs_interp, axis=1)
        b = jnp.abs(I0arg) * rescale_loglikelihood**2
        logb = jnp.log(b)

        p_broadcast = jnp.full(n_samples, p)
        logp_broadcast = jnp.full(n_samples, logp)

        # Evaluate integrator
        val = integrator_eval(
            integrator_funcs[0],
            integrator_funcs[1],
            integrator_funcs[2],
            integrator_limits,
            p_broadcast,
            b,
            logp_broadcast,
            logb,
        )

        # Add quadrature weight
        result = jnp.where(
            jnp.isfinite(val), val + u_log_weights[u_idx], u_log_weights[u_idx]
        )

        return compute_accum_logsumexp(result)

    # Vectorize over u and twopsi
    compute_u = vmap(lambda u_idx: compute_for_twopsi_u(0, u_idx))
    compute_twopsi_u = vmap(lambda twopsi_idx: compute_u(jnp.arange(nu)), in_axes=(0,))

    # Return final accumulated log-sum-exp
    accum_reduced = compute_twopsi_u(jnp.arange(ntwopsi))
    return compute_accum_logsumexp(accum_reduced.ravel())


@jit
def bsm_pixel_prob_jax(
    integrators,
    uniq,
    px,
    gmst,
    nifos,
    nsamples,
    sample_rate,
    epochs,
    snrs,
    responses,
    locations,
    horizons,
    rescale_loglikelihood,
):
    """
    Compute likelihood integrals over inclination and polarization.

    Parameters
    ----------
    integrators : list
        Tupled radial integrators (region functions and limits).
    uniq : int
        Uniq id of the pixel in the array.
    px : array_like
        Pixel row to modify.
    gmst : float
        Greenwich Mean Sidereal Time.
    nsamples : int
        Number of SNR time samples.
    nifos : int
        Number of detectors.
    sample_rate : float
        Sampling rate in Hz.
    epochs : array_like
        Arrival time offsets per detector.
    snrs : array_like
        Complex-valued SNR time series.
    responses : array_like
        Detector tensor responses.
    locations : array_like
        Detector locations.
    horizons : array_like
        Horizon distances.
    rescale_loglikelihood : float
        Factor for log-likelihood normalization.

    Returns
    -------
    array_like
        Pixels row with updated probability.

    """
    theta, phi = uniq2ang64(uniq)
    F = compute_F(responses, horizons, phi, theta, gmst, nifos)
    dt = toa_errors(theta, phi, gmst, locations, epochs)
    snrs_interp = compute_snrs_interp(snrs, dt, sample_rate, nsamples)

    result = compute_pixel_core(
        integrators, F, snrs_interp, 0, ntwopsi, rescale_loglikelihood
    )

    return px.at[1].set(result)


@jit
def bsm_pixel_dist_jax(
    integrators,
    uniq,
    px,
    gmst,
    nifos,
    nsamples,
    sample_rate,
    epochs,
    snrs,
    responses,
    locations,
    horizons,
    rescale_loglikelihood,
):
    """
    Compute likelihood integrals over inclination and polarization.

    Parameters
    ----------
    integrators : list
        Tupled radial integrators (region functions and limits).
    uniq : int
        Uniq id of the pixel in the array.
    px : array_like
        Pixel row to modify.
    gmst : float
        Greenwich Mean Sidereal Time.
    nifos : int
        Number of detectors.
    nsamples : int
        Number of SNR time samples.
    sample_rate : float
        Sampling rate in Hz.
    epochs : array_like
        Arrival time offsets per detector.
    snrs : array_like
        Complex-valued SNR time series.
    responses : array_like
        Detector tensor responses.
    locations : array_like
        Detector locations.
    horizons : array_like
        Horizon distances.
    rescale_loglikelihood : float
        Factor for log-likelihood normalization.

    Returns
    -------
    array_like
        Pixels row with updated distances.

    """
    theta, phi = uniq2ang64(uniq)
    F = compute_F(responses, horizons, phi, theta, gmst, nifos)
    dt = toa_errors(theta, phi, gmst, locations, epochs)
    snrs_interp = compute_snrs_interp(snrs, dt, sample_rate, nsamples)

    result1 = compute_pixel_core(
        integrators, F, snrs_interp, 1, ntwopsi, rescale_loglikelihood
    )
    result2 = compute_pixel_core(
        integrators, F, snrs_interp, 2, ntwopsi, rescale_loglikelihood
    )

    px = px.at[2].set(result1)
    px = px.at[3].set(result2)
    return px


@jit
def bsm_pixel_accum_jax(
    integrators,
    uniq,
    gmst,
    nifos,
    nsamples,
    sample_rate,
    epochs,
    snrs,
    responses,
    locations,
    horizons,
    rescale_loglikelihood,
):
    """
    Compute likelihood integrals over inclination and polarization.

    Parameters
    ----------
    integrators : list
        Tupled radial integrators (region functions and limits).
    uniq : int
        Uniq id of the pixel in the array.
    iifo : int
        Detector index.
    px : array_like
        Pixel row to modify.
    gmst : float
        Greenwich Mean Sidereal Time.
    nifos : int
        Number of detectors.
    nsamples : int
        Number of SNR time samples.
    sample_rate : float
        Sampling rate in Hz.
    epochs : array_like
        Arrival time offsets per detector.
    snrs : array_like
        Complex-valued SNR time series.
    responses : array_like
        Detector tensor responses.
    locations : array_like
        Detector locations.
    horizons : array_like
        Horizon distances.
    rescale_loglikelihood : float
        Factor for log-likelihood normalization.

    Returns
    -------
    Scalar value for accum calculation

    """
    theta, phi = uniq2ang64(uniq)
    F = compute_F(responses, horizons, phi, theta, gmst, nifos)
    dt = toa_errors(theta, phi, gmst, locations, epochs)
    snrs_interp = compute_snrs_interp(snrs, dt, sample_rate, nsamples)

    return compute_pixel_core(
        integrators, F, snrs_interp, 0, ntwopsi, rescale_loglikelihood
    )
