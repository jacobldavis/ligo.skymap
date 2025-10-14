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
from jax import jit

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
    Compute the complex antenna response F for each interferometer.
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
    nifos : int
        Number of detectors.
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


@partial(jit, static_argnames=["ntwopsi"])
def compute_twopsi_slice(F, snrs_interp, ntwopsi, rescale_loglikelihood):
    """
    Evaluate marginalized likelihood for all inclination angles at a fixed 2ψ.

    Parameters
    ----------
    F : array_like
        Antenna response.
    snrs_interp : array_like
        Interpolated SNR values.
    ntwopsi : int
        Total number of polarization angles.
    rescale_loglikelihood : float
        Normalization constant.

    Returns
    -------
    tuple
        Arrays of p, log(p), b values, and log(b) for each u.
    """
    twopsi_vals = (2 * jnp.pi / ntwopsi) * jnp.arange(ntwopsi)
    exp_i_twopsi_vals = jnp.exp(1j * twopsi_vals)

    u_vals = u_points_weights[:, 0]
    u2_vals = u_vals * u_vals

    tmp = F[None, None, :] * jnp.conj(exp_i_twopsi_vals[:, None, None])
    z_times_r = 0.5 * (1 + u2_vals[None, :, None]) * jnp.real(tmp) - 1j * u_vals[
        None, :, None
    ] * jnp.imag(tmp)

    p2 = jnp.sum(jnp.real(z_times_r) ** 2 + jnp.imag(z_times_r) ** 2, axis=2)
    p2 *= 0.5 * rescale_loglikelihood**2
    p = jnp.sqrt(p2)
    logp = jnp.log(p)

    I0arg = jnp.sum(
        jnp.conj(z_times_r[:, :, None, :]) * snrs_interp[None, None, :, :], axis=3
    )
    b = jnp.abs(I0arg) * rescale_loglikelihood**2
    logb = jnp.log(b)

    return p, logp, b, logb


@jit
def compute_accum(iint, accum):
    """
    Integrate over inclination and polarization to compute log-evidence.

    Parameters
    ----------
    iint : int
        Index of the integrator region (0, 1, or 2).
    accum : array_like
        Array of log-integrands.

    Returns
    -------
    float
        Log of total accumulated value.
    """
    accum_iint = accum[iint]
    accum_flat = accum_iint.reshape(-1)

    max_accum = jnp.max(accum_flat)

    accum_exp = jnp.exp(accum_flat - max_accum)
    accum1 = jnp.sum(accum_exp)

    return jnp.log(accum1) + max_accum


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
    # Initialize starting values
    theta, phi = uniq2ang64(uniq)

    # Look up antenna factors
    F = compute_F(responses, horizons, phi, theta, gmst, nifos)

    # Compute dt
    dt = toa_errors(theta, phi, gmst, locations, epochs)

    # Shift SNR time series by the time delay for this sky position
    snrs_interp = compute_snrs_interp(snrs, dt, sample_rate, nsamples)

    # Compute p and b values for all twopsi and u
    p, log_p, b, log_b = compute_twopsi_slice(
        F, snrs_interp, ntwopsi, rescale_loglikelihood
    )

    # Initialize accum with the integrator evaluation
    itwopsi_grid, iu_grid, isample_grid = jnp.meshgrid(
        jnp.arange(ntwopsi), jnp.arange(nu), jnp.arange(snrs.shape[1]), indexing="ij"
    )
    itwopsi_flat = itwopsi_grid.ravel()
    iu_flat = iu_grid.ravel()
    isample_flat = isample_grid.ravel()

    val = u_points_weights[iu_flat, 1] + integrator_eval(
        integrators[0][0][0],
        integrators[0][0][1],
        integrators[0][0][2],
        integrators[1][0],
        p[itwopsi_flat, iu_flat],
        b[itwopsi_flat, iu_flat, isample_flat],
        log_p[itwopsi_flat, iu_flat],
        log_b[itwopsi_flat, iu_flat, isample_flat],
    )
    accum_flat = jnp.where(jnp.isfinite(val), val, u_points_weights[iu_flat, 1])
    accum0 = accum_flat.reshape(ntwopsi, nu, snrs.shape[1])
    accum = jnp.array([accum0])

    # Return updated pixel row
    return px.at[1].set(compute_accum(0, accum))


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
    # Initialize starting values
    theta, phi = uniq2ang64(uniq)

    # Look up antenna factors
    F = compute_F(responses, horizons, phi, theta, gmst, nifos)

    # Compute dt
    dt = toa_errors(theta, phi, gmst, locations, epochs)

    # Shift SNR time series by the time delay for this sky position
    snrs_interp = compute_snrs_interp(snrs, dt, sample_rate, nsamples)

    # Compute p and b values for all twopsi and u
    p, log_p, b, log_b = compute_twopsi_slice(
        F, snrs_interp, ntwopsi, rescale_loglikelihood
    )

    # Initialize accum with the integrator evaluation
    itwopsi_grid, iu_grid, isample_grid = jnp.meshgrid(
        jnp.arange(ntwopsi), jnp.arange(nu), jnp.arange(snrs.shape[1]), indexing="ij"
    )
    itwopsi_flat = itwopsi_grid.ravel()
    iu_flat = iu_grid.ravel()
    isample_flat = isample_grid.ravel()

    val = u_points_weights[iu_flat, 1] + integrator_eval(
        integrators[0][1][0],
        integrators[0][1][1],
        integrators[0][1][2],
        integrators[1][1],
        p[itwopsi_flat, iu_flat],
        b[itwopsi_flat, iu_flat, isample_flat],
        log_p[itwopsi_flat, iu_flat],
        log_b[itwopsi_flat, iu_flat, isample_flat],
    )
    accum_flat = jnp.where(jnp.isfinite(val), val, u_points_weights[iu_flat, 1])
    accum1 = accum_flat.reshape(ntwopsi, nu, snrs.shape[1])

    val = u_points_weights[iu_flat, 1] + integrator_eval(
        integrators[0][2][0],
        integrators[0][2][1],
        integrators[0][2][2],
        integrators[1][2],
        p[itwopsi_flat, iu_flat],
        b[itwopsi_flat, iu_flat, isample_flat],
        log_p[itwopsi_flat, iu_flat],
        log_b[itwopsi_flat, iu_flat, isample_flat],
    )
    accum_flat = jnp.where(jnp.isfinite(val), val, u_points_weights[iu_flat, 1])
    accum2 = accum_flat.reshape(ntwopsi, nu, snrs.shape[1])
    accum = jnp.array([accum1, accum2])

    # Return updated pixel row
    px = px.at[2].set(compute_accum(0, accum))
    px = px.at[3].set(compute_accum(1, accum))
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
    # Initialize starting values
    theta, phi = uniq2ang64(uniq)

    # Look up antenna factors
    F = compute_F(responses, horizons, phi, theta, gmst, nifos)

    # Compute dt
    dt = toa_errors(theta, phi, gmst, locations, epochs)

    # Shift SNR time series by the time delay for this sky position
    snrs_interp = compute_snrs_interp(snrs, dt, sample_rate, nsamples)

    # Compute p and b values for all twopsi and u
    p, log_p, b, log_b = compute_twopsi_slice(
        F, snrs_interp, ntwopsi, rescale_loglikelihood
    )

    # Initialize accum with the integrator evaluation
    itwopsi_grid, iu_grid, isample_grid = jnp.meshgrid(
        jnp.arange(ntwopsi), jnp.arange(nu), jnp.arange(snrs.shape[1]), indexing="ij"
    )
    itwopsi_flat = itwopsi_grid.ravel()
    iu_flat = iu_grid.ravel()
    isample_flat = isample_grid.ravel()

    val = u_points_weights[iu_flat, 1] + integrator_eval(
        integrators[0][0][0],
        integrators[0][0][1],
        integrators[0][0][2],
        integrators[1][0],
        p[itwopsi_flat, iu_flat],
        b[itwopsi_flat, iu_flat, isample_flat],
        log_p[itwopsi_flat, iu_flat],
        log_b[itwopsi_flat, iu_flat, isample_flat],
    )
    accum_flat = jnp.where(jnp.isfinite(val), val, u_points_weights[iu_flat, 1])
    accum0 = accum_flat.reshape(ntwopsi, nu, snrs.shape[1])
    accum = jnp.array([accum0])

    # Return updated accum value
    return compute_accum(0, accum)
