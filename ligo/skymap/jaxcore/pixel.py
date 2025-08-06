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

from ligo.skymap.jaxcore.integrate import log_radial_integrator
from ligo.skymap.jaxcore.moc import M_PI_2, ang2vec, ntwopsi, nu, uniq2ang64


def extract_integrator_regions(integrators):
    """
    Extract region parameters from each log_radial_integrator.

    Parameters
    ----------
    integrators : list
        List of log_radial_integrator objects.

    Returns
    -------
    list of tuples
        Each tuple contains region0, region1, and region2 parameters.
    """
    result = []
    for integrator in integrators:
        result.append(
            (
                (
                    integrator.region0.fx,
                    integrator.region0.x0,
                    integrator.region0.xlength,
                    integrator.region0.a,
                ),
                (
                    integrator.region1.f,
                    integrator.region1.t0,
                    integrator.region1.length,
                    integrator.region1.a,
                ),
                (
                    integrator.region2.f,
                    integrator.region2.t0,
                    integrator.region2.length,
                    integrator.region2.a,
                ),
            )
        )
    return result


def extract_integrator_limits(integrators):
    """
    Extract the p0_limit, vmax, and ymax constants for each integrator.

    Parameters
    ----------
    integrators : list
        List of log_radial_integrator objects.

    Returns
    -------
    list of tuples
        Each tuple contains (p0_limit, vmax, ymax).
    """
    result = []
    for integrator in integrators:
        result.append((integrator.p0_limit, integrator.vmax, integrator.ymax))
    return result


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
def antenna_factor(D, ra, dec, gmst):
    """
    Compute antenna factors from the detector response tensor.

    Parameters
    ----------
    D : array_like
        Detector tensor (3,3).
    ra : float
        Right ascension.
    dec : float
        Declination.
    gmst : float
        Greenwich mean sidereal time.

    Returns
    -------
    complex
        Complex-valued antenna factor (F_plus + i F_cross).
    """
    gha = gmst - ra
    cosgha = jnp.cos(gha)
    singha = jnp.sin(gha)
    cosdec = jnp.cos(dec)
    sindec = jnp.sin(dec)
    X = jnp.array([-singha, -cosgha, 0.0])
    Y = jnp.array([-cosgha * sindec, singha * sindec, cosdec])
    F = 0

    def dxdy(i):
        DX = D[i][0] * X[0] + D[i][1] * X[1] + D[i][2] * X[2]
        DY = D[i][0] * Y[0] + D[i][1] * Y[1] + D[i][2] * Y[2]
        return (X[i] * DX - Y[i] * DY) + (X[i] * DY + Y[i] * DX) * 1j

    F = jnp.sum(vmap(dxdy)(jnp.arange(3)))
    return F


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

    Returns
    -------
    array
        Complex response factors for each detector.
    """

    def body(i):
        val = antenna_factor(responses[i], phi, M_PI_2 - theta, gmst)
        val *= horizons[i]
        return jnp.where(i < nifos, val, 0)

    return vmap(lambda i: body(i))(jnp.arange(responses.shape[0]))


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
def eval_snr(x, nsamples, t):
    """
    Evaluate interpolated complex SNR at fractional sample index.

    Parameters
    ----------
    x : array_like
        SNR values (magnitude, phase).
    nsamples : int
        Total number of SNR samples.
    t : float
        Time index to interpolate at.

    Returns
    -------
    complex
        Interpolated SNR value, or 0 outside valid range.
    """
    i = jnp.floor(t).astype(jnp.int32)
    f = jnp.float32(t - jnp.floor(t))
    cond = jnp.logical_and(i >= 1, i < nsamples - 2)

    mag = catrom(x[i - 1][0], x[i][0], x[i + 1][0], x[i + 2][0], f)
    phase = catrom(x[i - 1][1], x[i][1], x[i + 1][1], x[i + 2][1], f)

    val = mag * exp_i(phase)
    return jnp.where(cond, val, jnp.complex64(0.0 + 0.0j))


@jit
def bayestar_signal_amplitude_model(F, exp_i_twopsi, u, u2):
    """
    Compute the complex-valued signal amplitude model for a given inclination.

    Parameters
    ----------
    F : array_like
        Complex antenna factor.
    exp_i_twopsi : complex
        e^(i*2*psi), for polarization angle psi.
    u : float
        Cosine of the inclination angle.
    u2 : float
        Square of the cosine of inclination.

    Returns
    -------
    array_like
        Complex signal amplitude.
    """
    tmp = F * jnp.conj(exp_i_twopsi)
    return 0.5 * (1 + u2) * jnp.real(tmp) - 1j * u * jnp.imag(tmp)


@jit
def compute_samplewise_b(z_times_r, snrs_interp_sample, rescale_loglikelihood):
    """
    Compute the per-sample normalization constant b and its log.

    Parameters
    ----------
    z_times_r : array_like
        Complex signal template.
    snrs_interp_sample : array_like
        Interpolated SNR sample.
    rescale_loglikelihood : float
        Normalization factor for log-likelihood.

    Returns
    -------
    tuple
        b value and log(b).
    """
    I0arg = jnp.sum(jnp.conj(z_times_r) * snrs_interp_sample)
    b_val = jnp.abs(I0arg) * rescale_loglikelihood**2
    return b_val, jnp.log(b_val)


@jit
def compute_u_point(F, exp_i_twopsi, u, snrs_interp, rescale_loglikelihood):
    """
    Compute likelihood quantities for a specific inclination cosine u.

    Parameters
    ----------
    F : array_like
        Antenna response.
    exp_i_twopsi : complex
        e^(i*2*psi), for polarization angle psi.
    u : float
        Cosine of inclination.
    snrs_interp : array_like
        Interpolated SNR time series.
    rescale_loglikelihood : float
        Likelihood normalization factor.

    Returns
    -------
    tuple
        Marginalized likelihood values.
    """
    u2 = u * u

    z_times_r = vmap(
        lambda Fi: bayestar_signal_amplitude_model(Fi, exp_i_twopsi, u, u2)
    )(F)
    p2 = jnp.sum(jnp.real(z_times_r) ** 2 + jnp.imag(z_times_r) ** 2)
    p2 *= 0.5 * rescale_loglikelihood**2
    p = jnp.sqrt(p2)
    logp = jnp.log(p)

    b_vals, log_b_vals = vmap(
        lambda snr: compute_samplewise_b(z_times_r, snr, rescale_loglikelihood)
    )(snrs_interp)
    return p, logp, b_vals, log_b_vals


@partial(jit, static_argnames=["ntwopsi"])
def compute_twopsi_slice(F, snrs_interp, itwopsi, ntwopsi, rescale_loglikelihood):
    """
    Evaluate marginalized likelihood for all inclination angles at a fixed 2ψ.

    Parameters
    ----------
    F : array_like
        Antenna response.
    snrs_interp : array_like
        Interpolated SNR values.
    itwopsi : int
        Index of current polarization angle.
    ntwopsi : int
        Total number of polarization angles.
    rescale_loglikelihood : float
        Normalization constant.

    Returns
    -------
    tuple
        Arrays of p, log(p), b values, and log(b) for each u.
    """
    twopsi = (2 * jnp.pi / ntwopsi) * itwopsi
    exp_i_twopsi = exp_i(twopsi)

    def process_u(iu):
        u = u_points_weights[iu, 0]
        return compute_u_point(F, exp_i_twopsi, u, snrs_interp, rescale_loglikelihood)

    return vmap(process_u)(jnp.arange(u_points_weights.shape[0]))


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
def safe_eval(
    integrator_tuple,
    integrator_b,
    itwopsi,
    iu,
    isample,
    u_points_weights,
    p,
    b,
    log_p,
    log_b,
):
    val = u_points_weights[iu][1] + log_radial_integrator.integrator_eval(
        integrator_tuple[0],
        integrator_tuple[1],
        integrator_tuple[2],
        integrator_b,
        p[itwopsi][iu],
        b[itwopsi][iu][isample],
        log_p[itwopsi][iu],
        log_b[itwopsi][iu][isample],
    )
    return jnp.where(jnp.isfinite(val), val, u_points_weights[iu][1])


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
    flag : int
        Controls where results are stored (see below).
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
    snrs_interp = vmap(
        lambda isample: vmap(lambda x, tt: eval_snr(x, nsamples, tt))(
            snrs, isample - dt * sample_rate - 0.5 * (nsamples - 1)
        )
    )(jnp.arange(snrs.shape[1]))

    # Perform bayestar_signal_amplitude_model
    p, log_p, b, log_b = vmap(
        lambda itwopsi: compute_twopsi_slice(
            F, snrs_interp, itwopsi, ntwopsi, rescale_loglikelihood
        )
    )(jnp.arange(ntwopsi))

    # Initialize accum with the integrator evaluation
    accum0 = vmap(
        lambda itwopsi: vmap(
            lambda iu: vmap(
                lambda isample: safe_eval(
                    integrators[0][0],
                    integrators[1][0],
                    itwopsi,
                    iu,
                    isample,
                    u_points_weights,
                    p,
                    b,
                    log_p,
                    log_b,
                )
            )(jnp.arange(snrs.shape[1]))
        )(jnp.arange(nu))
    )(jnp.arange(ntwopsi))
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
    flag : int
        Controls where results are stored (see below).
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
    snrs_interp = vmap(
        lambda isample: vmap(lambda x, tt: eval_snr(x, nsamples, tt))(
            snrs, isample - dt * sample_rate - 0.5 * (nsamples - 1)
        )
    )(jnp.arange(snrs.shape[1]))

    # Perform bayestar_signal_amplitude_model
    p, log_p, b, log_b = vmap(
        lambda itwopsi: compute_twopsi_slice(
            F, snrs_interp, itwopsi, ntwopsi, rescale_loglikelihood
        )
    )(jnp.arange(ntwopsi))

    # Initialize accum with the integrator evaluation
    accum1 = vmap(
        lambda itwopsi: vmap(
            lambda iu: vmap(
                lambda isample: safe_eval(
                    integrators[0][1],
                    integrators[1][1],
                    itwopsi,
                    iu,
                    isample,
                    u_points_weights,
                    p,
                    b,
                    log_p,
                    log_b,
                )
            )(jnp.arange(snrs.shape[1]))
        )(jnp.arange(nu))
    )(jnp.arange(ntwopsi))
    accum2 = vmap(
        lambda itwopsi: vmap(
            lambda iu: vmap(
                lambda isample: safe_eval(
                    integrators[0][2],
                    integrators[1][2],
                    itwopsi,
                    iu,
                    isample,
                    u_points_weights,
                    p,
                    b,
                    log_p,
                    log_b,
                )
            )(jnp.arange(snrs.shape[1]))
        )(jnp.arange(nu))
    )(jnp.arange(ntwopsi))
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
    flag : int
        Controls where results are stored (see below).
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
    # Initialize starting values
    theta, phi = uniq2ang64(uniq)

    # Look up antenna factors
    F = compute_F(responses, horizons, phi, theta, gmst, nifos)

    # Compute dt
    dt = toa_errors(theta, phi, gmst, locations, epochs)

    # Shift SNR time series by the time delay for this sky position
    snrs_interp = vmap(
        lambda isample: vmap(lambda x, tt: eval_snr(x, nsamples, tt))(
            snrs, isample - dt * sample_rate - 0.5 * (nsamples - 1)
        )
    )(jnp.arange(snrs.shape[1]))

    # Perform bayestar_signal_amplitude_model
    p, log_p, b, log_b = vmap(
        lambda itwopsi: compute_twopsi_slice(
            F, snrs_interp, itwopsi, ntwopsi, rescale_loglikelihood
        )
    )(jnp.arange(ntwopsi))

    # Initialize accum with the integrator evaluation
    accum0 = vmap(
        lambda itwopsi: vmap(
            lambda iu: vmap(
                lambda isample: safe_eval(
                    integrators[0][0],
                    integrators[1][0],
                    itwopsi,
                    iu,
                    isample,
                    u_points_weights,
                    p,
                    b,
                    log_p,
                    log_b,
                )
            )(jnp.arange(snrs.shape[1]))
        )(jnp.arange(nu))
    )(jnp.arange(ntwopsi))
    accum = jnp.array([accum0])

    # Return updated accum value
    return compute_accum(0, accum)
