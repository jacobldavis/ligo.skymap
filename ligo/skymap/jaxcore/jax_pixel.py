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
from jax import jit, vmap, lax
import jax
import jax.numpy as jnp
from functools import partial
from ligo.skymap.jaxcore.jax_interp import *
from ligo.skymap.jaxcore.jax_integrate import *
from ligo.skymap.jaxcore.jax_cosmology import *
from ligo.skymap.jaxcore.jax_moc import *

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
        result.append((
            (integrator.region0.fx, integrator.region0.x0, integrator.region0.xlength, integrator.region0.a),
            (integrator.region1.f, integrator.region1.t0, integrator.region1.length, integrator.region1.a),
            (integrator.region2.f, integrator.region2.t0, integrator.region2.length, integrator.region2.a),
        ))
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
        Array of shape (nu, 2) where [:,0] are the u-values and [:,1] are log(weights).
    """
    points, weights = np.polynomial.legendre.leggauss(nu)

    u_points_weights = np.column_stack((points, np.log(weights)))
    return jnp.array(u_points_weights)

u_points_weights = u_points_weights_init(nu)

@jit
def antenna_factor(D, ra, dec, gmst):
    """
    Compute antenna factors from the detector response tensor and source sky location.

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
def compute_F(responses, horizons, phi, theta, gmst):
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
    nifos = responses.shape[0]

    def body(i):
        val = antenna_factor(lax.dynamic_slice(responses, (i, 0, 0), (1, responses.shape[1], responses.shape[2])), phi, M_PI_2 - theta, gmst) * lax.dynamic_slice(horizons, (i,), (1,)) 
        return val[0]

    return vmap(lambda i: body(i))(jnp.arange(nifos))

@jit
def catrom(x0, x1, x2, x3, t):
    return x1 + t*(-0.5*x0 + 0.5*x2 + t*(x0 - 2.5*x1 + 2.0*x2 - 0.5*x3 + t*(-0.5*x0 + 1.5*x1 - 1.5*x2 + 0.5*x3)))

@jit 
def exp_i(phi):
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
    f = t - i
    return jnp.where(jnp.logical_and(i >= 1, i < nsamples - 2), catrom(x[i-1][0], x[i][0], x[i+1][0], x[i+2][0], f) * exp_i(
            catrom(x[i-1][1], x[i][1], x[i+1][1], x[i+2][1], f)), 0)

@jit
def toa_errors(theta, phi, gmst, nifos, locs, toas):
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
        Marginalized likelihood p, log(p), samplewise b values and log(b) values.
    """
    u2 = u * u

    z_times_r = vmap(lambda Fi: bayestar_signal_amplitude_model(Fi, exp_i_twopsi, u, u2))(F)
    p2 = jnp.sum(jnp.real(z_times_r)**2 + jnp.imag(z_times_r)**2)
    p2 *= 0.5 * rescale_loglikelihood**2
    p = jnp.sqrt(p2)
    logp = jnp.log(p)

    b_vals, log_b_vals = vmap(lambda snr: compute_samplewise_b(z_times_r, snr, rescale_loglikelihood))(snrs_interp)
    return p, logp, b_vals, log_b_vals

@jit
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
def compute_accum(iint, snrs, accum):
    """
    Integrate over inclination and polarization to compute log-evidence.

    Parameters
    ----------
    iint : int
        Index of the integrator region (0, 1, or 2).
    snrs : array_like
        Original SNR data.
    accum : array_like
        Array of log-integrands.

    Returns
    -------
    float
        Log of total accumulated value.
    """
    nsamples = snrs.shape[1]
    max_accum = jnp.max(vmap(lambda itwopsi: vmap(lambda iu: vmap(lambda isample: accum[iint][itwopsi][iu][isample])(jnp.arange(nsamples)))(jnp.arange(nu)))(jnp.arange(ntwopsi)))
    accum1 = jnp.sum(vmap(lambda itwopsi: vmap(lambda iu: vmap(lambda isample: jnp.exp(accum[iint][itwopsi][iu][isample] - max_accum))(jnp.arange(nsamples)))(jnp.arange(nu)))(jnp.arange(ntwopsi)))
    return jnp.log(accum1) + max_accum

@jit
def bsm_pixel_jax(integrators, flag, i, iifo, pixels, gmst, nifos, nsamples, sample_rate, epochs, snrs, responses, locations, horizons, rescale_loglikelihood):
    """
    Compute likelihood integrals over inclination and polarization for one HEALPix pixel.

    Parameters
    ----------
    integrators : list
        Tupled radial integrators (region functions and limits).
    flag : int
        Controls where results are stored (see below).
    i : int
        Index of the pixel in the array.
    iifo : int
        Detector index.
    pixels : array_like
        Pixel data array to modify.
    gmst : float
        Greenwich Mean Sidereal Time.
    nifos : int
        Number of interferometers.
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
        Modified pixels array.

    Notes
    -----
    If flag == 1: store result in pixels[i][1]
    If flag == 2: store result in pixels[i][iifo]
    If flag == 3: store results in pixels[i][2] and pixels[i][3]
    """
    # Initialize starting values
    uniq = pixels[i, 0]
    theta, phi = uniq2ang64(uniq)
    
    # Look up antenna factors
    F = compute_F(responses, horizons, phi, theta, gmst)

    # Compute dt
    dt = toa_errors(theta, phi, gmst, nifos, locations, epochs)

    # Shift SNR time series by the time delay for this sky position
    @jit
    def snr_row(isample):
        t = isample - dt * sample_rate - 0.5 * (nsamples - 1) 
        return vmap(lambda x, tt: eval_snr(x, nsamples, tt))(snrs, t)  
    snrs_interp = vmap(snr_row)(jnp.arange(snrs.shape[1]))

    # Perform bayestar_singal_amplitude_model
    @jit
    def process_twopsi(itwopsi):
        p_u, log_p_u, b_u, log_b_u = compute_twopsi_slice(
            F, snrs_interp, itwopsi, ntwopsi, rescale_loglikelihood)
        return p_u, log_p_u, b_u, log_b_u
    p, log_p, b, log_b = vmap(process_twopsi)(jnp.arange(ntwopsi))
    
    # Initialize accum with the integrator evaluation
    @jit
    def safe_eval(integrator_tuple, integrator_b, itwopsi, iu, isample):
        val = u_points_weights[iu][1] + log_radial_integrator.log_radial_integrator_eval(
            integrator_tuple[0], integrator_tuple[1], integrator_tuple[2],
            integrator_b, p[itwopsi][iu], b[itwopsi][iu][isample],
            log_p[itwopsi][iu], log_b[itwopsi][iu][isample])
        return jnp.where(jnp.isfinite(val), val, u_points_weights[iu][1])

    accum0 = jnp.where(flag != 3, 
                    vmap(lambda itwopsi: vmap(lambda iu: vmap(lambda isample: 
                        safe_eval(integrators[0][0], integrators[1][0], itwopsi, iu, isample))
                    (jnp.arange(snrs.shape[1])))(jnp.arange(nu)))(jnp.arange(ntwopsi)), 
                    jnp.array([0]))
    accum1 = jnp.where(flag == 3, 
                    vmap(lambda itwopsi: vmap(lambda iu: vmap(lambda isample: 
                        safe_eval(integrators[0][1], integrators[1][1], itwopsi, iu, isample))
                    (jnp.arange(snrs.shape[1])))(jnp.arange(nu)))(jnp.arange(ntwopsi)), 
                    jnp.array([0]))
    accum2 = jnp.where(flag == 3, 
                    vmap(lambda itwopsi: vmap(lambda iu: vmap(lambda isample: 
                        safe_eval(integrators[0][2], integrators[1][2], itwopsi, iu, isample))
                    (jnp.arange(snrs.shape[1])))(jnp.arange(nu)))(jnp.arange(ntwopsi)), 
                    jnp.array([0]))
    accum = jnp.array([accum0, accum1, accum2])

    # Compute the final value with max_accum and accum1
    # Flag 1: Change the value at pixels[i][1]
    # Flag 2: Change the value at accum[i][iifo]
    # Flag 3: Change the values at pixels[i][2] and pixels[i][3]
    # NOTE: JAX arrays are immutable, meaning you must reassign the whole array to change a value, 
    # so this is my current solution for modifying the value array
    pixels = lax.cond(
        flag == 1,
        lambda px: px.at[i, 1].set(compute_accum(0, snrs, accum)),
        lambda px: px,
        pixels
    )
    pixels = lax.cond(
        flag == 2,
        lambda px: px.at[i, iifo].set(compute_accum(0, snrs, accum)),
        lambda px: px,
        pixels
    )
    pixels = lax.cond(
        flag == 3,
        lambda px: px.at[i, 2].set(compute_accum(1, snrs, accum)),
        lambda px: px,
        pixels
    )
    pixels = lax.cond(
        flag == 3,
        lambda px: px.at[i, 3].set(compute_accum(2, snrs, accum)),
        lambda px: px,
        pixels
    )
    return pixels

# --- TEST SUITE ---

def test_eval_snr():
    nsamples = 64
    x = np.zeros((nsamples, 2))

    # Populate data with samples of x(t) = t^2 * exp(i * t)
    for i in range(nsamples):
        x[i][0] = i ** 2      # real part
        x[i][1] = i           # imaginary part

    for t in np.arange(0, nsamples + 0.1, 0.1):
        result = eval_snr(x, nsamples, t)
        expected = t**2 * exp_i(t) if 1 < t < nsamples - 2 else 0

        print(result)
        print(expected)

def test_pixels_tracer():
    order0 = 4
    npix0 = 4000
    pixels = vmap(lambda ipix: jnp.concatenate([
            jnp.array([nest2uniq64(order0, ipix)]),
            jnp.zeros(3)
        ]))(jnp.arange(npix0))
    print(pixels)

    theta, phi = uniq2ang64(pixels[1690,0])
    jax.debug.print("theta: {}, phi: {}", theta, phi)

    def test_bug(pixels):
        def body(i, _):
            theta, phi = uniq2ang64(lax.dynamic_slice(pixels, (i, 0), (1, 1))[0, 0])
            jax.debug.print("i={}, uniq={}, theta={}, phi={}", i, pixels[i,0], theta, phi)
            return _
        return lax.fori_loop(0, pixels.shape[0], body, None)

    test_bug(pixels)