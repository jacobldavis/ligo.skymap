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
    result = []
    for integrator in integrators:
        result.append((
            (integrator.region0.fx, integrator.region0.x0, integrator.region0.xlength, integrator.region0.a),
            (integrator.region1.f, integrator.region1.t0, integrator.region1.length, integrator.region1.a),
            (integrator.region2.f, integrator.region2.t0, integrator.region2.length, integrator.region2.a),
        ))
    return result 

def extract_integrator_limits(integrators):
    result = []
    for integrator in integrators:
        result.append((integrator.p0_limit, integrator.vmax, integrator.ymax))
    return result

def u_points_weights_init(nu):
    points, weights = np.polynomial.legendre.leggauss(nu)

    u_points_weights = np.column_stack((points, np.log(weights)))
    return jnp.array(u_points_weights)

u_points_weights = u_points_weights_init(nu)

@jit
def antenna_factor(D, ra, dec, gmst):
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

def compute_F(responses, horizons, phi, theta, gmst):
    nifos = responses.shape[0]
    F_init = jnp.zeros(nifos, dtype=jnp.complex64)

    def body(i, F):
        val = antenna_factor(responses[i], phi, M_PI_2 - theta, gmst) * horizons[i]
        return F.at[i].set(val)

    return lax.fori_loop(0, nifos, body, F_init)

@jit
def catrom(x0, x1, x2, x3, t):
    return x1 + t*(-0.5*x0 + 0.5*x2 + t*(x0 - 2.5*x1 + 2.0*x2 - 0.5*x3 + t*(-0.5*x0 + 1.5*x1 - 1.5*x2 + 0.5*x3)))

@jit 
def exp_i(phi):
    return jnp.cos(phi) + 1j * jnp.sin(phi)

@jit 
def eval_snr(x, nsamples, t):
    i = jnp.floor(t).astype(jnp.int32)
    f = t - i
    return jnp.where(jnp.logical_and(i >= 1, i < nsamples - 2), catrom(x[i-1][0], x[i][0], x[i+1][0], x[i+2][0], f) * exp_i(
            catrom(x[i-1][1], x[i][1], x[i+1][1], x[i+2][1], f)), 0)

@jit
def toa_errors(theta, phi, gmst, nifos, locs, toas):
    n = ang2vec(theta, phi - gmst)
    dot = jnp.dot(locs, n)        
    dt = toas + dot             
    return dt

@jit
def bayestar_signal_amplitude_model(F, exp_i_twopsi, u, u2):
    tmp = F * jnp.conj(exp_i_twopsi)
    return 0.5 * (1 + u2) * jnp.real(tmp) - 1j * u * jnp.imag(tmp)

@jit
def compute_samplewise_b(z_times_r, snrs_interp_sample, rescale_loglikelihood):
    I0arg = jnp.sum(jnp.conj(z_times_r) * snrs_interp_sample)
    b_val = jnp.abs(I0arg) * rescale_loglikelihood**2
    return b_val, jnp.log(b_val)

@jit
def compute_u_point(F, exp_i_twopsi, u, snrs_interp, rescale_loglikelihood):
    u2 = u * u

    z_times_r = vmap(lambda Fi: bayestar_signal_amplitude_model(Fi, exp_i_twopsi, u, u2))(F)
    p2 = jnp.sum(jnp.abs(z_times_r) ** 2)
    p2 *= 0.5 * rescale_loglikelihood**2
    p = jnp.sqrt(p2)
    logp = jnp.log(p)

    b_vals, log_b_vals = vmap(lambda snr: compute_samplewise_b(z_times_r, snr, rescale_loglikelihood))(snrs_interp)
    return p, logp, b_vals, log_b_vals

@jit
def compute_twopsi_slice(F, snrs_interp, itwopsi, ntwopsi, rescale_loglikelihood):
    twopsi = (2 * jnp.pi / ntwopsi) * itwopsi
    exp_i_twopsi = exp_i(twopsi)

    def process_u(iu):
        u = u_points_weights[iu, 0]
        return compute_u_point(F, exp_i_twopsi, u, snrs_interp, rescale_loglikelihood)

    return vmap(process_u)(jnp.arange(u_points_weights.shape[0]))

@jit
def compute_accum(iint, snrs, accum):
    nsamples = snrs.shape[1]
    max_accum = jnp.max(vmap(lambda itwopsi: vmap(lambda iu: vmap(lambda isample: accum[iint][itwopsi][iu][isample])(jnp.arange(nsamples)))(jnp.arange(nu)))(jnp.arange(ntwopsi)))
    accum1 = jnp.sum(vmap(lambda itwopsi: vmap(lambda iu: vmap(lambda isample: jnp.exp(accum[iint][itwopsi][iu][isample] - max_accum))(jnp.arange(nsamples)))(jnp.arange(nu)))(jnp.arange(ntwopsi)))
    return jnp.log(accum1) + max_accum

@jit
def bsm_pixel_jax(integrators, nint, flag, i, iifo, uniq, pixels, gmst, nifos, nsamples, sample_rate, epochs, snrs, responses, locations, horizons, rescale_loglikelihood):
    # Initialize starting values
    theta, phi = uniq2ang64(uniq)

    # Look up antenna factors
    F = compute_F(responses, horizons, phi, theta, gmst)

    # Compute dt
    dt = toa_errors(theta, phi, gmst, nifos, locations, epochs)

    # Shift SNR time series by the time delay for this sky position
    def snr_row(isample):
        t = isample - dt * sample_rate - 0.5 * (nsamples - 1) 
        return vmap(lambda x, tt: eval_snr(x, nsamples, tt))(snrs, t)  
    snrs_interp = vmap(snr_row)(jnp.arange(snrs.shape[1]))

    # Perform bayestar_singal_amplitude_model
    def process_twopsi(itwopsi):
        p_u, log_p_u, b_u, log_b_u = compute_twopsi_slice(
            F, snrs_interp, itwopsi, ntwopsi, rescale_loglikelihood)
        return p_u, log_p_u, b_u, log_b_u
    p, log_p, b, log_b = vmap(process_twopsi)(jnp.arange(ntwopsi))
    
    # Initialize accum with the integrator evaluation
    accum0 = jnp.where(nint > 0, vmap(lambda itwopsi: vmap(lambda iu: vmap(lambda isample: 
                                 u_points_weights[iu][1] + log_radial_integrator.log_radial_integrator_eval(integrators[0][0][0], integrators[0][0][1], integrators[0][0][2], integrators[1][0], p[itwopsi][iu], b[itwopsi][iu][isample], log_p[itwopsi][iu], log_b[itwopsi][iu][isample]))
                                 (jnp.arange(snrs.shape[1])))(jnp.arange(nu)))(jnp.arange(ntwopsi)), jnp.array([0]))
    accum1 = jnp.where(nint > 1, vmap(lambda itwopsi: vmap(lambda iu: vmap(lambda isample: 
                                 u_points_weights[iu][1] + log_radial_integrator.log_radial_integrator_eval(integrators[0][1][0], integrators[0][1][1], integrators[0][1][2], integrators[1][1], p[itwopsi][iu], b[itwopsi][iu][isample], log_p[itwopsi][iu], log_b[itwopsi][iu][isample]))
                                 (jnp.arange(snrs.shape[1])))(jnp.arange(nu)))(jnp.arange(ntwopsi)), jnp.array([0]))
    accum2 = jnp.where(nint > 2, vmap(lambda itwopsi: vmap(lambda iu: vmap(lambda isample: 
                                 u_points_weights[iu][1] + log_radial_integrator.log_radial_integrator_eval(integrators[0][2][0], integrators[0][2][1], integrators[0][2][2], integrators[1][2], p[itwopsi][iu], b[itwopsi][iu][isample], log_p[itwopsi][iu], log_b[itwopsi][iu][isample]))
                                 (jnp.arange(snrs.shape[1])))(jnp.arange(nu)))(jnp.arange(ntwopsi)), jnp.array([0]))
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
        lambda px: px.at[i, 2].set(compute_accum(0, snrs, accum)),
        lambda px: px,
        pixels
    )
    pixels = lax.cond(
        flag == 3,
        lambda px: px.at[i, 3].set(compute_accum(1, snrs, accum)),
        lambda px: px,
        pixels
    )
    return pixels

# --- TESTS ---
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
