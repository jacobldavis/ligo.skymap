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
from cosmology import *
from moc import *
from interp import *
from integrate import *
from pixel import *
M_LN2 = jnp.log(2)

@jit
def logsumexp(accum, log_weight, ni, nj):
    max_accum = vmap(lambda i: jnp.max(accum[i]))(jnp.arange(ni))
    sum_accum = vmap(lambda i: jnp.sum(jnp.exp(accum[i] - max_accum)))(jnp.arange(ni))
    result = vmap(lambda j: jnp.log(sum_accum[j]) + max_accum[j] + log_weight)(jnp.arange(nj))
    return result

@jit
def bayestar_pixels_refine(pixels, last_n):
    len0 = pixels.shape[0]
    new_len = len0 + 3 * last_n

    to_refine = pixels[-last_n:] 
    base_uniq = to_refine[:, 0].astype(jnp.int64) * 4 

    def refine_one(i):
        value = to_refine[i, 1:4]
        uniq_base = base_uniq[i]
        uniqs = uniq_base + jnp.arange(4) 
        values = jnp.tile(value[None, :], (4, 1)) 
        return jnp.concatenate([uniqs[:, None], values], axis=1) 

    refined = jax.vmap(refine_one)(jnp.arange(last_n))
    refined = refined.reshape(-1, 4)

    keep = pixels[:len0 - last_n]

    return jnp.concatenate([keep, refined], axis=0), len

@jit
def bayestar_pixels_sort_prob(pixels, npix0):
    def compute_score(i):
        uniq = pixels[i, 0].astype(jnp.int64)
        value0 = pixels[i, 1]
        order = uniq2order64(uniq)
        score = value0 - 2 * M_LN2 * order
        return score

    scores = vmap(compute_score)(jnp.arange(npix0)) 
    sorted_indices = jnp.argsort(scores)
    return pixels[sorted_indices]

@jit 
def bayestar_pixels_sort_uniq(pixels, len):
    def get_uniq(i):
        return pixels[i,0]
    uniq = vmap(get_uniq)(jnp.arange(len))
    sorted_indices = jnp.argsort(uniq)
    return pixels[sorted_indices]

@jit
def bsm_jax(min_distance, max_distance, prior_distance_power, 
            cosmology, gmst, nifos, nsamples, sample_rate, epochs, snrs, 
            responses, locations, horizons, rescale_loglikelihood):
    # Initialize integrators
    pmax = jnp.sum(vmap(lambda iifo: jnp.pow(horizons[iifo], 2))(jnp.arange(nifos)))
    pmax = jnp.sqrt(0.5 * pmax)
    pmax *= rescale_loglikelihood
    integrators = [log_radial_integrator(min_distance, max_distance, prior_distance_power + 0, cosmology, pmax, default_log_radial_integrator_size),
                   log_radial_integrator(min_distance, max_distance, prior_distance_power + 1, cosmology, pmax, default_log_radial_integrator_size),
                   log_radial_integrator(min_distance, max_distance, prior_distance_power + 2, cosmology, pmax, default_log_radial_integrator_size)]
    
    # Initialize pixels
    order0 = 4
    nside = 1 << order0
    npix0 = 12 * nside * nside
    pixels = vmap(lambda ipix: jnp.concatenate([
        jnp.array([nest2uniq64(order0, ipix)], dtype=jnp.float64),
        jnp.zeros(3)
    ]))(jnp.arange(npix0))

    # Compute the coherent probability map and incoherent evidence at the lowest order
    # TODO: Modify integration function later
    log_norm = -jnp.log(2 * (2 * jnp.pi) * (4 * jnp.pi) * ntwopsi * nsamples) - log_radial_integrator_quadax.log_radial_integrator_eval_quadax(integrators[0].r1, integrators[0].r2, 0, 0, -jnp.inf, -jnp.inf)

    accum = jnp.zeros(npix0, nifos)

    @jit
    def accum_map(i, iifo, pixels, accum):
        return bsm_pixel_jax(
            integrators, 1, 2, i, iifo, pixels[i, 0], accum,
            gmst, 1, nsamples, sample_rate,
            epochs[iifo], snrs[iifo], responses[iifo],
            locations[iifo], horizons[iifo], rescale_loglikelihood
        )
    
    @jit
    def probability_map(i, pixels, accum):
        pixels_out = bsm_pixel_jax(
            integrators, 1, 1, i, 0, pixels[i, 0], pixels,
            gmst, nifos, nsamples, sample_rate,
            epochs, snrs, responses, locations, horizons, rescale_loglikelihood
        )

        def accum_iifo(iifo, accum_inner):
            return accum_map(i, iifo, pixels_out, accum_inner)

        accum_out = jax.lax.fori_loop(0, nifos, lambda j, acc: accum_iifo(j, acc), accum)
        return pixels_out, accum_out

    @jit
    def run_all(pixels, accum):
        def one_pixel(i, state):
            pixels, accum = state
            pixels_out, accum_out = probability_map(i, pixels, accum)
            return (pixels_out, accum_out)

        return jax.lax.fori_loop(0, npix0, one_pixel, (pixels, accum))
    
    pixels, accum = run_all(pixels, accum)
    
    log_weight = log_norm + jnp.log(uniq2pixarea64(pixels[0, 0]))
    log_evidence_incoherent = logsumexp(accum, log_weight, npix0, nifos)

    # Sort pixels by ascending posterior probability
    pixels = bayestar_pixels_sort_prob(pixels, npix0)

    # Adaptively refine until order=11
    @jit
    def refine(pixels, len):
        pixels, len = bayestar_pixels_refine(pixels, len, npix0 / 4)

        pixels = jax.lax.fori_loop(
        len - npix0, len,
        lambda i, pixels: bsm_pixel_jax(
            integrators, 1, 1, i, 0, pixels[i, 0], pixels,
            gmst, nifos, nsamples, sample_rate,
            epochs, snrs, responses, locations, horizons, rescale_loglikelihood),pixels
        )

        pixels = bayestar_pixels_sort_prob(pixels, len)

        return pixels, len
    
    len = 0
    pixels, len = jax.lax.fori_loop(order0, 11, refine, (pixels, len))

    # Evaluate distance layers
    pixels = jax.lax.fori_loop(0, len, lambda i, 
            pixels: bsm_pixel_jax(integrators, 1, 2, i, 0, pixels[i, 0], pixels,
            gmst, nifos, nsamples, sample_rate, epochs, snrs, responses, 
            locations, horizons, rescale_loglikelihood), pixels
    )

    # Rescale so that log(max) = 0
    max_logp = pixels[len-1, 1]

    @jit
    def log_rescale(i, pixels):
        pixels = pixels.at(i).at(1).set(pixels[i, 1] - max_logp)
        pixels = pixels.at(i).at(2).set(pixels[i, 2] - max_logp)
        pixels = pixels.at(i, 3).set(pixels[i, 3] - max_logp)
        return pixels

    pixels = jax.lax.fori_loop(0, len, log_rescale, pixels)

    # Determine normalization of map
    @jit
    def calc_dp(i, pixels):
        dA = uniq2pixarea64(pixels[i, 0])
        dP = jnp.exp(pixels[i, 1]) * dA 
        return dP
    norm = jnp.sum(vmap(jnp.where(calc_dp>0,calc_dp,0))(jnp.arange(len)))
    log_evidence_coherent = jnp.log(norm) + max_logp + log_norm 
    norm = 1 / norm

    # Rescale, normalize, and prepare output
    @jit
    def prepare_output(i, pixels):
        prob = jnp.exp(pixels[i, 1]) * norm 
        rmean = jnp.exp(pixels[i, 2] - pixels[i, 1])
        rstd = jnp.exp(pixels[i, 3] - pixels[i, 1]) - (rmean * rmean)
        rmean = jnp.where(rstd >= 0, jnp.inf, rmean)
        rstd = jnp.where(rstd >= 0, jnp.sqrt(rstd), 1)
        pixels.at(i).at(1).set(prob)
        pixels.at(i).at(2).set(rmean)
        pixels.at(i).at(3).set(rstd)
        return pixels
    
    pixels = jax.lax.fori_loop(0, len, prepare_output, pixels)

    # Sort pixels by ascending NUNIQ index
    pixels = bayestar_pixels_sort_uniq(pixels, len)

    # Calculate log Bayes factor and return
    log_bci = log_bsn = log_evidence_coherent
    log_bci -= jnp.sum(vmap(lambda i: log_evidence_incoherent[i])(jnp.arange(nifos)))

    return pixels, log_bci, log_bsn