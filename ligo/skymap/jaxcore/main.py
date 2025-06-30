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
# NOTE: For a pixel i, pixel[i][0] is the uniq id, pixel[i][1] is the values array

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
def bayestar_pixels_sort_prob(pixels, npix0):
    pixels = vmap(lambda i: jnp.append(pixels[i], pixels[i][1][0]-2*M_LN2*uniq2order64(pixels[i][0])))(jnp.arange(npix0))
    sorted_pixel_indices = jnp.argsort(pixels[:, 2])
    pixels = pixels[sorted_pixel_indices]
    pixels = jnp.delete(pixels, 2, axis=1)

@jit 
def assign_uniq(i, pixels, len, new_len):
    uniq = 4 * pixels[len - i - 1][0]
    pixels[new_len - (4 * i + 0) - 1][0] = 0 + uniq
    pixels[new_len - (4 * i + 1) - 1][0] = 1 + uniq
    pixels[new_len - (4 * i + 2) - 1][0] = 2 + uniq
    pixels[new_len - (4 * i + 3) - 1][0] = 3 + uniq

@jit
def bayestar_pixels_refine(pixels, len, last_n):
    new_len = len + 3 * last_n
    vmap(lambda i: assign_uniq(i, pixels, len, new_len))(jnp.arange(last_n))

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
    pixels = vmap(lambda ipix: jnp.array([nest2uniq64(order0, ipix), [0, 0, 0]]))(jnp.arange(npix0))

    # Compute the coherent probability map and incoherent evidence at the lowest order
    # TODO: Modify integration function later
    log_norm = -jnp.log(2 * (2 * jnp.pi) * (4 * jnp.pi) * ntwopsi * nsamples) - log_radial_integrator_quadax.log_radial_integrator_eval_quadax(integrators[0].r1, integrators[0].r2, 0, 0, -jnp.inf, -jnp.inf)

    accum = jnp.zeros(npix0, nifos)
    @jit
    def probability_map(i):
        bsm_pixel_jax(integrators, 1, 1, i, 0, pixels[i][0], pixels, 
                      gmst, nifos, nsamples, sample_rate, epochs, snrs, 
                      responses, locations, horizons, rescale_loglikelihood)
        vmap(lambda iifo: bsm_pixel_jax(integrators, 1, 2, i, iifo, pixels[i][0], accum, 
                                        gmst, 1, nsamples, sample_rate, epochs[iifo], snrs[iifo], responses[iifo], 
                                        locations[iifo], horizons[iifo], rescale_loglikelihood))(jnp.arange(nifos))
    vmap(lambda i: probability_map(i))(jnp.arange(npix0))
    log_weight = log_norm + jnp.log(uniq2pixarea64(pixels[0][0]))
    log_evidence_incoherent = logsumexp(accum, log_weight, npix0, nifos)

    # Sort pixels by ascending posterior probability
    bayestar_pixels_sort_prob(pixels, npix0)

    # Adaptively refine until order=11
    len = npix0

