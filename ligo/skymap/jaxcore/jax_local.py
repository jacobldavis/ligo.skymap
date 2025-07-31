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

from jax import jit, vmap, lax
import jax.numpy as jnp
from .jax_cosmology import *
from .jax_moc import *
from .jax_interp import *
from .jax_integrate import *
from .jax_pixel import *

@jit
def logsumexp(accum, log_weight):
    """
    Compute log-sum-exp across interferometers with numerical stability.

    Parameters
    ----------
    accum : array_like
        Log-likelihood values for each interferometer.
    log_weight : float
        Logarithmic weight applied to the result.

    Returns
    -------
    float
        Final normalized log-evidence value.
    """
    max_accum = jnp.max(accum, axis=0) 
    shifted = accum - max_accum 
    sum_accum = jnp.sum(jnp.exp(shifted), axis=0)
    result = jnp.log(sum_accum) + max_accum + log_weight
    return result

@partial(jit, static_argnames=['last_n'])
def bayestar_pixels_refine_core(pixels, last_n, new_pixels):
    """
    Refine pixels by splitting the last_n entries into 4 child pixels each.

    Parameters
    ----------
    pixels : array_like
        Current array of pixels.
    last_n : int
        Number of pixels to refine.
    new_pixels : array_like
        Pre-allocated array of size `len(pixels) + 3*last_n`.

    Returns
    -------
    tuple
        Refined pixel array and its new length.
    """
    length = pixels.shape[0]
    new_length = new_pixels.shape[0]
    prefix_len = length - last_n

    pixels_prefix = lax.dynamic_slice(pixels, (0, 0), (prefix_len, 4))
    new_pixels = lax.dynamic_update_slice(new_pixels, pixels_prefix, (0, 0))

    def refine_loop(i, new_pixels):
        base_uniq = 4 * pixels[length - i - 1, 0]
        child_uniqs = jnp.arange(4, dtype=pixels.dtype) + base_uniq
        child_aux = jnp.zeros((4, 3), dtype=pixels.dtype)
        new_children = jnp.concatenate([child_uniqs[:, None], child_aux], axis=1)

        dest_idx = new_length - 4 * i - 4
        new_pixels = lax.dynamic_update_slice(new_pixels, new_children, (dest_idx, 0))
        return new_pixels

    new_pixels = lax.fori_loop(0, last_n, refine_loop, new_pixels)
    return new_pixels, new_length

def bayestar_pixels_refine(pixels, last_n):
    """
    Interface for pixel refinement, allocating new space and invoking the core function.

    Parameters
    ----------
    pixels : array_like
        Input pixel array.
    last_n : int
        Number of pixels to split.

    Returns
    -------
    tuple
        Refined pixel array and new length.
    """
    length = pixels.shape[0]
    new_length = length + 3 * last_n
    new_pixels = jnp.zeros((new_length, 4), dtype=pixels.dtype) 
    return bayestar_pixels_refine_core(pixels, last_n, new_pixels)

@jit
def bayestar_pixels_sort_prob(pixels):
    """
    Sort pixels in ascending order of posterior probability corrected for pixel order.

    Parameters
    ----------
    pixels : array_like
        Pixel array with log-probability in column 1.

    Returns
    -------
    array_like
        Sorted pixel array.
    """
    def compute_score(pixel):
        uniq = pixel[0]
        logp = pixel[1]
        order = uniq2order64(uniq)
        return logp - 2 * M_LN2 * order

    scores = vmap(compute_score)(pixels)
    sorted_indices = jnp.argsort(scores)
    sorted_pixels = pixels[sorted_indices]
    return sorted_pixels

@jit 
def bayestar_pixels_sort_uniq(pixels):
    """
    Sort pixels by HEALPix UNIQ index.

    Parameters
    ----------
    pixels : array_like
        Pixel array.

    Returns
    -------
    array_like
        Sorted array.
    """
    def get_uniq(i):
        return pixels[i,0]
    uniq = vmap(get_uniq)(jnp.arange(pixels.shape[0]))
    sorted_indices = jnp.argsort(uniq)
    return pixels[sorted_indices]

@partial(jit, static_argnames=['nifos'])
def bsm_jax(min_distance, max_distance, prior_distance_power, 
            cosmology, gmst, nifos, nsamples, sample_rate, epochs, snrs, 
            responses, locations, horizons, rescale_loglikelihood,
            bs=24):
    """
    Compute Bayesian sky localization and distance posteriors.

    Parameters
    ----------
    min_distance, max_distance : float
        Bounds for source distance prior.
    prior_distance_power : float
        Power-law exponent for distance prior.
    cosmology : object
        Cosmology parameters (used in distance conversions).
    gmst : float
        Greenwich Mean Sidereal Time.
    nifos : int
        Number of detectors.
    nsamples : int
        Number of SNR time samples.
    sample_rate : float
        Sampling frequency in Hz.
    epochs : array_like
        Trigger times for each detector.
    snrs : array_like
        SNR time series.
    responses : array_like
        Detector tensor responses.
    locations : array_like
        Detector positions.
    horizons : array_like
        Horizon distances for each detector.
    rescale_loglikelihood : float
        Scaling factor for log-likelihood.
    bs : int
        batch_size for lax.map operations

    Returns
    -------
    tuple
        Pixel array with posterior, mean, std distance + log Bayes factors.

    NOTE: We use jax.lax.fori_loop and jax.lax.map to avoid overallocating memory in some cases.
          Tune the bs parameter (batch_size) or swap lax.map to vmap depending on your GPU's memory.
    """
    # Initialize integrators
    pmax = jnp.sqrt(0.5 * jnp.sum(jnp.square(horizons))) * rescale_loglikelihood
    integrators = [
        log_radial_integrator(min_distance, max_distance, prior_distance_power + k,
                              cosmology, pmax, default_log_radial_integrator_size)
        for k in range(3)
    ]
    regions = extract_integrator_regions(integrators)
    limits = extract_integrator_limits(integrators)
    integrators_values = (regions, limits)
    
    # Initialize pixels
    order0 = 4
    nside = 2 ** order0
    npix0 = 12 * nside * nside
    pixels = vmap(lambda ipix: jnp.concatenate([
        jnp.array([nest2uniq64(order0, ipix)]),
        jnp.zeros(3)
    ]))(jnp.arange(npix0))

    # Compute the coherent probability map and incoherent evidence at the lowest order
    log_norm = -jnp.log(2 * (2 * jnp.pi) * (4 * jnp.pi) * ntwopsi * nsamples) \
               - log_radial_integrator.log_radial_integrator_eval(regions[0][0], regions[0][1], regions[0][2], 
                                                                  (integrators[0].p0_limit, integrators[0].vmax, integrators[0].ymax), 
                                                                  0, 0, -jnp.inf, -jnp.inf)
    accum = jnp.zeros((npix0, nifos))

    def update_pixel_row(px_row):
        return bsm_pixel_prob_jax(
            integrators_values, px_row[0], px_row,
            gmst, nsamples, sample_rate,
            epochs, snrs, responses, locations, horizons,
            rescale_loglikelihood
        )
    
    def update_accum_row(px_row, iifo):
        return bsm_pixel_accum_jax(
            integrators_values, px_row[0],
            gmst, nsamples, sample_rate,
            lax.dynamic_slice(epochs, (iifo,), (1,)), 
            lax.dynamic_slice(snrs, (iifo, 0, 0), (1, snrs.shape[1], snrs.shape[2])), 
            lax.dynamic_slice(responses, (iifo, 0, 0), (1, responses.shape[1], responses.shape[2])),
            lax.dynamic_slice(locations, (iifo, 0), (1, locations.shape[1])), 
            lax.dynamic_slice(horizons, (iifo,), (1,)),
            rescale_loglikelihood
        )

    def run_all_vmap(pixels, accum):
        pixels_new_rows = lax.map(lambda px_row: update_pixel_row(px_row), pixels, batch_size=bs)
        pixels = pixels.at[:].set(pixels_new_rows)

        def update_incoherent(px_row):
            return vmap(lambda iifo: update_accum_row(px_row, iifo))(jnp.arange(nifos))

        accum = lax.map(update_incoherent, pixels, batch_size=bs)

        return pixels, accum

    pixels, accum = run_all_vmap(pixels, accum)
        
    log_weight = log_norm + jnp.log(uniq2pixarea64(pixels[0, 0]))
    log_evidence_incoherent = logsumexp(accum, log_weight)

    pixels = bayestar_pixels_sort_prob(pixels)

    # Adaptively refine until order=11
    def refine_vmap(pixels):
        pixels, length = bayestar_pixels_refine(pixels, npix0 // 4)

        new_rows = pixels[-(npix0):]
        updated_rows = lax.map(lambda px_row: update_pixel_row(px_row), new_rows, batch_size=bs)
        pixels = pixels.at[-(npix0):].set(updated_rows)

        pixels = bayestar_pixels_sort_prob(pixels)
        return pixels, length
    
    pixels, length = refine_vmap(pixels)
    pixels, length = refine_vmap(pixels)
    pixels, length = refine_vmap(pixels)
    pixels, length = refine_vmap(pixels)
    pixels, length = refine_vmap(pixels)
    pixels, length = refine_vmap(pixels)
    pixels, length = refine_vmap(pixels)

    # Evaluate distance layers
    def update_distance_row(px):
        return bsm_pixel_dist_jax(integrators_values, px[0], px,
                                gmst, nsamples, sample_rate,
                                epochs, snrs, responses, locations, horizons,
                                rescale_loglikelihood)

    distance_rows = lax.map(update_distance_row, pixels, batch_size=bs)
    pixels = pixels.at[:].set(distance_rows)

    # --- DONE SECTION ---

    # Rescale so that log(max) = 0
    max_logp = pixels[length - 1, 1]
    pixels = pixels.at[:, 1:].add(-max_logp)

    # Determine normalization of map
    pixareas = vmap(lambda px: uniq2pixarea64(px[0]))(pixels)
    dp_unnorm = jnp.exp(pixels[:, 1]) * pixareas
    norm = jnp.sum(jnp.where(dp_unnorm > 0, dp_unnorm, 0.0))
    log_evidence_coherent = jnp.log(norm) + max_logp + log_norm
    inv_norm = 1.0 / norm

    # Rescale, normalize, and prepare output
    def prepare_output(px):
        logp = px[1]
        prob = jnp.exp(logp) * inv_norm
        rmean = jnp.exp(px[2] - logp)
        rvar = jnp.exp(px[3] - logp) - rmean ** 2
        rmean = jnp.where(rvar >= 0, rmean, jnp.inf)
        rstd = jnp.where(rvar >= 0, jnp.sqrt(rvar), 1.0)
        return px.at[1].set(prob).at[2].set(rmean).at[3].set(rstd)

    pixels = vmap(prepare_output)(pixels)

    # Sort pixels by ascending NUNIQ index
    pixels = bayestar_pixels_sort_uniq(pixels)

    # Calculate log Bayes factor and return
    log_bci = log_bsn = log_evidence_coherent
    log_bci -= jnp.sum(vmap(lambda i: log_evidence_incoherent[i])(jnp.arange(nifos)))

    return pixels, log_bci, log_bsn

