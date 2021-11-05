import numpy as onp
from jax import jit, vmap, lax, remat
import jax.numpy as np
import jax.random as random
from functools import partial
import time


def softmax(u, scale=1, corr_factor=1):
    """
    Probability equation for multinomial logit.

    Parameters:
    ------------
    u: array-like. Array containing the utility estimate for each alternative.
    scale: (int, optional ) - Scaling factor for exp(scale * u). Default = 1
    axis: (None or int or tuple of ints, optional) – Axis or axes over which
        the sum is taken. By default axis is None, and all elements are summed.

    Returns:
    ----------
    (array) Probabilites for each alternative
    """
    exp_utility = np.exp(scale * u)
    sum_exp_utility = np.sum(exp_utility * corr_factor, keepdims=True)
    proba = exp_utility / sum_exp_utility
    return proba


def logsums(u, scale=1, axis=1, corr_factor=1):
    """
    Maximum expected utility
    Parameters:
    ------------
    u: array-like. Array containing the unscaled utility estimate for each
        alternative.
    scale: (int, optional ) - Scaling factor for exp(scale * u). Default = 1
    axis: (None or int or tuple of ints, optional) – Axis or axes over which
        the sum is taken. By default axis is None, and all elements are summed.

    return:
    -------
    Maximum expected utility of the nest
    """
    return (1 / scale) * np.log(
        np.sum(np.exp(scale * u) * corr_factor, axis=axis))


def get_mct(choosers, alts, var_mats=None, chooser_alts=None):

    num_choosers = choosers.shape[0]
    num_alts = alts.shape[0]
    sample_size = chooser_alts.shape[1]

    if chooser_alts is None:
        mct = np.tile(alts, (num_choosers, 1))
        mct = np.hstack((mct, choosers.repeat(num_alts)))
    else:
        mct = np.vstack(
            [alts[chooser_alts[i, :], :] for i in range(num_choosers)])
        mct = np.hstack((mct, choosers.repeat(sample_size).reshape(-1, 1)))

    return mct


class large_mnl():
    """
    Differentiable approach for multinomial logit with large alternative set
    """

    def __init__(
            self,
            model_object=None,
            coeffs=None,
            n_choosers=None,
            n_alts=None):
        self.weights = coeffs
        self.n_choosers = n_choosers
        self.n_alts = n_alts
        # self.constrains  = constrains
        # TO DO: yaml file

    def utilities(self, x):
        """ Calculates the utility fuction of weights w and data x
        Parameters:
        ------------
        x : Jax 2d array. Column names must match coefficient names. If a Jax
            array, order of columns should be the same order as the coeffient
            names.

        Return:
        ------------
        2-d jax numpy array with utilities for eahc alternative.
        """

        w = self.weights
        n = self.n_choosers
        j = self.n_alts
        return np.dot(x, w.T).reshape(n, j)

    def probabilities(self, x, scale=1, corr_factor=1):
        """ Estimates the probabilities for each chooser x alternative
        """
        utils = self.utilities(x)
        probs = vmap(
            softmax, in_axes=(0, None, None))(utils, scale, corr_factor)
        return probs

    def logsum(self, x):
        """ Estimates the maximum expected utility for all alternatives in the
        choice set, with Scale parameter normalized to 1.
        """
        utils = self.utilities(x)
        return vmap(logsums)(utils)

    def simulation(self, x, key):
        """
        Monte Carlo simulation.

        Parameters
        ----------
        x: 2-d Jax numpy array
        key: jax PRNG Key object

        Return
        -------
        - numpy.array

        """
        utils = self.utilities(x)
        shape = utils.shape
        keys = random.split(key, shape[0])

        @jit
        def single_simulation(u, key):
            return random.categorical(key, u)

        choices = vmap(single_simulation, in_axes=(0, 0))(utils, keys)
        return choices  # Assuming alternative name starts at 0


@jit
def swop_exp(key, alts_idxs):

    num_alts = alts_idxs.shape[0]
    w = 1 / num_alts
    E = -np.log(
        random.uniform(key, shape=(num_alts,), minval=0, maxval=1))
    E /= w
    shuffled = np.argsort(E)

    return shuffled


@partial(jit, static_argnums=[3])
def interaction_sample(
        alts, coeffs, pop_mean_prob, sample_size, chooser_val_key):

    chooser_val, key = chooser_val_key

    # perform interaction
    idx_alt_intx = -1
    interacted = alts[:, idx_alt_intx] / chooser_val
    alts2 = alts.at[:, idx_alt_intx].set(interacted)

    # compute probs
    true_logits = np.dot(alts2, coeffs.T)
    true_probs = softmax(true_logits)
    true_probs = true_probs.flatten()

    # sample
    total_alts = alts.shape[0]
    alts_idxs = np.arange(total_alts)
    shuffled = swop_exp(key, alts_idxs)
    samp_alts_idxs = shuffled[:sample_size]
    alts3 = alts2[samp_alts_idxs, :]

    # compute sample probs
    samp_logits = np.dot(alts3, coeffs.T)
    samp_probs = softmax(samp_logits)
    samp_probs = samp_probs.flatten()

    # compute metrics

    # pct alts with probs greater than true mean prob
    pct_probs_gt_pop_mean = np.sum(samp_probs > pop_mean_prob) / sample_size

    # true max vs true prob sampled max
    max_idx = np.nanargmax(samp_probs)
    true_idx = samp_alts_idxs[max_idx]

    # corrected max prob vs true max
    max_prob_corr = samp_probs[max_idx] * (sample_size / total_alts)

    del true_logits
    del true_probs
    del samp_logits
    del samp_probs
    del alts2
    del alts3
    del interacted
    del shuffled
    del alts_idxs
    del samp_alts_idxs

    return max_prob_corr, pct_probs_gt_pop_mean, true_idx


@jit
def interaction(alts, coeffs, pop_mean_prob, chooser_val_key):
    """ Compute logit probs with NO SAMPLING of alts
    """
    chooser_val, key = chooser_val_key
    # total_alts = alts.shape[0]

    # perform interaction
    idx_alt_intx = -1
    interacted = alts[:, idx_alt_intx] / chooser_val
    alts2 = alts.at[:, idx_alt_intx].set(interacted)

    # compute probs
    logits = np.dot(alts2, coeffs.T)
    probas = softmax(logits)
    probas = probas.flatten()

    # # compute metrics
    # pct_probs_gt_pop_mean = np.sum(probas > pop_mean_prob) / total_alts
    # max_prob = np.nanmax(probas)

    # del probas
    del logits
    del alts2
    del interacted

    return probas


@jit
@remat
def interaction_iters(
        alts, coeffs, pop_mean_prob, chooser_val_key, n_splits=10):

    chooser_val, key = chooser_val_key
    total_alts = alts.shape[0]

    # perform interaction
    idx_alt_intx = -1
    interacted = alts[:, idx_alt_intx] / chooser_val
    alts2 = alts.at[:, idx_alt_intx].set(interacted)

    # compute probs
    logits = np.dot(alts2, coeffs.T)
    probas = softmax(logits)
    probas = probas.flatten()
    true_max = np.nanmax(probas)
    pct_true_probs_gt_pop_mean = np.sum(
        probas > pop_mean_prob) / total_alts

    # sample splits
    alts_idxs = np.arange(total_alts)
    shuffled = swop_exp(key, alts_idxs)
    samples = np.array_split(shuffled, n_splits)

    # sample probs
    samp_alts_idxs = np.asarray([], dtype=int)
    result_arr = np.zeros((9, 14))
    for i, sample in enumerate(samples):

        if i == n_splits - 1:
            continue

        sample_rate = (i + 1) / n_splits
        sample_size = total_alts * sample_rate
        samp_alts_idxs = np.concatenate((samp_alts_idxs, sample))
        alts3 = alts2[samp_alts_idxs]
        samp_logits = np.dot(alts3, coeffs.T)
        samp_probs = softmax(samp_logits)
        samp_probs = samp_probs.flatten()
        samp_max_idx = np.nanargmax(samp_probs)
        samp_corr_max = samp_probs[samp_max_idx] * sample_rate
        true_prob_samp_max = probas[samp_alts_idxs[samp_max_idx]]
        pct_samp_probs_gt_pop_mean = np.sum(
            samp_probs > pop_mean_prob) / sample_size

        tmsm_err = true_max - true_prob_samp_max
        tmsm_pct_err = tmsm_err / true_max
        tmsm_sq_err = tmsm_err * tmsm_err
        tmsmc_err = true_max - samp_corr_max
        tmsmc_pct_err = tmsmc_err / true_max
        tmsmc_sq_err = tmsmc_err * tmsmc_err
        cf_err = true_prob_samp_max - samp_corr_max
        cf_pct_err = cf_err / true_prob_samp_max
        cf_sq_err = cf_err * cf_err
        ppgm_err = pct_true_probs_gt_pop_mean - pct_samp_probs_gt_pop_mean
        ppgm_pct_err = ppgm_err / pct_true_probs_gt_pop_mean
        ppgm_sq_err = ppgm_err * ppgm_err

        result_arr = result_arr.at[i, :].set(np.array([
            total_alts, sample_rate,
            tmsm_err, tmsm_pct_err, tmsm_sq_err,
            tmsmc_err, tmsmc_pct_err, tmsmc_sq_err,
            cf_err, cf_pct_err, cf_sq_err,
            ppgm_err, ppgm_pct_err, ppgm_sq_err]))

    # del probas
    del logits
    del probas
    del alts2
    del samp_logits
    del samp_probs
    del alts3
    del interacted
    del shuffled
    del samples

    return result_arr


def get_probs(
        choosers, alts, key, sample_size,
        batched=False, max_mct_size=1200000000):
    """VMAP the interaction function over all choosers' values"""

    num_choosers = choosers.shape[0]
    key_dim = key.shape[0]
    keys = random.split(key, num_choosers)
    num_alts = alts.shape[0]
    coeffs = np.array([-1, 1, 1, 1, 1])  # dist-to-cbd, sizes 1-3, intx term
    pop_mean_prob = 1 / num_alts
    mct_size = num_choosers * num_alts
    if (batched) & (mct_size > max_mct_size):

        n_chooser_batches = 1
        while True:
            n_chooser_batches += 1
            if num_choosers % n_chooser_batches != 0:
                continue
            elif (mct_size / n_chooser_batches) < max_mct_size / 3:
                break
        choosers_per_batch = int(num_choosers / n_chooser_batches)
        print(
            "Computing probabilities in {0} batches of {1} choosers".format(
                n_chooser_batches, choosers_per_batch))
        choosers = choosers.reshape(
            (n_chooser_batches, choosers_per_batch))
        keys = keys.reshape(
            (n_chooser_batches, choosers_per_batch, key_dim))

        if sample_size < num_alts:
            partial_interaction = partial(
                interaction_sample,
                alts,
                coeffs,
                pop_mean_prob,
                sample_size)
            results = lax.map(
                vmap(partial_interaction), (choosers, keys))
            results = [
                result.reshape((num_choosers, )) for result in results]
        else:
            partial_interaction = partial(
                interaction,
                alts,
                coeffs,
                pop_mean_prob)
            results = lax.map(
                vmap(partial_interaction), (choosers, keys))
            results = results.reshape((num_choosers, num_alts))

    else:
        if sample_size < num_alts:
            results = vmap(
                interaction_sample, in_axes=(None, None, None, None, 0))(
                alts, coeffs, pop_mean_prob, sample_size,
                (choosers, keys))
        else:
            results = vmap(
                interaction, in_axes=(None, None, None, 0))(
                alts, coeffs, pop_mean_prob, (choosers, keys))

    return results


def get_iter_probs(
        choosers, alts, key, sample_size,
        batched=False, max_mct_size=1200000000):
    """VMAP the interaction function over all choosers' values"""
    now = time.time()
    num_choosers = choosers.shape[0]
    key_dim = key.shape[0]
    keys = random.split(key, num_choosers)
    num_alts = alts.shape[0]
    coeffs = np.array([-1, 1, 1, 1, 1])  # dist-to-cbd, sizes 1-3, intx term
    pop_mean_prob = 1 / num_alts
    mct_size = num_choosers * num_alts
    if (batched) & (mct_size > max_mct_size):

        n_chooser_batches = 1
        while True:
            n_chooser_batches += 1
            if num_choosers % n_chooser_batches != 0:
                continue
            elif (mct_size / n_chooser_batches) < max_mct_size / 3:
                break
        choosers_per_batch = int(num_choosers / n_chooser_batches)
        print(
            "Computing probabilities in {0} batches of {1} choosers".format(
                n_chooser_batches, choosers_per_batch))
        choosers = choosers.reshape(
            (n_chooser_batches, choosers_per_batch))
        keys = keys.reshape(
            (n_chooser_batches, choosers_per_batch, key_dim))

        partial_interaction = partial(
            interaction_iters,
            alts,
            coeffs,
            pop_mean_prob)
        results = lax.map(
            vmap(partial_interaction), (choosers, keys))
        results = results.reshape((num_choosers, 9, 14))

    else:

        results = vmap(interaction_iters, in_axes=(None, None, None, 0))(
            alts, coeffs, pop_mean_prob, (choosers, keys))

    later = time.time()
    print(
        "Took {0} s to compute prob. metrics for all sample "
        "rates by chooser.".format(np.round(later - now)))

    return results
