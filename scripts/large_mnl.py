import numpy as onp
from jax import jit, vmap, lax, remat
import jax.numpy as np
import jax.random as random
from functools import partial


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


def create_data(num_alts, pop_to_alts_ratio, num_vars=5):

    # create choosers
    n_choosers = int(pop_to_alts_ratio * num_alts)
    choosers = onp.random.lognormal(0, 0.5, n_choosers)  # 1 chooser attribute

    # initialize alts
    alts = onp.zeros((num_alts, num_vars))

    # dist to CBD
    alts[:, 0] = onp.random.lognormal(0, 1, num_alts)  # Dist to CBD

    # size terms
    for i in range(1, num_vars - 1):
        split_val = int(onp.floor(num_alts * .5))
        alts[:split_val, i] = onp.random.normal(
            1, 1, split_val)  # first 1/2 of alts have mean = 1
        alts[split_val:, i] = onp.random.normal(
            0.5, 1, num_alts - split_val)  # rest of alts have mean 0.5

    # interaction term
    alts[:, -1] = onp.random.lognormal(1, .5, num_alts)

    return np.array(choosers), np.array(alts)


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


@partial(jit, static_argnums=[3])
def interaction_sample(
        alts, coeffs, pop_mean_prob, sample_size, chooser_val_key):

    chooser_val, key = chooser_val_key
    total_alts = alts.shape[0]

    # w = 1 / total_alts
    # E = -np.log(random.uniform(key, shape=(total_alts,), minval=0, maxval=1))
    # E /= w
    # shuffled = np.argsort(E)

    shuffled = random.permutation(key, np.arange(total_alts))
    samp_alts_idxs = shuffled[:sample_size]

    alts2 = alts[samp_alts_idxs, :]

    # perform interaction
    idx_alt_intx = -1
    interacted = alts2[:, idx_alt_intx] / chooser_val
    alts3 = alts2.at[:, idx_alt_intx].set(interacted)

    # compute probs
    logits = np.dot(alts3, coeffs.T)
    probas = softmax(logits)
    probas = probas.flatten()

    # compute metrics
    pct_probs_gt_pop_mean = np.sum(probas > pop_mean_prob) / sample_size
    max_prob = np.nanmax(probas)

    del logits
    del probas
    del alts2
    del alts3
    del interacted
    del shuffled
    # del E
    del samp_alts_idxs

    return max_prob, pct_probs_gt_pop_mean


@jit
@remat
def interaction(alts, coeffs, pop_mean_prob, chooser_val_key):

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

    # compute metrics
    pct_probs_gt_pop_mean = np.sum(probas > pop_mean_prob) / total_alts
    max_prob = np.nanmax(probas)

    del probas
    del logits
    del alts2
    del interacted

    return max_prob, pct_probs_gt_pop_mean


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
            elif (mct_size / n_chooser_batches) < max_mct_size:
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
        else:
            partial_interaction = partial(
                interaction,
                alts,
                coeffs,
                pop_mean_prob)

        results = lax.map(vmap(partial_interaction), (choosers, keys))
        results = [
            result.reshape((num_choosers, )) for result in results]

    else:
        if sample_size < num_alts:
            results = vmap(
                interaction_sample, in_axes=(None, None, None, None, 0))(
                alts, coeffs, pop_mean_prob, sample_size, (choosers, keys))
        else:
            results = vmap(
                interaction, in_axes=(None, None, None, 0))(
                alts, coeffs, pop_mean_prob, (choosers, keys))

    return results
