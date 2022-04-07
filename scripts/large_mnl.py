from jax import jit, vmap, lax, remat, devices, device_put
import jax.numpy as np
import jax.random as random
from functools import partial
import time
import numpy as onp
from tqdm import tqdm

N_SAMP_RATES = 10
ERR_METRIC_ROW_SIZE = 19
CHOOSER_PROB_ROW_SIZE = 6


def softmax(u, corr_factor=0):
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
    exp_utility = np.exp(u - corr_factor)
    sum_exp_utility = np.sum(exp_utility, keepdims=True)
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


def swop_exp(key, alts_idxs):

    num_alts = alts_idxs.shape[0]
    w = 1 / num_alts
    E = -np.log(
        random.uniform(key, shape=(num_alts,), minval=0, maxval=1))
    E /= w
    shuffled = np.argsort(E)

    return shuffled


@jit
def all_samp_idxs(key, alts_idxs):
    samps = random.shuffle(key, alts_idxs)
    return samps


@jit
def interact_choosers_alts(chooser_val, alts, alt_intx_idx=-1):

    idx_alt_intx = -1
    interacted = np.sqrt(alts[:, idx_alt_intx] / chooser_val)
    alts2 = alts.at[:, idx_alt_intx].set(interacted)

    return alts2


@jit
def interact_choosers_alts_disc(chooser_val, alts, alt_intx_idx=-1):

    interacted = alts[:, alt_intx_idx] * chooser_val
    alts2 = alts.at[:, alt_intx_idx].set(interacted)

    return alts2


def interaction_sample_utils(
        alts, coeffs, scale, sample_size, chooser_val_key):
    """ Compute logit utils with SAMPLING of alts.
    """
    chooser_val, key = chooser_val_key

    # perform interaction
    data = interact_choosers_alts(chooser_val, alts)

    # sample
    total_alts = data.shape[0]
    alts_idxs = np.arange(total_alts)
    shuffled = swop_exp(key, alts_idxs)

    samp_alts_idxs = shuffled[:sample_size]
    data = data[samp_alts_idxs, :]

    # compute utils
    logits = np.dot(data, coeffs.T) * scale

    del data
    del alts_idxs
    del shuffled

    return samp_alts_idxs, logits


@jit
def interaction_utils(alts, coeffs, scale, chooser_val):
    """ Compute logit utils NO SAMPLING of alts.
    """

    # perform interaction
    data = interact_choosers_alts(chooser_val, alts)

    # compute probs
    logits = np.dot(data, coeffs.T) * scale
    del data

    return logits


@partial(jit, static_argnums=[2, 3])
def interaction_sample_probs(
        alts, coeffs, scale, sample_size, chooser_val_key):

    samp_alt_idxs, utils = interaction_sample_utils(
        alts, coeffs, scale, sample_size, chooser_val_key)

    samp_probs = softmax(utils)  # scale already incorporated into utils
    samp_probs = samp_probs.flatten()

    del utils

    return samp_alt_idxs, samp_probs


@jit
def interaction_probs(alts, coeffs, scale, chooser_val):
    """ Compute logit probs with NO SAMPLING of alts.

    Returns probs for all alts for each chooser. Will prob run out of
    memory around n_alts == 20000.
    """
    # total_alts = alts.shape[0]

    utils = interaction_utils(alts, coeffs, scale, chooser_val)

    probas = softmax(utils)  # scale already incorporated into utils
    probas = probas.flatten()

    # cleanup
    del utils

    return probas


@jit
def interaction_probs_all(alts, coeffs, scale, chooser_val_key):
    """
    """
    total_alts = alts.shape[0]
    chooser_val, key = chooser_val_key

    # interact choosers/alts
    full_data = interact_choosers_alts(chooser_val, alts)

    # interact and take dot product
    full_utils = np.dot(full_data, coeffs.T) * scale

    # sample splits
    alts_idxs = np.arange(total_alts)
    shuffled = swop_exp(key, alts_idxs)
    samples = np.array_split(shuffled, N_SAMP_RATES)

    result_arr = np.zeros((N_SAMP_RATES, total_alts), dtype=np.float32)
    samp_alts_idxs = np.asarray([], dtype=int)
    for i, sample in enumerate(samples):

        samp_alts_idxs = np.concatenate((samp_alts_idxs, sample))

        # compute sample probs
        samp_utils = full_utils[samp_alts_idxs]
        samp_probs = softmax(samp_utils).flatten()
        del samp_utils

        # sparsify
        probs_samp_sparse = np.zeros_like(full_utils, dtype=np.float32)
        probs_samp_sparse = probs_samp_sparse.at[samp_alts_idxs].set(
            samp_probs)
        del samp_probs

        result_arr = result_arr.at[i, :].set(probs_samp_sparse)

        del probs_samp_sparse

    # cleanup
    del full_utils
    del full_data
    del alts_idxs
    del shuffled
    del samples
    del samp_alts_idxs

    return result_arr


@jit
def interaction_probs_all_disc(alts, coeffs, scale, chooser_val_key):
    """ For discretionary activity location choice
    """
    total_alts = alts.shape[0]
    chooser_val, key = chooser_val_key

    # interact choosers/alts
    full_data = interact_choosers_alts_disc(chooser_val, alts)

    # interact and take dot product
    full_utils = np.dot(full_data, coeffs.T) * scale

    # sample splits
    alts_idxs = np.arange(total_alts)
    shuffled = swop_exp(key, alts_idxs)
    samples = np.array_split(shuffled, N_SAMP_RATES)

    result_arr = np.zeros((N_SAMP_RATES, total_alts), dtype=np.float32)
    samp_alts_idxs = np.asarray([], dtype=int)
    for i, sample in enumerate(samples):

        samp_alts_idxs = np.concatenate((samp_alts_idxs, sample))

        # compute sample probs
        samp_utils = full_utils[samp_alts_idxs]
        samp_probs = softmax(samp_utils).flatten()
        del samp_utils

        # sparsify
        probs_samp_sparse = np.zeros_like(full_utils, dtype=np.float32)
        probs_samp_sparse = probs_samp_sparse.at[samp_alts_idxs].set(
            samp_probs)
        del samp_probs

        result_arr = result_arr.at[i, :].set(probs_samp_sparse)

        del probs_samp_sparse

    # cleanup
    del full_utils
    del full_data
    del alts_idxs
    del shuffled
    del samples
    del samp_alts_idxs

    return result_arr


@jit
def interaction_probs_all_w_max(alts, coeffs, scale, chooser_val_key):
    """ n-1 samp + max
    """
    total_alts = alts.shape[0]
    chooser_val, key = chooser_val_key

    # interact choosers/alts
    full_data = interact_choosers_alts(chooser_val, alts)

    # interact and take dot product
    full_utils = np.dot(full_data, coeffs.T) * scale

    # true probs
    true_probs = softmax(full_utils).flatten()
    max_alt = np.argmax(true_probs)
    del true_probs

    # sample splits
    alts_idxs = np.arange(total_alts)
    shuffled = swop_exp(key, alts_idxs)
    samples = np.array_split(shuffled, N_SAMP_RATES)

    result_arr = np.zeros((N_SAMP_RATES, total_alts), dtype=np.float32)
    samp_alts_idxs = np.asarray([], dtype=int)
    for i, sample in enumerate(samples):

        samp_alts_idxs = np.concatenate((samp_alts_idxs, sample))
        n_rands = len(samp_alts_idxs) - 1
        mask = np.where(samp_alts_idxs != max_alt, size=n_rands)
        this_idxs = samp_alts_idxs.at[mask].get()
        this_idxs = np.append(this_idxs, max_alt)

        # compute sample probs
        samp_utils = full_utils[this_idxs]
        samp_probs = softmax(samp_utils).flatten()
        del samp_utils

        # sparsify
        probs_samp_sparse = np.zeros_like(full_utils, dtype=np.float32)
        probs_samp_sparse = probs_samp_sparse.at[this_idxs].set(samp_probs)
        del samp_probs

        result_arr = result_arr.at[i, :].set(probs_samp_sparse)

        del probs_samp_sparse

    # cleanup
    del full_utils
    del full_data
    del alts_idxs
    del shuffled
    del samples
    del samp_alts_idxs
    del this_idxs
    del mask

    return result_arr


@jit
def interaction_probs_all_w_half_max(alts, coeffs, scale, chooser_val_key):
    """ n-1 samp + max
    """
    total_alts = alts.shape[0]
    chooser_val, key = chooser_val_key
    foo

    # interact choosers/alts
    full_data = interact_choosers_alts(chooser_val, alts)

    # interact and take dot product
    full_utils = np.dot(full_data, coeffs.T) * scale
    argsort = np.argsort(full_utils)
    del full_data

    # half maxs
    rates = np.linspace(.1, 1, 10)
    half_maxs = [argsort[-(total_alts * rate):] for rate in rates]

    # sample splits
    alts_idxs = np.arange(total_alts)
    shuffled = swop_exp(key, alts_idxs)
    samples = np.array_split(shuffled, N_SAMP_RATES * 2)

    result_arr = np.zeros((N_SAMP_RATES, total_alts), dtype=np.float32)
    samp_alts_idxs = np.asarray([], dtype=int)
    for i, sample in enumerate(samples):

        samp_alts_idxs = np.concatenate((samp_alts_idxs, sample))
        n_rands = int(len(samp_alts_idxs) / 2)
        max_idxs = half_maxs[i]
        mask = np.where(
            np.isin(samp_alts_idxs, max_idxs, invert=True), size=n_rands)
        this_idxs = samp_alts_idxs.at[mask].get()
        this_idxs = np.append(this_idxs, max_idxs)

        # compute sample probs
        samp_utils = full_utils[this_idxs]
        samp_probs = softmax(samp_utils).flatten()
        del samp_utils

        # sparsify
        probs_samp_sparse = np.zeros_like(full_utils, dtype=np.float32)
        probs_samp_sparse = probs_samp_sparse.at[this_idxs].set(samp_probs)
        del samp_probs

        result_arr = result_arr.at[i, :].set(probs_samp_sparse)

        del probs_samp_sparse

    # cleanup
    del full_utils
    del alts_idxs
    del shuffled
    del samples
    del samp_alts_idxs
    del max_idxs
    del this_idxs
    del mask
    del argsort

    return result_arr


@jit
def interaction_probs_all_weighted(alts, coeffs, scale, chooser_val_key):
    """ resample according to true probs
    """
    total_alts = alts.shape[0]
    chooser_val, key = chooser_val_key

    # interact choosers/alts
    full_data = interact_choosers_alts(chooser_val, alts)

    # take dot product and apply scale param
    full_utils = np.dot(full_data, coeffs.T) * scale
    del full_data

    # true probs
    true_probs = softmax(full_utils).flatten()

    # sample
    alts_idxs = np.arange(total_alts)
    result_arr = np.zeros((N_SAMP_RATES, total_alts), dtype=np.float32)
    result_arr = result_arr.at[9, :].set(true_probs)

    for i in range(9):
        n_samp = int((i + 1) * .1 * total_alts)
        samp_alts_idxs = random.choice(
            key, alts_idxs, (n_samp,), replace=False, p=true_probs)

        # compute sample probs
        samp_utils = full_utils[samp_alts_idxs] - np.log(1 / true_probs[samp_alts_idxs])
        samp_probs = softmax(samp_utils).flatten()
        del samp_utils

        # sparsify
        probs_samp_sparse = np.zeros_like(full_utils, dtype=np.float32)
        probs_samp_sparse = probs_samp_sparse.at[samp_alts_idxs].set(samp_probs)
        del samp_probs

        result_arr = result_arr.at[i, :].set(probs_samp_sparse)

        del probs_samp_sparse

    # cleanup
    del full_utils
    del alts_idxs
    del samp_alts_idxs
    del true_probs

    return result_arr


@jit
def interaction_probs_all_strat(alts, coeffs, scale, chooser_val_key):
    """ resample according to true probs
    """
    num_strata = 10
    total_alts = alts.shape[0]
    chooser_val, key = chooser_val_key

    # interact choosers/alts
    full_data = interact_choosers_alts(chooser_val, alts)

    # interact and take dot product
    full_utils = np.dot(full_data, coeffs.T) * scale
    del full_data

    # true probs
    true_probs = softmax(full_utils).flatten()

    # strata probs
    idxs_sorted = np.argsort(true_probs)
    strata = np.array_split(idxs_sorted, num_strata)

    # sample
    alts_idxs = np.arange(total_alts)
    result_arr = np.zeros((N_SAMP_RATES, total_alts), dtype=np.float32)
    result_arr = result_arr.at[9, :].set(true_probs)

    for i in range(9):
        n_samp = (i + 1) * .1 * total_alts
        samp_alts_idxs = np.zeros(int(n_samp), dtype=int)
        strata_n_samp = int(n_samp / num_strata)
        for s in range(num_strata):
            strata_samp_idxs = random.choice(key, strata[s], (strata_n_samp,), replace=False)
            start_idx = s * strata_n_samp
            end_idx = start_idx + strata_n_samp
            samp_alts_idxs = samp_alts_idxs.at[start_idx:end_idx].set(strata_samp_idxs)
            # samp_alts_idxs = random.choice(
            #     key, alts_idxs, (int(n_samp),), replace=False, p=strat_probs)

        # compute sample probs
        samp_utils = full_utils[samp_alts_idxs] #+ np.log(strat_probs[samp_alts_idxs])
        samp_probs = softmax(samp_utils).flatten()
        del samp_utils

        # sparsify
        probs_samp_sparse = np.zeros_like(full_utils, dtype=np.float32)
        probs_samp_sparse = probs_samp_sparse.at[samp_alts_idxs].set(samp_probs)
        del samp_probs

        result_arr = result_arr.at[i, :].set(probs_samp_sparse)

        del probs_samp_sparse

    # cleanup
    del full_utils
    del alts_idxs
    del samp_alts_idxs
    del true_probs

    return result_arr


@jit
def interaction_prob_errs_all(alts, coeffs, scale, chooser_val_key):
    """ 
    """
    total_alts = alts.shape[0]
    chooser_val, key = chooser_val_key

    full_data = interact_choosers_alts(chooser_val, alts)

    true_utils = interaction_utils(alts, coeffs, scale, chooser_val)
    true_probs = softmax(true_utils, scale=scale).flatten()
    true_probs_sd = true_probs.std()

    # sample splits
    alts_idxs = np.arange(total_alts)
    shuffled = swop_exp(key, alts_idxs)
    samples = np.array_split(shuffled, N_SAMP_RATES)

    result_arr = np.zeros((total_alts, 10), dtype=np.float16)
    samp_alts_idxs = np.asarray([], dtype=int)
    for i, sample in enumerate(samples):

        sample_rate = (i + 1) / N_SAMP_RATES
        sample_size = total_alts * sample_rate
        samp_alts_idxs = np.concatenate((samp_alts_idxs, sample))
        samp_data = full_data[samp_alts_idxs, :]

        # compute sample probs
        samp_utils = np.dot(samp_data, coeffs.T) * scale
        samp_probs = softmax(samp_utils).flatten()

        # sparsify
        probs_samp_sparse = np.zeros_like(true_probs)
        probs_samp_sparse = probs_samp_sparse.at[samp_alts_idxs].set(samp_probs)

        # errs
        result_arr = result_arr.at[:, i].set(np.float16(probs_samp_sparse - true_probs))

        del samp_data
        del samp_utils
        del samp_probs
        del probs_samp_sparse

    # cleanup
    del true_utils
    del full_data
    del true_probs
    del alts_idxs
    del shuffled
    del samples
    del samp_alts_idxs

    return result_arr


def get_probs(
        choosers, alts, coeffs, key, sample_size=None,
        scale=1, utils=False, hed=False, sum_probs=False,
        max_mct_size=1200000000):
    """ Original function to get jitted prob metrics

    DEPRECATED now bc the jitted funcs return all chooser alt-probs, which
    will run out of memory at ~20k alts.
    """
    now = time.time()
    num_choosers = choosers.shape[0]
    key_dim = key.shape[0]
    keys = random.split(key, num_choosers)
    num_alts = alts.shape[0]
    mct_size = num_choosers * num_alts
    batch_size_dict = {20000: 5, 200000: 1000, 2000000: 50000}

    if sample_size:
        gpu_func = interaction_sample_probs
        if utils:
            gpu_func = interaction_sample_utils
    else:
        gpu_func = interaction_probs
        if utils:
            gpu_func = interaction_utils
    if (batched) & (mct_size > max_mct_size):

        n_chooser_batches = batch_size_dict[num_alts]
        choosers_per_batch = int(num_choosers / n_chooser_batches)
        if sample_size is None:
            print(
                "Computing probabilities in {0} batches of {1} choosers".format(
                    n_chooser_batches, choosers_per_batch))
        choosers = choosers.reshape(
            (n_chooser_batches, choosers_per_batch))
        keys = keys.reshape(
            (n_chooser_batches, choosers_per_batch, key_dim))

        if sample_size:
            partial_interaction = partial(
                gpu_func,
                alts,
                coeffs,
                scale,
                sample_size)
            if sum_probs:
                results = []
                for i in tqdm(range(n_chooser_batches)):
                    chooser_batch = choosers[i]
                    key_batch = keys[i]
                    alt_idxs, probs = vmap(partial_interaction)((chooser_batch, key_batch))
                    probs_samp_sparse = onp.zeros((choosers_per_batch, num_alts))
                    probs_samp_sparse[
                        np.arange(choosers_per_batch).repeat(sample_size),
                        alt_idxs.flatten()] = probs.flatten()
                    result = probs_samp_sparse.sum(axis=0)
                    results.append(result)
                results = np.array(results).sum(axis=0)
            else:
                alt_idxs, probs = lax.map(
                    vmap(partial_interaction), (choosers, keys))
                results = probs.reshape((num_choosers, num_alts))
        else:
            partial_interaction = partial(
                gpu_func,
                alts,
                coeffs,
                scale)
            if sum_probs:
                results = []
                for chooser_batch in choosers:
                    probs = vmap(partial_interaction)(chooser_batch)
                    result = probs.sum(axis=0)
                    results.append(result)
                results = np.array(results).sum(axis=0)
            else:
                probs = lax.map(
                    vmap(partial_interaction), (choosers))
                results = results.reshape((num_choosers, num_alts))

    else:
        if sample_size:
            alt_idxs, probs = vmap(
                gpu_func,
                in_axes=(None, None, None, None, 0))(
                alts, coeffs, scale, sample_size,
                (choosers, keys))
            if sum_probs:
                results = onp.zeros((num_choosers, num_alts))
                results[
                    np.arange(num_choosers).repeat(sample_size),
                    alt_idxs.flatten()] = probs.flatten()
        else:
            results = vmap(
                gpu_func, in_axes=(None, None, None, 0))(
                alts, coeffs, scale, (choosers))
        if sum_probs:
            results = results.sum(axis=0)
    later = time.time()
    if sample_size is None:
        print(
            "Took {0} s to compute true probabilities.".format(
                np.round(later - now), 1))

    return results


def get_all_probs(
        choosers, alts, coeffs, key,
        scale=1, batched=False, verbose=True,
        max_mct_size=1200000000):
    """ Original function to get jitted probs
    """
    now = time.time()
    num_choosers = choosers.shape[0]
    key_dim = key.shape[0]
    keys = random.split(key, num_choosers)
    num_alts = alts.shape[0]
    mct_size = num_choosers * num_alts
    batch_size_dict = {
        20000: 8,
        200000: 750,
        750000: 8,
        2000000: int(num_choosers / 100) if num_choosers % 100 == 0 else int(num_choosers / 75),
        7500000: 7500
        }
    gpu_func = interaction_probs_all
    batch_size_dict = {20000: 8, 200000: 1000, 2000000: 75000}
    # gpu_func = interaction_probs_all_w_max
    # batch_size_dict = {20000: 10, 200000: 1000, 2000000: 75000}
    # gpu_func = interaction_probs_all_w_half_max
    # batch_size_dict = {20000: 10, 200000: 1000, 2000000: 100000}
    # gpu_func = interaction_probs_all_weighted
    # gpu_func = interaction_probs_all_strat
    if (batched) & (mct_size > max_mct_size):

        if len(str(num_choosers)) > len(str(num_alts)):
            batch_size_lookup = num_choosers
        else:
            batch_size_lookup = num_alts 
        n_chooser_batches = batch_size_dict[batch_size_lookup]
        choosers_per_batch = int(num_choosers / n_chooser_batches)
        if verbose:
            print(
                "Computing probabilities in {0} batches of "
                "{1} choosers".format(
                    n_chooser_batches, choosers_per_batch))
        choosers = choosers.reshape(
            (n_chooser_batches, choosers_per_batch))
        keys = keys.reshape(
            (n_chooser_batches, choosers_per_batch, key_dim))

        partial_interaction = partial(
            gpu_func,
            alts,
            coeffs,
            scale)

        # store results on cpu to preserve space on GPU for computation
        cpu = devices("cpu")[0]
        results = device_put(onp.zeros((10, num_alts)), device=cpu)

        # process chooser batches
        disable = not verbose
        for i in tqdm(range(n_chooser_batches), disable=disable):
            chooser_batch = choosers[i]
            key_batch = keys[i]
            probs = vmap(partial_interaction)((chooser_batch, key_batch))
            assert np.isclose(np.nansum(probs, axis=2), 1.0).all()
            results = results + np.nansum(probs, axis=0)
            del probs

    else:
        results = vmap(
            gpu_func, in_axes=(None, None, None, 0))(
            alts, coeffs, scale, (choosers, keys))
        assert np.isclose(np.nansum(results, axis=2), 1.0).all()
        results = results.sum(axis=0)

    later = time.time()
    if verbose:
        print(
            "Took {0} s to compute true probabilities.".format(
                np.round(later - now), 1))

    return results


def get_all_prob_errs(
        choosers, alts, coeffs, key,
        scale=1, batched=False, max_mct_size=1200000000):
    """ Original function to get jitted prob metrics

    DEPRECATED now bc the jitted funcs return all chooser alt-probs, which
    will run out of memory at ~20k alts.
    """
    now = time.time()
    num_choosers = choosers.shape[0]
    key_dim = key.shape[0]
    keys = random.split(key, num_choosers)
    num_alts = alts.shape[0]
    mct_size = num_choosers * num_alts
    batch_size_dict = {20000: 8, 200000: 600, 2000000: 50000}

    gpu_func = interaction_prob_errs_all
    if (batched) & (mct_size > max_mct_size):

        n_chooser_batches = batch_size_dict[num_alts]
        choosers_per_batch = int(num_choosers / n_chooser_batches)
        print(
            "Computing probabilities in {0} batches of {1} choosers".format(
                n_chooser_batches, choosers_per_batch))
        choosers = choosers.reshape(
            (n_chooser_batches, choosers_per_batch))
        keys = keys.reshape(
            (n_chooser_batches, choosers_per_batch, key_dim))

        partial_interaction = partial(
            gpu_func,
            alts,
            coeffs,
            scale)

        # store results on cpu to preserve space on GPU for computation
        cpu = devices("cpu")[0]
        massing_err = device_put(onp.zeros((num_alts, 10)), device=cpu)

        # process chooser batches
        for i in tqdm(range(n_chooser_batches)):
            chooser_batch = choosers[i]
            key_batch = keys[i]
            errs = vmap(partial_interaction)((chooser_batch, key_batch))
            massing_err = massing_err + errs.sum(axis=0)
            del errs

    else:
        errs = vmap(
            gpu_func, in_axes=(None, None, None, 0))(
            alts, coeffs, scale, (choosers, keys))
        massing_err = errs.sum(axis=0)
        del errs
        
    later = time.time()
    print(
        "Took {0} s to compute true probabilities.".format(
            np.round(later - now), 1))

    tot_abs_err = np.abs(massing_err).sum(axis=0)
    rmse = np.sqrt(np.mean(massing_err * massing_err, axis=0))

    del massing_err

    return np.vstack((tot_abs_err, rmse)).T


def reshape_batched_iter_results(results, num_choosers, num_metric_cols):
    """ DEPRECATED

    Use reshape_batched_iter_chooser_results instead
    """

    if len(results) == 2:
        true_choices = results[0]
        true_choices = true_choices.reshape((num_choosers,))
        samp_metrics = results[1]
        samp_metrics = samp_metrics.reshape((
            num_choosers, N_SAMP_RATES, num_metric_cols))

        results = (true_choices, samp_metrics)

    else:
        results = results.reshape((
            num_choosers, N_SAMP_RATES, num_metric_cols))

    return results


def get_iter_probs(choosers, alts, coeffs, key, batched=False, num_metric_cols=14, max_mct_size=1200000000):
    """ DEPRECATED by method that gets choice counts for alt shares computation

    Use get_iter_chooser_errs_w_choices
    """
    now = time.time()
    num_choosers = choosers.shape[0]
    key_dim = key.shape[0]
    keys = random.split(key, num_choosers)
    num_alts = alts.shape[0]
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
            interaction_iters_chooser_err_metrics,
            alts,
            coeffs,
            pop_mean_prob)
        results = lax.map(
            vmap(partial_interaction), (choosers, keys))
        results = reshape_batched_iter_results(
            results, num_choosers, num_metric_cols)

    else:

        results = vmap(
            interaction_iters_chooser_err_metrics, in_axes=(
                None, None, None, 0))(
            alts, coeffs, pop_mean_prob, (choosers, keys))

    later = time.time()
    print(
        "Took {0} s to compute prob. metrics for all sample "
        "rates by chooser.".format(np.round(later - now)))

    return results


@jit
@remat
def interaction_iters_chooser_err_metrics(alts, coeffs, pop_mean_prob, chooser_val_key):
    """ DEPRECATED

    Use interaction_iters_chooser_err_metrics_w_choices
    """

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
    samples = np.array_split(shuffled, N_SAMP_RATES)

    # sample probs
    samp_alts_idxs = np.asarray([], dtype=int)
    result_arr = np.zeros((N_SAMP_RATES - 1, 14))
    for i, sample in enumerate(samples):

        if i == N_SAMP_RATES - 1:
            continue
        sample_rate = (i + 1) / N_SAMP_RATES
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


def get_iter_chooser_errs_w_choices(choosers, alts, coeffs, key, batched=False, num_metric_cols=ERR_METRIC_ROW_SIZE, max_mct_size=1200000000):
    """ DEPRECATED

    Use get_iter_metrics, which combined this method with
    get_iter_chooser_errs_w_choices

    Computes error metrics for each sample rate, plus the true choices and
    sample choices for computing alt shares.

    Since the array is a matrix of error metrics comparing sampled
    results to true (unsampled), there are no entries in the matrix for
    the true (unsampled) results.
    """
    now = time.time()
    num_choosers = choosers.shape[0]
    key_dim = key.shape[0]
    keys = random.split(key, num_choosers)
    num_alts = alts.shape[0]
    pop_mean_prob = 1 / num_alts
    mct_size = num_choosers * num_alts
    if (batched) & (mct_size > max_mct_size):
        n_chooser_batches = batch_size_dict[num_alts]
        choosers_per_batch = int(num_choosers / n_chooser_batches)
        print(
            "Computing probabilities in {0} batches of {1} choosers".format(
                n_chooser_batches, choosers_per_batch))
        choosers = choosers.reshape(
            (n_chooser_batches, choosers_per_batch))
        keys = keys.reshape(
            (n_chooser_batches, choosers_per_batch, key_dim))

        partial_interaction = partial(
            interaction_iters_chooser_err_metrics_w_choices,
            alts,
            coeffs,
            pop_mean_prob)
        results = lax.map(
            vmap(partial_interaction), (choosers, keys))
        results = reshape_batched_iter_chooser_results(
            results, num_choosers, N_SAMP_RATES, num_metric_cols)

    else:

        results = vmap(
            interaction_iters_chooser_err_metrics_w_choices, in_axes=(
                None, None, None, 0))(
            alts, coeffs, pop_mean_prob, (choosers, keys))

    later = time.time()
    print(
        "Took {0} s to compute prob. metrics for all sample "
        "rates by chooser.".format(np.round(later - now)))

    return results


def get_iter_chooser_probs(choosers, alts, coeffs, key, batched=False, max_mct_size=1200000000):
    """ DEPRECATED

    Use get_iter_metrics, which combined this method with
    get_iter_chooser_errs_w_choices
    """

    now = time.time()
    num_choosers = choosers.shape[0]
    key_dim = key.shape[0]
    keys = random.split(key, num_choosers)
    num_alts = alts.shape[0]
    pop_mean_prob = 1 / num_alts
    mct_size = num_choosers * num_alts
    if (batched) & (mct_size > max_mct_size):
        n_chooser_batches = batch_size_dict[num_alts]
        choosers_per_batch = int(num_choosers / n_chooser_batches)
        print(
            "Computing probabilities in {0} batches of {1} choosers".format(
                n_chooser_batches, choosers_per_batch))
        choosers = choosers.reshape(
            (n_chooser_batches, choosers_per_batch))
        keys = keys.reshape(
            (n_chooser_batches, choosers_per_batch, key_dim))

        partial_interaction = partial(
            interaction_iters_chooser_probs,
            alts,
            coeffs,
            pop_mean_prob)
        results = lax.map(
            vmap(partial_interaction), (choosers, keys))
        results = reshape_batched_iter_chooser_results(
            results, num_choosers, N_SAMP_RATES + 1, CHOOSER_PROB_ROW_SIZE)

    else:

        results = vmap(
            interaction_iters_chooser_probs, in_axes=(None, None, None, 0))(
            alts, coeffs, pop_mean_prob, (choosers, keys))

    later = time.time()
    print(
        "Took {0} s to compute prob. metrics for all sample "
        "rates by chooser.".format(np.round(later - now, 1)))

    return results


@jit
@remat
def interaction_iters_chooser_err_metrics_w_choices(
        alts, coeffs, pop_mean_prob, chooser_val_key):
    """ Generate chooser-level error metrics for each sample-rate

    Returns n x 16 matrix of prob-metrics, where n is the number
    of sample rates used in the experiment. Includes true choice
    and sample choice columns to faciliate computation of alts
    shares across population of choosers.
    """

    chooser_val, key = chooser_val_key
    total_alts = alts.shape[0]

    # perform interaction
    idx_alt_intx = -1
    interacted = np.sqrt(alts[:, idx_alt_intx] / chooser_val)
    alts2 = alts.at[:, idx_alt_intx].set(interacted)

    # compute probs
    logits = np.dot(alts2, coeffs.T)
    probas = softmax(logits)
    probas = probas.flatten()
    true_choice = np.nanargmax(probas)
    true_max = probas[true_choice]
    pct_true_probs_gt_pop_mean = np.sum(
        probas > pop_mean_prob) / total_alts

    # sample splits
    alts_idxs = np.arange(total_alts)
    shuffled = swop_exp(key, alts_idxs)
    samples = np.array_split(shuffled, N_SAMP_RATES)

    # sample probs
    samp_alts_idxs = np.asarray([], dtype=int)
    result_arr = np.zeros((N_SAMP_RATES, ERR_METRIC_ROW_SIZE))
    for i, sample in enumerate(samples):
        sample_rate = (i + 1) / N_SAMP_RATES
        sample_size = total_alts * sample_rate
        samp_alts_idxs = np.concatenate((samp_alts_idxs, sample))
        alts3 = alts2[samp_alts_idxs]
        samp_logits = np.dot(alts3, coeffs.T)
        samp_probs = softmax(samp_logits)
        samp_probs = samp_probs.flatten()
        samp_max_idx = np.nanargmax(samp_probs)
        samp_max = samp_probs[samp_max_idx]
        samp_corr_max = samp_max * sample_rate
        samp_choice = samp_alts_idxs[samp_max_idx]
        true_prob_samp_max = probas[samp_choice]
        pct_samp_probs_gt_pop_mean = np.sum(
            samp_probs > pop_mean_prob) / sample_size
        pct_samp_corr_probs_gt_pop_mean = np.sum(
            (samp_probs * sample_rate) > pop_mean_prob) / sample_size

        tmsm_err = true_prob_samp_max - true_max
        tmsm_pct_err = tmsm_err / true_max
        tmsm_sq_err = tmsm_err * tmsm_err

        tmsmc_err = samp_corr_max - true_max
        tmsmc_pct_err = tmsmc_err / true_max
        tmsmc_sq_err = tmsmc_err * tmsmc_err

        cf_true = true_prob_samp_max / samp_max
        cf_err = sample_rate - cf_true
        cf_pct_err = cf_err / cf_true
        cf_sq_err = cf_err * cf_err

        ppgm_err = pct_samp_probs_gt_pop_mean - pct_true_probs_gt_pop_mean
        ppgm_pct_err = ppgm_err / pct_true_probs_gt_pop_mean
        ppgm_sq_err = ppgm_err * ppgm_err

        pcpgm_err = pct_samp_corr_probs_gt_pop_mean - \
            pct_true_probs_gt_pop_mean
        pcpgm_pct_err = pcpgm_err / pct_true_probs_gt_pop_mean
        pcpgm_sq_err = pcpgm_err * pcpgm_err

        result_arr = result_arr.at[i, :].set(np.array([
            total_alts, sample_rate, true_choice, samp_choice,
            tmsm_err, tmsm_pct_err, tmsm_sq_err,
            tmsmc_err, tmsmc_pct_err, tmsmc_sq_err,
            cf_err, cf_pct_err, cf_sq_err,
            ppgm_err, ppgm_pct_err, ppgm_sq_err,
            pcpgm_err, pcpgm_pct_err, pcpgm_sq_err]))

    del chooser_val
    del key
    del chooser_val_key
    del interacted
    del alts2
    del logits
    del probas
    del alts_idxs
    del shuffled
    del samples
    del samp_alts_idxs
    del samp_logits
    del samp_probs

    return result_arr


@jit
@remat
def interaction_iters_chooser_probs(
        alts, coeffs, pop_mean_prob, chooser_val_key):
    """ Generate chooser-level choices/max. probs plus metrics (e.g.
        ppgm, dispersion err) that can only be computed when the
        full probs are available.

    Returns n x 5 matrix of prob-metrics, where n is 1 + the
    number of sample rates used in the experiment.
    """

    chooser_val, key = chooser_val_key
    total_alts = alts.shape[0]

    # perform interaction
    idx_alt_intx = -1
    interacted = np.sqrt(alts[:, idx_alt_intx] / chooser_val)
    alts2 = alts.at[:, idx_alt_intx].set(interacted)

    # compute probs
    logits = np.dot(alts2, coeffs.T)
    probas = softmax(logits)
    probas = probas.flatten()
    true_choice = np.nanargmax(probas)
    true_max = probas[true_choice]
    pct_true_probs_gt_pop_mean = np.sum(
        probas > pop_mean_prob) / total_alts
    result_arr = np.array([
        -1, true_choice, true_max, np.nan, pct_true_probs_gt_pop_mean, np.nan])

    # sample splits
    alts_idxs = np.arange(total_alts)
    shuffled = swop_exp(key, alts_idxs)
    samples = np.array_split(shuffled, N_SAMP_RATES)

    # sample probs
    samp_alts_idxs = np.asarray([], dtype=int)
    for i, sample in enumerate(samples):
        sample_rate = (i + 1) / N_SAMP_RATES
        sample_size = total_alts * sample_rate
        samp_alts_idxs = np.concatenate((samp_alts_idxs, sample))
        alts3 = alts2[samp_alts_idxs]
        samp_logits = np.dot(alts3, coeffs.T)
        samp_probs = softmax(samp_logits)
        samp_probs = samp_probs.flatten()
        samp_max_idx = np.nanargmax(samp_probs)
        samp_max_prob = samp_probs[samp_max_idx]
        samp_choice = samp_alts_idxs[samp_max_idx]
        true_prob_samp_max = probas[samp_choice]
        pct_samp_probs_gt_pop_mean = np.sum(
            samp_probs > pop_mean_prob) / sample_size
        pct_samp_corr_probs_gt_pop_mean = np.sum(
            samp_probs * sample_rate > pop_mean_prob) / sample_size

        result_arr = np.vstack((result_arr, [
            sample_rate, samp_choice, samp_max_prob, true_prob_samp_max,
            pct_samp_probs_gt_pop_mean, pct_samp_corr_probs_gt_pop_mean]))

    del chooser_val
    del key
    del chooser_val_key
    del interacted
    del alts2
    del logits
    del probas
    del alts_idxs
    del shuffled
    del samples
    del samp_alts_idxs
    del samp_logits
    del samp_probs

    return result_arr


@jit
@remat
def interaction_iters_chooser_probs_w_massing(
        alts, coeffs, pop_mean_prob, alt_err_totals, chooser_val_key):
    """ Generate chooser-level choices/max. probs plus metrics (e.g.
        ppgm, dispersion err) that can only be computed when the
        full probs are available.

    Returns n x 5 matrix of prob-metrics, where n is 1 + the
    number of sample rates used in the experiment.
    """

    chooser_val, key = chooser_val_key
    total_alts = alts.shape[0]

    # perform interaction
    idx_alt_intx = -1
    interacted = np.sqrt(alts[:, idx_alt_intx] / chooser_val)
    alts2 = alts.at[:, idx_alt_intx].set(interacted)

    # compute probs
    logits = np.dot(alts2, coeffs.T)
    probas = softmax(logits)
    probas = probas.flatten()
    true_choice = np.nanargmax(probas)
    true_max = probas[true_choice]
    pct_true_probs_gt_pop_mean = np.sum(
        probas > pop_mean_prob) / total_alts
    result_arr = np.array([
        -1, true_choice, true_max, np.nan, pct_true_probs_gt_pop_mean, np.nan])

    # sample splits
    alts_idxs = np.arange(total_alts)
    shuffled = swop_exp(key, alts_idxs)
    samples = np.array_split(shuffled, N_SAMP_RATES)

    # sample probs
    samp_alts_idxs = np.asarray([], dtype=int)
    for i, sample in enumerate(samples):
        probs_samp_sparse = np.zeros_like(probas)
        sample_rate = (i + 1) / N_SAMP_RATES
        sample_size = total_alts * sample_rate
        samp_alts_idxs = np.concatenate((samp_alts_idxs, sample))
        alts3 = alts2[samp_alts_idxs]
        samp_logits = np.dot(alts3, coeffs.T)
        samp_probs = softmax(samp_logits)
        samp_probs = samp_probs.flatten()
        probs_samp_sparse = probs_samp_sparse.at[samp_alts_idxs].set(samp_probs)
        probs_err = (probs_samp_sparse - probas)
        err_totals = alt_err_totals.at[i, :].get()
        new_err_total = err_totals + probs_err
        alt_err_totals = alt_err_totals.at[i, :].set(new_err_total)
        samp_max_idx = np.nanargmax(samp_probs)
        samp_max_prob = samp_probs[samp_max_idx]
        samp_choice = samp_alts_idxs[samp_max_idx]
        true_prob_samp_max = probas[samp_choice]
        pct_samp_probs_gt_pop_mean = np.sum(
            samp_probs > pop_mean_prob) / sample_size
        pct_samp_corr_probs_gt_pop_mean = np.sum(
            samp_probs * sample_rate > pop_mean_prob) / sample_size

        result_arr = np.vstack((result_arr, [
            sample_rate, samp_choice, samp_max_prob, true_prob_samp_max,
            pct_samp_probs_gt_pop_mean, pct_samp_corr_probs_gt_pop_mean]))

    del chooser_val
    del key
    del chooser_val_key
    del interacted
    del alts2
    del logits
    del probas
    del alts_idxs
    del shuffled
    del samples
    del samp_alts_idxs
    del samp_logits
    del samp_probs

    return result_arr


def reshape_batched_iter_chooser_results(
        results, num_choosers, num_sample_iters, num_metric_cols):
    results = results.reshape((
        num_choosers, num_sample_iters, num_metric_cols))

    return onp.array(results)


def get_iter_metrics(
        choosers, alts, coeffs, key,
        batched=False, max_mct_size=1200000000, gpu_func='probs'):

    num_choosers = choosers.shape[0]
    key_dim = key.shape[0]
    keys = random.split(key, num_choosers)
    num_alts = alts.shape[0]
    pop_mean_prob = 1 / num_alts
    mct_size = num_choosers * num_alts

    # compute err metrics one chooser at a time inside the jitted function
    if gpu_func == 'errs':
        batch_size_dict = {20000: 6, 200000: 600, 2000000: 60000}
        interaction_func = interaction_iters_chooser_err_metrics_w_choices
        num_metric_cols = ERR_METRIC_ROW_SIZE
        n_samp_idxs = N_SAMP_RATES

    # jitted function just gets prob metrics for each chooser. error
    # metrics for each sample rate are computed on CPU for all choosers
    # at once and then stored to disk. requires an extra post-processing
    # step, but it reduces the GPU memory footprint which allows us to
    # process the choosers in fewer, larger batches, reducing runtimes.
    elif gpu_func == 'probs':
        batch_size_dict = {20000: 5, 200000: 500, 2000000: 50000}
        interaction_func = interaction_iters_chooser_probs_w_massing
        alt_err_totals = np.zeros((N_SAMP_RATES, num_alts))
        num_metric_cols = CHOOSER_PROB_ROW_SIZE
        n_samp_idxs = N_SAMP_RATES + 1

    if (batched) & (mct_size > max_mct_size):
        n_chooser_batches = batch_size_dict[num_alts]
        choosers_per_batch = int(num_choosers / n_chooser_batches)
        print(
            "Computing probabilities in {0} batches of {1} choosers".format(
                n_chooser_batches, choosers_per_batch))
        choosers = choosers.reshape(
            (n_chooser_batches, choosers_per_batch))
        keys = keys.reshape(
            (n_chooser_batches, choosers_per_batch, key_dim))

        partial_interaction = partial(
            interaction_func,
            alts,
            coeffs,
            pop_mean_prob, alt_err_totals)
        results = lax.map(
            vmap(partial_interaction), (choosers, keys))
        results = reshape_batched_iter_chooser_results(
            results, num_choosers, n_samp_idxs, num_metric_cols)

    else:

        results = vmap(
            interaction_func, in_axes=(None, None, None, None, 0))(
            alts, coeffs, pop_mean_prob, alt_err_totals, (choosers, keys))

    return results
