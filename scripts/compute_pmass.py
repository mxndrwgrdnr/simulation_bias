import pandas as pd
import numpy as np
import jax.random as random
import time
from jax.lib import xla_bridge
import argparse
import os

import large_mnl as lmnl

OUTFILE = "iter_pmass_err_v4.csv"
DATA_DIR = "../data"


def create_data(num_alts, num_choosers, num_vars=5):

    # create choosers
    choosers = np.random.lognormal(0, 0.5, num_choosers)  # 1 chooser attribute

    # initialize alts
    alts = np.zeros((num_alts, num_vars))

    # dist to CBD
    alts[:, 0] = np.random.lognormal(0, 1, num_alts)  # Dist to CBD

    # size terms
    for i in range(1, num_vars - 1):
        split_val = int(np.floor(num_alts * .5))
        alts[:split_val, i] = np.random.normal(
            1, 1, split_val)  # first 1/2 of alts have mean = 1
        alts[split_val:, i] = np.random.normal(
            0.5, 1, num_alts - split_val)  # rest of alts have mean 0.5

    # interaction term
    alts[:, -1] = np.random.lognormal(1, .5, num_alts)

    return choosers, alts


def run(
        num_alts, pop_to_alts_ratio, sample_rates, coeffs,
        key, batched, gpu_func):
    """ Get probs for one sample rate at a time
    """

    num_alts = int(num_alts)
    num_choosers = int(pop_to_alts_ratio * num_alts)
    choosers, alts = create_data(num_alts, num_choosers)
    num_choosers = choosers.shape[0]
    print(num_alts, " ALTS, ", num_choosers, " CHOOSERS")

    true_prob_mass = lmnl.get_probs(
        choosers, alts, coeffs, key, scale=1, batched=True, sum_probs=True)
    true_prob_mass_std = true_prob_mass.std()

    err_metrics = []
    now = time.time()
    for samp_rate in sample_rates:

        samp_size = int(samp_rate * num_alts)
        samp_prob_mass = lmnl.get_probs(
            choosers, alts, coeffs, key, sample_size=samp_size,
            scale=1, batched=True, sum_probs=True)
        samp_prob_mass_std = samp_prob_mass.std()

        massing_err = (samp_prob_mass - true_prob_mass)
        total_massing_err = np.abs(massing_err).sum()
        rmse = np.sqrt(np.mean(massing_err * massing_err))
        mape = np.mean(np.abs(massing_err / true_prob_mass))
        stddev_err = samp_prob_mass_std - true_prob_mass_std 
        stddev_pct_err = stddev_err / true_prob_mass_std
        err_metrics.append([samp_rate, total_massing_err, rmse, mape, stddev_err])

    later = time.time()
    print(
        "Took {0} s to compute all sampled probabilities.".format(
            np.round(later - now), 1))

    iter_df = pd.DataFrame(
        err_metrics, columns=['sample_rate', 'total_abs_err', 'rmse', 'mape', 'stddev_pe'])
    iter_df['num_alts'] = num_alts
    iter_df['num_choosers'] = num_choosers

    return iter_df


def run_v2(
        num_alts, pop_to_alts_ratio, sample_rates, coeffs,
        key, batched, scale=1, debug=True, verbose=True):
    """ Get probs for all sample rates at once
    """

    num_alts = int(num_alts)
    num_choosers = int(pop_to_alts_ratio * num_alts)
    choosers, alts = create_data(num_alts, num_choosers)
    num_choosers = choosers.shape[0]
    if verbose:
        print(num_alts, " ALTS, ", num_choosers, " CHOOSERS")

    all_prob_mass = lmnl.get_all_probs(
        choosers, alts, coeffs, key, scale=scale, batched=True,
        verbose=verbose)

    if debug:
        assert np.isclose(
            all_prob_mass.sum(axis=1), num_choosers, rtol=3e-4).all()

    true_prob_mass = all_prob_mass[-1, :]

    if debug:
        assert any(np.isnan(true_prob_mass)) is False

    true_prob_mass_std = true_prob_mass.std()

    err_metrics = []
    now = time.time()
    for i, samp_rate in enumerate(sample_rates):
        samp_prob_mass = all_prob_mass[i, :]
        samp_prob_mass_std = samp_prob_mass.std()
        massing_err = (samp_prob_mass - true_prob_mass)
        total_massing_err = np.abs(massing_err).sum()
        rmse = np.sqrt(np.mean(massing_err * massing_err))
        mape = np.nanmean(np.abs(massing_err / true_prob_mass))
        stddev_err = samp_prob_mass_std - true_prob_mass_std
        stddev_pct_err = stddev_err / true_prob_mass_std
        err_metrics.append([
            samp_rate, total_massing_err, rmse, mape, stddev_pct_err])

    later = time.time()
    if verbose:
        print(
            "Took {0} s to compute all sampled probabilities.".format(
                np.round(later - now), 1))

    iter_df = pd.DataFrame(
        err_metrics, columns=[
        'sample_rate', 'total_abs_err', 'rmse', 'mape', 'sd_pct_err'])
    iter_df['num_alts'] = num_alts
    iter_df['num_choosers'] = num_choosers

    del all_prob_mass
    del true_prob_mass
    del err_metrics

    return iter_df


def run_errs(
        num_alts, pop_to_alts_ratio, sample_rates, coeffs,
        key, batched, gpu_func):
    """ Get probs for all sample rates at once
    """

    num_alts = int(num_alts)
    num_choosers = int(pop_to_alts_ratio * num_alts)
    pop_mean_prob = 1 / num_alts
    choosers, alts = create_data(num_alts, num_choosers)
    num_choosers = choosers.shape[0]
    print(num_alts, " ALTS, ", num_choosers, " CHOOSERS")

    err_metrics = lmnl.get_all_prob_errs(
        choosers, alts, coeffs, key, scale=1, batched=True)

    iter_df = pd.DataFrame(
         np.hstack((np.array(sample_rates).reshape(-1,1), err_metrics)),
         columns=['sample_rate', 'tot_abs_err', 'rmse'])
    iter_df['num_alts'] = num_alts
    iter_df['num_choosers'] = num_choosers

    del err_metrics

    return iter_df


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batched", action="store_true")
    parser.add_argument("-n", "--num_bootstraps", action="store", default=0)
    parser.add_argument("-g", "--gpu_func", action="store", default='probs')
    parser.add_argument("-a", "--append", action="store_true")
    args = parser.parse_args()
    batched = args.batched
    gpu_func = args.gpu_func
    append = args.append

    output_fpath = os.path.join(DATA_DIR, OUTFILE)

    assert xla_bridge.get_backend().platform == 'gpu'

    alts_sizes = [
        # 200, 2000, 20000, 200000,
        2000000
    ]
    pop_to_alts_ratio = 750 / 200
    sample_rates = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
    coeffs = np.array([-1, 1, 1, 1, 1])  # dist-to-cbd, sizes 1-3, intx term

    if append & os.path.exists(output_fpath):
        pmass_err_df = pd.read_csv(output_fpath)
        run_id = pmass_err_df['run_id'].max()
        run_id += 1

    else:
        pmass_err_df = pd.DataFrame()
        run_id = 1

    key = random.PRNGKey(0)
    for i in range(10):

        print("Executing RUN ID #{0}".format(run_id))

        for num_alts in alts_sizes:

            # run for num_alts
            iter_err_df = run_v2(
                num_alts, pop_to_alts_ratio, sample_rates,
                coeffs, key, batched)

            # append to metrics df
            iter_err_cols = list(iter_err_df.columns)
            iter_err_df.loc[:, 'run_id'] = run_id
            iter_err_df = iter_err_df[['run_id'] + iter_err_cols]
            pmass_err_df = pd.concat((pmass_err_df, iter_err_df))

            # save intermediate outputs
            pmass_err_df.to_csv(output_fpath, index=False)

        run_id += 1
