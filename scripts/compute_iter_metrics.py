import pandas as pd
import numpy as np
import jax.random as random
import time
from jax.lib import xla_bridge
import argparse

import large_mnl as lmnl


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


def process_iter_prob_metrics(num_alts, result_metrics, sample_rates):

    iter_metrics = pd.DataFrame()
    now = time.time()
    for i, sample_rate in enumerate(sample_rates):
        sample_metrics = result_metrics[:, i, :]
        tmsm_mean_err = np.nanmean(sample_metrics[:, 2])
        tmsm_mean_abs_pct_err = np.nanmean(np.abs(sample_metrics[:, 3]))
        tmsm_rmse = np.sqrt(np.nanmean(sample_metrics[:, 4]))
        tmsmc_mean_err = np.nanmean(sample_metrics[:, 5])
        tmsmc_mean_abs_pct_err = np.nanmean(np.abs(sample_metrics[:, 6]))
        tmsmc_rmse = np.sqrt(np.nanmean(sample_metrics[:, 7]))
        cf_mean_err = np.nanmean(sample_metrics[:, 8])
        cf_mean_abs_pct_err = np.nanmean(np.abs(sample_metrics[:, 9]))
        cf_rmse = np.sqrt(np.nanmean(sample_metrics[:, 10]))
        ppgm_mean_err = np.nanmean(sample_metrics[:, 11])
        ppgm_mean_abs_pct_err = np.nanmean(np.abs(sample_metrics[:, 12]))
        ppgm_rmse = np.sqrt(np.nanmean(sample_metrics[:, 13]))

        iter_metrics = pd.concat((iter_metrics, pd.DataFrame([{
            'num_alts': num_alts,
            'num_choosers': num_choosers,
            'sample_rate': sample_rate,
            'tmsm_mean_err': tmsm_mean_err,
            'tmsm_mean_abs_pct_err': tmsm_mean_abs_pct_err,
            'tmsm_rmse': tmsm_rmse,
            'tmsmc_mean_err': tmsmc_mean_err,
            'tmsmc_mean_abs_pct_err': tmsmc_mean_abs_pct_err,
            'tmsmc_rmse': tmsmc_rmse,
            'cf_mean_err': cf_mean_err,
            'cf_mean_abs_pct_err': cf_mean_abs_pct_err,
            'cf_rmse': cf_rmse,
            'ppgm_mean_err': ppgm_mean_err,
            'ppgm_mean_abs_pct_err': ppgm_mean_abs_pct_err,
            'ppgm_rmse': ppgm_rmse}])))

    later = time.time()
    print(
        "Took {0} s to compute sample metrics for all "
        "sample rates.".format(np.round(later - now, 1)))

    return iter_metrics


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batched", action="store_true")
    parser.add_argument("-n", "--num_bootstraps", action="store")
    args = parser.parse_args()
    batched = args.batched

    print("Running on", xla_bridge.get_backend().platform, "!")

    alts_sizes = [200, 2000, 2e4, 2e5, 2e6]
    pop_to_alts_ratio = 750 / 200
    sample_rates = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
    coeffs = np.array([-1, 1, 1, 1, 1])

    metrics = pd.DataFrame()

    key = random.PRNGKey(0)
    for num_alts in alts_sizes:

        # get data
        num_alts = int(num_alts)
        num_choosers = int(pop_to_alts_ratio * num_alts)
        pop_mean_prob = 1 / num_alts
        choosers, alts = create_data(num_alts, num_choosers)
        num_choosers = choosers.shape[0]
        print(num_alts, " ALTS, ", num_choosers, " CHOOSERS")

        # compute probs and metrics
        result_metrics = lmnl.get_iter_probs(
            choosers, alts, key, num_alts, batched)
        num_alt_metrics = process_iter_prob_metrics(
            num_alts, result_metrics, sample_rates)

        # save intermediate results
        metrics = pd.concat((metrics, num_alt_metrics))
        metrics.to_csv("../iter_err_metrics.csv", index=False)
