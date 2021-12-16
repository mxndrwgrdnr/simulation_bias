import pandas as pd
import numpy as np
import jax.random as random
import time
from jax.lib import xla_bridge
import argparse
import os

import large_mnl as lmnl

N_SAMP_RATES = 10
ERR_METRIC_ROW_SIZE = 19
OUTFILE = "iter_err_metrics.csv"
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


# def process_iter_prob_metrics(num_alts, result_metrics, sample_rates):
#     """ DEPRECATED
#     """
#     iter_metrics = pd.DataFrame()
#     now = time.time()

#     for i, sample_rate in enumerate(sample_rates):
#         sample_metrics = result_metrics[:, i, :]
#         tmsm_mean_err = np.nanmean(sample_metrics[:, 2])
#         tmsm_mean_abs_pct_err = np.nanmean(np.abs(sample_metrics[:, 3]))
#         tmsm_rmse = np.sqrt(np.nanmean(sample_metrics[:, 4]))
#         tmsmc_mean_err = np.nanmean(sample_metrics[:, 5])
#         tmsmc_mean_abs_pct_err = np.nanmean(np.abs(sample_metrics[:, 6]))
#         tmsmc_rmse = np.sqrt(np.nanmean(sample_metrics[:, 7]))
#         cf_mean_err = np.nanmean(sample_metrics[:, 8])
#         cf_mean_abs_pct_err = np.nanmean(np.abs(sample_metrics[:, 9]))
#         cf_rmse = np.sqrt(np.nanmean(sample_metrics[:, 10]))
#         ppgm_mean_err = np.nanmean(sample_metrics[:, 11])
#         ppgm_mean_abs_pct_err = np.nanmean(np.abs(sample_metrics[:, 12]))
#         ppgm_rmse = np.sqrt(np.nanmean(sample_metrics[:, 13]))

#         iter_metrics = pd.concat((iter_metrics, pd.DataFrame([{
#             'num_alts': num_alts,
#             'num_choosers': num_choosers,
#             'sample_rate': sample_rate,
#             'tmsm_mean_err': tmsm_mean_err,
#             'tmsm_mean_abs_pct_err': tmsm_mean_abs_pct_err,
#             'tmsm_rmse': tmsm_rmse,
#             'tmsmc_mean_err': tmsmc_mean_err,
#             'tmsmc_mean_abs_pct_err': tmsmc_mean_abs_pct_err,
#             'tmsmc_rmse': tmsmc_rmse,
#             'cf_mean_err': cf_mean_err,
#             'cf_mean_abs_pct_err': cf_mean_abs_pct_err,
#             'cf_rmse': cf_rmse,
#             'ppgm_mean_err': ppgm_mean_err,
#             'ppgm_mean_abs_pct_err': ppgm_mean_abs_pct_err,
#             'ppgm_rmse': ppgm_rmse}])))

#     later = time.time()
#     print(
#         "Took {0} s to compute sample metrics for all "
#         "sample rates.".format(np.round(later - now, 1)))

#     return iter_metrics


def get_agg_metrics(
        num_alts, num_choosers, result_metrics, sample_rates):
    """ Compute pop-wide error metrics by sample rate

    Data represented as one point per sample-rate
    """
    # store error metrics to disk
    fname = '../data/chooser_err_metrics_{0}_alts.npy'.format(
        num_alts)
    with open(fname, 'wb') as f:
        np.save(f, result_metrics)

    iter_metrics = pd.DataFrame()
    metric_to_col = {
        'num_alts': 0, 'sample_rate': 1,
        'true_choice': 2, 'samp_choice': 3,
        'tmsm_err': 4, 'tmsm_pct_err': 5, 'tmsm_sq_err': 6,
        'tmsmc_err': 7, 'tmsmc_pct_err': 8, 'tmsmc_sq_err': 9,
        'cf_err': 10, 'cf_pct_err': 11, 'cf_sq_err': 12,
        'ppgm_err': 13, 'ppgm_pct_err': 14, 'ppgm_sq_err': 15,
        'pcpgm_err': 16, 'pcpgm_pct_err': 17, 'pcpgm_sq_err': 18}

    now = time.time()
    true_counts = np.zeros((num_alts,))
    true_choices = result_metrics[:, 0, metric_to_col['true_choice']]
    true_choice_idxs, true_choice_counts = np.unique(
        true_choices, return_counts=True)
    true_counts[true_choice_idxs.astype(int)] = true_choice_counts
    true_alt_shares = true_counts / num_choosers
    true_alt_shares = true_alt_shares[np.argwhere(true_alt_shares > 0)]

    for i in range(10):

        sample_rate = sample_rates[i]
        samp_alt_counts = np.zeros((num_alts, ))
        sample_metrics = result_metrics[:, i, :]

        # true max. probs vs. samp. max. probs
        tmsm_mean_err = np.mean(
            sample_metrics[:, metric_to_col['tmsm_err']])
        tmsm_mean_abs_pct_err = np.mean(np.abs(
            sample_metrics[:, metric_to_col['tmsm_pct_err']]))
        tmsm_rmse = np.sqrt(np.mean(
            sample_metrics[:, metric_to_col['tmsm_sq_err']]))

        # true max. probs vs. samp. max. corrected probs
        tmsmc_mean_err = np.mean(
            sample_metrics[:, metric_to_col['tmsmc_err']])
        tmsmc_mean_abs_pct_err = np.mean(np.abs(
            sample_metrics[:, metric_to_col['tmsmc_pct_err']]))
        tmsmc_rmse = np.sqrt(np.mean(
            sample_metrics[:, metric_to_col['tmsmc_sq_err']]))

        # correction factor
        cf_mean_err = np.mean(
            sample_metrics[:, metric_to_col['cf_err']])
        cf_mean_abs_pct_err = np.mean(np.abs(
            sample_metrics[:, metric_to_col['cf_pct_err']]))
        cf_rmse = np.sqrt(np.mean(
            sample_metrics[:, metric_to_col['cf_sq_err']]))

        # % probs > true mean prob.
        ppgm_mean_err = np.mean(
            sample_metrics[:, metric_to_col['ppgm_err']])
        ppgm_mean_abs_pct_err = np.mean(np.abs(
            sample_metrics[:, metric_to_col['ppgm_pct_err']]))
        ppgm_rmse = np.sqrt(np.mean(
            sample_metrics[:, metric_to_col['ppgm_sq_err']]))

        # % corrected probs > true mean prob.
        pcpgm_mean_err = np.mean(
            sample_metrics[:, metric_to_col['pcpgm_err']])
        pcpgm_mean_abs_pct_err = np.mean(np.abs(
            sample_metrics[:, metric_to_col['pcpgm_pct_err']]))
        pcpgm_rmse = np.sqrt(np.mean(
            sample_metrics[:, metric_to_col['pcpgm_sq_err']]))

        # alts shares
        samp_choices = sample_metrics[
            :, metric_to_col['samp_choice']].astype(int)
        samp_choice_idxs, samp_choice_counts = np.unique(
            samp_choices, return_counts=True)
        samp_alt_counts[samp_choice_idxs] = samp_choice_counts
        samp_alt_shares = samp_alt_counts / num_choosers
        samp_alt_shares = samp_alt_shares[np.argwhere(true_alt_shares > 0)]
        alt_shares_err = samp_alt_shares - true_alt_shares
        alt_shares_pct_err = alt_shares_err / true_alt_shares
        alt_shares_sq_err = alt_shares_err * alt_shares_err
        alt_shares_mean_err = np.mean(alt_shares_err)
        alt_shares_mape = np.mean(np.abs(alt_shares_pct_err))
        alt_shares_rmse = np.sqrt(np.mean(alt_shares_sq_err))

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
            'ppgm_rmse': ppgm_rmse,
            'pcpgm_mean_err': pcpgm_mean_err,
            'pcpgm_mean_abs_pct_err': pcpgm_mean_abs_pct_err,
            'pcpgm_rmse': pcpgm_rmse,
            'alt_shares_mean_err': alt_shares_mean_err,
            'alt_shares_mape': alt_shares_mape,
            'alt_shares_rmse': alt_shares_rmse,
        }])), ignore_index=True)

    later = time.time()
    print(
        "Took {0} s to compute agg metrics for all "
        "sample rates.".format(np.round(later - now, 1)))

    return iter_metrics


def get_err_metrics_from_chooser_prob_metrics(
        num_alts, num_choosers, result_metrics, sample_rates):
    """ Compute chooser-level error metrics probs

    Store result to disk as numpy object for each sample rate. Allows
    distributions of err to be visualized/compared.
    """

    now = time.time()

    # store GPU results to disk
    fname = '../data/chooser_prob_metrics_{0}_alts.npy'.format(num_alts)
    with open(fname, 'wb') as f:
        np.save(f, result_metrics)

    output_arr = np.zeros((
        num_choosers, N_SAMP_RATES, ERR_METRIC_ROW_SIZE))
    metric_to_col = {
        'sample_rate': 0, 'choice': 1, 'max_prob': 2, 'max_prob_true': 3,
        'ppgm': 4, 'pcpgm': 5}

    true_metrics = result_metrics[:, 0, :]
    true_counts = np.zeros((num_alts,))
    true_choices = true_metrics[:, metric_to_col['choice']]
    true_choice_idxs, true_choice_counts = np.unique(
        true_choices, return_counts=True)
    true_counts[true_choice_idxs.astype(int)] = true_choice_counts
    true_alt_shares = true_counts / num_choosers
    true_counts_pos = np.argwhere(true_alt_shares > 0)
    true_alt_shares = true_alt_shares[true_counts_pos]
    true_maxs = true_metrics[:, metric_to_col['max_prob']]
    true_ppgm = true_metrics[:, metric_to_col['ppgm']]

    for i in range(N_SAMP_RATES):

        sample_rate = sample_rates[i]
        sample_metrics = result_metrics[:, i + 1, :]  # +1 to skip true probs
        samp_maxs = sample_metrics[:, metric_to_col['max_prob']]
        true_samp_maxs = sample_metrics[:, metric_to_col['max_prob_true']]
        samp_choices = sample_metrics[:, metric_to_col['choice']]
        samp_probs_corrected = (
            sample_metrics[:, metric_to_col['max_prob']] * sample_rate)

        tmsm_err = true_samp_maxs - true_maxs
        tmsm_pct_err = tmsm_err / true_maxs
        tmsm_sq_err = tmsm_err * tmsm_err

        tmsmc_err = samp_probs_corrected - true_maxs
        tmsmc_pct_err = tmsmc_err / true_maxs
        tmsmc_sq_err = tmsmc_err * tmsmc_err

        cf_true = true_samp_maxs / samp_maxs
        cf_err = sample_rate - cf_true
        cf_pct_err = cf_err / cf_true
        cf_sq_err = cf_err * cf_err

        ppgm_err = sample_metrics[:, metric_to_col['ppgm']] - true_ppgm
        ppgm_pct_err = ppgm_err / true_ppgm
        ppgm_sq_err = ppgm_err * ppgm_err

        pcpgm_err = sample_metrics[:, metric_to_col['pcpgm']] - true_ppgm
        pcpgm_pct_err = pcpgm_err / true_ppgm
        pcpgm_sq_err = pcpgm_err * pcpgm_err

        num_alts_arr = np.full(true_choices.shape, num_alts)
        sample_rate_arr = np.full(true_choices.shape, sample_rate)
        output_arr[:, i, :] = np.hstack([arr.reshape(-1, 1) for arr in [
            num_alts_arr, sample_rate_arr,
            true_choices, samp_choices,
            tmsm_err, tmsm_pct_err, tmsm_sq_err,
            tmsmc_err, tmsmc_pct_err, tmsmc_sq_err,
            cf_err, cf_pct_err, cf_sq_err,
            ppgm_err, ppgm_pct_err, ppgm_sq_err,
            pcpgm_err, pcpgm_pct_err, pcpgm_sq_err]])

    later = time.time()
    print(
        "Took {0} s to compute chooser error metrics for all "
        "sample rates on the CPU.".format(np.round(later - now, 1)))

    return output_arr


def run(choosers, alts, sample_rates, coeffs, key, batched, gpu_func='probs'):

    num_alts = alts.shape[0]
    num_choosers = choosers.shape[0]

    now = time.time()
    result_metrics = lmnl.get_iter_metrics(
        choosers, alts, coeffs, key, batched, gpu_func=gpu_func)
    later = time.time()
    print(
        "Took {0} s to compute chooser {1} metrics for all choosers "
        "and sample rates on the GPU".format(
            np.round(later - now, 1), gpu_func))

    # jitted function just gets prob metrics for each chooser. have to
    # compute error metrics for each sample rate on CPU now for all
    # choosers at once, store probs to disk, then compute agg metrics
    # like normal. this reduces the per-chooser GPU memory consumption,
    # making it possible to process the probabilities in fewer, larger
    # batches, which reduces the runtime for the largest sample sizes. but
    # it means we have to do an extra processing step on CPU here to convert
    # the probs to errors.
    if gpu_func == 'probs':

        result_metrics = get_err_metrics_from_chooser_prob_metrics(
            num_alts, num_choosers, result_metrics, sample_rates)

    # aggregate the chooser-level error metrics and return as a dataframe
    iter_metrics_df = get_agg_metrics(
        num_alts, num_choosers, result_metrics, sample_rates)

    del result_metrics
    return iter_metrics_df


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batched", action="store_true")
    parser.add_argument("-n", "--num_bootstraps", action="store", default=0)
    parser.add_argument("-g", "--gpu_func", action="store", default='probs')
    parser.add_argument("-a", "--append", action="store_true")
    parser.add_argument("-r", "--resume", action="store_true")
    args = parser.parse_args()
    batched = args.batched
    gpu_func = args.gpu_func
    append = args.append
    resume = args.resume
    if args.resume:
        append = True

    output_fpath = os.path.join(DATA_DIR, OUTFILE)

    assert xla_bridge.get_backend().platform == 'gpu'

    alts_sizes = [200, 2000, 20000, 200000, 2000000]
    pop_to_alts_ratio = 750 / 200
    sample_rates = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
    coeffs = np.array([-1, 1, 1, 1, 1])  # dist-to-cbd, sizes 1-3, intx term

    if append & os.path.exists(output_fpath):
        agg_metrics_df = pd.read_csv(output_fpath)
        run_id = agg_metrics_df['run_id'].max()
        if not resume:
            run_id += 1

    else:
        agg_metrics_df = pd.DataFrame()
        run_id = 1

    key = random.PRNGKey(0)
    for i in range(10):

        print("Executing RUN ID #{0}".format(run_id))

        for num_alts in alts_sizes:
            if resume:
                exists_in_df = (
                    agg_metrics_df['num_alts'] == num_alts) & (
                    agg_metrics_df['run_id'] == run_id)
                if len(agg_metrics_df[exists_in_df]) > 0:
                    print((
                        "Already got results for run ID {0} "
                        "with {1} alts.".format(run_id, num_alts)))
                    continue

            # get data and store it to disk
            num_alts = int(num_alts)
            num_choosers = int(pop_to_alts_ratio * num_alts)
            pop_mean_prob = 1 / num_alts
            choosers, alts = create_data(num_alts, num_choosers)
            num_choosers = choosers.shape[0]
            chooser_fname = '../data/choosers_{0}_alts.npy'.format(num_alts)
            with open(chooser_fname, 'wb') as f:
                np.save(f, choosers)
            alts_fname = '../data/alts_{0}_alts.npy'.format(num_alts)
            with open(alts_fname, 'wb') as f:
                np.save(f, alts)
            print(num_alts, " ALTS, ", num_choosers, " CHOOSERS")

            # run for num_alts
            iter_metrics_df = run(
                choosers, alts, sample_rates, coeffs, key, batched, gpu_func)

            # append to metrics df
            iter_metrics_cols = list(iter_metrics_df.columns)
            iter_metrics_df.loc[:, 'run_id'] = run_id
            iter_metrics_df = iter_metrics_df[['run_id'] + iter_metrics_cols]
            agg_metrics_df = pd.concat((agg_metrics_df, iter_metrics_df))

            # save intermediate outputs
            agg_metrics_df.to_csv(output_fpath, index=False)

            del choosers
            del alts
            del iter_metrics_df

        run_id += 1
