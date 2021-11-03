import pandas as pd
import numpy as np
import jax.random as random
import time
from jax.lib import xla_bridge
import argparse

import large_mnl as lmnl


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batched", action="store_true")
    args = parser.parse_args()
    batched = args.batched

    print("Running on", xla_bridge.get_backend().platform, "!")

    alts_sizes = [200, 2000, 2e4, 2e5, 2e6]
    pop_to_alts_ratio = 750 / 200
    sample_rates = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
    coeffs = np.array([-1, 1, 1, 1, 1])

    metrics = pd.DataFrame(
        columns=[
            'num_alts', 'num_choosers', 'sample_rate',
            'med_pct_probs_gt_pop_mean', 'med_max_corr_prob'])

    key = random.PRNGKey(0)
    for num_alts in alts_sizes:
        now = time.time()
        num_alts = int(num_alts)
        pop_mean_prob = 1 / num_alts

        choosers, alts = lmnl.create_data(num_alts, pop_to_alts_ratio)
        num_choosers = choosers.shape[0]
        print(num_alts, " ALTS, ", num_choosers, " CHOOSERS")
        probs_true_max, pct_true_probs_gt_pop_mean = lmnl.get_probs(
            choosers, alts, key, num_alts, batched)
        later = time.time()
        print("Took {0} s to compute true probs.".format(
            np.round(later - now)))

        now = time.time()
        med_pct_probs_gt_pop_mean = np.median(pct_true_probs_gt_pop_mean)
        med_max_prob = np.median(probs_true_max)
        later = time.time()
        print("Took {0} s to compute true prob. metrics".format(
            np.round(later - now)))

        metrics = pd.concat((metrics, pd.DataFrame([{
            'num_alts': num_alts,
            'num_choosers': num_choosers,
            'sample_rate': 1,
            'med_pct_probs_gt_pop_mean': med_pct_probs_gt_pop_mean,
            'med_max_corr_prob': med_max_prob}])))

        for sample_rate in sample_rates:
            now = time.time()

            sample_size = int(num_alts * sample_rate)
            probs_samp_max, pct_samp_probs_gt_pop_mean = lmnl.get_probs(
                choosers, alts, key, sample_size, batched)
            later = time.time()
            print("Took {0} s to compute {1}% sample probs.".format(
                np.round(later - now), sample_rate * 100))

            # metrics
            now = time.time()
            med_pct_probs_gt_pop_mean = np.median(pct_samp_probs_gt_pop_mean)
            med_max_corr_prob = np.median(probs_samp_max) * sample_rate
            later = time.time()
            print("Took {0} s to compute {1}% sample metrics.".format(
                np.round(later - now), sample_rate * 100))

            metrics = pd.concat((metrics, pd.DataFrame([{
                'num_alts': num_alts,
                'num_choosers': num_choosers,
                'sample_rate': sample_rate,
                'med_pct_probs_gt_pop_mean': med_pct_probs_gt_pop_mean,
                'med_max_corr_prob': med_max_corr_prob}])))
