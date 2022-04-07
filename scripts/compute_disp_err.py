import pandas as pd
from tqdm import tqdm
import numpy as np
import jax.random as random
import os
import time

import compute_pmass as pm

# "fixed_ratio", "fixed_alts", "fixed_choosers"
MODE = "fixed_ratio"
VERBOSE = True
DEBUG = True

ALTS_SIZES = [
    200,
    # 2000,
    # 20000,
    # 200000,
    # 2000000
]

CHOOSER_SIZES = [
    # 750,
    # 7500,
    # 75000,
    # 750000,
    7500000
]


ALTS_TO_ITERS = {200: 10, 2000: 10, 20000: 10, 200000: 10, 2000000: 10}
POP_TO_ALTS_RATIO = 750 / 200
SCALE_PARAMS = [.25, .5, .75, 1, 1.25, 1.5, 1.75]
SAMPLE_RATES = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
COEFFS = np.array([-1, 1, 1, 1, 1])  # dist-to-cbd, sizes 1-3, intx term
OUTFILE = '../data/disp_err_scale_iters_importance.csv'


def run(mode=MODE):
    key = random.PRNGKey(0)
    if os.path.exists(OUTFILE):
        df = pd.read_csv(OUTFILE)
    else:
        df = pd.DataFrame()
    for num_choosers in CHOOSER_SIZES:
        if MODE == 'fixed_ratio':
            ALTS_SIZES = [int(num_choosers) * (2.0 / 7.5)]
        for num_alts in ALTS_SIZES:

            run_id = df['run_id'].max() + 1 if len(df) > 0 else 1

            num_iters = ALTS_TO_ITERS[num_alts]
            print("RUNNING {0} ITERATIONS WITH {1} ALTS and {2} CHOOSERS".format(
                num_iters, num_alts, num_choosers))

            for i in tqdm(range(num_iters)):
                
                iter_df = pd.DataFrame()
                disable = (max(num_alts, num_choosers) < 200000) and (VERBOSE is False)
                for scale_param in tqdm(SCALE_PARAMS, disable=disable):
                    sttm = time.time()
                    scale_df = pm.run_v2(
                        num_alts, num_choosers, SAMPLE_RATES, COEFFS, key,
                        batched=True, scale=scale_param, debug=DEBUG, verbose=VERBOSE)
                    endtm = time.time()
                    scale_df['scale'] = scale_param
                    scale_df['runtime'] = endtm - sttm
                    iter_df = pd.concat((iter_df, scale_df), ignore_index=True)
                iter_df['run_id'] = run_id
                iter_df['num_alts'] = num_alts
                iter_df['num_choosers'] = num_choosers
                df = pd.concat((df, iter_df), ignore_index=True)
                df.to_csv(OUTFILE, index=False)
                run_id += 1

    df = df.replace(np.inf, np.nan)

    return df


if __name__ == '__main__':

    df = run()
