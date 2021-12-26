import pandas as pd
from tqdm import tqdm
import numpy as np
import jax.random as random
import os

import compute_pmass as pm

ALTS_SIZES = [
    # 200, 2000,
    # 20000,
    # 200000,
    2000000
]
SCALE_PARAMS = [.25, .5, .75, 1, 1.25, 1.5, 1.75]
ALTS_TO_ITERS = {200: 100, 2000: 50, 20000: 10, 200000: 10, 2000000: 4}
POP_TO_ALTS_RATIO = 750 / 200
SAMPLE_RATES = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
COEFFS = np.array([-1, 1, 1, 1, 1])  # dist-to-cbd, sizes 1-3, intx term
OUTFILE = '../data/disp_err_scale_iters_v2.csv'


def run():
    key = random.PRNGKey(0)
    if os.path.exists(OUTFILE):
        df = pd.read_csv(OUTFILE)
    else:
        df = pd.DataFrame()
    for num_alts in ALTS_SIZES:
        if len(df) > 0:
            run_id = df['run_id'].max()
        else:
            run_id = 1
        num_iters = ALTS_TO_ITERS[num_alts]
        print("RUNNING {0} ITERATIONS WITH {1} ALTS".format(
            num_iters, num_alts))
        for i in tqdm(range(num_iters)):
            run_id += 1
            iter_df = pd.DataFrame()
            disable = num_alts < 200000
            for scale_param in tqdm(SCALE_PARAMS, disable=disable):
                scale_df = pm.run_v2(
                    num_alts, POP_TO_ALTS_RATIO, SAMPLE_RATES, COEFFS, key,
                    batched=True, scale=scale_param, debug=True, verbose=False)
                scale_df['scale'] = scale_param
                iter_df = pd.concat((iter_df, scale_df), ignore_index=True)
            iter_df['run_id'] = run_id
            iter_df['num_alts'] = num_alts
            iter_df['num_choosers'] = int(POP_TO_ALTS_RATIO * num_alts)
            df = pd.concat((df, iter_df), ignore_index=True)
        df.to_csv(OUTFILE, index=False)

    df = df.replace(np.inf, np.nan)

    return df


if __name__ == '__main__':

    df = run()
