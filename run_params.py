import os
import subprocess 
import json 
import sys

from multiprocessing import Pool
import pandas as pd
import numpy as np


def run_subprocess(script, param_args):
    
    result = subprocess.run(['python', script, param_args], capture_output=True, text=True)
    output = json.loads(result.stdout)
    return output   

clip_files_dir = "/mnt/rds/gwa18/projects/kozlovlab_rds2/live/Grace/data/clips"
clip_lag_info = pd.read_csv("clip_delays.csv")
clip_names = clip_lag_info.drop_duplicates(subset = ['A', 'B'])

param_list = []
n_iter = 1
for index, row in clip_names.iterrows():
    clipA = os.path.join(clip_files_dir, row['A'] + '.wav')
    clipB = os.path.join(clip_files_dir, row['B'] + '.wav')
    #for delay in [-0.02, -0.01, 0, 0.01, 0.02]:
    for delay in [-0.02]:
        for b_val in [200]:
            for nmda in [0.0]:
                for ampa in [0.55]: # default is 0.5
                    for gaba in [0.0]:
                        for alpha in [1.0]:
                            for rep in np.arange(n_iter):
                                _params = ' '.join(
                                    [f'A_PATH={clipA}',
                                    f'B_PATH={clipB}',
                                    f'DELAY={delay}',
                                    f'B_CURRENT={b_val}*pA',
                                    f'NMDA_WEIGHT={nmda}',
                                    f'AMPA_WEIGHT={ampa}', 
                                    f'GABA_WEIGHT={gaba}',
                                    f'ALPHA={alpha}', 
                                    f'ITER={rep}'])
                                param_list.append(_params)

scripts = ['run_dendrite_model.py']*len(param_list)
#scripts = ['script_calculate_dendrite_overlap.py']*len(param_list)
 
os.environ["NUMEXPR_MAX_THREADS"] = "50"
pool = Pool(processes=50)
outputs = pool.starmap(run_subprocess, zip(scripts, param_list)) 
spktimes_df = pd.DataFrame(outputs)

spktimes_df.to_pickle("b.pkl")