
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from functools import partial

def bin_counts(x, window):
    '''
    window in seconds 
    '''
    tbins = np.arange(window[0], window[1], 0.01)
    spiketimes = np.array(x)
    searchsorted_idx = np.squeeze(np.searchsorted(spiketimes, window))
    spike_counts = np.histogram(spiketimes[searchsorted_idx[0]:searchsorted_idx[1]], tbins)[0]
    return np.insert(spike_counts, 0, 0)

bin_counts2 = partial(bin_counts, window = [0, 0.13])

def calc_smi(df): 

    """
    
    """
    vals = {}
    for clip in ['A', 'B', 'AB']: 
        spkmat = df[clip].map(bin_counts2).values.tolist()
        avg = np.mean(spkmat, 0)
        vals.update({clip:np.max(avg)})
        #vals.append(np.max(avg))
    Ra = min(vals['A'], vals['B'])
    Rb = max(vals['A'], vals['B'])
    Rab = vals['AB']
    smi = np.round((Rab - Rb)/Ra, 2)
    return Ra, Rb, Rab, smi

def calc_max_smi(x): 
    """
    """
    max_index = x['rab'].idxmax()
    max_smi = x.loc[max_index]['smi']
    lag = x.loc[max_index]['DELAY']
    return max_smi, lag

def plot_hist(df, fig_dims): 
    """
    
    """
    fig = plt.figure(figsize=fig_dims)
    g1 = sns.histplot(data=df, x="smi", element="step", hue="params", bins=np.arange(-2, 4, 0.5), fill=None, legend=False)
    plt.xlim(-2.3, 4)
    g1.set(xlabel="SmI")
    g1.axvline(0, linestyle='--', color='k', linewidth=1)
    g1.axvline(1, linestyle='--', color='k', linewidth=1)
    sns.despine()
    g1.set(xticks=np.arange(-2, 5, 1))
    plt.tight_layout()

    return fig 