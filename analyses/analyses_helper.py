
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa 
from functools import partial

from brian2.units import kHz
from brian2 import second  
import brian2hears as bh 
from brian2hears import dB

bh.set_default_samplerate(250*kHz)

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

def combine_clips(clip_path1, clip_path2, lag): 
    '''
    silence_length in seconds to prepend to the second clip
    '''
    soundA = bh.loadsound(clip_path1)
    soundB = bh.loadsound(clip_path2)
    soundA.level = 70*dB
    soundB.level = 70*dB

    silence_length = abs(lag)

    if lag > 0: 
        soundB_pad = bh.sequence([bh.silence(silence_length*second), soundB])
        soundA_pad = soundA
    else:
        soundA_pad = bh.sequence([bh.silence(silence_length*second), soundA])
        soundB_pad = soundB

    soundAB = soundA_pad + soundB_pad   

    return(soundA_pad, soundB_pad, soundAB)

def calc_spec(sound, sr):
    y= np.array(sound) 
    nfft = 1024

    D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=nfft)), ref=np.max)
    freq_bins = np.linspace(0, sr/2, D.shape[0])
    dur = librosa.get_duration(y=y, sr=sr)
    time_bins = np.linspace(0, dur, D.shape[1])

    return freq_bins, time_bins, D

def plot_func(df, clip_files_dir, out_folder, window):
    
    """
    function to plot individual sets
    """ 
    
    g_cmap = plt.cm.get_cmap('gray')
    r_map = g_cmap.reversed()

    colors = ["#264653", "#e76f51", "#2a9d8f"]

    fig, axes = plt.subplots(2, 3, figsize=(6, 4))
    #figname = x['clipname'].values + '.png'

    #spec_d = features[(features['clipA'] == x['clipA'].values[0]) & (features['clipB'] == x['clipB'].values[0]) & features['DELAY'] == x['DELAY'].values[0]]['dists'].values[0]

    clip_path1 = os.path.join(clip_files_dir, df['clipA'].values[0] + '.wav')
    clip_path2 = os.path.join(clip_files_dir, df['clipB'].values[0] + '.wav')
    lag_time = float(df['DELAY'].values[0])

    clipA_name = df['clipA'].values[0]
    clipB_name = df['clipB'].values[0]
    delay = round(lag_time*1000)

    #d_spec = features[(features['clipA'] == clipA_name) & (features['clipB'] == clipB_name) & (features['delay'] == delay)]['eu'].values[0]
    #d_spec = str(round(float(d_spec), 2))
    #print(d_spec)

    figname = os.path.join(out_folder, df['clipname'].values[0] + '.png')

    soundA, soundB, soundAB = combine_clips(clip_path1, clip_path2, lag_time)
    clip_set = [soundA.T[0], soundB.T[0], soundAB.T[0]]

    ii = 0
    vals = {}
    for clip in ['A', 'B', 'AB']: 
        ax1 = axes[1, ii]
        ax2 = ax1.twinx()
        spkmat = df[clip].map(lambda x: bin_counts(x, window)).values.tolist()
        avg = np.mean(spkmat, 0)
        vals.update({clip:np.max(avg)})

        ht = sns.heatmap(spkmat, vmin=0, vmax=2, cmap=r_map, ax=ax1, cbar=False, alpha=0.45)
        ax1.invert_yaxis()
        ax1.get_yaxis().set_visible(False)
        
        psth = np.mean(spkmat, axis=0)
        ln = sns.lineplot(x=np.arange(0, psth.shape[0]) + 0.5, y=psth, ax=ax2, color=colors[ii], linewidth=2.5)
        ln.set(ylim=(0, 2.2))
        ln.set_yticks([0, 1, 2])
        ln.set_xticks([0, 5, 10])
        ln.set_xticklabels(np.arange(0, 0.13, 0.05))

        freq_bins, time_bins, D = calc_spec(clip_set[ii], 250000)
        ht2 = sns.heatmap(D, ax=axes[0, ii], cbar=False, cmap=r_map, vmin=-65, rasterized=True)
        axes[0, ii].axhline(color="k")
        axes[0, ii].axvline(color="k")
        axes[0, ii].invert_yaxis()
        axes[0, ii].get_yaxis().set_visible(False)
        axes[0, ii].get_xaxis().set_visible(False)
        fig.tight_layout()
        
        ii += 1

    Ra = min(vals['A'], vals['B'])
    Rb = max(vals['A'], vals['B'])
    Rab = vals['AB']
    smi = np.round((Rab - Rb)/Ra, 2)

    [ax.set_xlim(0, 128) for ax in axes[0, :]] # set arbitrarily to 118 which corresponds to the number of samples in a 0.12 long clip 
    axes[1, 0].set_title("smi = " + str(smi))
    #axes[1, 2].set_title(str(d_spec))
    
    fig.suptitle(df['clipname'].values[0])
    fig.savefig(figname, facecolor='white', transparent=False) 