from brian2 import second, clip, Inf
from brian2.units import (ms, ohm, mV, mvolt)
import brian2hears as bh
from brian2hears import dB

def IHCmodel(filtered, scalefactor): 
    
    params = {
        'El' : -60*mV, 
        'R' : 0.2*ohm, 
        'tau': 1*ms, 
        'mu' : 0*mV, 
        'sigma':3*mV,
        'scale_factor' : scalefactor
    }

    eqs = '''
            dv/dt = (-(v-El)+R*I2)/tau + mu/tau + sigma*xi*(2/tau)**0.5 : volt
            I2 = scale_factor*I*amp : amp
            I: 1
            '''

    G = bh.FilterbankGroup(filtered, 'I', eqs, reset="v=-20*mV", threshold="v>-40*mV", refractory=5*ms, namespace=params)
    G.v = params['El']

    return(G)

def filter_sound(sound, minfreq, maxfreq, nfilters): 
    cf = bh.erbspace(minfreq, maxfreq, nfilters)
    gammatone = bh.Gammatone(sound, cf)
    cochlea = bh.FunctionFilterbank(gammatone, lambda x: 3*clip(x, 0, Inf)**(1.0/3.0))

    return(cochlea)

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
 



