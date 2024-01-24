from brian2 import *
import brian2hears as bh
from brian2.units import (ms, um, pA, nS, uS, ohm, cm, mV, uF, kHz)
from brian2hears import (dB)
from dendrify import Soma, Dendrite, NeuronModel
#from soundsig.sound import BioSound
import scipy.io.wavfile as wav

def ihc_model(sound, minfreq, maxfreq, nfilters): 
    cf = linspace(minfreq*kHz, maxfreq*kHz, nfilters)
    gfb = bh.Gammatone(sound, cf)
    ihc = bh.FunctionFilterbank(gfb, lambda x: 3*clip(x, 0, Inf)**(1.0/3.0))
    # Leaky integrate-and-fire model with noise and refractoriness
    eqs = '''
        dv/dt = (I-v)/(1*ms)+0.2*xi*(2/(1*ms))**.5 : 1 (unless refractory)
        I : 1
        '''
    G = bh.FilterbankGroup(ihc, 'I', eqs, reset='v=0', threshold='v>1', refractory=5*ms)

    return G

def ihc_model2(filtered, scalefactor): 
    
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

def ihc_model3(filtered, scalefactor): 
    
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
    
    G = bh.FilterbankGroup(filtered, 'I', eqs, reset="v=-20*mV", threshold="v>-50*mV", refractory=4*ms, namespace=params)
    G.v = params['El']

    return(G)


def filtersound(sound, minfreq, maxfreq, nfilters): 
    cf = bh.erbspace(minfreq, maxfreq, nfilters)
    gammatone = bh.Gammatone(sound, cf)
    cochlea = bh.FunctionFilterbank(gammatone, lambda x: 3*clip(x, 0, Inf)**(1.0/3.0))

    return(cochlea)


#def calculate_specs(soundpath):
#    fs, sound_sig = wav.read(soundpath)
#    myBioSound = BioSound(sound_sig, fs=fs)
#    myBioSound.spectrum(f_high=120000)
#    properties = np.array([myBioSound.q1, myBioSound.q2, myBioSound.q3, myBioSound.meanspect, myBioSound.stdspect])
#    return properties




