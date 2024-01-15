import os
import json
import sys
import numpy as np
import brian2hears as bh

from string import ascii_lowercase as alc
from brian2 import prefs, start_scope, SpikeMonitor, run, NeuronGroup, Synapses, SpikeGeneratorGroup
from brian2.units import (ms, um, pA, nS, uS, ohm, cm, mV, uF, kHz, mvolt)
from dendrify import Soma, Dendrite, NeuronModel
from helper_funcs import IHCmodel, filter_sound, combine_clips
# test 

prefs.codegen.target = 'numpy'

# read in variables and values
params_dict = {}
# sys.argv[1:][0] is 'N_BRANCHES=10 B=100*pA'
for i in sys.argv[1:][0].split(' '):
    var, value = i.split('=')
    if 'PATH' in var: 
        exec(f'{var} = "{value}"') # read in as string
    else:
        exec(f'{var} = {value}')
    params_dict[f'{var}'] = value

# encode stimuli into spikes 
bh.set_default_samplerate(250*kHz)
clip_path1 = A_PATH
clip_path2 = B_PATH
soundA, soundB, soundAB = combine_clips(clip_path1, clip_path2, DELAY)

clip_set = [soundA, soundB, soundAB]

minFreq = 30*kHz
maxFreq = 120*kHz
nFilters = 3000

input_spikes = {}

clipnames = ['A', 'B', 'AB']
for ii, clip in enumerate(clip_set):
    start_scope()
    cochlea = filter_sound(clip, minFreq, maxFreq, nFilters)
    G = IHCmodel(cochlea, 0.13)
    M = SpikeMonitor(G)
    run(clip.duration)
    input_spikes[clipnames[ii]] = M

#create neuron model
nBranches = 10

soma = Soma('soma', model='adaptiveIF', length=25*um, diameter=25*um)
soma.noise(tau=20*ms, sigma=20*pA, mean=0*pA)

d = {}
for branch_no in np.arange(0, nBranches):
    d["apical{0}".format(branch_no)] = Dendrite('ap'+str(branch_no), length=100*um, diameter=1*um)

ct = 0
edges = []
for key, val in d.items():
    val.synapse('AMPA', pre=alc[ct], g=0.8*nS, t_rise=0.2*ms, t_decay=3*ms)
    val.synapse('GABA', pre=alc[ct], g=0.8*nS, t_rise=2*ms, t_decay=8*ms)
    val.synapse('NMDA', pre=alc[ct], g=0.8*nS, t_rise=2*ms, t_decay=60*ms)
    #val.dspikes('Na', threshold=-35*mV, g_rise=7*nS, g_fall=6*nS)
    edges.append((soma, val, 5*nS))
    ct += 1 

pyr_model = NeuronModel(edges, cm=0.5*uF/(cm**2), gl=50*uS/(cm**2),
                            v_rest=-60*mV, r_axial=150*ohm*cm,
                            scale_factor=3, spine_factor=1.5)

#1/d['apical0'].g_leakage *d['apical0'].capacitance

params_adapt = {
    'a_soma' : 0.8*nS, # coupling to membrane potential 
    'tauw_soma' : 100*ms, # decay of w, higher values mean that tau decays slower 
    'Vth_soma' : -50*mV,
    'b' : B_CURRENT
}

pyr_model.add_params(params_adapt)

start_scope() 

# we are creating 3 instances of the same neuron to simulate what happens with inputs Clip A, ClipB and ClipAB (combined)
global pyr_group # because when calling the pyr_model.link function doesn't work otherwise
pyr_group = NeuronGroup(3, model=pyr_model.equations, method='euler', 
                    threshold='V_soma > Vth_soma', reset= 'V_soma =0*mV; w_soma += b',
                    refractory=4*ms, namespace=pyr_model.parameters, events=pyr_model.events)

pyr_model.link(pyr_group)
second_reset = Synapses(pyr_group, pyr_group, on_pre='V_soma=-58*mV', delay=0.2*ms)
second_reset.connect(j='i')

#synaptic connections 
indices = np.arange(0, nFilters)
binsize = nFilters // nBranches
n_in = int(ALPHA*binsize) # alpha is the parameter controlling the fraction of synaptic connections from a frequency bin. 
n_out = binsize - n_in

inTune = [np.random.choice(indices[i*binsize: (i + 1)*binsize], n_in, replace=False) for i in np.arange(nBranches)]
randomized = np.setdiff1d(indices, np.concatenate(inTune))
np.random.shuffle(randomized)
outTune = [randomized[i*n_out:(i+1)*n_out] for i in np.arange(nBranches)]
connections = [np.concatenate((x, y)) for x, y in zip(outTune, inTune)]

for clip_index in range(3): # for ['A', 'B', 'AB']
    spike_object = input_spikes[clipnames[clip_index]]
    globals()[f"spike_object{clip_index}"] = SpikeGeneratorGroup(nFilters, spike_object.i, spike_object.t)   
    for branch_no in np.arange(0, nBranches):
        ampa_w = "wampa_" + alc[branch_no]
        nmda_w = "wnmda_" + alc[branch_no]
        gaba_w = "wgaba_" + alc[branch_no]
        syn_mod = ampa_w + " : 1" + "\n" + nmda_w + " : 1" + "\n" + gaba_w + " : 1"
        s_var_ampa = "s_AMPA_" + alc[branch_no] + "_ap" + str(branch_no)
        s_var_nmda = "s_NMDA_" + alc[branch_no] + "_ap" + str(branch_no)
        s_var_gaba = "s_GABA_" + alc[branch_no] + "_ap" + str(branch_no)
        #pre_effect = s_var + " += " + ampa_w + ";" + s_var + " = clip(" + s_var + ", 0, 0.1)" 
        pre_effect = s_var_ampa + " += " + ampa_w + " ; " + s_var_nmda + " += " + nmda_w + " ; " + s_var_gaba + " += " + gaba_w
        globals()[f"syn_object{branch_no}{clip_index}"] = Synapses(globals()[f"spike_object{clip_index}"], pyr_group, model=syn_mod, on_pre=pre_effect) # need to create a new object this way 
        #globals()[f"syn_object{branch_no}"].connect(i=np.arange(np.int(branch_no*fibers_per_dendrite), np.int((branch_no+1)*fibers_per_dendrite)), j=0)

        globals()[f"syn_object{branch_no}{clip_index}"].connect(i=connections[branch_no], j=clip_index)
        setattr(globals()[f"syn_object{branch_no}{clip_index}"], ampa_w, AMPA_WEIGHT) # set weights here 
        setattr(globals()[f"syn_object{branch_no}{clip_index}"], nmda_w, NMDA_WEIGHT) # set weights here 
        setattr(globals()[f"syn_object{branch_no}{clip_index}"], gaba_w, GABA_WEIGHT) # set weights here 

#track_vm = ["V_ap" + str(branch_no) for branch_no in range(nbranches)]
#track_vm.append("V_soma")

#Mvm = StateMonitor(pyr_group, track_vm, record=True)
S = SpikeMonitor(pyr_group)

run(150*ms)

params_dict['clipA'] = os.path.basename(clip_path1).strip('.wav')
params_dict['clipB'] = os.path.basename(clip_path2).strip('.wav')
del params_dict['A_PATH']
del params_dict['B_PATH']

#data out 
serialized_dict = {}

for key, value in S.spike_trains().items():
    serialized_dict[clipnames[key]] = np.array(value).tolist()

serialized_dict.update(params_dict)

json_data = json.dumps(serialized_dict)
print(json_data)

