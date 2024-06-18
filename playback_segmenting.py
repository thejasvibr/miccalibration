# -*- coding: utf-8 -*-
"""
Segmenting playback sounds
==========================

Created on Wed May 29 06:42:00 2024

@author: theja
"""
import numpy as np 
import scipy.signal as signal
from calibration_utility import absmax_norm

def give_expected_segments():
    '''No input. These start-stop indices only work 
    if the microphone audio is well aligned in the 
    first place.
    
    Returns
    -------
    start_stops : list with tuples
        Each tuple holds the start and stop indices of 
        a playback sound type
    
    '''
    fs = 44100

    # sweep durations
    durns = np.concatenate((np.array([3e-3, 5e-3, 7e-3, 5]),
                            np.tile(0.1,7)))

    # on either side.
    silence_samples = int(fs*0.1)
    start_stops = []
    start_sample = 0
    for d in durns:
        numsamples = int(fs*d)
        total_samples = numsamples + silence_samples*2
        
        end_sample = start_sample + total_samples
        start_stops.append((start_sample, end_sample))
        start_sample += total_samples 
    return start_stops

def align_to_outputfile(mic_audio, digital_out, use_numsamples=44100):
    '''
    '''
    cc_mic = signal.correlate(absmax_norm(mic_audio[:use_numsamples]), digital_out[:use_numsamples])
    delaypeak = int(np.argmax(cc_mic) - cc_mic.size*0.5)
    numsamps = abs(delaypeak)
    print(delaypeak)
    if delaypeak<0:
        timealigned_audio = np.concatenate((np.zeros(numsamps),
                                           mic_audio[delaypeak:]))
    elif delaypeak>0:
        timealigned_audio = np.concatenate((mic_audio[numsamps:],
                                           np.zeros(int(numsamps))))
    return timealigned_audio, delaypeak

