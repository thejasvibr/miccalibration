# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 22:48:09 2024

@author: theja
"""
import scipy.signal as signal 
import scipy.ndimage as ndi
import numpy as np 
import sys 
sys.path.append('../')
import tqdm
from calibration_utility import get_freqband_rms


def segment_sounds_v2(X, smooth_window, pctile_threshold=30):
    '''
    Makes the Hilbert envelope, smooths out the envelope and threshold it. 
    Finally, segments the sounds into chunks. 
    
    '''
    envelope = abs(signal.hilbert(X))
    # smoothen out the clippoint a bit
    smoothened_envelope = signal.convolve(envelope, 
                                          np.ones(smooth_window)/smooth_window, 'same')
    pctile_threshold_value = np.percentile(smoothened_envelope,
                                           pctile_threshold)
    above_thresh = smoothened_envelope > pctile_threshold_value
    segmented_tones, numtones = ndi.label(above_thresh)
    segmeneted_chunks = ndi.find_objects(segmented_tones)
    return segmeneted_chunks, smoothened_envelope

def calculate_rms_for_freqbins(X, fs, centrefreqs):
    '''
    '''
        
    centrefreq_dist = np.unique(np.diff(centrefreqs)) # Hz
    halfwidth = centrefreq_dist*0.5
    bandwise_tgtmic = np.empty(centrefreqs.size)

    for i,each in tqdm.tqdm(enumerate(centrefreqs)):
        bandrms_tgt = get_freqband_rms(X, fs,
                                freq_range=(each-halfwidth, each+halfwidth))
        
        bandwise_tgtmic[i] = bandrms_tgt
    return bandwise_tgtmic