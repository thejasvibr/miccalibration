# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 22:48:09 2024

@author: theja
"""
import scipy.signal as signal 
import scipy.ndimage as ndi
import soundfile as sf
import numpy as np 
import sys 
sys.path.append('../')
import tqdm
from calibration_utility import get_freqband_rms
import warnings

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


def load_default_shortsweep():
    audio, fs = sf.read('..\multisignal_calibaudio.wav')
    short_sweep = audio[int(0.1*fs):int((0.1+3e-3)*fs)]
    return short_sweep

def extract_out_signalparts(audio, fs, **kwargs):
    '''
    Extracts out a set of repeating sweeps by cross-correlating a known template
    sweep and the timing between sweeps. 
    
    Parameters
    ----------
    audio : np.array 
        Audio clip, with the short sweep as the first sound in it. 
    short_sweep : np.array, optional 
        The template sweep signal. Defaults to the 'multisignal_calibaudio'
        3 ms sweep
    inter_sweep_durn : float, optional 
        Time between the start/end of sweeps. Defaults to 0.1 s after and 
        before each sweep. 

    Returns
    -------    
    chunk_startstops : list
        List with adjacent start-stop indices of each of the sound types in
        the following order [short sweep, medium sweep, long sweep, white noise, 
                             7 tones]
    audio_segs : list of np.arrays
        Each of the audio clips segmented using chunk_startstops
    '''
    short_sweep = kwargs.get('short_sweep', load_default_shortsweep())    
    inter_sweep_durn = kwargs.get('inter_sweep_durn', 0.2)
    cross_cor = signal.correlate(audio[:int(inter_sweep_durn*fs)],
                                 short_sweep, 'same')
    maxind = np.argmax(cross_cor)
    short_sweep_start = int(maxind - short_sweep.size*0.5 - int(0.1*fs))
    
    
    
    # get start of the short sweep index, and proceed to calculate start-stop
    # for the rest of the sweeps and their repeats. 
    # 3 sweeps, 1 long noise, 7 tones and their durations 
    chunk_lengths = [203, 205, 207, 5200, 2100] # ms
    chunk_lengths = np.array(chunk_lengths)*1e-3 # in seconds
    chunk_lengths = np.int64(fs*chunk_lengths)
    
    chunk_startstops =  np.cumsum(np.concatenate((np.array([short_sweep_start]),
                                             chunk_lengths)))
    audio_segs = []
    for start, stop in zip(chunk_startstops, chunk_startstops[1:]):
        if np.logical_and(start>=0, stop<audio.size):
            snippet = audio[start:stop]
        elif start<0:
            snippet = np.append(np.zeros(abs(start)), audio[:stop])
        elif stop>=audio.size:
            warnings.warn(f'{stop} is greater than audio size')
            snippet = np.concatenate((audio[start:],
                                    np.zeros(int(stop-audio.size))))
        audio_segs.append(snippet)
    return chunk_startstops, audio_segs
