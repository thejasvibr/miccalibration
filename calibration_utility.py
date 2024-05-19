# -*- coding: utf-8 -*-
"""
Utility functions
=================
Created on Mon May 13 10:45:34 2024

@author: theja
"""
import numpy as np 
import scipy.signal as signal 
import scipy.ndimage as ndi

def rms(X):
    return np.sqrt(np.mean(X**2))

dB = lambda X: 20*np.log10(abs(np.array(X).flatten()))


def segment_sounds(X, smooth_window, threshold):
    '''
    Makes the Hilbert envelope, smooths out the envelope and threshold it. 
    Finally, segments the sounds into chunks. 
    '''
    envelope = abs(signal.hilbert(X))
    # smoothen out the clippoint a bit
    smoothened_envelope = signal.convolve(envelope, 
                                          np.ones(smooth_window)/smooth_window, 'same')
    above_thresh = smoothened_envelope > threshold
    segmented_tones, numtones = ndi.label(above_thresh)
    segmeneted_chunks = ndi.find_objects(segmented_tones)
    return segmeneted_chunks, smoothened_envelope

def absmax_norm(X):
    '''
    Take the absolute max value and normalise the signal. 
    REturns a new array absmax_norm
    '''
    abs_max = np.max(np.abs(X))
    absmaxnorm = X.copy()
    absmaxnorm /= abs_max
    return absmaxnorm

def get_energywindow(X, percentile_thresh):
    '''
    '''
    remaining =  100 - percentile_thresh
    min_pctile = remaining*0.5
    max_pctile = 100 - min_pctile
    X_sqcumsum = np.cumsum(X**2)
    lower, higher =  np.percentile(X_sqcumsum, [min_pctile, max_pctile])
    lower_ind = np.argmin(abs(X_sqcumsum-lower))
    max_ind = np.argmin(abs(X_sqcumsum-higher))
    return X[lower_ind:max_ind]
    

# bin width clarification https://stackoverflow.com/questions/10754549/fft-bin-width-clarification
def get_rms_from_spectrum(freqs, spectrum, **kwargs):
    '''Use Parseval's theorem to get the RMS level of each frequency component
    This only works for RFFT spectrums!!!
    
    '''
    minfreq, maxfreq = kwargs['freq_range']
    relevant_freqs = np.logical_and(freqs>=minfreq, freqs<=maxfreq)
    spectrum_copy = spectrum.copy()
    spectrum_copy[~relevant_freqs] = 0
    mean_sigsquared = np.sum(abs(spectrum_copy)**2)/spectrum.size
    root_mean_squared = np.sqrt(mean_sigsquared/(2*spectrum.size-1))
    return root_mean_squared


def get_freqband_rms(X, fs, **kwargs):
    fft_x = np.fft.rfft(X)
    freqs_x = np.fft.rfftfreq(fft_x.size*2 - 1, 1/fs)
    rms = get_rms_from_spectrum(freqs_x, fft_x, **kwargs)
    return rms

def spllevel_from_audio(rms_or_peak, mic_sensitivity, ref=20e-6):
    '''
    Parameters
    ----------
    rms_or_peak : float, array-like
        RMS or peak values
    mic_sensitivity : float
        RMS or peak value sensitivity in V/Pa
    ref : float, optional
        Defaults to 20 microPascals
    
    Returns 
    -------
    db_spl : float, array-like
        dB Sound pressure level 
    '''
    soundpressure = rms_or_peak/mic_sensitivity
    db_spl = dB(soundpressure/ref)
    return db_spl

    