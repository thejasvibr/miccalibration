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
import tqdm 

def rms(X):
    return np.sqrt(np.mean(X**2))

dB = lambda X: 20*np.log10(abs(np.array(X).flatten()))

db_to_linear = lambda X: 10**(X/20)

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
def get_rms_from_fft(freqs, spectrum, **kwargs):
    '''Use Parseval's theorem to get the RMS level of each frequency component
    This only works for RFFT spectrums!!!
    
    Parameters
    ----------
    freqs : (Nfreqs,) np.array >0 values
    spectrum : (Nfreqs,) np.array (complex)
    freq_range : (2,) array-like
        Min and max values
    
    Returns 
    -------
    root_mean_squared : float
        The RMS of the signal within the min-max frequency range
   
    '''
    minfreq, maxfreq = kwargs['freq_range']
    relevant_freqs = np.logical_and(freqs>=minfreq, freqs<=maxfreq)
    spectrum_copy = spectrum.copy()
    spectrum_copy[~relevant_freqs] = 0
    mean_sigsquared = np.sum(abs(spectrum_copy)**2)/spectrum.size
    root_mean_squared = np.sqrt(mean_sigsquared/(2*spectrum.size-1))
    return root_mean_squared


def calc_native_freqwise_rms(X, fs):
    '''
    Converts the FFT spectrum into a band-wise rms output. 
    The frequency-resolution of the spectrum/audio size decides
    the frequency resolution in general. 
    
    Parameters
    ----------
    X : np.array
        Audio
    fs : int
        Sampling rate in Hz
    
    Returns 
    -------
    fftfreqs, freqwise_rms : np.array
        fftfreqs holds the frequency bins from the RFFT
        freqwise_rms is the RMS value of each frequency bin. 
    '''
    rfft = np.fft.rfft(X)
    fftfreqs = np.fft.rfftfreq(X.size, 1/fs)
    # now calculate the rms per frequency-band
    freqwise_rms = []
    for each in rfft:
        mean_sq_freq = np.sum(abs(each)**2)/rfft.size
        rms_freq = np.sqrt(mean_sq_freq/(2*rfft.size-1))
        freqwise_rms.append(rms_freq)
    return fftfreqs, freqwise_rms


def get_freqband_rms(X, fs, **kwargs):
    '''
    Get rms within a given min-max frequency range using the fft method.
    
    Parameters
    ----------
    X : (Nsamples,) np.array
    fs : float>0
        Freq. of sampling in Hz
    freq_range : (2,) array-like
        (min, max) frequency to calculate rms from in Hz.
    
    Returns 
    -------
    freqwise_rms : np.array 
        Frequency bin-wise rms values. 
    
    
    '''
    fft_x = np.fft.rfft(X)
    freqs_x = np.fft.rfftfreq(fft_x.size*2 - 1, 1/fs)
    freqwise_rms = get_rms_from_fft(freqs_x, fft_x, **kwargs)       
    return freqwise_rms


def get_centrefreq_rms(X, fs, **kwargs):
    '''
    Given an audio input - get the RMS for each centre frequency from 
    an rfft.
    
    Parameters
    ----------
    X : np.array
        Audio clip
    fs : float>0
        Sampling rate in Hz
    
    Returns 
    -------
    freqs_x : np.array 
        Band centre-frequencies
    freqwise_rms_values : np.array 
        RMS values for each frequency-band
    
    '''
    fft_x = np.fft.rfft(X)
    freqs_x = np.fft.rfftfreq(fft_x.size*2 - 1, 1/fs)
    binwidth = np.diff(freqs_x).max()
    half_binwidth = binwidth*0.5
    freqwise_rms_values = np.zeros(freqs_x.size)
    for i, centre_freq in tqdm.tqdm(enumerate(freqs_x)):
        kwargs['freq_range'] = (centre_freq-half_binwidth, centre_freq+half_binwidth)   
        freqwise_rms_values[i] = get_rms_from_fft(freqs_x, fft_x, **kwargs)
    return freqs_x, freqwise_rms_values



def get_customband_rms(X, fs, **kwargs):
    '''
    Given an audio input - get the RMS for user-defined centre-frequencies
    and bin-widths
    
    TODO:
        Throw ValueError if the bin widths and audio size don't match
    
    '''
    fft_x = np.fft.rfft(X)
    freqs_x = np.fft.rfftfreq(fft_x.size*2 - 1, 1/fs)
    binwidth = np.diff(freqs_x).max()
    half_binwidth = binwidth*0.5
    freqwise_rms_values = np.zeros(freqs_x.size)
    for i, centre_freq in tqdm.tqdm(enumerate(freqs_x)):
        kwargs['freq_range'] = (centre_freq-half_binwidth, centre_freq+half_binwidth)   
        freqwise_rms_values[i] = get_rms_from_fft(freqs_x, fft_x, **kwargs)
    return freqs_x, freqwise_rms_values



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

def pascal_to_dbspl(X):
    '''
    Converts Pascals to dB SPL re 20 uPa
    '''
    return dB(X/20e-6)
    
    