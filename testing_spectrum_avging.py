# -*- coding: utf-8 -*-
"""
What does spectrum averaging do?
================================
Lena suggested actually using spectrum averaging over a signal to even out 
the weird dips and peaks we're seeing the audio.

Created on Thu Aug 22 06:04:17 2024

@author: theja
"""
import glob
import pandas as pd
import scipy.signal as signal 
import soundfile as sf
import numpy as np 
np.random.seed(78464)
import matplotlib.pyplot as plt
import os 
from calibration_utility import get_centrefreq_rms, get_freqband_rms
#%%
dB = lambda X: 20*np.log10(X)



def calc_spectral_average(X, fs, centre_freqs, num_parts):
    rec_parts = np.array_split(X, num_parts)
    bandwise_tgtmic = []
    halfwidth = np.unique(abs(np.diff(centre_freqs)))*0.5

    for j in range(len(rec_parts)):
        allband_rms = []
        for i,each in enumerate(centre_freqs):
            bandrms_tgt = get_freqband_rms(rec_parts[j], fs,
                                    freq_range=(each-halfwidth, each+halfwidth))
            allband_rms.append(bandrms_tgt)
        bandwise_tgtmic.append(allband_rms)

    stacked_parts = np.array(bandwise_tgtmic)
    mean_rms = np.mean(stacked_parts, axis=0)
    return mean_rms

if __name__ == "__main__":
    
    #%%
    fs = 44100
    durn = 5
    t_durn = np.linspace(0, durn, int(fs*durn))
    # recording = signal.chirp(t_durn, 15000, t_durn[-1], 200)
    recording = np.random.normal(0,0.1,int(fs*durn))
    #recording *= signal.windows.tukey(recording.size)
    #%% Normal spectrum 
    recording_fft = np.fft.rfft(recording)
    spectrum = dB(abs(recording_fft))
    rec_freqs = np.fft.rfftfreq(recording.size, d=1/fs)
    
    #%% RMS of our custom frequency bins
    # Use the Bartlett method
    rec_parts = np.array_split(recording, 5)
    
    centrefreq_dist = 172 # Hz
    halfwidth = centrefreq_dist*0.5
    centrefreqs = np.arange(500, 18e3+centrefreq_dist, centrefreq_dist)
    bandwise_tgtmic = []
    
    for j in range(len(rec_parts)):
        allband_rms = []
        for i,each in enumerate(centrefreqs):
            bandrms_tgt = get_freqband_rms(rec_parts[j], fs,
                                    freq_range=(each-halfwidth, each+halfwidth))
            allband_rms.append(bandrms_tgt)
        bandwise_tgtmic.append(allband_rms)
    
    stacked_parts = np.array(bandwise_tgtmic)
    mean_rms = np.mean(stacked_parts, axis=0)
    
    bandrms_wholetgt_freqrms = []
    for i,each in enumerate(centrefreqs):
        bandrms_wholetgt = get_freqband_rms(recording, fs,
                                freq_range=(each-halfwidth, each+halfwidth))
        bandrms_wholetgt_freqrms.append(bandrms_wholetgt)

    
    #%%
    plt.figure()
    plt.plot(centrefreqs, bandrms_wholetgt_freqrms)
    plt.plot(centrefreqs, mean_rms)
    
    #%%
    segment_size = 256
    centrefreqs_welch, avgspec_welch = signal.welch(recording,
                                                    return_onesided=True,
                                                    window='boxcar',
                                                    nperseg=segment_size, 
                                                    scaling='spectrum', 
                                                    fs=fs)
    
    
    plt.plot(centrefreqs_welch, np.sqrt(avgspec_welch))
    
