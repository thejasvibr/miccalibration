# -*- coding: utf-8 -*-
"""
Testing the effect of frequency resolution on the frequency-band rms
====================================================================


Testing and implementing a 'native' frequency bin-wise rms calculator. 
This means, the frequency resolution is set naturally by the length of the 
input audio, thus removing the ambiguity of overlapping frequency-bins caused
by user-defined frequency bins. 

Created on Wed Sep 18 16:39:30 2024

@author: theja
"""

import scipy.signal as signal 
import numpy as np 
import matplotlib.pyplot as plt 
from freqresp_utils import calculate_rms_for_freqbins
import sys 
sys.path.append('../')
from calibration_utility import get_rms_from_fft

rms = lambda X: np.sqrt(np.mean(X**2))

freq_resolution = 100
freq_bands = np.arange(200, 19200, freq_resolution)


# synthetic signal with known rms sine waves
fs = 44100 # Hz
sine_freq = [2500, 5000, 6500] # Hz
durn = 50e-3
sine_waves = np.zeros(int(fs*0.05))
sine_wave_rms = []
t = np.linspace(0, durn, sine_waves.size)

for f in sine_freq:
    rand_phase = np.random.choice(np.linspace(0,2*np.pi,1000), 1)
    sine_wave_freq = np.sin(2*np.pi*f*t + rand_phase)/len(sine_freq)
    sine_wave_freq *= signal.windows.tukey(sine_wave_freq.size, alpha=0.1)
    sine_wave_rms.append(rms(sine_wave_freq))
    sine_waves += sine_wave_freq

freqband_rms = calculate_rms_for_freqbins(sine_waves , fs, freq_bands)
padded = np.pad(sine_waves, pad_width=[int(fs*0.01)]*2, mode='constant', 
                constant_values=[0,0])
paddedfreqband_rms = calculate_rms_for_freqbins(padded , fs, freq_bands)

plt.figure()
plt.plot(sine_freq, sine_wave_rms, '-*')
plt.plot(freq_bands, freqband_rms)


#%% 
rfft = np.fft.rfft(sine_waves)
freqs = np.fft.rfftfreq(sine_waves.size, 1/fs)
get_rms_from_fft(freqs,rfft, freq_range=[900, 1040])

mean_sigsquared = np.sum(abs(rfft)**2)/rfft.size
root_mean_squared = np.sqrt(mean_sigsquared/(2*rfft.size-1))

def calc_freqwise_rms(X, fs):
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

# make a freq-wise rms calculation 
freqwise_rms = []
for each in rfft:
    mean_sq_freq = np.sum(abs(each)**2)/rfft.size
    rms_freq = np.sqrt(mean_sq_freq/(2*rfft.size-1))
    freqwise_rms.append(rms_freq)

ff, ffrms = calc_freqwise_rms(sine_waves, fs)