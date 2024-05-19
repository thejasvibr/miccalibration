# -*- coding: utf-8 -*-
"""
Created on Wed May  8 09:44:27 2024

@author: theja
"""

import scipy.signal as signal 
import numpy as np 
import matplotlib.pyplot as plt 
import sounddevice as sd
import soundfile as sf

#%%

def powerspec(X, **kwargs):
    fft_X = np.fft.rfft(X)
    fft_freqs = np.fft.rfftfreq(X.size, d=1/kwargs['fs'])
    return fft_freqs, 20*np.log10(abs(fft_X))

def maxnorm_powerspec(X, **kwargs):
    fftfreqs, spectrum = powerspec(X, **kwargs)
    spectrum -= np.max(spectrum)
    return fftfreqs, spectrum
def rms(X):
    return np.sqrt(np.mean(X**2))
#%%
# make a sweep
durns = np.array([3, 5, 7] )*1e-3
fs = 44100 # Hz

all_sweeps = []
for durn in durns:
    t = np.linspace(0, durn, int(fs*durn))
    start_f, end_f = 15e3, 200
    sweep = signal.chirp(t, start_f, t[-1], end_f)
    sweep *= signal.windows.tukey(sweep.size, 0.95)
    sweep *= 0.8
    sweep_padded = np.pad(sweep, pad_width=[int(fs*0.1)]*2, constant_values=[0,0])
    all_sweeps.append(sweep_padded)
    
    
    freqs, power = powerspec(sweep, fs=fs)
    plt.figure()
    plt.subplot(211)
    plt.plot(freqs, power)
    
    plt.subplot(212)
    plt.plot(sweep_padded)
    

sweeps_combined = np.concatenate(all_sweeps)
sf.write('playback_sweeps.wav', sweeps_combined, samplerate=fs)
#%% 
# Also make a gaussian white noise playbacks
noise_durn = 5 # seconds
noise = np.random.normal(0,0.5e-1, int(fs*noise_durn))
noise /= noise.max()
noise *= 0.5

noise_padded = np.pad(noise, pad_width=[int(fs*0.1)]*2, constant_values=[0,0])

#%%
# tones to test if everything corroborates one another
tone_freqs = np.arange(1000, 14000, 2000)
tone_durn = 0.1 # seconds
t_sine = np.linspace(0, tone_durn, int(fs*tone_durn))
all_sines = []
for freq in tone_freqs:
    sine = np.sin(2*np.pi*freq*t_sine)
    sine *= signal.windows.hann(sine.size)
    sine *= 0.8
    sine_padded = np.pad(sine, pad_width=[int(fs*0.1)]*2, constant_values=[0,0])
    all_sines.append(sine_padded)
all_sines_padded = np.concatenate(all_sines)    
plt.figure()
plt.plot(all_sines_padded)


#%% Combine everything into one playback
all_signals_pbk = np.concatenate((sweeps_combined, noise_padded, all_sines_padded))
sf.write('multisignal_calibaudio.wav', all_signals_pbk, samplerate=fs)
