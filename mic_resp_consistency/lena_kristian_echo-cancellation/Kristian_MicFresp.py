# -*- coding: utf-8 -*-
"""
Lena's Python module with Thejasvi's modifications on the go. 
=============================================================
Created on Fri Sep 13 15:35:22 2024

Inputs 
------
* Sweep snippets from main audio file (target and gras microphone)
* GRAS 1 Pa tone sensitivity in dB rms (gain compensated)
* Start/end freqs & duration of the sweep
* Find the peak in the cross-correlation (post-convolution) - this can be automated with a peak-finder alg.


Changes made
------------
* Changes GRAS sweep files to keep naming convention consistent


@author: Lena
"""

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
from scipy import fft
from pathlib import WindowsPath
import os
import glob

plt.close("all")
from utility_Kristian_MicFresp import * 
    

#%%

wav_folder = "sweeps/"

wav_files = glob.glob(os.path.join(wav_folder, '*.wav'))

sweep_type = 'Sweep3' # other options are Sweep1, Sweep2, Sweep3
tagsweeps_files = [t for t in wav_files if sweep_type in t]
print(tagsweeps_files)

# choose the corresponding GRAS mic sweep
GrasRec_file = [each for each in tagsweeps_files if 'gras' in each][0]


#%%

# make the magic xc template
sweep_durations = {'Sweep3': 7e-3, 'Sweep2': 5e-3, 'Sweep1': 3e-3} # seconds

D = sweep_durations[sweep_type]

# Start & end frequencies
SF, EF = 15e3, 0.2e3 # Hz

## calculate sweep rate
Sweeprate = (EF-SF)/D

## gras mic data
Sens_gras = -17.27-36 # dB RMS a.u. of the 1 Pa tone. 

Gain = 20+36

#%%

# process grass mic rec

## load, select channel 1, remove DC offset
Gras, fs_g = sf.read(GrasRec_file)
# Gras = Gras[:, 0]
Gras = DC_remove(Gras)

## Kristian's magic 
GrasXC, modu = Sweeptovertical(Gras, Sweeprate=Sweeprate, fs=fs_g)
plt.show()

## find the peak (by hand on graph)
detect = [0.0492, 0.0504]
detect = [0.047, 0.0485]

## select the part without the echo
CleanXC = GrasXC
CleanXC[0: int(detect[0]*fs_g)]=0
CleanXC[int(detect[1]*fs_g):-1]=0

## back transform to sweep
Clean_gras = Cleansweep(CleanXC, modu)

#%%
fftsize = 64
overlap = fftsize - 1 
plt.figure();
plt.subplot(211);plt.specgram(Gras, Fs=fs_g,NFFT=fftsize, noverlap=overlap);
plt.xticks([]);plt.ylabel('Frequency, Hz');plt.title('Raw microphone audio')
plt.subplot(212);plt.specgram(Clean_gras, Fs=fs_g, NFFT=fftsize, noverlap=overlap)
plt.title('Processed - only template & direct path')
plt.xlabel('Time, s')
#%%
## spectrum
fr_g,spec_Gras = signal.periodogram(Gras, fs=fs_g)
fr_g,spec_CleanGras = signal.periodogram(Clean_gras, fs=fs_g)

fig, ax = plt.subplots(2, 1)
ax[0].specgram(Clean_gras, Fs=fs_g)
ax[1].plot(fr_g,spec_Gras)
ax[1].plot(fr_g,spec_CleanGras)
plt.show()

## energy
E = np.sum(Clean_gras**2)
Ei = np.sum(Gras**2)

RMS_Gras = RMS(Clean_gras)
RMS_Orig = dB(RMS_Gras) + 94 - Sens_gras - Gain
#%%

# process tag mics

Sensitivity ={}
Frequency_response = pd.DataFrame()

for tag in tagsweeps_files:
    
    # read and remove DC offset
    sweeppath = tag
    Sweep, fs_s = sf.read(sweeppath)
    Sweep = DC_remove(Sweep)

    ## Kristian's magic 
    SweepXC, modu = Sweeptovertical(Sweep, Sweeprate=Sweeprate, fs=fs_s)
    
    # Cross-correlation peak finding part 
    mx = np.where(SweepXC == np.max(SweepXC))[0][0]
    
    detect = [(mx/fs_s)-0.001, (mx/fs_s)+0.001]
    
    CleanXC = SweepXC
    CleanXC[0:int(detect[0]*fs_s)]=0
    CleanXC[int(detect[1]*fs_s):-1]=0
    
    t = np.linspace(0, len(CleanXC)/fs_s, len(CleanXC))
    plt.plot(t, CleanXC)
    
    
    ## back transform to sweep - recover the echo-cancelled playback
    Clean_Sweep = Cleansweep(CleanXC, modu)

    ## spectrum
    # fr_s,spec_Sweep = signal.welch(Sweep, nfft=1024, fs=fs_s)
    # fr_s,spec_CleanSweep = signal.welch(Clean_Sweep, nfft=1024, fs=fs_s)
    
    ## spectrum
    fr_s,spec_Sweep = signal.periodogram(Sweep, fs=fs_s)
    fr_s,spec_CleanSweep = signal.periodogram(Clean_Sweep, fs=fs_s)
    
    spec_CleanGras_dB = 10*np.log10(spec_CleanGras/np.max(spec_CleanGras))
    spec_CleanSweep_dB = 10*np.log10(spec_CleanSweep/np.max(spec_CleanSweep))
    
    ## calcualte frequency response
    Fr, Fresp = DiffSpecs(spec_CleanGras_dB, f_g= fr_g, spec_Sweep_dB= spec_CleanSweep_dB, f_s= fr_s)
    Fresp = np.where(Fr<1000*min(SF, EF)+1500, 0, Fresp)
    Fresp = np.where(Fr>1000*max(SF, EF)-1500, 0, Fresp)
    Fr[-1]=fs_s/2
    
    if tag == tagsweeps_files[0]:
        Frequency_response["Frequency_Hz"] = Fr
    
    fig, ax = plt.subplots(3, 1)
    ax[0].specgram(Clean_Sweep, Fs=fs_s, noverlap = 32, NFFT=64)
    ax[1].plot(fr_s,spec_CleanSweep_dB)
    ax[1].plot(fr_g, spec_CleanGras_dB)
    ax[2]. plot(Fr, Fresp)
    plt.show()
    
    # correct thingy
    Filt = 10**((0-Fresp)/20)
    Impulse_response = signal.firwin2(513, freq= Fr, gain = Filt, fs = fs_s, antisymmetric=False)
    correct = signal.lfilter(Impulse_response, 1, Clean_Sweep)
    
    # f, s = signal.welch(Sweep)
    # f2, s2 = signal.welch(correct)
    
    f, s = signal.periodogram(Sweep)
    f2, s2 = signal.periodogram(correct)
    
    # plt.figure()
    # plt.plot(f, 10*np.log10(s))
    # plt.plot(f2, 10*np.log10(s2))
    # plt.show()
    
    ## energy
    E = np.sum(Clean_Sweep**2)
    Ei = np.sum(Sweep**2)

    ## sensitivity
    RMS_Sweep = RMS(correct)
    Sens_mic = 94 + dB(RMS_Sweep) - RMS_Orig 

    ## save
    Frequency_response[f"Gain_dB_{tag}"] = Fresp
    
    Sensitivity[ f'Sens_dB_{tag}']= Sens_mic

    

FrespPath = os.path.join(wav_folder, f"Fresp_{pd.to_datetime('today').strftime('%Y-%m-%d')}.csv")
Frequency_response.to_csv(FrespPath, index=False)

SensPath = os.path.join(wav_folder, f"Sens_{pd.to_datetime('today').strftime('%Y-%m-%d')}.csv")
(pd.DataFrame.from_dict(data=Sensitivity, orient='index').to_csv(SensPath, header=["Sensitivity_dB_re_1UAPa"]))
#%%
plt.plot(Frequency_response['Frequency_Hz'])

