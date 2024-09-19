# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 15:35:22 2024

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

plt.close("all")

# import basicfunctions as bf

def DC_remove(audio):
    audio2 = audio-np.mean(audio)
    return audio2

def normalize_energy(audio):
    norm = audio/np.sqrt(np.sum(audio**2))
    return norm

def Sweeptovertical (audio, Sweeprate, fs, plot = True):
    swp=DC_remove(audio)
    L=len(swp)
    G=fft.fft(swp)
    f = np.arange(0, L, 1)/L*fs
    t = f*L/fs**2
    gdl=f/Sweeprate

    modu=np.exp(1j*np.cumsum(gdl)*2*np.pi/L*fs)
    modu[int(len(modu)/2):-1]=0
    modu[-1]=0

    modsig=np.real(fft.ifft(G*modu))*2

    if plot: 
        fig, ax = plt.subplots(3, 1)
        ax[0].specgram(swp, Fs=fs, NFFT=64, noverlap = 32)
        ax[1].specgram(modsig, Fs= fs, NFFT=64, noverlap = 32)
        ax[2].plot(t, modsig)
        plt.draw()
        
    return modsig, modu

def Cleansweep (modsig, modu):

    G2 = fft.fft(modsig)
    modsig2=np.real(fft.ifft(G2*np.flip(modu)))*2

    return modsig2


def Energy (audio):
    return np.sum(audio**2)

def RMS(x):
    # calculate root mean square
    rms = np.sqrt(np.mean(x*x))
    return(rms)

def dB(x, ref=None):
    if ref==None:
        ref= 1
    dBx = 20*np.log10(x/ref)
    return dBx

def undB(x, ref=None):
    if ref==None:
        ref= 1
    dBx = ref * 10**(x/20)
    return dBx

def DiffSpecs (spec_Gras_dB, f_g, spec_Sweep_dB, f_s):
    if f_s[-1]<f_g[-1]: 
        spec_Gras_dB = spec_Gras_dB[f_g<f_s[-1]]        
        f=f_s
    elif f_g[-1]<f_s[-1]: 
        spec_Sweep_dB = spec_Sweep_dB[f_s<f_g[-1]]
        f=f_g
    else:
        f = f_g
    
    f=resample_by_interpolation(f ,n_out=513)
    spec_Gras_dB = resample_by_interpolation(spec_Gras_dB ,n_out=513)
    spec_Sweep_dB = resample_by_interpolation(spec_Sweep_dB ,n_out=513)

    DiffSpec = [Sweep - Gras for Sweep, Gras in zip (spec_Sweep_dB, spec_Gras_dB)]
    return f, DiffSpec


def resample_by_interpolation(signal, input_fs=None, output_fs=None, n_out=None):
    # DISCLAIMER: This function is copied from https://github.com/nwhitehead/swmixer/blob/master/swmixer.py, 
    #             which was released under LGPL. 
    
    if n_out is None:
        scale = output_fs / input_fs
        # calculate new length of sample
        n = round(len(signal) * scale)
    elif input_fs is None:
        n=n_out

    # use linear interpolation
    # endpoint keyword means than linspace doesn't go all the way to 1.0
    # If it did, there are some off-by-one errors
    # e.g. scale=2.0, [1,2,3] should go to [1,1.5,2,2.5,3,3]
    # but with endpoint=True, we get [1,1.4,1.8,2.2,2.6,3]
    # Both are OK, but since resampling will often involve
    # exact ratios (i.e. for 44100 to 22050 or vice versa)
    # using endpoint=False gets less noise in the resampled sound
    resampled_signal = np.interp(
        np.linspace(0.0, 1.0, n, endpoint=True),  # where to interpret
        np.linspace(0.0, 1.0, len(signal), endpoint=True),  # known positions
        signal,  # known data points
    )
    return resampled_signal
    

#%%
   
# wav_folder = WindowsPath("D://Lena//panama//TagFresp")
wav_folder = WindowsPath("D://Lena//0HBrumm//Mic_calib//2024-08-19//sweeps")

wav_files = [p.stem for p in wav_folder.iterdir() if p.is_file() and p.suffix == ".wav"]

# tagsweeps_files = [t for t in wav_files if "Sweep" in t]
tagsweeps_files = [t for t in wav_files if "Sweep3" in t]
print(tagsweeps_files)


# GrasRec_file = wav_folder / "mic4_t40dB_40cm_kl14_lodretposition_5VADPeak_14.wav"
GrasRec_file = wav_folder / "Sweep_3_gras.wav"

# InitPb_file = wav_folder / "sweepkorr.wav"


#%%

# make the magic xc template

## fill in by hand the calibration sweep
# PB, fs_i = sf.read(InitPb_file)

# Start Frequency (kHz)
SF = 10
SF = 15
# End Frequency (kHz)
EF = 94
EF = 1
# Duration (sec)
D = 0.6
D = 0.0065

## calculate sweep rate
Sweeprate = (EF-SF)*1000/D


## gras mic data

Sens_gras = -60.28
Sens_gras = -17.27-36

Gain = 40
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
    sweeppath = wav_folder / f"{tag}.wav"
    Sweep, fs_s = sf.read(sweeppath)
    Sweep = DC_remove(Sweep)

    ## Kristian's magic 
    SweepXC, modu = Sweeptovertical(Sweep, Sweeprate=Sweeprate, fs=fs_s)
    

    mx = np.where(SweepXC == np.max(SweepXC))[0][0]
    
    detect = [(mx/fs_s)-0.001, (mx/fs_s)+0.001]
    
    CleanXC = SweepXC
    CleanXC[0:int(detect[0]*fs_s)]=0
    CleanXC[int(detect[1]*fs_s):-1]=0
    
    t = np.linspace(0, len(CleanXC)/fs_s, len(CleanXC))
    plt.plot(t, CleanXC)
    
    
    ## back transform to sweep
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

    

FrespPath = wav_folder / f"Fresp_{pd.to_datetime('today').strftime('%Y-%m-%d')}.csv"
Frequency_response.to_csv(FrespPath, index=False)

SensPath = wav_folder / f"Sens_{pd.to_datetime('today').strftime('%Y-%m-%d')}.csv"
(pd.DataFrame.from_dict(data=Sensitivity, orient='index').to_csv(SensPath, header=["Sensitivity_dB_re_1UAPa"]))

#%%


