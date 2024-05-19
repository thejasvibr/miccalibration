# -*- coding: utf-8 -*-
"""
Measuring the calibration sounds with the GRAS mic
==================================================


Created on Thu May 16 16:36:21 2024

@author: theja
"""

from fullscale_calculations import *
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import soundfile as sf
import scipy.signal as signal 
import scipy.ndimage as ndi
import scipy.stats as stats
import os 
from calibration_utility import *
import tqdm


#%%
edirol_fs_vp = 6.72 # Vpeak
edirol_fs_rms = vpp2rms(edirol_fs_vp*2)
gras_sensitivity = 3.48e-3 # Vrms/Pa
gras_sensitivity_peak = vrms2vp2p(3.48e-3)*0.5 # Vpeak/Pa
gras_gain = 20 + 40 # 20 dB gain on the power module + max gain on the Edirol UA-25
gras_gaincomp = gras_gain*-1
other_datafolder = os.path.join('multisignal_recordings','Calibwthejasvi2024-05-08')
audio_filepath = os.path.join(other_datafolder, 'Calib20240508alltogether.wav')
fs = sf.info(audio_filepath).samplerate
b,a = signal.butter(2, np.array([200, 15e3])/(fs/2), 'bandpass')
gras_audio, fs = sf.read(audio_filepath,
                         stop=int(fs*8))
gras_audio = signal.filtfilt(b,a, gras_audio)
gras_audio *= 2 # because it was recorded with Audacity using Windows DirectSound

gras_segs, env = segment_sounds(gras_audio, 882, 5e-3)
gras_audio *= 10**(gras_gaincomp/20) # account for gain
audio_clips = [gras_audio[each[0].start:each[0].stop]for each in gras_segs]

t_clip = np.linspace(0, audio_clips[0].size/fs, audio_clips[0].size )
plt.figure()
plt.subplot(211)
plt.plot(t_clip, audio_clips[0])
plt.subplot(212)
plt.specgram(audio_clips[0], Fs=fs, NFFT=256, noverlap=128)

#%%
# Now let's measure the Vrms of the sweep at each frequency band
# now compensate for the gain and then check if the sensitivity of the mic 

rms_audioclips = [rms(each) for each in audio_clips]
peak_audioclips = [np.max(abs(each)) for each in audio_clips]
dbrms = [dB(each) for each in rms_audioclips]
dbpeak = [dB(each) for each in peak_audioclips]

Vrms_clips = np.array([edirol_fs_rms*10**(each/20) for each in dbrms])
Vpeak_clips = np.array([edirol_fs_vp*10**(each/20) for each in dbpeak])

dbspl_peak = spllevel_from_audio(Vpeak_clips, gras_sensitivity_peak)
dbspl_pressure_rms = spllevel_from_audio(Vrms_clips, gras_sensitivity) # dB SPL re 20muPa

#%% And now also estimate the SPL levels of each frequency band
input_audio = audio_clips[2]
fftaudio = np.fft.rfft(input_audio)
freqs = np.fft.rfftfreq(fftaudio.size*2 - 1 , 1/fs)
halfbin = fs/input_audio.size
relevant_freqs = freqs[np.logical_and(freqs>=200, freqs<=18e3)]
freq_bins = [ (centre-halfbin, centre+halfbin) for centre in relevant_freqs]
rms_freqbins = []
for freq_bin in tqdm.tqdm(freq_bins):
    rms_freqbins.append(get_rms_from_spectrum(freqs, fftaudio, freq_range=freq_bin))

dbspl_rms = spllevel_from_audio(np.array(rms_freqbins), gras_sensitivity)

plt.figure()
plt.plot(relevant_freqs, dbspl_rms,'-*')

#%% Now, load the target microphone audio - MK66 microphone
tgtmic_filepath = audio_filepath = os.path.join(other_datafolder, 'CalibExpemic.wav')
tgtmicaudio, fs = sf.read(tgtmic_filepath)
tgtmicaudio = signal.filtfilt(b,a, tgtmicaudio[:,0])
tgtmicaudio *= 2 # because it was recorded with Audacity using Windows DirectSound
tgtmicgain = 34 # dB gain from the TASCAM 
tgtmic_gaincomp = -1*tgtmicgain
tgt_segments, env = segment_sounds(tgtmicaudio, 882, 7e-3)
tgtmicaudio *= 10**(tgtmic_gaincomp/20) # account for gain
tgtmic_clips = [tgtmicaudio[each[0].start:each[0].stop]for each in tgt_segments]

plt.figure()
plt.plot(env)
plt.hlines(25e-3, 0, env.size)



