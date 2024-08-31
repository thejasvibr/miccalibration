# -*- coding: utf-8 -*-
"""
Are the microphone frequency responses consistent across sessions?
==================================================================
Let's choose two sessions where we know the Sennheiser ME66 was recorded without
the windshield: 2024-08-19 and 2024-06-24.

Observations
------------
* Just the 1 Pa calibration tone itself can have 0.5 dBrms measurement variation across sessions
* When recorded w a GRAS mic, the white noise has a constant-ish dB rms at the start and beginning. 
  What seems to vary by 1-1.5 dB is the frequency-bin-wise dB rms ...?? How/why?

Notes/corrections in raw data
----------------------------------
* The audio_rec_data.csv - has one corrected gain value. The gain for '2024-06-24\240618_0318.wav' is
originally noted as 66 dB. This is inconsistent with other recordings, and has been corrected to 46 dB.

* The two sessions have different sampling rates. This is important to consider.

Created on Fri Aug 30 22:46:07 2024

@author: thejasvi
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import os
import soundfile as sf
import sys
sys.path.append('../')
from calibration_utility import *
from playback_segmenting import give_expected_segments, align_to_outputfile
from freqresp_utils import segment_sounds_v2, calculate_rms_for_freqbins
#%% Load the calibration mic audio recordings 
rec_data = pd.read_csv('audio_rec_data.csv')
rec_data['session'] = rec_data['filename'].apply(lambda X: os.path.split(X)[0])
rec_data['pure_filename'] = rec_data['filename'].apply(lambda X: os.path.split(X)[1])
rec_by_mic = rec_data.groupby(['mic_model','rec_name'])

gras_1Pa_all = rec_by_mic.get_group(('GRAS1/4', '1Pa_tone'))


gras_1Pa_all['au_rms'] = np.tile(np.nan, gras_1Pa_all.shape[0])
onePa_audio = []
gras_rms_1Pa = []
for i, row in gras_1Pa_all.iterrows():
    final_path = os.path.join('..', 'multisignal_recordings', row['session'], 
                              row['pure_filename'])
    audio, fs  = sf.read(final_path)
    b,a = signal.butter(1, 2e3/(fs*0.5), 'low')
    audio_rec = audio[:,0]
    audio_rec = signal.filtfilt(b,a, audio_rec)
    audio_gaincomp = audio_rec*10**(-row['gain_dB']/20)
    print(row['gain_dB'])
    onePa_audio.append(audio_gaincomp)
    gras_1Pa_all.loc[i,'au_rms'] = rms(audio_gaincomp)
    gras_rms_1Pa.append(gras_1Pa_all.loc[i,'au_rms'])

gras_rms_1Pa = list(map(rms, onePa_audio)) 
print(np.around(gras_rms_1Pa, 4), np.around(dB(gras_rms_1Pa),2))
grasrms_1Pa_avg = np.mean(gras_rms_1Pa) # sensitivity in a.u. rms/Pa

#%% Generate the SPL vs frequency profile from the calibration microphone
gras_pbkrec_all = rec_by_mic.get_group(('GRAS1/4', 'playback_rec'))

ind = 9
row = gras_pbkrec_all.loc[ind,:]
final_path = os.path.join('..', 'multisignal_recordings', row['session'], 
                          row['pure_filename'])
audio, fs  = sf.read(final_path)

if np.isnan(row['pbk_start']):
    audio_rec = audio[:,0]
else:
    audio_rec = audio[int(fs*row['pbk_start']):int(fs*row['pbk_stop']),0]
audio_gaincomp = audio_rec*10**(-row['gain_dB']/20)

#Make the hilbert envelope, and cross-correlate with the hilbert envelope of the 
# synthetic noise signal
import scipy.ndimage as ndi

noise_envelope = np.ones(int(fs*5))*0.8
rec_envelope = abs(signal.hilbert(audio_gaincomp/audio_gaincomp.max()))
rec_envelope = signal.convolve(rec_envelope, np.ones(int(fs*0.01))/int(fs*0.01))

noise_cc = signal.correlate(rec_envelope, noise_envelope, 'same')
noise_cc /= noise_cc.max()
peaks, info = signal.find_peaks(noise_cc, height=0.9, distance=int(fs*3.5))

# plt.figure()
# a0 = plt.subplot(211)
# plt.plot(rec_envelope)
# plt.hlines(np.percentile(rec_envelope, 30),0, rec_envelope.size)
# plt.subplot(212, sharex=a0)
# plt.plot(noise_cc)
# plt.plot(peaks, noise_cc[peaks], '*')

chunks, envelope = segment_sounds_v2(audio_gaincomp, int(fs*5e-3), 25)
#plt.plot([each[0].stop-each[0].start for each in chunks])

long_snips = []
for each in chunks:
    chunk = each[0]
    if (chunk.stop - chunk.start) >= int(fs*4):
        print('yes')
        #plt.plot(audio_gaincomp[chunk.start:chunk.stop])
        long_snips.append(audio_gaincomp[chunk.start:chunk.stop])
    
for each in long_snips:
    print(dB(rms(each)))
#%%
freqbin_separation = 500
freq_bins = np.arange(200, 19000+freqbin_separation, freqbin_separation)

for each in long_snips:
    plt.subplot(211)
    fft = np.fft.rfft(each)
    fftfreqs = np.fft.rfftfreq(each.size, d=1/fs)
    plt.plot(fftfreqs, dB(abs(fft)))
    plt.subplot(212)
    freqbins_rms = calculate_rms_for_freqbins(each, fs, freq_bins)
    plt.plot(freq_bins, dB(freqbins_rms), label=ind)
plt.legend()
    




#%% Load the target microphone recordings 




#%% Generate the frequency response of the target microphones across different sessions





