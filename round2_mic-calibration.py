# -*- coding: utf-8 -*-
"""
Round 2 microphone calibration 
==============================
Lena did another round of recordings on 2024-05-27 with the following changes.

* Avisoft Vifa speaker instead of the 2-way speaker used before
* Microphones recorded individually from the same position (height, distance)
* GRAS microphone recording of the playbacks at the beginning of the session AND
at the end of the session 
* All recordings done on the same recorder (TASCAM Portacapture X-06)


Thoughts/notes
~~~~~~~~~~~~~~
* The Sennheiser estimated sensitivity microphone seems to be lower by ~6 dB -- typo in gain value?
* 

Created on Tue May 28 12:49:27 2024

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
from scipy.optimize import minimize_scalar
import os 
from calibration_utility import *
from playback_segmenting import give_expected_segments, align_to_outputfile
import tqdm
import glob

playback_start_stops = give_expected_segments()

#%%
# Find all audio files for the session
session_folder = os.path.join('multisignal_recordings', '2025-05-27')
audio_files = glob.glob(os.path.join(session_folder,'*.wav'))
# manually input the total gain from the notes
mic_gains = [40, 60, 40, 50, 56, 46, 62] # dB

#%% Load the 94 dB SPL rms recording 
onePa_tone , fs = sf.read(audio_files[0])
nyq_fs = fs*0.5
onePa_tone = onePa_tone[:,0]*db_to_linear(-mic_gains[0]) # gain compensation
b,a = signal.butter(2, np.array([100, 2000])/nyq_fs, 'bandpass')
onePa_tone_bp = signal.lfilter(b,a, onePa_tone)

# for 1 Pa we have X a.u. rms and a.u. peak. What is the peak Pressure
# that this recording chain can handle?
rms_au_onePa = rms(onePa_tone_bp)
peak_au_onePa = vrms2vpeak(rms_au_onePa)
ref_Pascals_rms = 1 # sound pressure in Pascals RMS (94 dB SPL is 1Pa)
ref_Pascals_peak = vrms2vpeak(ref_Pascals_rms)

clip_rms_value = 1/np.sqrt(2) # remember the max value of RMS is 1/np.sqrt x Vpeak
rms_clipPa = clip_rms_value*(ref_Pascals_rms/rms_au_onePa) # rms clip point in Pascals
peak_clipPa = ref_Pascals_peak/peak_au_onePa # peak clip point in Pascals
#%% 
# Align the GRAS mic recording to the digital audio file 
gras_stereo, fs = sf.read(audio_files[1])
gras_audio = gras_stereo[:,0]*db_to_linear(-mic_gains[1])
digital_out, fs = sf.read('multisignal_calibaudio.wav')
timealigned_gras = align_to_outputfile(gras_audio, digital_out) 
plt.figure()
plt.plot(absmax_norm(timealigned_gras[:fs]))
plt.plot(digital_out)
#%%
# Knowing the peak clip pressure in the recording chain.
# We can now convert the a.u. audio samples of the speaker playbacks
# into audio samples in Pascals
timeal_gras_pressuresig = timealigned_gras*peak_clipPa

#%% Target microphone 1 : Sennheiser ME66 with 
# start at 7.7 s so that the audio starts with the sweeps.
sennheiser_stereo, fs = sf.read(audio_files[2], start=int(fs*7.7))
senn_audio = sennheiser_stereo[:,0]*db_to_linear(-mic_gains[2])
timealigned_senn = align_to_outputfile(senn_audio, digital_out)

#%% Calcluate the sensitivity of the target microphone for one playback chunk
# Also manually choose a 200 ms silent period to assess SNR. 

# Create audio segments for each playback type
calibmic_chunk_silence = timealigned_gras[int(fs*16.14):int(fs*16.34)] 
calibmic_chunks = [timealigned_gras[start:stop] for (start, stop) in playback_start_stops]
calibmic_chunks.append(calibmic_chunk_silence)

tgtmic1_chunks = [timealigned_senn[start:stop] for (start, stop) in playback_start_stops]
tgtmic1_silence = timealigned_senn[int(7.94*fs):int(8.14*fs)]
tgtmic1_chunks.append(tgtmic1_silence)

sound_index = 3


#%%
plt.figure()
plt.plot(tgtmic1_chunks[sound_index])
plt.plot(calibmic_chunks[sound_index]*12)
#%%

centrefreq_dist = 250 # Hz
halfwidth = centrefreq_dist*0.5
centrefreqs = np.arange(500, 15e3+centrefreq_dist, centrefreq_dist)
bandwise_tgtmic = np.empty(centrefreqs.size)
bandwise_grasmic = np.empty(centrefreqs.size)
bandwise_grasmic_Pa = np.empty(centrefreqs.size)
for i,each in enumerate(centrefreqs):
    bandrms_tgt = get_freqband_rms(tgtmic1_chunks[sound_index], fs,
                            freq_range=(each-halfwidth, each+halfwidth))
    bandrms_grasmicPa = get_freqband_rms(calibmic_chunks[sound_index]*peak_clipPa, fs,
                            freq_range=(each-halfwidth, each+halfwidth))
    bandrms_gras = get_freqband_rms(calibmic_chunks[sound_index], fs,
                            freq_range=(each-halfwidth, each+halfwidth))
    bandwise_tgtmic[i] = bandrms_tgt
    bandwise_grasmic[i] = bandrms_gras
    bandwise_grasmic_Pa[i] = bandrms_grasmicPa

gras_silence_bandwise = np.empty(centrefreqs.size)
tgt_silence_bandwise = np.empty(centrefreqs.size)

for k, each in enumerate(centrefreqs):
    tgt_silence_bandwise[k] = get_freqband_rms(tgtmic1_chunks[-1], fs,
                            freq_range=(each-halfwidth, each+halfwidth))
    gras_silence_bandwise[k] = get_freqband_rms(calibmic_chunks[-1], fs,
                            freq_range=(each-halfwidth, each+halfwidth))
#%% Estimate the SNR at each band for target and calibration mics

snr_target = dB(bandwise_tgtmic/tgt_silence_bandwise)
snr_gras = dB(bandwise_grasmic/gras_silence_bandwise)

plt.figure()
plt.plot(centrefreqs, snr_target, label='Target mic (Sennheiser)')
plt.plot(centrefreqs, snr_gras, label='GRAS mic')
plt.ylabel('Signal-to-noise ratio, dB', fontsize=12)
plt.xlabel('Band centre-frequencies, Hz', fontsize=12)
plt.legend()

# Convert the band-wise rms to band-wise Pascal. 
# We know the max-value is 1/sqrt(2) and the clip Pa
# 
grasmic_bandwise_pascals = rms_clipPa*(bandwise_grasmic/clip_rms_value) # Pascals
tgtmic1_sensitivity_au = bandwise_tgtmic/grasmic_bandwise_pascals # a.u. rms/Pa

plt.figure()

plt.subplot(211)
plt.plot(centrefreqs, tgtmic1_sensitivity_au)
plt.ylabel('Sensit. a.u.$_{rms}$//Pa')
plt.title('Sennheiser ME66 sensitivity')
# We also know the TASCAM has a clip rms of 2 dBu, which is 0.975 Vrms, and also 
# that we expect a ~50 mV/Pa +/- 2.5 dB for this mic. 
Vrms_clip = dbu2vrms(2)
senn_vrms_bandwise = Vrms_clip*(tgtmic1_sensitivity_au/clip_rms_value)

plt.subplot(212)
plt.plot(centrefreqs, senn_vrms_bandwise*1e3)
plt.ylabel('Sensit. mV$_{rms}$/Pa')
plt.gca().set_ylim(10, 80)
plt.hlines(50*db_to_linear(np.array([0,-2.5,2.5])), 0,centrefreqs[-1],
           label='Manuf. specs Nominal mV/Pa $\pm 2.5 dB$')
plt.legend()



