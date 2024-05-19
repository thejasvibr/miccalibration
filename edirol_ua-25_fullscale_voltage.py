# -*- coding: utf-8 -*-
"""
Estimating the 

Created on Thu May  9 08:33:47 2024

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
#%%
# specs say -60 to -20 dBu is the "nominal input level" for XLR
# and -36 to +4 dBu for the 1/4" TRS jack input. 
# The nominal input level only gives the general working range of the 
# device, and NOT the maximum. We need to figure that out ourselves. 


datafolder = os.path.join('multisignal_recordings','CalibEdirol_2024-05-13')

#%% Gain
# The sinusoids were recorded at knob position 10.
position_gain = pd.read_csv('roland_edirol_ua-25_knobpos_vs_gain.csv')
gain = float(position_gain.groupby('knob_position').get_group(10)['gain_dB'])


#%% Input Voltage data
#   ==================
# All recordings were made at 'vertical'  gain, which is  knob position 10
input_vpp = np.array([0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1, 1.25, 1.5, 1.75, 2, 2.25, ])*10**(gain/20)
input_vp = input_vpp/2

#%%
# load audio and detect tones at various gain settings
# IMPORTANT: the audio was recorded on the XLR input channel

clippoint_raw, fs = sf.read(os.path.join(datafolder,'ClippingPoint_ppV_2023-05-13.wav'))
clippoint_envelope = abs(signal.hilbert(clippoint_raw))
# smoothen out the clippoint a bit
smoothened_envelope = signal.convolve(clippoint_envelope, np.ones(100)/100, 'same')
above_thresh = smoothened_envelope > 0.01 
segmented_tones, numtones = ndi.label(above_thresh)
tone_segmented = ndi.find_objects(segmented_tones)

# plt.figure()
# plt.subplot(211)
# plt.plot(smoothened_envelope)
# plt.subplot(212)
# plt.plot(segmented_tones)


# now extract the audio from each of the regions above threshold
# also compensate for the gain
audio_snippets = []
snippets_rms = []
snippets_peak = []
for each in tone_segmented:
    indices = each[0]
    audioclip = clippoint_raw[indices.start:indices.stop]
    audio_snippets.append(audioclip)
    rms_audioclip = rms(audioclip)
    vpeak_audioclip = np.sqrt(2)*rms_audioclip
    snippets_rms.append(rms_audioclip)
    snippets_peak.append(vpeak_audioclip)
snippets_rms = np.array(snippets_rms)
snippets_peak = np.array(snippets_peak)
#%%
dBVp = dB(input_vp)
dB_peak = dB(snippets_peak)
slope, intcpt, r_value, p_value, stderr = stats.linregress(dBVp, dB_peak)

pred_line = slope*dBVp + intcpt

#%% Estimating the dBV for 0 dB FS.
# 
# Set y = 0 in the y = mx + c, and estimate the dBV required to get 0 dB FS
# x = c/m
FS_dBV = -intcpt/slope
FS_Vp = 10**(FS_dBV/20)
FS_Vpp = FS_Vp*2
FS_dbu = vpp2dbu(FS_Vpp)


plt.figure()
plt.plot(dBVp, dB_peak, '*', label='observed')
plt.plot( dBVp, pred_line,  'r', label='line-fit')
plt.grid()
plt.ylim(-40,0);plt.xlim(-30, 20)
plt.legend()
plt.gca().set_aspect('equal')

plt.xlabel('Input signal dBVp (re 1V)')
plt.ylabel('Measured dBFS (peak amp re 1)')
plt.title(f'Linear response (slope is ~ 1!! ({np.round(slope,2)} dBpeak/1dBV input) \
          \n Est. max Vp is: {np.round(FS_Vp,1)} V')
plt.savefig('EDIROLUA-25_FSVp_estimation.png')

#%%
# Now let's double check this with the Edirol at MIN gain (position 0), and various voltages. 
# If the min gain corresponds to 0 dB gain, then we expect to recover the same full-scale voltage
# as that derived from the measurements utilising the other gains estimated from knob positions.

mingain_audio, fs = sf.read(os.path.join(datafolder,'1_2_5Vpp_gainMinEdirol_2023-05-15.wav'))
mingain_audio_envelope = abs(signal.hilbert(mingain_audio))
smooth_envelope = signal.convolve(mingain_audio_envelope, np.ones(100)/100, 'same')
segmented_signals, numsignals = ndi.label(smooth_envelope > 0.01 )
mingain_segments = ndi.find_objects(segmented_signals)
input_vpp = np.array([1, 2, 5])
input_vp = input_vpp*0.5

mingain_rms = np.zeros(len(mingain_segments))
mingain_vp = np.zeros(mingain_rms.size)
for i,each in enumerate(mingain_segments):
    start, stop = each[0].start, each[0].stop
    audio = mingain_audio[start:stop]
    audiorms = rms(audio)
    audiovp = vrms2vp2p(audiorms)/2
    mingain_rms[i] = audiorms
    mingain_vp[i] = audiovp

plt.figure()
plt.plot(dB(input_vp), dB(mingain_vp),'-*')
plt.ylim(-40,0)

slope, intcpt, r_value, p_value, stderr = stats.linregress(dB(input_vp), dB(mingain_vp))
FS_dBV_mingain = -intcpt/slope
FS_Vp_mingain = 10**(FS_dBV_mingain/20)
FS_Vpp_mingain = FS_Vp_mingain*2
FS_dbu_mingain = vpp2dbu(FS_Vpp_mingain)
#%% Double-checking the FS Vp of the Edirol with a known mic sensitivity
# The sensitivity of the GRAS 1/4" is 3.4 mV/Pa. We expect that it will output
# a 3.4 mV rms signal for the  94 dB SPL calibration tone (94 dB SPL re 20muPa is 1Pa).
# If our gain estimates and FS Vp estimates are correct, then the expected and predicted
# values should match. 

gras_sens = 3.48e-3 # V/Pa for a GRAS 1/4" microphone. 
gras_sens_vp = vrms2vp2p(gras_sens)*0.5 # sensitivity in Vp/Pa
gras_gain = 20 + 40 # 20 dB gain on the power module + max gain on the Edirol UA-25
gras_gaincomp = gras_gain*-1
other_datafolder = os.path.join('multisignal_recordings','Calibwthejasvi2024-05-08')
gras_audio, fs = sf.read(os.path.join(other_datafolder, 'Calib20240508alltogether.wav'))
gras_audio *= 2 # because it was recorded with DirectSound
gras_segs, env = segment_sounds(gras_audio, 22050, 0.3)

# now compensate for the gain and then check if the sensitivity of the mic 
gras_audio *= 10**(gras_gaincomp/20)

calib_tone = gras_audio[gras_segs[-2][0].start:gras_segs[-2][0].stop]
calibtone_rms = rms(calib_tone)
calibtone_peak = vrms2vp2p(calibtone_rms)*0.5

# Now, if the FS Vp is correct the obtained dBVp and the expected Vp given the 
# GRAS sensitivity should match well. 
meas_dBFS_vp = dB(calibtone_peak) # measured dB Vp re 1 at 1 Pascal pressure
expected_dBFS_vp = dB(gras_sens_vp/FS_Vp) # expected dBVp FS at 1 Pascal pressure
meas_pred_error = meas_dBFS_vp - expected_dBFS_vp
print(f'The expected dB_Vp re FS is {expected_dBFS_vp}, obtained is: {meas_dBFS_vp}')
print(f'The error in obtained-expected is : {meas_pred_error} dB')

#%% Wrapping it up
#   --------------
# We can now be sure that the 




