# -*- coding: utf-8 -*-
"""
Edirol sliding knob gain estimation
===================================
Created on Mon May 13 13:04:17 2024

What was done
-------------
Lena fed in a 1 kHz signal with a known p2p amplitude at various
positions using a oscilloscope. 


@author: theja
"""
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import os 
import soundfile as sf
import scipy.signal as signal 
import scipy.ndimage as ndi
from calibration_utility import rms, dB
from fullscale_calculations import vpp2rms

input_vp2p = np.array([1]*7 + [0.75, 0.5, 0.5, 0.25, 0.25, ] + [0.1]*3)
vp2p_db = dB(input_vp2p)
input_rms = vpp2rms(input_vp2p)
gain_positions = np.array([0, 2, 4, 6, 8, 10] + [12]*3 + [14, 14, 16, 16, 18, 20])



calibaudio_path = os.path.join('multisignal_recordings',
                               'CalibEdirol_2024-05-13',
                               'SensEdirolNEW_2023-05-13.wav')

audio, fs = sf.read(calibaudio_path)
clippoint_envelope = abs(signal.hilbert(audio))
# smoothen out the clippoint a bit
smoothened_envelope = signal.convolve(clippoint_envelope, np.ones(10)/10, 'same')
above_thresh = smoothened_envelope >= 0.01
segmented_tones, numtones = ndi.label(above_thresh)
tone_segmented = ndi.find_objects(segmented_tones)
[each[0].stop-each[0].start for each in tone_segmented]


# now extract the audio from each of the regions above threshold
audio_snippets = []
snippets_rms = []
snippets_peak = []
for each in tone_segmented:
    indices = each[0]
    audioclip = audio[indices.start:indices.stop]
    audio_snippets.append(audioclip)
    rms_audioclip = rms(audioclip)
    vpeak_audioclip = np.sqrt(2)*rms_audioclip
    snippets_rms.append(rms_audioclip)
    snippets_peak.append(vpeak_audioclip)
snippets_rms = np.array(snippets_rms)
snippets_peak = np.array(snippets_peak)
#%%
# The Vmeasured = gain x signal_level x sensitivity 
# dB(Vmeasured) = dB(gain) + dB(signal_level) + dB(sensitivity)
# To get the difference in gains we take the difference of two measurements
# dB(gain1/gain_ref) = dB(Vm1/Vm_ref) - dB(level1/level_ref)
# Here let's set gain_ref as the 0th position 

dB_rel_gain = dB(snippets_rms/snippets_rms[0]) - dB(input_rms/input_rms[0])

measurement_data = pd.DataFrame(data={'dB_gain':dB_rel_gain,
                                      'knob_position': gain_positions})

avged_measurement_data = measurement_data.groupby('knob_position').apply(lambda X: dB(np.mean(10**(X['dB_gain']/20))))
avged_gain_knob = pd.DataFrame(data={'gain_dB': np.array(avged_measurement_data), 
                                     'knob_position': sorted(np.unique(gain_positions))})

avged_gain_knob.to_csv('roland_edirol_ua-25_knobpos_vs_gain.csv')

plt.figure()
plt.plot(avged_gain_knob['knob_position'], avged_gain_knob['gain_dB'], '*') 
for x,y in zip(avged_gain_knob['knob_position'], avged_gain_knob['gain_dB']):
    plt.text(x,y+0.5,str(np.round(y,1))) 
plt.xticks(avged_gain_knob['knob_position']);
plt.xlabel('Knob position, marking number (0 : min, 14 : vertical, 20 : max)', fontsize=12)
plt.ylabel('Rel. gain, dB.\n Min gain position assumed at 0 dB gain', fontsize=12)
plt.title('Roland Edirol UA-25 knob position vs gain.')
plt.grid()
plt.savefig('roland_edirol_UA-25_knobposition_vs_gain.png')
