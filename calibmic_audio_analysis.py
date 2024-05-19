# -*- coding: utf-8 -*-
"""
Calibration microphone audio analysis
=====================================



Note
----
We found out later that when using the Windows MME protocol on Audacity (v 3.4.2)
clipping happens already at +/-0.5 as somehow Windows halves all sample values 
for stereo recordings (even though the recording is Mono). 

When the protocol is changes to Windows WASAPI, the same signal results in clipping 
anyway at +/- 1 - so it's all self-consistent.




Created on Mon May 13 10:19:24 2024

@author: theja
"""
import soundfile as sf 
import scipy.signal as signal 
import numpy as np 
import matplotlib.pyplot as plt
import os 
from calibration_utility import *


gras_gain = 20 # dB 
edirol_gain = 0 # baseline
deamplif_dB = -(gras_gain + edirol_gain)
calibaudio_path = os.path.join('multisignal_recordings',
                               'Calibwthejasvi2024-05-08',
                               'Calib20240508alltogether.wav')
#%% 
fs = sf.info(calibaudio_path).samplerate
refsignalaudio, fs = sf.read(calibaudio_path, stop=int(fs*29.5))
refsignalaudio *= 2 # to compensate for the weird halving by Windows

b,a = signal.butter(1, np.array([0.5e3,2e3])/(fs*0.5), 'bandpass')
reference_toen = signal.filtfilt(b,a,refsignalaudio[-int(3*fs):-int(2.5*fs)])
rms_reftone = rms(reference_toen)*10**(deamplif_dB/20)
dbrms_reftone = dB(rms_reftone) # re 1 a.u. rms 

#% At 94 dB SPL, the mic produces 0.05 a.u. rms 



#%% 








