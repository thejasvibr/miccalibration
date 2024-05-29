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
import tqdm
import glob

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
# Knowing the peak clip pressure in the recording chain.
# We can now convert the a.u. audio samples of the speaker playbacks
# into audio samples in Pascals








