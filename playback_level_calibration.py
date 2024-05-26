# -*- coding: utf-8 -*-
"""
Measuring the calibration sounds with the GRAS mic & Sennheiser ME66 
====================================================================


Some notes on the data
~~~~~~~~~~~~~~~~~~~~~~
This was a test run and so a few things didn't work out particularly well in the end. 

Audio file levels
-----------------
The recordings were done using Audacity running on a Windows laptop. Apparently one 
of the Windows audio protocols automatically halves the audio levels as it makes a
stereo recording. This is horrible because you would normally think values >= 1 mean
clipping, but actually here values >= 0.5 mean clipping. The white noise recordings
are thus very clipped - and there was no way to find out until after the experiments. 

Two recorders
-------------
The target microphone (Sennheiser ME66) was recorded using a TASCAM X-06 Portacapture. 
The calibration microphone (GRAS 1/4") + calibration tone was recorded using an EDIROL UA-25. Two
recording devices were needed because at that point of time the we didn't have a BNC->XLR 
connector. 

Having two recorders adds some amount of complexity into the picture - but for now its been handled
by incorporating/measuring the fullscale voltage of the two recorders. 

The speaker
-----------
The speaker used on 2024-05-08 and 2024-05-13 was a two-way speaker. I suspect this also
contributes a bit to a weird speaker frequency response - especially for the short sweeps. 
There may also be more funky things happening in terms of directionality, but hopefully we were
far enough away from the speaker (3.5 m or so).

What to improve in the next round
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Use one recording device for everything. We could bypass relying on knowing/measuring 
full-scale voltages, and instead directly use the 'clip-pressure' of the recording chain. 
* Check the recording protocol - and thereby also for clipping in the audio files. 
* Use a single-driver speaker. 



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
from scipy.optimize import minimize_scalar
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
minfreq, maxfreq = 200, 15e3
b,a = signal.butter(2, np.array([minfreq, maxfreq])/(fs/2), 'bandpass')
gras_audio, fs = sf.read(audio_filepath,
                         stop=int(fs*8))
gras_audio = signal.filtfilt(b,a, gras_audio)
gras_audio *= 2 # because it was recorded with Audacity using Windows DirectSound
gras_audio *= 10**(gras_gaincomp/20) # account for gain

#%%
# Cross-correlate with the digital signal so you can segment properly. 
digital_out, fs = sf.read('multisignal_calibaudio.wav')
cc_gras = signal.correlate(gras_audio[:fs], digital_out[:fs])
delaypeak_gras = int(np.argmax(cc_gras) - cc_gras.size*0.5)

if delaypeak_gras<0:
    timealigned_gras = np.concatenate((np.zeros(int(delaypeak_gras)),
                                       gras_audio[delaypeak_gras:]))
elif delaypeak_gras>0:
    timealigned_gras = np.concatenate((gras_audio[delaypeak_gras:],
                                       np.zeros(int(delaypeak_gras))))

plt.figure()
plt.plot(digital_out)
plt.plot(absmax_norm(timealigned_gras))


#%%
durns = np.concatenate((np.array([3e-3, 5e-3, 7e-3, 5]),
                        np.tile(0.1,7)))

# on either side.
silence_samples = int(fs*0.1)
start_stops = []
start_sample = 0
gras_chunks = []
for d in durns:
    numsamples = int(fs*d)
    total_samples = numsamples + silence_samples*2
    
    end_sample = start_sample + total_samples
    start_stops.append((start_sample, end_sample))
    audio_chunk = timealigned_gras[start_sample:end_sample]
    start_sample += total_samples 
    gras_chunks.append(audio_chunk)

#%%
# Now let's measure the Vrms of the sweep at each frequency band
# now compensate for the gain and then check if the sensitivity of the mic 

rms_audioclips = [rms(each) for each in gras_chunks]
peak_audioclips = [np.max(abs(each)) for each in gras_chunks]
dbrms = [dB(each) for each in rms_audioclips]
dbpeak = [dB(each) for each in peak_audioclips]

Vrms_clips = np.array([edirol_fs_rms*10**(each/20) for each in dbrms])
Vpeak_clips = np.array([edirol_fs_vp*10**(each/20) for each in dbpeak])

dbspl_peak = spllevel_from_audio(Vpeak_clips, gras_sensitivity_peak)
dbspl_pressure_rms = spllevel_from_audio(Vrms_clips, gras_sensitivity) # dB SPL re 20muPa

#%% And now also estimate the SPL levels of each frequency band
sound_index = 2
input_audio = gras_chunks[sound_index]

#%% Now, load the target microphone audio - MK66 microphone 
# and align the target mic audio with the calib mic audio 
tgtmic_filepath = audio_filepath = os.path.join(other_datafolder, 'CalibExpemic.wav')
pbk_replicates = [(0,8), (8,16), (16, 23.9)] # (start,stop) in s

startind, stopind = np.array(pbk_replicates[0])*fs
tgtmicaudio, fs = sf.read(tgtmic_filepath, start=startind, stop=stopind)
tgtmicaudio = signal.filtfilt(b,a, tgtmicaudio[:,0])
tgtmicaudio *= 2 # because it was recorded with Audacity using Windows DirectSound
tgtmicgain = 34 # dB gain from the TASCAM 
tgtmic_gaincomp = -1*tgtmicgain

# The target microphone was recorded on a TASCAM Portacapture X-06
# on the XLR channel. The audio was set to MIC
# The max signal level as per the specs (https://tascam.com/us/product/portacapture_x6/spec)
# is then 2 dBu

tascam_fs_rms_dbu = 2 # dBu rms 
tascam_fs_vrms = dbu2vrms(tascam_fs_rms_dbu)
tascam_fs_vp = vrms2vp2p(tascam_fs_vrms)*0.5

# So now compensate the gain, but also the lower FS Vp because 0-1 is only ~1.37 p on the 
# Tascam, while 0-1 in 6.37 Vp on the EDIROL

tgtmicaudio *= 10**(tgtmic_gaincomp/20)

#%%
# Time align the target mic audio to the digital out
cc_tgt = signal.correlate(tgtmicaudio[:fs], digital_out[:fs])
delaypeak_tgt = int(np.argmax(cc_tgt) - cc_tgt.size*0.5)

if delaypeak_gras<0:
    timealigned_tgt = np.concatenate((np.zeros(int(delaypeak_tgt)),
                                       tgtmicaudio[delaypeak_tgt:]))
elif delaypeak_gras>0:
    timealigned_tgt = np.concatenate((tgtmicaudio[delaypeak_tgt:],
                                       np.zeros(int(delaypeak_tgt))))
tgt_chunks = [timealigned_tgt[start:stop] for (start, stop) in start_stops]

#%%


# also just convert the audio in raw voltages - makes for an easier comparison

v_tgtchunks = [each*tascam_fs_vp for each in tgt_chunks]
v_grasaudio = [each*edirol_fs_vp for each in gras_chunks]



#%%
# Get the Vrms every 100 Hz, with a bandwidth of +/- 50 Hz of the centre freqs.
centrefreq_dist = 250 # Hz
halfwidth = centrefreq_dist*0.5
centrefreqs = np.arange(500, 15e3+centrefreq_dist, centrefreq_dist)
bandwise_tgtmic = []
bandwise_grasmic = []
for each in centrefreqs:
    bandrms_tgt = get_freqband_rms(v_tgtchunks[sound_index], fs,
                            freq_range=(each-halfwidth, each+halfwidth))
    bandrms_gras = get_freqband_rms(v_grasaudio[sound_index], fs,
                            freq_range=(each-halfwidth, each+halfwidth))
    bandwise_tgtmic.append(bandrms_tgt)
    bandwise_grasmic.append(bandrms_gras)

bandwise_pa = np.array(bandwise_grasmic)/gras_sensitivity
bandwise_tgtmic_sens = bandwise_tgtmic/bandwise_pa
# ME 66 sensitivity from this link: https://assets.sennheiser.com/global-downloads/file/801/ME_66.pdf
plt.figure()
plt.plot(centrefreqs, bandwise_tgtmic_sens*1e3)
plt.ylabel('Target mic sensitivity, mV/Pa'); plt.xlabel('Freq., Hz')
plt.hlines(50, centrefreqs[0], centrefreqs[-1], label='Tech specs expected', linestyles='dashed',)
plt.hlines([50*db_to_linear(2.5), 50*db_to_linear(-2.5)], centrefreqs[0], centrefreqs[-1],
           linestyles='dotted', label='Tech. specs $\pm 2.5$ dB'); plt.title(f'Sennheiser ME66 - w sweep {sound_index}')
plt.ylim(10,100);plt.yticks(np.arange(0,100,10))
plt.legend()
plt.savefig(f'brummgroup_sennheiser_me66_sensitivity_testsound_{sound_index}.png')
#%% 
# Align the two spectra vertically. 
def spectral_distance(compfactor):
    ''' Calculate 'vertical' offset between the gras mic and the target mic
    This broadly corresponds to the overall difference in sensitivity
    '''
    comp_calibmic = dB(bandwise_grasmic) + compfactor
    diff = dB(bandwise_tgtmic) - comp_calibmic 
    return np.median(abs(diff))

comp_factor = minimize_scalar(spectral_distance, bounds=[-100,100])
sensitivity_offset = comp_factor.x # dB

offset_bandwise_gras = sensitivity_offset + dB(bandwise_grasmic)

# The compensated frequency response which needs to be implemented for a flat
# response. 
freq_compensation_tgtmic = offset_bandwise_gras - dB(bandwise_tgtmic) 

plt.figure()
plt.plot(centrefreqs, freq_compensation_tgtmic)    
plt.ylabel('Target mic compensation level, dB');plt.xlabel('Freq., Hz')


#%% Measure the Vrms without compensation of the test tones + noise:
tones_gras_rms = [ rms(each[silence_samples:-silence_samples])  for each in v_grasaudio]  
tones_tgtmic_rms = [ rms(each[silence_samples:-silence_samples])  for each in v_tgtchunks]    

# Now make a compensation filter and pass it over the target audio 
desired = np.pad(db_to_linear(freq_compensation_tgtmic),
                                constant_values=[1,1], 
                                pad_width=1)
bands =  np.pad(centrefreqs,
                                constant_values=[0,fs*0.5], 
                                pad_width=1)
num_taps = 1023
#fir = signal.firls(num_taps, bands, desired, fs=fs)
fir = signal.firwin2(num_taps, bands, desired, fs=fs)
freq, response = signal.freqz(fir)


compensated_tgtmicaudio = [signal.convolve(each[silence_samples:-silence_samples], fir, 'same') for each in v_tgtchunks]
tones_tgtmic_rms_comp = [ rms(each)  for each in compensated_tgtmicaudio]    

# Compare the compensated and uncompensated rms values for each audio chunk. 
plt.figure()
plt.plot(dB(absmax_norm(tones_tgtmic_rms)),'.-' , label='uncompensated')
plt.plot(dB(absmax_norm(tones_tgtmic_rms_comp)),'.-' , label='compensated')
plt.plot(dB(absmax_norm(tones_gras_rms)), '.-' ,label='calibration mic')
plt.grid()
plt.legend()

#%%
# Does the filter actually do its job?
plt.figure()
for chunk_ind in range(4):
    freqs , fft_uncomp = np.fft.rfftfreq(v_tgtchunks[chunk_ind][silence_samples:-silence_samples].size, 1/fs), np.fft.rfft(v_tgtchunks[chunk_ind][silence_samples:-silence_samples])
    freqs , fft_comp = np.fft.rfftfreq(compensated_tgtmicaudio[chunk_ind].size, 1/fs), np.fft.rfft(compensated_tgtmicaudio[chunk_ind])

    plt.plot(freqs, dB(abs(fft_comp))-dB(abs(fft_uncomp)), label=f'test sound {chunk_ind}')
plt.plot(centrefreqs, freq_compensation_tgtmic, '.-', label='expected')    
plt.legend(); plt.title('Microphone frequency compensation - expected vs obtained ')



