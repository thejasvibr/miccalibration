# -*- coding: utf-8 -*-
"""
Tegernsee retreat calibration 
=============================

Speaker model: Avisoft speaker (with ERC 1)
Microphone models used: 
    * Sennheiser MKH<> with windshield
    * Sennheiser ME66 (naked)
    * Behringer <> (calibration microphone)


"""


from fullscale_calculations import *
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import soundfile as sf
import scipy.signal as signal 
import scipy.ndimage as ndi
import scipy.stats as stats
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
import os 
from calibration_utility import *
from playback_segmenting import give_expected_segments, align_to_outputfile
import tqdm
import glob
#%%
playback_start_stops_raw = give_expected_segments() # for 44100 Hz sampling rate
# but now also make sure you have it working for 96 kHz sampling. Thus multiple everything

conversion_factor = 96000/44100
playback_start_stops = []
for each in playback_start_stops_raw:
    start, stop = each
    playback_start_stops.append((int(start*conversion_factor),
                                 int(stop*conversion_factor))
                                )
    

#%%
# Find all audio files for the session
session_folder = os.path.join('multisignal_recordings', '2024-06-24')
rec_data = pd.read_csv(os.path.join(session_folder,'recording_parameters.csv'))
by_recname = rec_data.groupby('rec_name')

calibration_recs = by_recname.get_group('1Pa_tone').reset_index(drop=True)

#%% Load the 94 dB SPL rms recording 
onePa_tone , fs = sf.read(os.path.join(session_folder, calibration_recs.loc[0,'filename']))
nyq_fs = fs*0.5
onePa_tone = onePa_tone[:,0]*db_to_linear(-calibration_recs.loc[0,'gain_dB']) # gain compensation
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
calib_ind = 0
calibmic_recs = by_recname.get_group('calib_playback').reset_index(drop=True)
calibmic_start_t = calibmic_recs.loc[calib_ind,'playback_start']
calibmic_startind = int(fs*calibmic_start_t)
gras_stereo, fs = sf.read(os.path.join(session_folder, calibmic_recs.loc[calib_ind,'filename']),
                          start=calibmic_startind)
gras_audio = gras_stereo[:,0]*db_to_linear(-calibmic_recs.loc[calib_ind,'gain_dB'])
digital_out, fs_pbk = sf.read('multisignal_calibaudio.wav')

# Check if the recording audio is the same as the digital out audio samplerate. 
upsampled_digitalout = signal.resample(digital_out, int(digital_out.size*(fs/fs_pbk)))

#raise IndexError('Some weird stuff happening with alignment!!')
timealigned_gras, gras_delpeak = align_to_outputfile(gras_audio, upsampled_digitalout) 
plt.figure()
plt.plot(absmax_norm(timealigned_gras[:fs]))
plt.plot(upsampled_digitalout)
    #%%
# Knowing the peak clip pressure in the recording chain.
# We can now convert the a.u. audio samples of the speaker playbacks
# into audio samples in Pascals
timeal_gras_pressuresig = timealigned_gras*peak_clipPa

#%% Target microphone audio
mic_names = list(by_recname.groups.keys())
target_micname = mic_names[1]
tgt_mic = by_recname.get_group(target_micname).reset_index(drop=True)
start_ind = int(fs*tgt_mic.loc[:,'playback_start'])
tgtmic_gain = int(tgt_mic.loc[:,'gain_dB'])
audiofile_path = os.path.join(session_folder, tgt_mic.loc[:,'filename'].to_list()[0])
tgtmic_stereo, fs = sf.read(audiofile_path,
                            start=start_ind)
tgtmic_audio = tgtmic_stereo[:,0]*db_to_linear(-tgtmic_gain)
timealigned_tgtmic, tgtmic_delpeak = align_to_outputfile(tgtmic_audio, upsampled_digitalout)

plt.figure()
plt.plot(absmax_norm(timealigned_tgtmic))
plt.plot(upsampled_digitalout)

#%% Calcluate the sensitivity of the target microphone for one playback chunk
# Also manually choose a 200 ms silent period to assess SNR. 

# Create audio segments for each playback type
calibmic_silence_startstop = calibmic_recs.loc[calib_ind,['silence_start', 'silence_stop']]
calibmic_silence_start, calibmic_silence_stop = calibmic_silence_startstop - calibmic_start_t
calibmic_chunk_silence = gras_audio[int(fs*calibmic_silence_start):int(fs*calibmic_silence_stop)] 
calibmic_chunks = [timealigned_gras[start:stop] for (start, stop) in playback_start_stops]
calibmic_chunks.append(calibmic_chunk_silence)


tgtmic1_chunks = [timealigned_tgtmic[start:stop] for (start, stop) in playback_start_stops]
tgtmic_sil_startstop = tgt_mic.loc[0,['silence_start', 'silence_stop']]
tgtmic_startsil, tgtmic_stopsil = tgtmic_sil_startstop - start_ind/fs
tgtmic1_silence = tgtmic_audio[int(tgtmic_startsil*fs):int(tgtmic_stopsil*fs)]
tgtmic1_chunks.append(tgtmic1_silence)

sound_index = 3
#%%
plt.figure()
plt.plot(absmax_norm(tgtmic1_chunks[sound_index]))
plt.plot(absmax_norm(calibmic_chunks[sound_index]))
#%%

centrefreq_dist = 250 # Hz
halfwidth = centrefreq_dist*0.5
centrefreqs = np.arange(500, 18e3+centrefreq_dist, centrefreq_dist)
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

# Convert the band-wise rms to band-wise Pascal. 
# We know the max-value is 1/sqrt(2) and the clip Pa
# 
grasmic_bandwise_pascals = rms_clipPa*(bandwise_grasmic/clip_rms_value) # Pascals
tgtmic1_sensitivity_au = bandwise_tgtmic/grasmic_bandwise_pascals # a.u. rms/Pa

plt.figure()

plt.subplot(311)
plt.plot(centrefreqs, tgtmic1_sensitivity_au)
plt.ylabel('Sensit. a.u.$_{rms}$//Pa')
plt.title('sensitivity - with gain values as given')
# We also know the TASCAM has a clip rms of 2 dBu, which is 0.975 Vrms, and also 
# that we expect a ~50 mV/Pa +/- 2.5 dB for this mic. 
file_ind = 2
Vrms_clip = dbu2vrms(2)
senn_vrms_bandwise = Vrms_clip*(tgtmic1_sensitivity_au/clip_rms_value)

sens = tgt_mic['expected_mvPa']*1e-3 # from mV to V

plt.subplot(312)
plt.plot(centrefreqs, senn_vrms_bandwise*1e3)
plt.ylabel('Sensit. mV$_{rms}$/Pa')
# plt.gca().set_ylim(10, 80)
plt.hlines([sens*1e3, sens*db_to_linear(-2.5)*1e3,
            sens*db_to_linear(2.5)*1e3], 0, centrefreqs[-1],
           label=f'user-manual value {np.around(sens*1e3, 2)} mV/Pa $\pm 2.5$ dB')
# plt.hlines(50*db_to_linear(np.array([0,-2.5,2.5])), 0,centrefreqs[-1],
#            label='Manuf. specs Nominal mV/Pa $\pm 2.5 dB$')
plt.legend()

plt.subplot(313)

plt.plot(centrefreqs, snr_target, label='Target mic (Sennheiser)')
plt.plot(centrefreqs, snr_gras, label='GRAS mic')
plt.ylabel('Bandwise SNR, dB', fontsize=12)
plt.xlabel('Band centre-frequencies, Hz', fontsize=12)
plt.legend()
#%%
plt.figure()
plt.plot(centrefreqs, dB(senn_vrms_bandwise))
plt.ylabel('sensitivity - dBV/Pa')
plt.xlabel('Frequency, Hz')
plt.gca().set_xscale('log')

#%%
plt.figure()
plt.title('2024-06-24 audio')
plt.plot(centrefreqs, senn_vrms_bandwise*1e3)
plt.plot(centrefreqs, senn_vrms_bandwise*1e3*db_to_linear(6))
plt.ylabel('Target mic sensitivity, mV/Pa'); plt.xlabel('Freq., Hz')
plt.hlines(sens*1e3, centrefreqs[0], centrefreqs[-1], label='Tech specs expected', linestyles='dashed',)
plt.hlines([sens*1e3*db_to_linear(2.5), sens*1e3*db_to_linear(-2.5)], centrefreqs[0], centrefreqs[-1],
           linestyles='dotted', label='Tech. specs $\pm 2.5$ dB'); 
plt.ylim(10,100);plt.yticks(np.arange(0,100,10))
plt.legend();plt.grid()
#plt.savefig(f'brummgroup_sennheiser_2024-05-27.png')

plt.figure()
plt.title('2024-06-24 audio')
plt.plot(centrefreqs, senn_vrms_bandwise*1e3)
plt.plot(centrefreqs, senn_vrms_bandwise*1e3*db_to_linear(6))
plt.gca().set_xscale('log')
plt.ylabel('Target mic sensitivity, mV/Pa'); plt.xlabel('Freq., Hz')
plt.hlines(sens*1e3, centrefreqs[0], centrefreqs[-1], label='Tech specs expected', linestyles='dashed',)
plt.hlines([sens*1e3*db_to_linear(2.5), sens*1e3*db_to_linear(-2.5)], centrefreqs[0], centrefreqs[-1],
           linestyles='dotted', label='Tech. specs $\pm 2.5$ dB'); 
plt.ylim(10,100);plt.yticks(np.arange(0,100,10))
plt.legend();plt.grid()

#%% Having the current frequency response, let's now compensate for it, and calculate the overall 
# 
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
plt.plot(centrefreqs, freq_compensation_tgtmic+dB(bandwise_tgtmic))
plt.plot(centrefreqs, dB(bandwise_grasmic))
plt.plot(centrefreqs, freq_compensation_tgtmic)    

plt.ylabel('Target mic compensation level, dB');plt.xlabel('Freq., Hz')


#%%
# What is the sensitivity of the Sennheiser after 'flattening'?
compensated_sennheiser_vrms = db_to_linear(freq_compensation_tgtmic+dB(senn_vrms_bandwise))
compensated_sennheiser_aurms =  db_to_linear(freq_compensation_tgtmic+dB(tgtmic1_sensitivity_au))

plt.figure()
plt.title(f'2024-05-27 {target_micname}')
plt.plot(centrefreqs, senn_vrms_bandwise*1e3)
plt.plot(centrefreqs, compensated_sennheiser_vrms*1e3, label='post-compensation sensit.') 
plt.ylabel('Target mic sensitivity, mV/Pa'); plt.xlabel('Freq., Hz')
#plt.ylim(10,100);plt.yticks(np.arange(0,100,10))
plt.legend();plt.grid()
plt.savefig(f'2024-05-27 {target_micname}_sensitivity.png')
#%%
gras_rms = np.empty(len(calibmic_chunks))
for i,chunk in enumerate(calibmic_chunks):
    if i<len(calibmic_chunks)-1:
        gras_rms[i] = rms(chunk[int(fs*0.1):-int(fs*0.1)])
        
    else:
        gras_rms[i] = rms(chunk)


# And now see if compensating for the 

# Now make a compensation filter and pass it over the target audio 
desired = np.pad(db_to_linear(freq_compensation_tgtmic),
                                constant_values=[0,0], 
                                pad_width=1)
bands =  np.pad(centrefreqs,
                                constant_values=[0,fs*0.5], 
                                pad_width=1)
num_taps = 1023
#fir = signal.firls(num_taps, bands, desired, fs=fs)
fir = signal.firwin2(num_taps, bands, desired, fs=fs)
freq, response = signal.freqz(fir)

#%% Pass the Sennheiser audio through the compensation filter and calculate the SPL
sennheiser_comp_chunks = [signal.convolve(each, fir) for each in tgtmic1_chunks]

tgtmic_comp_rms = np.empty(len(sennheiser_comp_chunks))
for i,(chunk, compchunk) in enumerate(zip(tgtmic1_chunks, sennheiser_comp_chunks)):
    if i<len(tgtmic1_chunks)-1:
        tgtmic_comp_rms[i] = rms(compchunk[int(fs*0.1):-int(fs*0.1)])
        
    else:
        tgtmic_comp_rms[i] = rms(compchunk)
   
# The GRAS mic has a FLAT response, so one value for all 
# frequencies makes sense, and the RMS can be calculated considering EVERYTHING
gras_Pa = rms_clipPa*gras_rms/clip_rms_value 
gras_dbspl = dB((rms_clipPa*gras_rms/clip_rms_value)/20e-6) # dB SPL

#%% The IFFT method to 'compensate' the microphone frequency response. 
# Take the FFT of the audio, compensate each frequency bin by the mic response
# Take IFFT of the compensated FFT -> this is the freq. response compensated audio 

mic_comp_linear = db_to_linear(freq_compensation_tgtmic)


tgtmic_ifft_compaudio = []
tgtmic_ifft_rms = []

for i, audio_chunk in enumerate(tgtmic1_chunks):
    if i<len(tgtmic1_chunks)-1:
        eg_chunk = audio_chunk[int(fs*0.1):-int(fs*0.1)]        
    else:
        eg_chunk = audio_chunk.copy()  
    fft_x = np.fft.rfft(eg_chunk)
    freqs_x = np.fft.rfftfreq(fft_x.size*2 - 1, 1/fs)
    
    # Make an interpolation function 
    # create an interpolation function to compensate for the in-between freqs 
    compensation_interpfn = interp1d(centrefreqs, mic_comp_linear, 
                                     kind='cubic', bounds_error=False, fill_value=1)

    # interpolate the freq. response of the mic to intermediate freqs
    tgtmicsens_comp = compensation_interpfn(freqs_x)
    comp_fft = fft_x*tgtmicsens_comp
    comp_audio = np.fft.irfft(comp_fft)
    tgtmic_ifft_compaudio.append(comp_audio)
    tgtmic_ifft_rms.append(rms(comp_audio))

#%%
# If we're not interesting in creating a 'compensated' audio ...
# The sennheiser has an uneven freq. response. We must somehow incorporate this
# while calculating the RMS of the sound. 


sennheiser_Pa_rms = []
for i, audio_chunk in enumerate(tgtmic1_chunks):
    print(i)
    if i<len(tgtmic1_chunks)-1:
        eg_chunk = audio_chunk[int(fs*0.1):-int(fs*0.1)]        
    else:
        eg_chunk = audio_chunk.copy()  
    fft_x = np.fft.rfft(eg_chunk)
    freqs_x = np.fft.rfftfreq(fft_x.size*2 - 1, 1/fs)
    
    # Make an interpolation function 
    tgtmic_sens_interpfn = interp1d(centrefreqs, tgtmic1_sensitivity_au,
                                    kind='cubic', bounds_error=False, fill_value=np.min(fft_x))
    # interpolate the sensitivity of the mic to intermediate freqs
    tgtmicsens_interp = tgtmic_sens_interpfn(freqs_x)
    
    # now convert the FFT output to Pascal units 
    fft_Pa_x = fft_x/tgtmicsens_interp
    tgtmic_rmsPa = get_rms_from_fft(freqs_x, fft_Pa_x, freq_range=[0,20e3])
    sennheiser_Pa_rms.append(tgtmic_rmsPa)

sennheiser_dbspl = dB(np.array(sennheiser_Pa_rms)/20e-6)
#%%
# What is the received level calculated when no frequency response compensation is done

tgtmic_nocomp_dbspl = np.zeros(len(tgtmic1_chunks))
for i, audio_chunk in enumerate(tgtmic1_chunks):
    if i<len(tgtmic1_chunks)-1:
        eg_chunk = audio_chunk[int(fs*0.1):-int(fs*0.1)]        
    else:
        eg_chunk = audio_chunk.copy()  
    tgtmic_nocomp_Vrms = Vrms_clip*rms(eg_chunk)/clip_rms_value
    # convert from Vrms to Pa knowing the sensitivity
    tgtmic_nocomp_Parms = tgtmic_nocomp_Vrms/sens
    print(tgtmic_nocomp_Parms)
    tgtmic_nocomp_dbspl[i] = dB(tgtmic_nocomp_Parms/20e-6)

#%%
# Compare the received levels of the sounds with the reference mic & target mic

reclevel_ifftcomp = np.array(tgtmic_ifft_rms)/np.median(compensated_sennheiser_aurms)
reclevel_compsennheiser =  tgtmic_comp_rms/np.median(compensated_sennheiser_aurms)


sennheiser_comp_dbspl = dB(reclevel_compsennheiser/20e-6)
sennheiser_ifftcomp_dbspl = dB(reclevel_ifftcomp/20e-6)

sound_labels = ['sweep-short', 'sweep-med.', 'sweep-long', 'noise'] + ['tone'+str(i) for i in range(1,8)] + ['silence']
plt.figure(figsize=(12,5))
plt.plot(gras_dbspl, '-*', label='Ref. mic')
plt.plot(sennheiser_dbspl,'-*', label='Sens. comp. RMS')
plt.xticks(range(12), sound_labels, rotation=10, fontsize=12)
#plt.plot(sennheiser_comp_dbspl, '-*',label='Filter-based. RMS')
#plt.plot(sennheiser_ifftcomp_dbspl,'-*', label='IFFT-based RMS')
plt.plot(tgtmic_nocomp_dbspl, '-*', label='No compensation')
plt.legend(); plt.yticks(np.arange(10,80,6))
plt.ylabel('dB SPL, re 20$\mu$Pa rms', fontsize=12)
plt.grid()
plt.savefig('received-levels-comparison_.png')




