# -*- coding: utf-8 -*-
"""
Are the microphone frequency responses consistent across sessions -SWEEP VERSION
================================================================================
Let's choose two sessions where we know the Sennheiser ME66 was recorded without
the windshield: 2024-08-19 and 2024-06-24.

The predecessor of this module is 'are_freq_responses_consistent.py'. The idea here
is to test if changing the analysed signal affects the microphone sensitivity calculated. 

Observations
~~~~~~~~~~~~




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
from freqresp_utils import extract_out_signalparts

#%% Load the calibration mic audio recordings 
rec_data = pd.read_csv('sweeps_audio_rec_data.csv')
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

#%%
# load the playback signal 
digital_pbk, fs = sf.read('..\multisignal_calibaudio.wav')
short_sweep = digital_pbk[int(fs*0.1):int(fs*(0.1+3e-3))]

resampling_factor = 96000/44100
short_sweep_highfs = signal.resample(short_sweep,
                                     int(short_sweep.size*resampling_factor))

#%% Generate the SPL vs frequency profile from the calibration microphone
gras_pbkrec_all = rec_by_mic.get_group(('GRAS1/4', 'playback_rec'))
# keep all those with valid linear sweep recordings
gras_pbkrec_all = gras_pbkrec_all.dropna(subset='sweepset_start')

#%%
all_spl_vs_freq = []

clips = []
for ind, row in gras_pbkrec_all.iterrows():
    row = gras_pbkrec_all.loc[ind,:]
    final_path = os.path.join('..', 'multisignal_recordings', row['session'], 
                              row['pure_filename'])
    audio, fs  = sf.read(final_path)

    if np.isnan(row['sweepset_start']):
        raise ValueError('this row should not be here...')
    else:
        audio_rec = audio[int(fs*row['sweepset_start']):,0]
    audio_gaincomp = audio_rec*10**(-row['gain_dB']/20)
    b,a = signal.butter(1, 100/(fs*0.5), 'high')
    audio_gaincomp = signal.filtfilt(b,a, audio_gaincomp)
    inds, segments = extract_out_signalparts(audio_gaincomp, fs)
    for i, segment in enumerate(segments[:1]):
        if fs == 96000:
            template = short_sweep_highfs
        elif fs==44100:
            template = short_sweep
        else:
            raise ValueError(f'{fs} is an unknown sampling rate')
            
        cc = signal.correlate(segment, template, 'same')
        index = np.argmax(cc)
        signal_start = int(index-template.size*0.5)
        signal_stop = signal_start + template.size
        rec_signal = segment[signal_start:signal_stop]
        
        freqbins, freq_rms = calc_native_freqwise_rms(rec_signal, fs)
        
        clips.append(rec_signal)
        spl_vs_freq = pd.DataFrame(data=None,
                                   columns=['session', 'filename', 'freq_bin', 'au_rms'], 
                                   index=range(freqbins.size))
        spl_vs_freq['freq_bin'] = freqbins
        spl_vs_freq['au_rms'] = freq_rms
        spl_vs_freq['session'] = row['session']
        spl_vs_freq['filename'] = row['pure_filename']
        spl_vs_freq['replicate_num'] = i
        all_spl_vs_freq.append(spl_vs_freq)

all_spl_vs_freq = pd.concat(all_spl_vs_freq)
all_spl_vs_freq['pa_rms'] = all_spl_vs_freq['au_rms']/grasrms_1Pa_avg
all_spl_vs_freq['dBspl'] = dB(all_spl_vs_freq['pa_rms']/20e-6)
#%%
plt.figure()
for key, subdf in all_spl_vs_freq.groupby(['session', 'filename']):
    subdf_so = subdf.sort_values(by='freq_bin', ascending=True)
    plt.plot(subdf_so['freq_bin'], subdf_so['dBspl'],'-*', label=key[0]+' '+key[1][-9:-4])
plt.legend()
plt.grid();plt.ylabel('dB SPL$_{rms}$, re 20$\mu$Pa', fontsize=12)
plt.xlabel(f'Frequency, Hz {(freqbins[0])} to {freqbins[-1]}', fontsize=12)
plt.title('only sweep')
plt.ylim(-30, 75)
#%% Load the target microphone recordings 
tgt_mic_recs = rec_by_mic.get_group(('SENNHEISER-ME66-ONLY', 'playback_rec'))


all_rms_vs_freq = []
for ind, row in tgt_mic_recs.iterrows():
    row = tgt_mic_recs.loc[ind,:]
    final_path = os.path.join('..', 'multisignal_recordings', row['session'], 
                              row['pure_filename'])
    audio, fs  = sf.read(final_path)
    
    if np.isnan(row['sweepset_start']):
        continue  
    else:
        audio_rec = audio[int(fs*row['sweepset_start']):,0]
        
   
    
    
    audio_gaincomp = audio_rec*10**(-row['gain_dB']/20)
    inds, segments = extract_out_signalparts(audio_gaincomp, fs)
    for i, segment in enumerate(segments[:1]):
        if fs == 96000:
            template = short_sweep_highfs
        elif fs==44100:
            template = short_sweep
        else:
            raise ValueError(f'{fs} is an unknown sampling rate')
            
        cc = signal.correlate(segment, template, 'same')
        index = np.argmax(cc)
        signal_start = int(index-template.size*0.5)
        signal_stop = signal_start + template.size
        rec_signal = segment[signal_start:signal_stop]
        
        freqbins, freq_rms = calc_native_freqwise_rms(rec_signal, fs)
        
        clips.append(rec_signal)
        rms_vs_freq = pd.DataFrame(data=None,
                                   columns=['session', 'filename', 'freq_bin', 'au_rms'], 
                                   index=range(freqbins.size))
        rms_vs_freq['freq_bin'] = freqbins
        rms_vs_freq['au_rms'] = freq_rms
        rms_vs_freq['session'] = row['session']
        rms_vs_freq['filename'] = row['pure_filename']
        rms_vs_freq['replicate_num'] = i
        all_rms_vs_freq.append(rms_vs_freq)

all_rms_vs_freq = pd.concat(all_rms_vs_freq)
all_rms_vs_freq['pa_rms'] = all_rms_vs_freq['au_rms']/grasrms_1Pa_avg
all_rms_vs_freq['dBspl'] = dB(all_rms_vs_freq['pa_rms']/20e-6)
#%%
# Does the speaker also continue to show the same kind of playback variation? 

plt.figure()
for key, subdf in all_rms_vs_freq.groupby(['session', 'filename']):
    subdf_so = subdf.sort_values(by='freq_bin', ascending=True)
    plt.plot(subdf_so['freq_bin'], dB(subdf_so['au_rms']),'-*', label=key[0]+' '+key[1][-9:-4])
plt.legend()
plt.grid();plt.ylabel('dB a.u. rms, re 1', fontsize=12)
plt.xlabel(f'Frequency, Hz {(freqbins[0])} to {freqbins[-1]}', fontsize=12)
#plt.title('Consistent noise spectra within a playback. \n Sometimes inconsistent noise spectra across playbacks')


#%% Generate the frequency rms of the target microphones across different sessions
rms_v_freq_avged_by_pbk = all_rms_vs_freq.groupby(['session', 'filename', 'freq_bin'])['au_rms'].mean()
tgt_rms_v_freq_avged = rms_v_freq_avged_by_pbk.reset_index(name='avg_au_rms')

tgt_rms_v_freq_avged_by_session = tgt_rms_v_freq_avged.groupby(['session'])

#%% Calib mic generate avg spectra of the recorded sounds
spl_vs_freq_by_pbk = all_spl_vs_freq.groupby(['session', 'filename', 'freq_bin'])['au_rms'].mean()
spl_vs_freq_by_pbk  = spl_vs_freq_by_pbk.reset_index(name='avg_au_rms')
spl_vs_freq_by_pbk['avg_pa_rms'] = spl_vs_freq_by_pbk['avg_au_rms']/grasrms_1Pa_avg

#%% 
# Now calculate the sensitivity based on each reference recording we have. 

tgtmic_sensitivity = []
for group_name, subdf in spl_vs_freq_by_pbk.groupby(['session', 'filename']):
    print(group_name)
    session_date, filename = group_name
    tgtmic_sens_df = pd.DataFrame(data=[], columns=['session', 'calibmic_filename',
                                                     'freq_bin', 'au_rms_perPa'], 
                                  index=range(subdf['freq_bin'].size))
    known_pa = subdf['avg_pa_rms'].reset_index(drop=True)
    
    relevant_tgtmic = tgt_rms_v_freq_avged_by_session.get_group((session_date,))
    tgtmic_rms = relevant_tgtmic['avg_au_rms'].reset_index(drop=True)
    
    tgtmic_sens_df['freq_bin'] = subdf['freq_bin'].reset_index(drop=True)
    tgtmic_sens_df['session'] = session_date
    tgtmic_sens_df.loc[:,'calibmic_filename'] = filename
    tgtmic_sens_df['tgtmic_filename'] = relevant_tgtmic['filename'].reset_index(drop=True)
    tgtmic_sens_df['au_rms_perPa'] = tgtmic_rms/known_pa
    tgtmic_sensitivity.append(tgtmic_sens_df)
    

all_tgtmic_sens = pd.concat(tgtmic_sensitivity)
    
#%%
plt.figure()
for group, subdf in all_tgtmic_sens.groupby(['session', 'calibmic_filename', 'tgtmic_filename']):
    print(group)
    plt.plot(subdf['freq_bin'], dB(subdf['au_rms_perPa']), label=group)

ylims = np.around(dB(np.percentile(all_tgtmic_sens['au_rms_perPa'], [0,100])))
plt.ylim(ylims[0]-3, ylims[1]+3)
plt.yticks(np.arange(ylims[0], ylims[1]+4, 2));
plt.grid();plt.ylabel('Sensitivity, dB( $\\frac{au_{rms}}{Pa}$) re 1', fontsize=12);plt.xlabel('Frequency, Hz', fontsize=12)
plt.vlines([2.5e3, 12.5e3], -44,-10, color='k', label='-20 dB limits')
plt.legend()
plt.gca().get_legend().set_title("Date , Calibration mic recording, Target mic recording")
plt.title('Sennheiser ME 66 shows consistent sensitivity for 3ms sweeps')
plt.xlim(0, 19.2e3)
plt.savefig(f'sennheiserme66_sweeps_Hz_separation.png')

plt.figure()
for group, subdf in all_tgtmic_sens.groupby(['session', 'calibmic_filename', 'tgtmic_filename']):
    print(group)
    plt.plot(subdf['freq_bin'], dB(subdf['au_rms_perPa']), label=group)

ylims = np.around(dB(np.percentile(all_tgtmic_sens['au_rms_perPa'], [0,100])))
plt.gca().set_xscale('log')
plt.ylim(ylims[0]-3, ylims[1]+3)
plt.yticks(np.arange(ylims[0], ylims[1]+4, 2));
plt.grid();plt.ylabel('Sensitivity, dB( $\\frac{au_{rms}}{Pa}$) re 1', fontsize=12);plt.xlabel('Frequency, Hz', fontsize=12)
plt.vlines([2.5e3, 12.5e3], -44,-10, color='k', label='-20 dB limits')
plt.legend()
plt.gca().get_legend().set_title("Date , Calibration mic recording, Target mic recording")
plt.title('Sennheiser ME 66 shows consistent sensitivity for 3ms sweeps')
plt.xlim(0, 19.2e3)

#%%
# Are we in the correct range, in dBV/Pa 
tascam_clip_vrms =  0.975 # Vrms clip
# convert aurms re 1 to aurms re Clip (1/sqrt(2))
clip_rms = 1/np.sqrt(2)
# Sennheiser ME 66 - specs link: https://assets.sennheiser.com/global-downloads/file/11897/SpecSheet_ME_66_EN.pdf
raise NotImplementedError(('Not yet complete -- see if you can recover 50 mVrms/Pa overall'))

for group, subdf in all_tgtmic_sens.groupby(['session', 'calibmic_filename', 'tgtmic_filename']):
    subdf['au_rms_reclip'] = subdf['au_rms_perPa']/clip_rms
    plt.plot(subdf['freq_bin'], dB(subdf['au_rms_perPa']), label=group)

#%%
# Get the 
rfft_digitalsweep = np.fft.rfft(short_sweep)
freqs_digitalsweep = np.fft.rfftfreq(short_sweep.size, 1/fs)
dB_powerspec = dB(abs(rfft_digitalsweep))

plt.figure()
plt.plot(freqs_digitalsweep, dB_powerspec)
plt.hlines(np.max(dB_powerspec)-12, 0, freqs_digitalsweep.max())
