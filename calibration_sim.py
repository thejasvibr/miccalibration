# -*- coding: utf-8 -*-
"""
Simulating a calibration workflow
=================================
Created on Sat May  4 09:22:13 2024

@author: theja
"""
import scipy.signal as signal 
import numpy as np 
import matplotlib.pyplot as plt 
np.random.seed(82319)
fs = 44100

durn = 0.003 # s
t = np.linspace(0, durn, int(fs*durn))
chirp = signal.chirp(t, 18000, t[-1], 200, 'linear')

# Your reference signal 
ref_signal =  chirp.copy() #np.random.normal(0,1e-2, 256)
ref_signal *= signal.windows.tukey(ref_signal.size, alpha=0.97)
padwidth = 100
ref_signal = np.pad(ref_signal, pad_width=100, constant_values=(0,0))
# The reference signal recorded on the calibration microphone
# the simulated calib microphone has a low sensitivity and thus the
# sound levels are much fainter

calibmic_audio = ref_signal*1e-2


#%% the target microphone.
# Simulate a weird frequency response by making a custom frequency filter
# Overall, let's say the target mic is 10x more sensitive than the calibration
# microphone

desired = np.array([1,1,0.5,0.75,0.25,0.1, 0.1, 1,1,1])
#desired *= 0.1
bands =  np.linspace(0, fs*0.5, len(desired))
num_taps = 1023
#fir = signal.firls(num_taps, bands, desired, fs=fs)
fir = signal.firwin2(num_taps, bands, desired, fs=fs)
freq, response = signal.freqz(fir)

plt.figure()
ax = plt.subplot(111)
for band, gains in zip(zip(bands[::2], bands[1::2]),

                           zip(desired[::2], desired[1::2])):

        ax.semilogy(band, np.maximum(gains, 1e-7), 'k--', linewidth=2)
plt.plot(bands, desired)
# Verify the FIR's match with the expected
# 
fft_fir = np.fft.rfft(fir)
fft_freqs = np.fft.rfftfreq(fir.size, d=1/fs)
plt.figure()
plt.plot(fft_freqs, 20*np.log10(abs(fft_fir)))
for bandstart, bandstop,  value in zip(bands[1:], bands[:-1], desired):
    plt.plot(np.mean([bandstart, bandstop]), 20*np.log10(value),'*')
    
# convolve the FIR with the ref signal to mimic target mic recording
# also make the mic about 10X more sensitive than the calibration mic
targetmic_audio = signal.convolve(ref_signal*1e-1, fir, 'same')
# normalise to the peak value


#%%
# Now compare the FFTs of the reference with that of the target microphone

calibmic_audio_in = calibmic_audio[padwidth:-padwidth]
calibmic_audio_in /= max(abs(calibmic_audio)) 
# normalise audio so only the frequency response is calculated.
targetmic_audio_in = targetmic_audio[padwidth:-padwidth]
targetmic_audio_in /= max(abs(targetmic_audio_in)) 

fft_refsignal = np.fft.rfft(calibmic_audio_in)
fft_targetmic = np.fft.rfft(targetmic_audio_in)

compensation_fft = fft_refsignal/fft_targetmic
fftfreqs = np.fft.rfftfreq(ref_signal[padwidth:-padwidth].size, d=1/fs)
# get the minimum power within the calibration frequency range
min_freq, max_freq = [1000,15000]
relevant_freqs_inds = np.logical_and(fftfreqs>=min_freq, fftfreqs<=max_freq)
compensation_fft_power = min(20*np.log10(abs(compensation_fft))[relevant_freqs_inds])
compensation_fft_power *= -1


# take the lowest value within the FFT power range and set that as the new 0
compensation_fft_amped = compensation_fft*10**(compensation_fft_power/20)


plt.figure()
plt.plot(fftfreqs, 20*np.log10(abs(compensation_fft)))
plt.plot(fftfreqs, 20*np.log10(abs(compensation_fft_amped)))

# Make the IFFT filter
freq_comp_filter = np.fft.irfft(compensation_fft_amped)
freq_comp_filter  = np.roll(freq_comp_filter, freq_comp_filter.size//2 +1)

plt.figure();plt.plot(freq_comp_filter)

#%% Make a frequency compensated version of the target mic audio
targetmic_freqcomped = signal.convolve(targetmic_audio_in, freq_comp_filter, 'same')

#%% make a composite to compare the freq-comp with raw audio of the target mic
composite_audio = np.concatenate((targetmic_freqcomped, np.zeros(441), targetmic_audio_in))
composite_audio = np.pad(composite_audio, pad_width=100, constant_values=[0,0])
composite_audio += np.random.normal(0,1e-4,composite_audio.size)
plt.figure()
spec, freqs, t, im = plt.specgram(composite_audio, Fs=fs, NFFT=64, noverlap=32)

def powerspec(X, **kwargs):
    fft_X = np.fft.rfft(X)
    fft_freqs = np.fft.rfftfreq(X.size, d=1/kwargs['fs'])
    return fft_freqs, 20*np.log10(abs(fft_X))

def maxnorm_powerspec(X, **kwargs):
    fftfreqs, spectrum = powerspec(X, **kwargs)
    spectrum -= np.max(spectrum)
    return fftfreqs, spectrum
def rms(X):
    return np.sqrt(np.mean(X**2))
#%%
plt.figure()
plt.title('Max normalised power spectra')
plt.plot(*maxnorm_powerspec(targetmic_freqcomped, fs=fs), label='freq. compensated')
plt.plot(*maxnorm_powerspec(targetmic_audio_in, fs=fs), label='raw target mic')
plt.plot(*maxnorm_powerspec(ref_signal, fs=fs), label='original signal')
plt.plot(*powerspec(fir, fs=fs), label='FIR freq response')


plt.figure()
plt.title('Power spectra')
plt.plot(*powerspec(targetmic_freqcomped, fs=fs), label='freq. compensated')
plt.plot(*powerspec(targetmic_audio_in, fs=fs), label='raw target mic')



plt.figure()
plt.subplot(211)
plt.plot(targetmic_freqcomped, label='compensated')
plt.plot(targetmic_audio_in, label='raw')
plt.plot(calibmic_audio[padwidth:-padwidth], label='calibration')
plt.legend()
plt.subplot(212)
plt.plot(ref_signal[padwidth:-padwidth])
plt.legend()


#%%
tonedurn = 0.2
t = np.linspace(0,tonedurn, int(fs*tonedurn))
tone1 = np.sin(2*np.pi*1e3*t)
tone2 = np.sin(2*np.pi*0.8e3*t)
tone = tone1 + tone2

tone *= signal.windows.hann(tone.size)
tonefft = np.fft.rfft(tone)
tone_rms = rms(tone)

freqs, spectrum = powerspec(tone, fs=fs)



plt.figure()
plt.plot(freqs, 10**(spectrum/20))

# testing Parseval's relation 
# http://www.dspguide.com/ch10/7.htm
coefficients = 10**(spectrum/20)
total_freq_power = np.sum(coefficients**2)/coefficients.size # no 2x for a real powered signal 
sig_squared = np.sum(tone**2)

def get_rms_from_spectrum(freqs, spectrum, **kwargs):
    '''Use Parseval's theorem to get the RMS level of each frequency component
    This only works for RFFT spectrums!!!
    
    '''
    minfreq, maxfreq = kwargs['freq_range']
    relevant_freqs = np.logical_and(freqs>=minfreq, freqs<=maxfreq)
    spectrum_copy = spectrum.copy()
    spectrum_copy[~relevant_freqs] = 0
    mean_sigsquared = np.sum(spectrum_copy**2)/spectrum.size
    root_mean_squared = np.sqrt(mean_sigsquared/(2*spectrum.size-1))
    return root_mean_squared
# bin width clarification https://stackoverflow.com/questions/10754549/fft-bin-width-clarification

bin_widths = fs/coefficients.size
rms_from_spec = get_rms_from_spectrum(freqs, coefficients, freq_range=[0,200])
















