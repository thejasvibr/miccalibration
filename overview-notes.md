# Overview of the mic-calibration 

## The current situation: when is a calibration 'good enough'? 
When we had the Brumm group retreat at Schloss Ringberg everything seemed nice and rosy. The SNR of recordings were good, we managed to get consistent-ish microphone sensitivity profiles, and were able to recover the dB SPL of the playback sounds correctly (when compared to the calibration GRAS microphone). 

As of now, I've been realising there're some weird quirks in the calibration workflow and measurement. To start with, Lena has always been concerned about the heavy variation in the microphone sensitivity from frequency-band to frequency-band. Irrespective of the actual frequency resolution you carry out the workflow for, there can be 3-10 dB jumps from one frequency-band to the next. Especially when you consider that we can't think of any physical reason why a single microphone should react so differently for on-axis sound that is just 50-100 Hz apart.

## Summary of meeting on 26/8/2024 

### Some findings: 

* Even just the 1 Pa (94 dB SPL) 1 kHz tones can vary by 0.5 dBrms in recorded level. This already sets the minimum expected variation in received level measurement. 

* The recorded sound had 3 linear sweeps (3, 5, 7 ms each), a 5 second long white noise sound, and a series of tones (1-14 kHz in 2 kHz jumps). The speaker produced too much varaition in playback levels for the tones, and so they are useless for our purposes. Each recording had at least 2-3 repeats of the playback signal. 
	* The white noise playbacks are overall consistent within a recording. However, the white noise playback itself can change over the course of an hour or so. And this can lead to 2-5 dB variation in band-wise calculated sensitivity - depending on which calibration mic recording is used as the reference'. 


	![](.//mic_resp_consistency//sennheiserme66_200_Hz_separation.png)
	
	Figure 1: *White-noise based average sensitivity measures on two different days, but using two different GRAS mic recordings as the reference. There is a broad consistency, but there can also be up to >= 4 dB within the same frequency band...*

### Next steps:

* While in discussion with Bioacoustics StackExchange contributor 'WMXZ', Lena found that we expect white-noise to be *on average* spectrally uniform, but not on a recording-basis. While this may be true, it still doesn't make sense that the white-noise recordings vary so much from band to band. We still expect some kind of consistency across the span of one hour..? Moreover, considering the fundamental principle of substitution - we *don't* expect major changes in recordings as we made sure the microphones were really in the same place w.r.t the speaker. However, white-noise recordings aren't necessarily reverberation free, and so may be this plays a big role too?

   * Maybe it makes sense to look into the linear sweeps. We expect them to be relatively flat (discounting speaker frequency response) - and we also don't expect any fancy stochasticity in its structure.










	
	
Authors/contributors : Thejasvi Beleyur, Lena de Framond