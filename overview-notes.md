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

## Summary of meeting 18/9/2024 

### Update from Lena's end - meeting with Kristian Beedlholm. 

After meeting Kristian Beedlholm (Uni Aarhus), Lena found out the following:

* KB says to use sweeps. Long noise is prone to echoes & reverb, not worth the effort
* KB shared his echo-filtering code, it works by convolving the recording with the template sweep, and then recovering the direct sweep audio through deconvolution and echo-suppression using the cross-correlation & some FFT magic. 
* KB has <= 1.5 dB band-wise variation across calibration runs of the same microphone
* KB & co however use a different calibration setup for their tags. A speaker is kept 40 cm from the microphone, and their sweeps starts at 10 kHz - which is far-field enough for ultrasound. 

### Updates from Thejasvi's end -
* Started segmenting out the sweeps. Just when plotting the short 3 ms sweep, TB could see how consistent the GRAS microphone recordings were. 
* However, when you start looking even at the sweeps - depending on frequency resolution it can get 'rough' (esp. when it's 100 Hz frequency resolution)


###  Post-discussion: echoes cause spikiness 
While discussing, we realised that the 3 ms sweeps provide a nice example to check the effect of background noise & echoes. Each sweep is sandwiched by 100 ms of silence on either side when played back. When recorded however, the preceding silence+sweep will only have background noise & signal. The sweep+following silence will have echoes too. If the echoes & reverb are the cause for the inconsistent spikiness - then the spectra with signal+proceeding silence should be weird -- and voila!

<img src="mic_resp_consistency/echoes_cause_spikiness.png" width="900" >

*Figure 2: Left: preceding silence + sweep, Middle: Only 3 ms sweep, Right: sweep+proceeding silence.  The presence of reflections in the recording clearly causes spikiness in the spectrum while also creating the weird variation in spectral power across frequency-bands in recordings made just ~30 minutes apart-- this is despite the geometry of the GRAS mic & speaker setup being replicated as well as possible!!*

### Next steps: 
* Only use the sweeps for any further analysis. They are much cleaner to handle and analyse. 
* First check how consistent the spectra are for the 3 ms sweeps within and across recordings. 

### Longer term steps:
* Implement the 'echo-cancellation' workflow that Kristian Beedlholm uses. This reminds TB of the workflows in the swept-sine room impulse-response experiments. 

	
Authors/contributors : Thejasvi Beleyur, Lena de Framond